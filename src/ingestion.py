"""PDF extraction: text, Markdown tables, and vision captions become RAG chunks."""

import base64
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from src.config import settings


def _table_to_markdown(table: list[list[object | None]]) -> str:
    """Serialize a pdfplumber table into a compact Markdown table."""
    rows = [[str(cell or "").replace("\n", " ").strip() for cell in row] for row in table]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    return "\n".join([
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
        *["| " + " | ".join(row) + " |" for row in rows[1:]],
    ])


def _caption_image(image_bytes: bytes, extension: str) -> str:
    """Ask the configured multimodal model for a factual, retrieval-friendly caption."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to caption embedded images.")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.vision_model,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this document image factually. Include labels, figures, and relevant text; do not invent details."},
                {"type": "image_url", "image_url": {"url": f"data:image/{extension};base64,{encoded}"}},
            ],
        }],
    )
    return response.choices[0].message.content or "Image with no generated caption."


def _page_documents(pdf_path: Path) -> Iterable[Document]:
    source = pdf_path.name
    pdf = fitz.open(pdf_path)
    try:
        with pdfplumber.open(pdf_path) as plumber_pdf:
            for index, page in enumerate(pdf):
                page_number = index + 1
                metadata = {"source": source, "page": page_number, "content_type": "text"}
                text = page.get_text("text").strip()
                if text:
                    yield Document(page_content=text, metadata=metadata)

                for table_index, table in enumerate(plumber_pdf.pages[index].extract_tables()):
                    markdown = _table_to_markdown(table)
                    if markdown:
                        yield Document(
                            page_content=f"Table {table_index + 1} (page {page_number}):\n{markdown}",
                            metadata={**metadata, "content_type": "table", "table_index": table_index + 1},
                        )

                for image_index, image in enumerate(page.get_images(full=True)):
                    xref = image[0]
                    extracted = pdf.extract_image(xref)
                    if not extracted:
                        continue
                    caption = _caption_image(extracted["image"], extracted.get("ext", "png"))
                    yield Document(
                        page_content=f"Image {image_index + 1} (page {page_number}) caption: {caption}",
                        metadata={**metadata, "content_type": "image", "image_index": image_index + 1},
                    )
    finally:
        pdf.close()


def ingest_pdf(pdf_path: str | Path) -> list[Document]:
    """Extract a PDF and split every extracted item into 800/100 character chunks."""
    path = Path(pdf_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be ingested.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(_page_documents(path)))
