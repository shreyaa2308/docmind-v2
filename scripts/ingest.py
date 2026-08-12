"""CLI entrypoint for indexing a PDF into DocMind v2.

Usage:
    python scripts/ingest.py data/raw/example.pdf
    python scripts/ingest.py  # uses the sole PDF in data/raw/
"""

import argparse
from pathlib import Path
import sys

# Allow direct execution (``python scripts/ingest.py``) from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.ingestion import ingest_pdf
from src.retriever import save_documents
from src.vectorstore import index_documents


def _pdf_from_default_directory() -> Path:
    candidates = sorted(Path("data/raw").glob("*.pdf"))
    if len(candidates) != 1:
        raise SystemExit(
            "Pass a PDF path explicitly, or put exactly one .pdf file in data/raw/. "
            f"Found {len(candidates)}."
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and index a PDF into DocMind v2.")
    parser.add_argument("pdf", nargs="?", type=Path, help="Path to the PDF to ingest")
    args = parser.parse_args()
    pdf = args.pdf or _pdf_from_default_directory()
    if not pdf.is_file():
        raise SystemExit(f"PDF not found: {pdf}")
    documents = ingest_pdf(pdf)
    if not documents:
        raise SystemExit("No extractable text, tables, or embedded images were found.")
    ids = index_documents(documents)
    save_documents(documents)
    print(f"Indexed {len(ids)} chunks from {pdf}.")


if __name__ == "__main__":
    main()
