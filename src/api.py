"""FastAPI entrypoint for live document ingestion and question answering."""

import json
from pathlib import Path
import shutil
import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from langchain_core.documents import Document

from src.chain import answer_question, build_chain
from src.config import settings
from src.ingestion import ingest_pdf
from src.retriever import build_retriever
from src.vectorstore import index_documents

PROCESSED_DOCS_PATH = Path("data/processed/processed_docs.json")
_all_docs: list[Document] = []
_retriever = None
_chain = None
_rebuild_lock = threading.Lock()

app = FastAPI(title="DocMind v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_processed_docs() -> list[Document]:
    if not PROCESSED_DOCS_PATH.exists():
        return []
    records = json.loads(PROCESSED_DOCS_PATH.read_text(encoding="utf-8"))
    return [Document(**record) for record in records]


def _save_processed_docs(documents: list[Document]) -> None:
    """Atomically persist the BM25 corpus for the next API process."""
    PROCESSED_DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    temporary_path = PROCESSED_DOCS_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(PROCESSED_DOCS_PATH)


def _rebuild_retriever_and_chain() -> None:
    """Atomically replace live BM25/dense retrieval and LCEL components."""
    global _chain, _retriever
    with _rebuild_lock:
        if not _all_docs:
            _retriever = None
            _chain = None
            return
        retriever = build_retriever(_all_docs)
        _retriever = retriever
        _chain = build_chain(retriever)


@app.on_event("startup")
def initialize_retrieval() -> None:
    """Restore the sparse corpus and make persisted uploads queryable at startup."""
    global _all_docs
    with _rebuild_lock:
        _all_docs = _load_processed_docs()
    _rebuild_retriever_and_chain()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "docmind-v2"}


@app.get("/")
def frontend():
    return FileResponse(Path("frontend/index.html"))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / Path(file.filename).name
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        documents = ingest_pdf(destination)
        if not documents:
            raise ValueError("No text, tables, or embedded images could be extracted.")
        ids = index_documents(documents)
        with _rebuild_lock:
            _all_docs.extend(documents)
            _save_processed_docs(_all_docs)
        _rebuild_retriever_and_chain()
        return {"filename": destination.name, "chunks_indexed": len(ids)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> dict:
    try:
        if _retriever is None or _chain is None:
            raise RuntimeError("No documents have been ingested yet. Upload a PDF to /ingest first.")
        return answer_question(request.question, retriever=_retriever, chain=_chain)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
