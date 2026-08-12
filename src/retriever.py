"""Hybrid BM25 + Qdrant retrieval with FlashRank reranking."""

import json
from pathlib import Path

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import settings
from src.vectorstore import vector_store


def save_documents(documents: list[Document]) -> None:
    """Persist chunks locally so sparse BM25 retrieval survives API restarts."""
    path = Path(settings.document_store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_documents()
    records = [{"page_content": d.page_content, "metadata": d.metadata} for d in existing + documents]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def load_documents() -> list[Document]:
    path = Path(settings.document_store_path)
    if not path.exists():
        return []
    return [Document(**record) for record in json.loads(path.read_text(encoding="utf-8"))]


def build_retriever(documents: list[Document] | None = None):
    """Build a hybrid retriever over supplied documents or the persisted corpus."""
    documents = load_documents() if documents is None else documents
    if not documents:
        raise RuntimeError("No documents have been ingested yet. Upload a PDF to /ingest first.")
    sparse = BM25Retriever.from_documents(documents)
    sparse.k = 12
    dense = vector_store().as_retriever(search_kwargs={"k": 12})
    hybrid = EnsembleRetriever(
        retrievers=[sparse, dense], weights=[settings.sparse_weight, settings.dense_weight]
    )
    reranker = FlashrankRerank(top_n=3)
    return ContextualCompressionRetriever(base_retriever=hybrid, base_compressor=reranker)
