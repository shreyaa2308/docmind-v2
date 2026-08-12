"""Environment-backed configuration for DocMind v2."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _int("PORT", 8000)
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "docmind_v2")
    document_store_path: str = os.getenv("DOCUMENT_STORE_PATH", "data/documents.json")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "*")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    vision_model: str = os.getenv("VISION_MODEL", "gpt-4o-mini")
    chunk_size: int = _int("CHUNK_SIZE", 800)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 100)
    sparse_weight: float = _float("SPARSE_WEIGHT", 0.4)
    dense_weight: float = _float("DENSE_WEIGHT", 0.6)
    rerank_top_n: int = _int("RERANK_TOP_N", 3)
    eval_faithfulness_min: float = _float("EVAL_FAITHFULNESS_MIN", 0.75)
    eval_answer_relevancy_min: float = _float("EVAL_ANSWER_RELEVANCY_MIN", 0.75)
    eval_context_precision_min: float = _float("EVAL_CONTEXT_PRECISION_MIN", 0.70)
    eval_context_recall_min: float = _float("EVAL_CONTEXT_RECALL_MIN", 0.70)


settings = Settings()
