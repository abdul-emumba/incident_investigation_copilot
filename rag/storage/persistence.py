import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).parent.parent / "storage"
FAISS_INDEX_DIR = STORAGE_DIR / "faiss_index"
BM25_CORPUS_PATH = STORAGE_DIR / "bm25_corpus.pkl"
CHUNKS_METADATA_PATH = STORAGE_DIR / "chunks_metadata.json"
LOCK_FILE_PATH = STORAGE_DIR / "index.lock"

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "rag"


def _compute_source_hash() -> str:
    """SHA-256 of all source file modification times."""
    paths = sorted(DATA_DIR.rglob("*"))
    mtime_strs = [f"{p}:{p.stat().st_mtime}" for p in paths if p.is_file()]
    combined = "\n".join(mtime_strs)
    return hashlib.sha256(combined.encode()).hexdigest()


def write_lock_file() -> None:
    LOCK_FILE_PATH.write_text(_compute_source_hash())
    logger.info("Written index.lock")


def check_lock_file() -> Tuple[bool, bool]:
    """Returns (lock_exists, hash_matches)."""
    if not LOCK_FILE_PATH.exists():
        return False, False
    stored = LOCK_FILE_PATH.read_text().strip()
    current = _compute_source_hash()
    return True, stored == current


def storage_exists() -> bool:
    return (
        FAISS_INDEX_DIR.exists()
        and BM25_CORPUS_PATH.exists()
        and CHUNKS_METADATA_PATH.exists()
        and LOCK_FILE_PATH.exists()
    )


def save_bm25(bm25_retriever) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(
            {"bm25": bm25_retriever.bm25, "corpus": bm25_retriever.corpus, "docs": bm25_retriever.docs},
            f,
        )
    logger.info("Saved BM25 corpus to %s", BM25_CORPUS_PATH)


def load_bm25():
    if not BM25_CORPUS_PATH.exists():
        raise FileNotFoundError("Index not found. Run with --build-index first.")
    with open(BM25_CORPUS_PATH, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded BM25 corpus from %s", BM25_CORPUS_PATH)
    return data


def save_chunks_metadata(docs: List[Document]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    metadata_list = [
        {"page_content": d.page_content, "metadata": d.metadata} for d in docs
    ]
    with open(CHUNKS_METADATA_PATH, "w") as f:
        json.dump(metadata_list, f, default=str)
    logger.info("Saved chunks metadata (%d chunks) to %s", len(docs), CHUNKS_METADATA_PATH)


def load_chunks_metadata() -> List[Document]:
    if not CHUNKS_METADATA_PATH.exists():
        raise FileNotFoundError("Index not found. Run with --build-index first.")
    with open(CHUNKS_METADATA_PATH) as f:
        data = json.load(f)
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    logger.info("Loaded %d chunks from metadata", len(docs))
    return docs
