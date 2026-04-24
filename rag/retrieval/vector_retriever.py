import logging
import sys
from typing import List

import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024


class BGEEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings wrapper that prepends the BGE query prefix."""

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        return super().embed_query(prefixed)


def load_embeddings() -> BGEEmbeddings:
    try:
        embeddings = BGEEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Loaded embedding model %s", MODEL_NAME)
        return embeddings
    except Exception as e:
        logger.error("Failed to load %s: %s", MODEL_NAME, e)
        print(
            f"Failed to load {MODEL_NAME}.\n"
            "Delete ~/.cache/huggingface and retry to re-download."
        )
        sys.exit(1)


def build_faiss_store(docs: List[Document], embeddings: BGEEmbeddings) -> FAISS:
    """Embed docs and build a FAISS index with inner-product similarity."""
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    logger.info("Embedding %d documents with %s ...", len(texts), MODEL_NAME)
    store = FAISS.from_documents(docs, embeddings)
    logger.info("FAISS index built with %d vectors", store.index.ntotal)
    return store


def load_faiss_store(index_dir: str, embeddings: BGEEmbeddings) -> FAISS:
    store = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("Loaded FAISS index from %s (%d vectors)", index_dir, store.index.ntotal)
    return store
