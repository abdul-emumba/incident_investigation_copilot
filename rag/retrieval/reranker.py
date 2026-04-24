import logging
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        logger.info("Loading cross-encoder model %s ...", RERANKER_MODEL)
        self.model = CrossEncoder(RERANKER_MODEL)
        logger.info("Cross-encoder loaded")

    def rerank(self, query: str, docs: List[Document], top_k: int = 8) -> List[Document]:
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        scored_docs = sorted(
            zip(scores, docs), key=lambda x: x[0], reverse=True
        )

        results = []
        for score, doc in scored_docs[:top_k]:
            enriched_meta = {**doc.metadata, "cross_encoder_score": float(score)}
            results.append(Document(page_content=doc.page_content, metadata=enriched_meta))

        logger.info(
            "Reranker: input=%d docs, output=%d docs (top_k=%d)",
            len(docs), len(results), top_k,
        )
        return results
