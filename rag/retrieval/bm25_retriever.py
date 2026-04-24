import logging
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

CANONICAL_SERVICES = {
    "auth", "payments", "orders", "gateway", "redis",
    "orders-db", "inventory", "search", "notifications",
}


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _boosted_tokens(doc: Document) -> List[str]:
    """Return a token list with field-boosted repetitions."""
    tokens = _tokenize(doc.page_content)
    meta = doc.metadata

    boosted: List[str] = []

    # base tokens × 1 (kept as-is; title/message/symptom tokens are part of page_content)
    boosted.extend(tokens)

    # incident_id × 3 total (add 2 more copies)
    inc_id = meta.get("incident_id") or ""
    if inc_id:
        inc_tokens = _tokenize(str(inc_id))
        boosted.extend(inc_tokens * 2)

    # service × 2 total (add 1 more copy)
    service = meta.get("service") or ""
    if service:
        svc_tokens = _tokenize(str(service))
        boosted.extend(svc_tokens)

    # severity / level × 2 total (add 1 more copy)
    for field in ("severity", "level"):
        val = meta.get(field) or ""
        if val:
            boosted.extend(_tokenize(str(val)))

    # title / message / symptom tokens already in page_content at ×1;
    # add an extra 0.5 pass by appending them once more for effective ×1.5
    # (BM25 works on integer token counts so we approximate with whole copies)
    boosted.extend(tokens)

    return boosted


def build_bm25_index(
    docs: List[Document],
) -> "BM25Retriever":
    corpus = [_boosted_tokens(d) for d in docs]
    bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
    return BM25Retriever(docs=docs, bm25=bm25, corpus=corpus)


class BM25Retriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    docs: List[Document] = Field(default_factory=list)
    bm25: Any = Field(default=None)
    corpus: List[List[str]] = Field(default_factory=list)
    top_k: int = Field(default=40)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # pair (score, index) and sort descending
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = scored[: self.top_k]

        results = []
        for idx, score in top:
            doc = self.docs[idx]
            enriched_meta = {**doc.metadata, "bm25_score": float(score)}
            results.append(Document(page_content=doc.page_content, metadata=enriched_meta))

        logger.debug("BM25 returned %d docs for query: %.60s", len(results), query)
        return results
