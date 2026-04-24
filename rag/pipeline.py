import logging
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import tiktoken
from dotenv import load_dotenv
from groq import Groq
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document

from ingestion.loaders import (
    load_incident_tickets,
    load_production_logs,
    load_deployment_records,
    load_runbooks,
    load_incident_event_log,
    load_incident_responses,
)
from ingestion.chunker import build_all_chunks
from retrieval.bm25_retriever import build_bm25_index, BM25Retriever
from retrieval.vector_retriever import load_embeddings, build_faiss_store, load_faiss_store
from retrieval.reranker import Reranker
from storage.persistence import (
    FAISS_INDEX_DIR,
    storage_exists,
    check_lock_file,
    write_lock_file,
    save_bm25,
    load_bm25,
    save_chunks_metadata,
)

load_dotenv()

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
CANONICAL_SERVICES = {
    "auth", "payments", "orders", "gateway", "redis",
    "orders-db", "inventory", "search", "notifications",
}
MAX_CONTEXT_TOKENS = 6000
TIKTOKEN_ENCODING = "cl100k_base"


def _load_system_prompt() -> str:
    return (PROMPTS_DIR / "system_prompt.txt").read_text()


def _count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return len(enc.encode(text))


def _parse_filters(query: str) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}

    inc_match = re.search(r"\bINC-\d+\b", query, re.IGNORECASE)
    if inc_match:
        filters["incident_id"] = inc_match.group(0).upper()

    for svc in CANONICAL_SERVICES:
        if svc.lower() in query.lower():
            filters["service"] = svc
            break

    sev_match = re.search(r"\b(P1|P2|P3)\b", query, re.IGNORECASE)
    if sev_match:
        filters["severity"] = sev_match.group(1).upper()

    if re.search(r"\bbreaking\b", query, re.IGNORECASE):
        filters["breaking_change"] = True

    level_match = re.search(r"\b(ERROR|WARN)\b", query)
    if level_match:
        filters["level"] = level_match.group(1)

    return filters


def _apply_filters(docs: List[Document], filters: Dict[str, Any]) -> List[Document]:
    if not filters:
        return docs

    result = []
    for doc in docs:
        meta = doc.metadata
        match = True

        if "incident_id" in filters:
            doc_inc = str(meta.get("incident_id") or "").upper()
            if doc_inc != filters["incident_id"]:
                match = False

        if match and "service" in filters:
            doc_svc = str(meta.get("service") or "").lower()
            if filters["service"].lower() not in doc_svc:
                match = False

        if match and "severity" in filters:
            doc_sev = str(meta.get("severity") or "").upper()
            if doc_sev != filters["severity"]:
                match = False

        if match and "breaking_change" in filters:
            if not meta.get("breaking_change"):
                match = False

        if match and "level" in filters:
            doc_lvl = str(meta.get("level") or "").upper()
            if doc_lvl != filters["level"]:
                match = False

        if match:
            result.append(doc)

    return result if result else docs  # fall back to unfiltered if nothing passes


def _assemble_context(docs: List[Document]) -> str:
    lines = []
    total_tokens = 0
    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    for i, doc in enumerate(docs):
        meta = doc.metadata
        source_dataset = meta.get("source_dataset", "unknown")
        service = meta.get("service") or "N/A"
        incident_id = meta.get("incident_id") or "N/A"
        timestamp = meta.get("timestamp") or "N/A"

        header = f"[SOURCE {i + 1}: {source_dataset} | {service} | {incident_id} | {timestamp}]"
        block = f"{header}\n{doc.page_content}\n---"
        block_tokens = len(enc.encode(block))

        if total_tokens + block_tokens > MAX_CONTEXT_TOKENS:
            logger.warning("Context truncated at source %d to stay within %d tokens", i + 1, MAX_CONTEXT_TOKENS)
            break

        lines.append(block)
        total_tokens += block_tokens

    return "\n\n".join(lines)


class HybridRagPipeline:
    def __init__(self):
        self._docs: Optional[List[Document]] = None
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._faiss_store = None
        self._embeddings = None
        self._reranker: Optional[Reranker] = None
        self._ensemble: Optional[EnsembleRetriever] = None
        self._groq_client: Optional[Groq] = None
        self._system_prompt: str = _load_system_prompt()

    # build and load index
    def build_index(self) -> None:
        logger.info("=== Building index from source datasets ===")

        tickets = load_incident_tickets()
        logs = load_production_logs()
        deployments = load_deployment_records()
        runbooks = load_runbooks()
        event_log = load_incident_event_log()
        responses_text = load_incident_responses()

        self._docs, counts = build_all_chunks(tickets, logs, deployments, runbooks, event_log, responses_text)

        print("\n--- Chunk counts per dataset ---")
        for dataset, count in counts.items():
            print(f"  {dataset}: {count}")
        print(f"  TOTAL: {len(self._docs)}\n")

        # Embeddings + FAISS
        t0 = time.time()
        self._embeddings = load_embeddings()
        self._faiss_store = build_faiss_store(self._docs, self._embeddings)
        logger.info("Embedding + FAISS build took %.1fs", time.time() - t0)

        # BM25
        t0 = time.time()
        self._bm25_retriever = build_bm25_index(self._docs)
        logger.info("BM25 index build took %.1fs", time.time() - t0)

        # Persist
        self._faiss_store.save_local(str(FAISS_INDEX_DIR))
        save_bm25(self._bm25_retriever)
        save_chunks_metadata(self._docs)
        write_lock_file()

        self._init_ensemble()
        self._init_reranker()
        self._init_groq()

        logger.info("=== Index build complete ===")

    #  Index load                                                          
    def load_index(self) -> None:
        if not storage_exists():
            raise FileNotFoundError("Index not found. Run with --build-index first.")

        lock_exists, hash_matches = check_lock_file()
        if lock_exists and not hash_matches:
            logger.warning(
                "Source data has changed since last index build. "
                "Run with --build-index to rebuild."
            )

        print("Loaded index from storage — skipping ingestion")
        logger.info("Loading FAISS and BM25 from storage ...")

        self._embeddings = load_embeddings()
        self._faiss_store = load_faiss_store(str(FAISS_INDEX_DIR), self._embeddings)

        bm25_data = load_bm25()
        self._bm25_retriever = BM25Retriever(
            docs=bm25_data["docs"],
            bm25=bm25_data["bm25"],
            corpus=bm25_data["corpus"],
        )

        self._init_ensemble()
        self._init_reranker()
        self._init_groq()

    def auto_load(self) -> None:
        if storage_exists():
            self.load_index()
        else:
            logger.info("No storage found — building index ...")
            self.build_index()

    # init helpers
    def _init_ensemble(self) -> None:
        vector_retriever = self._faiss_store.as_retriever(search_kwargs={"k": 40})
        self._ensemble = EnsembleRetriever(
            retrievers=[self._bm25_retriever, vector_retriever],
            weights=[0.4, 0.6],
        )
        logger.info("EnsembleRetriever initialised (BM25=0.4, vector=0.6)")

    def _init_reranker(self) -> None:
        self._reranker = Reranker()

    def _init_groq(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set — LLM calls will fail")
        self._groq_client = Groq(api_key=api_key)


    # query method
    def query(self, question: str, top_k: int = 8) -> Dict[str, Any]:
        if self._ensemble is None:
            raise RuntimeError("Index not loaded. Call load_index() or auto_load() first.")

        start = time.time()
        filters = _parse_filters(question)
        if filters:
            logger.info("Parsed filters: %s", filters)

        # Retrieval
        t0 = time.time()
        merged_docs = self._ensemble.invoke(question)
        retrieve_ms = (time.time() - t0) * 1000
        logger.info(
            "Retrieval: %d merged docs in %.0fms (filters: %s)",
            len(merged_docs), retrieve_ms, filters or "none",
        )

        # Post-filter
        if filters:
            filtered_docs = _apply_filters(merged_docs, filters)
            # pad with unfiltered results if filtered set is small
            if len(filtered_docs) < top_k:
                filtered_set = {id(d) for d in filtered_docs}
                for d in merged_docs:
                    if id(d) not in filtered_set:
                        filtered_docs.append(d)
                    if len(filtered_docs) >= 80:
                        break
            merged_docs = filtered_docs

        # Take up to 80 for reranker input
        rerank_input = merged_docs[:80]

        # Re-rank
        t0 = time.time()
        top_docs = self._reranker.rerank(question, rerank_input, top_k=top_k)
        rerank_ms = (time.time() - t0) * 1000
        logger.info(
            "BM25: ~%d docs, Vector: ~%d docs, After rerank: %d docs (%.0fms)",
            int(len(rerank_input) * 0.4),
            int(len(rerank_input) * 0.6),
            len(top_docs),
            rerank_ms,
        )

        # Assemble context
        context = _assemble_context(top_docs)

        # LLM call
        t0 = time.time()
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        chat_response = self._groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        llm_ms = (time.time() - t0) * 1000
        logger.info("LLM call took %.0fms", llm_ms)

        answer = chat_response.choices[0].message.content
        total_ms = (time.time() - start) * 1000

        return {
            "question": question,
            "answer": answer,
            "sources": [doc.metadata for doc in top_docs],
            "latency_ms": round(total_ms, 2),
        }
