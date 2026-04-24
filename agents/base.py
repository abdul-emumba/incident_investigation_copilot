from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generator

from groq import Groq

if TYPE_CHECKING:
    from graph.queries import GraphQueries
    from rag.pipeline import HybridRagPipeline

from .models import AgentResult

# Source priority: logs carry the most ground-truth signal; runbooks the least.
_SOURCE_RANK: dict[str, float] = {
    "log": 1.0,
    "deployment": 0.85,
    "event": 0.75,
    "ticket": 0.65,
    "runbook_issue": 0.45,
    "runbook_step": 0.40,
    "response": 0.35,
}

_MODEL = "llama-3.3-70b-versatile"
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class BaseAgent(ABC):
    def __init__(
        self,
        rag: HybridRagPipeline,
        graph: GraphQueries | None,
        llm: Groq,
    ) -> None:
        self.rag = rag
        self.graph = graph
        self.llm = llm

    # LLM helpers
    def ask_llm(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = self.llm.chat.completions.create(
            model=_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

    def ask_llm_json(self, system: str, user: str) -> dict:
        """Call the LLM and parse the response as JSON. Returns {} on failure."""
        raw = self.ask_llm(
            system + "\n\nRespond ONLY with valid JSON. No markdown fences, no commentary.",
            user,
        )
        raw = _JSON_FENCE.sub("", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Last resort: extract the outermost {...} or [...]
            for pattern in (r"\{.*\}", r"\[.*\]"):
                m = re.search(pattern, raw, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group())
                    except json.JSONDecodeError:
                        pass
        return {}

    #RAG helpers
    def rag_query(self, question: str) -> tuple[str, list[dict]]:
        """Returns (answer_text, list_of_source_metadata_dicts)."""
        result = self.rag.query(question)
        return result["answer"], result.get("sources", [])

    def ranked_rag_query(self, question: str) -> tuple[str, list[dict]]:
        """Same as rag_query but sources are sorted by reliability tier."""
        answer, sources = self.rag_query(question)
        sources_sorted = sorted(
            sources,
            key=lambda s: _SOURCE_RANK.get(s.get("doc_type", ""), 0.3),
            reverse=True,
        )
        return answer, sources_sorted

    def stream_text(self, system: str, user: str) -> Generator[str, None, None]:
        """Stream LLM response token-by-token using Groq's streaming API."""
        stream = self.llm.chat.completions.create(
            model=_MODEL,
            stream=True,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    #Abstract interface
    @abstractmethod
    def run(self, context: dict[str, Any]) -> AgentResult:
        """Execute the agent given the shared investigation context."""
