from __future__ import annotations

from typing import Any

from .base import BaseAgent
from .incident_memory_store import IncidentMemoryStore
from .models import AgentResult, MemoryResult, SimilarIncident

_SYSTEM = """You are an incident pattern analyst. You have access to summaries of past incidents.
Given a list of similar historical incidents, identify recurring patterns, common root causes,
and actionable insights that could accelerate the current investigation.

Return a single JSON object — no markdown, no explanation:
{
  "pattern_insights": [
    "Insight 1: ...",
    "Insight 2: ..."
  ]
}

Rules:
- Generate 2-4 concrete, specific insights based only on the provided incidents.
- Focus on: recurring services, repeated root causes, deployment patterns, or resolution strategies.
- If no clear patterns exist, return an empty list.
- Each insight must be one clear sentence."""


class MemoryAgent(BaseAgent):
    """Queries incident memory for similar past incidents and extracts patterns."""

    def __init__(self, *args: Any, memory_store: IncidentMemoryStore, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._store = memory_store

    def run(self, context: dict[str, Any]) -> AgentResult:
        query: str = context.get("query", "")

        similar = self._store.find_similar(query, top_k=3)

        if not similar:
            return AgentResult(
                "memory",
                True,
                MemoryResult(similar_incidents=[], pattern_insights=[]),
            )

        pattern_insights = self._extract_patterns(query, similar)

        return AgentResult(
            "memory",
            True,
            MemoryResult(similar_incidents=similar, pattern_insights=pattern_insights),
        )

    def _extract_patterns(self, query: str, similar: list[SimilarIncident]) -> list[str]:
        summaries = "\n".join(
            f"- [{s.incident_id}] Root cause: {s.root_cause}. "
            f"Services: {', '.join(s.affected_services) or 'unknown'}. "
            f"Summary: {s.investigation_summary[:200]}"
            for s in similar
        )
        user_msg = (
            f"Current investigation query: {query}\n\n"
            f"Similar past incidents:\n{summaries}"
        )
        data = self.ask_llm_json(_SYSTEM, user_msg)
        return data.get("pattern_insights", []) if isinstance(data, dict) else []
