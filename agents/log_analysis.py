from __future__ import annotations

from typing import Any

from .base import BaseAgent
from .models import AgentResult, LogAnalysisResult, LogAnomaly

_SYSTEM = """You are an expert SRE analyzing production logs for anomalies and failure patterns.

Given log evidence retrieved from multiple sources, identify:
1. Error anomalies — sudden spikes, new error types, cascading failures, unusually high rates
2. Error clusters — groups of related errors that likely share the same root cause
3. Which services are involved

Source reliability (highest to lowest): logs > deployment records > event log > tickets > runbooks.
When fields are missing or inconsistent, use "unknown" rather than guessing.

Return JSON (no markdown):
{
  "anomalies": [
    {
      "service": "service-name or unknown",
      "error_type": "e.g. TokenValidationError",
      "first_seen": "ISO timestamp or unknown",
      "last_seen": "ISO timestamp or unknown",
      "count": 42,
      "severity": "P1 | P2 | P3 | unknown",
      "example_messages": ["log line 1", "log line 2"]
    }
  ],
  "error_clusters": {
    "auth_token_failures": ["auth-service", "orders-service"],
    "db_connection_pool": ["orders-db"]
  },
  "affected_services": ["auth-service", "orders-service"],
  "analysis_summary": "One paragraph narrative of what the logs reveal"
}"""


class LogAnalysisAgent(BaseAgent):
    def run(self, context: dict[str, Any]) -> AgentResult:
        plan = context.get("plan")
        query: str = context.get("query", "")

        queries = self._pick_queries(plan, query)
        evidence_parts: list[str] = []

        for q in queries:
            answer, sources = self.ranked_rag_query(q)
            # Annotate each piece of evidence with highest-priority source
            top_src = sources[0].get("source_dataset", "?") if sources else "?"
            evidence_parts.append(f"[Source: {top_src}]\nQ: {q}\nA: {answer}")

        combined = "\n\n---\n\n".join(evidence_parts)
        data = self.ask_llm_json(_SYSTEM, f"Log evidence:\n{combined}")

        if not data:
            return AgentResult(
                "log_analysis", False,
                LogAnalysisResult([], {}, [], evidence_parts),
                error="LLM returned no structured data",
            )

        anomalies = [
            LogAnomaly(
                service=a.get("service", "unknown"),
                error_type=a.get("error_type", "unknown"),
                first_seen=a.get("first_seen", "unknown"),
                last_seen=a.get("last_seen", "unknown"),
                count=int(a["count"]) if str(a.get("count", "0")).isdigit() else 0,
                severity=a.get("severity", "unknown"),
                example_messages=a.get("example_messages", []),
            )
            for a in data.get("anomalies", [])
        ]

        return AgentResult(
            "log_analysis", True,
            LogAnalysisResult(
                anomalies=anomalies,
                error_clusters=data.get("error_clusters", {}),
                affected_services=data.get("affected_services", []),
                raw_evidence=evidence_parts,
            ),
        )

    def _pick_queries(self, plan, fallback: str) -> list[str]:
        if plan:
            for step in plan.steps:
                if step.agent == "log_analysis" and step.queries:
                    return step.queries
        return [
            f"What errors and anomalies appear in the production logs? {fallback}",
            "Which services show ERROR or WARN log spikes?",
            "What are the most frequent error messages and their timestamps?",
        ]
