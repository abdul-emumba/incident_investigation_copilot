from __future__ import annotations

from typing import Any

from .base import BaseAgent
from .models import AgentResult, TimelineEvent, TimelineResult

_SYSTEM = """You are an SRE reconstructing a precise incident timeline from heterogeneous sources.

Evidence may come from production logs, deployment records, incident tickets, and event logs.
Sources have different reliability: logs > deployment records > event log > tickets > runbooks.

Your job:
- Extract every discrete event (deployments, first errors, alert firings, ticket creation, remediation steps)
- Sort them chronologically
- Identify key transition points (e.g. "deployment completed → first errors appear 7 min later")
- Handle missing/inconsistent timestamps: mark them "unknown" rather than fabricating values
- Flag if ticket timestamps conflict with log timestamps for the same event

Return JSON (no markdown):
{
  "events": [
    {
      "timestamp": "ISO timestamp or unknown",
      "event_type": "deployment | error | alert | ticket | recovery | config_change | other",
      "description": "concise description of what happened",
      "service": "service-name or null",
      "source": "logs | tickets | deployment_records | event_log | runbooks"
    }
  ],
  "key_transitions": [
    "13:55 – Deployment v2.1.3 of auth-service completed",
    "14:02 – First authentication errors observed in auth-service logs (7 min after deploy)"
  ]
}

Sort events by timestamp ascending. Place "unknown" timestamps last."""


class TimelineAgent(BaseAgent):
    def run(self, context: dict[str, Any]) -> AgentResult:
        plan = context.get("plan")
        query: str = context.get("query", "")

        queries = self._pick_queries(plan, query)
        evidence_parts: list[str] = []

        for q in queries:
            answer, sources = self.ranked_rag_query(q)
            top_src = sources[0].get("source_dataset", "?") if sources else "?"
            evidence_parts.append(f"[Source: {top_src}]\nQ: {q}\nA: {answer}")

        combined = "\n\n---\n\n".join(evidence_parts)
        data = self.ask_llm_json(_SYSTEM, f"Evidence:\n{combined}")

        if not data:
            return AgentResult(
                "timeline", False,
                TimelineResult([], [], evidence_parts),
                error="LLM returned no structured data",
            )

        raw_events = data.get("events", [])
        events = [
            TimelineEvent(
                timestamp=e.get("timestamp", "unknown"),
                event_type=e.get("event_type", "other"),
                description=e.get("description", ""),
                service=e.get("service"),
                source=e.get("source", "unknown"),
            )
            for e in raw_events
        ]

        # Sort: known timestamps first, then unknowns
        events.sort(key=lambda e: (e.timestamp == "unknown", e.timestamp))

        return AgentResult(
            "timeline", True,
            TimelineResult(
                events=events,
                key_transitions=data.get("key_transitions", []),
                raw_evidence=evidence_parts,
            ),
        )

    def _pick_queries(self, plan, fallback: str) -> list[str]:
        if plan:
            for step in plan.steps:
                if step.agent == "timeline" and step.queries:
                    return step.queries
        return [
            f"What is the timeline of events for this incident? {fallback}",
            "What deployments were made before the incident started? What versions were deployed?",
            "When were the first errors detected in logs? When were incident tickets created?",
            "What remediation actions were taken and when did the incident resolve?",
        ]
