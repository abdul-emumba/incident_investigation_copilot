from __future__ import annotations

from typing import Any

from .base import BaseAgent
from .models import AgentResult, InvestigationPlan, InvestigationStep

_SYSTEM = """You are an incident investigation planner for an SRE team.
Decompose the user's incident query into a structured investigation plan.

Available specialist agents:
- log_analysis   : detects error spikes, anomalies, and clusters related errors
- timeline       : builds the chronological sequence of events
- graph          : traces dependency chains and blast radius across services
- hypothesis     : generates ranked root cause explanations
- critic         : verifies claims against evidence and flags conflicts

Return a single JSON object — no markdown, no explanation:
{
  "incident_summary": "one-sentence description of the incident",
  "incident_id": "INC-XXXX or null",
  "focus_services": ["service-a", "service-b"],
  "time_window": "e.g. 2024-01-15 14:00 – 14:30 UTC",
  "steps": [
    {
      "step_id": 1,
      "agent": "<agent_name>",
      "objective": "what this step must determine",
      "queries": ["specific question 1", "specific question 2"]
    }
  ]
}

Rules:
- Extract the incident ID (INC-XXXX) from the query if present, else null.
- List all services mentioned or implied.
- Write 2-3 targeted RAG queries per step that will surface the most relevant evidence.
- Steps must cover all five agents in the order listed above."""


class PlannerAgent(BaseAgent):
    def run(self, context: dict[str, Any]) -> AgentResult:
        query: str = context.get("query", "")
        data = self.ask_llm_json(_SYSTEM, f"User query: {query}")

        if not data:
            plan = self._fallback_plan(query)
            return AgentResult("planner", True, plan)

        steps = [
            InvestigationStep(
                step_id=s.get("step_id", i + 1),
                agent=s.get("agent", ""),
                objective=s.get("objective", ""),
                queries=s.get("queries", [query]),
            )
            for i, s in enumerate(data.get("steps", []))
        ]
        # Ensure all required agents appear even if LLM omitted some
        present = {s.agent for s in steps}
        for agent in ("log_analysis", "timeline", "graph", "hypothesis", "critic"):
            if agent not in present:
                steps.append(InvestigationStep(
                    step_id=len(steps) + 1,
                    agent=agent,
                    objective=f"Investigate: {query}",
                    queries=[query],
                ))

        plan = InvestigationPlan(
            incident_summary=data.get("incident_summary", query),
            focus_services=data.get("focus_services", []),
            time_window=data.get("time_window", "unknown"),
            incident_id=data.get("incident_id"),
            steps=steps,
        )
        return AgentResult("planner", True, plan)

    def _fallback_plan(self, query: str) -> InvestigationPlan:
        return InvestigationPlan(
            incident_summary=query,
            focus_services=[],
            time_window="unknown",
            incident_id=None,
            steps=[
                InvestigationStep(1, "log_analysis", "Detect anomalies in production logs", [
                    f"What errors appear in the logs? {query}",
                    "Which services show error spikes?",
                ]),
                InvestigationStep(2, "timeline", "Build chronological event sequence", [
                    f"What is the incident timeline? {query}",
                    "When did the first errors occur relative to deployments?",
                ]),
                InvestigationStep(3, "graph", "Identify affected services and blast radius", [
                    f"Which services are affected and how are they connected? {query}",
                ]),
                InvestigationStep(4, "hypothesis", "Generate root cause hypotheses", [
                    f"What is the root cause? {query}",
                    "What deployment or change triggered the failures?",
                ]),
                InvestigationStep(5, "critic", "Verify hypotheses against evidence", [
                    f"Is there evidence supporting or contradicting the root cause? {query}",
                ]),
            ],
        )
