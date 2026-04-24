from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamEvent:
    """
    A single event emitted by Orchestrator.stream_investigate().

    kind values:
        "agent_start"  – an agent has begun running (data=None)
        "agent_done"   – an agent has finished    (data=AgentResult)
        "complete"     – the whole investigation is done (data=InvestigationResult)
    """
    kind: str        # agent_start | agent_done | complete
    agent: str       # planner | memory | log_analysis | timeline | graph | hypothesis | critic | report
    data: Any = None
    error: str | None = None


# Ordered list of agents shown in the UI progress panel
PIPELINE_AGENTS: list[str] = [
    "planner",
    "memory",
    "log_analysis",
    "timeline",
    "graph",
    "hypothesis",
    "critic",
    "report",
]

AGENT_LABELS: dict[str, str] = {
    "planner":      "Planner",
    "memory":       "Incident Memory",
    "log_analysis": "Log Analysis",
    "timeline":     "Timeline",
    "graph":        "Graph / Dependencies",
    "hypothesis":   "Hypothesis",
    "critic":       "Critic",
    "report":       "Report Generator",
}

AGENT_DESCRIPTIONS: dict[str, str] = {
    "planner":      "Breaks the query into investigation steps",
    "memory":       "Retrieves similar past incidents and extracts recurring patterns",
    "log_analysis": "Detects anomalies and clusters related errors",
    "timeline":     "Builds the chronological sequence of events",
    "graph":        "Traces cascading failures through the dependency graph",
    "hypothesis":   "Generates ranked root-cause explanations",
    "critic":       "Verifies claims and surfaces conflicting signals",
    "report":       "Synthesises the final incident report",
}
