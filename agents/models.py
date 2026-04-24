from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Planner
@dataclass
class InvestigationStep:
    step_id: int
    agent: str
    objective: str
    queries: list[str] = field(default_factory=list)


@dataclass
class InvestigationPlan:
    incident_summary: str
    focus_services: list[str]
    time_window: str
    incident_id: str | None
    steps: list[InvestigationStep]


# Log Analysis
@dataclass
class LogAnomaly:
    service: str
    error_type: str
    first_seen: str
    last_seen: str
    count: int
    severity: str
    example_messages: list[str] = field(default_factory=list)


@dataclass
class LogAnalysisResult:
    anomalies: list[LogAnomaly]
    error_clusters: dict[str, list[str]]   # cluster_name → [service_ids]
    affected_services: list[str]
    raw_evidence: list[str]


# Timeline
@dataclass
class TimelineEvent:
    timestamp: str
    event_type: str   # deployment | error | alert | ticket | recovery | config_change | other
    description: str
    service: str | None
    source: str       # logs | tickets | deployment_records | event_log | runbooks


@dataclass
class TimelineResult:
    events: list[TimelineEvent]
    key_transitions: list[str]
    raw_evidence: list[str]


# Graph
@dataclass
class AffectedService:
    service_id: str
    role: str          # root_cause_candidate | directly_affected | transitively_affected
    depth: int
    has_circuit_breaker: bool
    has_fallback: bool
    team: str | None = None


@dataclass
class GraphAnalysisResult:
    affected_services: list[AffectedService]
    critical_path: list[str]
    shared_dependencies: list[str]
    dependency_chains: list[list[str]]
    graph_available: bool = True


# Hypothesis
@dataclass
class Hypothesis:
    id: int
    claim: str
    services_involved: list[str]
    confidence: float
    supporting_evidence: list[str] = field(default_factory=list)


@dataclass
class HypothesisResult:
    hypotheses: list[Hypothesis]
    primary_hypothesis: Hypothesis | None
    raw_evidence: list[str]


# Critic
@dataclass
class VerifiedHypothesis:
    hypothesis: Hypothesis
    status: str          # confirmed | refuted | uncertain
    evidence_for: list[str]
    evidence_against: list[str]
    final_confidence: float
    conflicts_detected: list[str] = field(default_factory=list)


@dataclass
class CriticResult:
    verified_hypotheses: list[VerifiedHypothesis]
    conflicting_signals: list[str]
    best_hypothesis: VerifiedHypothesis | None


# Report
@dataclass
class IncidentReport:
    incident_id: str | None
    root_cause: str
    timeline: list[TimelineEvent]
    affected_services: list[str]
    evidence: list[str]
    confidence_score: float
    recommended_actions: list[str]
    conflicting_signals: list[str]
    investigation_summary: str


# Incident Memory
@dataclass
class SimilarIncident:
    incident_id: str
    query: str
    root_cause: str
    affected_services: list[str]
    confidence: float
    similarity_score: float
    investigation_summary: str
    timestamp: str


@dataclass
class MemoryResult:
    similar_incidents: list[SimilarIncident]
    pattern_insights: list[str]


# Shared
@dataclass
class AgentResult:
    agent_name: str
    success: bool
    data: Any
    error: str | None = None
