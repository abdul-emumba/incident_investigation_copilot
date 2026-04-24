from __future__ import annotations

from typing import Any, Generator

from .base import BaseAgent
from .models import AgentResult, IncidentReport

_SYSTEM = """You are generating the final incident investigation report for an SRE team.
Synthesize all evidence and verified hypotheses into a precise, actionable report.

Return JSON (no markdown):
{
  "root_cause": "Clear, specific, falsifiable root cause statement. Name the service, version, and mechanism.",
  "investigation_summary": "2-3 sentence narrative: what happened, why, and how it propagated.",
  "affected_services": ["auth-service", "orders-service", "gateway"],
  "evidence": [
    "Log evidence: TokenValidationError spike in auth-service logs starting 14:02",
    "Deployment evidence: v2.1.3 deployed to auth-service at 13:55 with a breaking change in token parsing",
    "Graph evidence: orders-service and gateway depend on auth-service via synchronous calls"
  ],
  "recommended_actions": [
    "Immediate: Roll back auth-service to v2.1.2",
    "Short-term: Add contract tests for token parsing to the auth-service CI pipeline",
    "Long-term: Implement circuit breaker on orders-service → auth-service dependency"
  ],
  "investigation_notes": "Any important caveats, data gaps, or remaining unknowns."
}"""


class ReportGenerator(BaseAgent):
    def run(self, context: dict[str, Any]) -> AgentResult:
        plan = context.get("plan")
        timeline_res = context.get("timeline")
        graph_res = context.get("graph")
        critic_res = context.get("critic")

        summary = _build_synthesis(plan, context.get("log_analysis"), timeline_res, graph_res, critic_res)
        data = self.ask_llm_json(_SYSTEM, summary)

        # Pull structured data that was already computed by prior agents
        timeline_events = timeline_res.data.events if (timeline_res and timeline_res.success) else []

        affected_services: list[str] = []
        if graph_res and graph_res.success:
            affected_services = [s.service_id for s in graph_res.data.affected_services]
        elif plan:
            affected_services = list(plan.focus_services)

        conflicting_signals: list[str] = []
        confidence_score = 0.5
        best_hyp = None
        if critic_res and critic_res.success:
            conflicting_signals = critic_res.data.conflicting_signals
            best_hyp = critic_res.data.best_hypothesis
            if best_hyp:
                confidence_score = best_hyp.final_confidence

        incident_id = plan.incident_id if plan else None

        if not data:
            root_cause = best_hyp.hypothesis.claim if best_hyp else "Root cause undetermined"
            report = IncidentReport(
                incident_id=incident_id,
                root_cause=root_cause,
                timeline=timeline_events,
                affected_services=affected_services,
                evidence=[],
                confidence_score=confidence_score,
                recommended_actions=["Gather more data and re-run investigation"],
                conflicting_signals=conflicting_signals,
                investigation_summary="Report synthesis failed; see raw agent outputs.",
            )
        else:
            # Prefer graph-computed services; fall back to LLM output
            if not affected_services:
                affected_services = data.get("affected_services", [])

            report = IncidentReport(
                incident_id=incident_id,
                root_cause=data.get(
                    "root_cause",
                    best_hyp.hypothesis.claim if best_hyp else "Undetermined",
                ),
                timeline=timeline_events,
                affected_services=affected_services,
                evidence=data.get("evidence", []),
                confidence_score=confidence_score,
                recommended_actions=data.get("recommended_actions", []),
                conflicting_signals=conflicting_signals,
                investigation_summary=data.get("investigation_summary", ""),
            )

        return AgentResult("report", True, report)

    def stream_narrative(self, context: dict[str, Any]) -> Generator[str, None, None]:
        """Stream a concise investigation narrative token-by-token using Groq streaming."""
        synthesis = _build_synthesis(
            context.get("plan"),
            context.get("log_analysis"),
            context.get("timeline"),
            context.get("graph"),
            context.get("critic"),
        )
        yield from self.stream_text(
            "You are an expert SRE writing a concise incident investigation narrative. "
            "In 4-6 sentences describe: the root cause, when it started, how it propagated, "
            "and the primary recommended action. Be specific — name services, versions, and timestamps.",
            synthesis,
        )


# synthesis helper

def _build_synthesis(plan, log_res, timeline_res, graph_res, critic_res) -> str:
    parts: list[str] = []

    if plan:
        parts.append(f"INCIDENT: {plan.incident_summary}")
        parts.append(f"ID: {plan.incident_id or 'unknown'}")
        parts.append(f"Services: {', '.join(plan.focus_services)}")
        parts.append(f"Window: {plan.time_window}")

    if log_res and log_res.success:
        lr = log_res.data
        parts.append(f"\nLOG ANOMALIES ({len(lr.anomalies)}):")
        for a in lr.anomalies[:6]:
            parts.append(
                f"  [{a.severity}] {a.service}: {a.error_type} "
                f"count={a.count} first={a.first_seen}"
            )

    if timeline_res and timeline_res.success:
        tr = timeline_res.data
        parts.append(f"\nTIMELINE ({len(tr.events)} events):")
        for e in tr.events[:12]:
            line = f"  {e.timestamp:<32} [{e.event_type}] {e.description}"
            if e.service:
                line += f" | {e.service}"
            parts.append(line)
        for t in tr.key_transitions[:4]:
            parts.append(f"  KEY: {t}")

    if graph_res and graph_res.success:
        gr = graph_res.data
        parts.append(f"\nDEPENDENCY IMPACT (graph_available={gr.graph_available}):")
        parts.append(f"  Critical path: {' → '.join(gr.critical_path)}")
        for s in gr.affected_services[:8]:
            parts.append(f"  {s.service_id} ({s.role}, depth={s.depth})")

    if critic_res and critic_res.success:
        cr = critic_res.data
        if cr.best_hypothesis:
            bh = cr.best_hypothesis
            parts.append(f"\nVERIFIED HYPOTHESIS ({bh.status}, confidence={bh.final_confidence}):")
            parts.append(f"  {bh.hypothesis.claim}")
            for ev in bh.evidence_for[:3]:
                parts.append(f"  FOR: {ev}")
            for ev in bh.evidence_against[:2]:
                parts.append(f"  AGAINST: {ev}")
            for cf in bh.conflicts_detected[:3]:
                parts.append(f"  CONFLICT: {cf}")
        if cr.conflicting_signals:
            parts.append(f"\nGLOBAL CONFLICTS:")
            for sig in cr.conflicting_signals:
                parts.append(f"  ⚠ {sig}")

    return "\n".join(parts) if parts else "No investigation evidence available."
