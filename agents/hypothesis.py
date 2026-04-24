from __future__ import annotations

from typing import Any

from .base import BaseAgent
from .models import AgentResult, Hypothesis, HypothesisResult

_SYSTEM = """You are a senior SRE generating ranked root cause hypotheses for an incident.

Synthesize evidence from logs, timeline, and the dependency graph.

Reasoning guidelines:
- Prefer hypotheses that explain the EARLIEST anomaly in the timeline.
- A deployment just before the first error is strong causal evidence.
- A service failure at depth=0 that cascades to depth=1, 2, … fits a top-down hypothesis.
- If logs and tickets disagree, note the conflict and discount the weaker source (tickets < logs).
- Produce 2-4 hypotheses ranked by confidence descending.

Return JSON (no markdown):
{
  "hypotheses": [
    {
      "id": 1,
      "claim": "Specific, falsifiable root cause statement",
      "services_involved": ["auth-service", "orders-service"],
      "confidence": 0.87,
      "supporting_evidence": [
        "Deployment v2.1.3 completed at 13:55, first errors at 14:02",
        "TokenValidationError appears in 100% of auth-service ERROR logs after deploy"
      ]
    }
  ],
  "primary_hypothesis_id": 1,
  "reasoning": "Brief explanation of why hypothesis 1 is ranked first"
}"""


class HypothesisAgent(BaseAgent):
    def run(self, context: dict[str, Any]) -> AgentResult:
        plan = context.get("plan")
        query: str = context.get("query", "")

        # Combine structured evidence from prior agents
        evidence_summary = _build_evidence_summary(
            context.get("log_analysis"),
            context.get("timeline"),
            context.get("graph"),
        )

        # Supplement with targeted RAG queries
        rag_parts: list[str] = []
        for q in self._pick_queries(plan, query):
            answer, sources = self.ranked_rag_query(q)
            top = sources[0].get("source_dataset", "?") if sources else "?"
            rag_parts.append(f"[Source: {top}]\nQ: {q}\nA: {answer}")

        full_context = (
            f"{evidence_summary}\n\n"
            f"=== SUPPLEMENTAL RAG EVIDENCE ===\n"
            + "\n\n---\n\n".join(rag_parts)
        )
        data = self.ask_llm_json(_SYSTEM, full_context)

        if not data or not data.get("hypotheses"):
            fallback = Hypothesis(
                id=1,
                claim=f"Root cause undetermined from available evidence for: {query}",
                services_involved=[],
                confidence=0.1,
            )
            return AgentResult(
                "hypothesis", False,
                HypothesisResult([fallback], fallback, rag_parts),
                error="LLM returned no hypotheses",
            )

        hypotheses = [
            Hypothesis(
                id=h.get("id", i + 1),
                claim=h.get("claim", ""),
                services_involved=h.get("services_involved", []),
                confidence=float(h.get("confidence", 0.5)),
                supporting_evidence=h.get("supporting_evidence", []),
            )
            for i, h in enumerate(data["hypotheses"])
        ]

        primary_id = data.get("primary_hypothesis_id", 1)
        primary = next((h for h in hypotheses if h.id == primary_id), hypotheses[0])

        return AgentResult(
            "hypothesis", True,
            HypothesisResult(
                hypotheses=hypotheses,
                primary_hypothesis=primary,
                raw_evidence=rag_parts,
            ),
        )

    def _pick_queries(self, plan, fallback: str) -> list[str]:
        if plan:
            for step in plan.steps:
                if step.agent == "hypothesis" and step.queries:
                    return step.queries
        return [
            f"What is the most likely root cause of this incident? {fallback}",
            "What deployment or configuration change triggered the failures?",
        ]


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_evidence_summary(log_res, timeline_res, graph_res) -> str:
    parts: list[str] = []

    if log_res and log_res.success:
        lr = log_res.data
        parts.append(f"=== LOG ANOMALIES ({len(lr.anomalies)}) ===")
        for a in lr.anomalies[:6]:
            parts.append(
                f"  [{a.severity}] {a.service}: {a.error_type} "
                f"(count={a.count}, first={a.first_seen}, last={a.last_seen})"
            )
        if lr.error_clusters:
            for cluster, svcs in list(lr.error_clusters.items())[:4]:
                parts.append(f"  Cluster '{cluster}': {svcs}")

    if timeline_res and timeline_res.success:
        tr = timeline_res.data
        parts.append(f"\n=== TIMELINE ({len(tr.events)} events) ===")
        for e in tr.events[:12]:
            line = f"  [{e.timestamp}] [{e.event_type}] {e.description}"
            if e.service:
                line += f" ({e.service})"
            line += f" [src: {e.source}]"
            parts.append(line)
        for t in tr.key_transitions[:4]:
            parts.append(f"  KEY: {t}")

    if graph_res and graph_res.success:
        gr = graph_res.data
        parts.append(f"\n=== DEPENDENCY BLAST RADIUS ===")
        if gr.critical_path:
            parts.append(f"  Critical path: {' → '.join(gr.critical_path)}")
        for s in gr.affected_services[:10]:
            cb = "✓CB" if s.has_circuit_breaker else "✗CB"
            fb = "✓FB" if s.has_fallback else "✗FB"
            parts.append(f"  {s.service_id} role={s.role} depth={s.depth} {cb} {fb}")
        if gr.shared_dependencies:
            parts.append(f"  Shared SPOFs: {gr.shared_dependencies[:5]}")

    return "\n".join(parts) if parts else "No structured evidence from prior agents."
