from __future__ import annotations

from typing import Any

from .base import BaseAgent
from .models import AgentResult, CriticResult, VerifiedHypothesis

_SYSTEM = """You are a critical reviewer verifying incident root cause hypotheses for an SRE team.

For each hypothesis, search the provided evidence for support and contradictions.

Pay special attention to:
- Source conflicts: does a ticket claim X while logs show Y? Prefer logs.
- Temporal inconsistency: does the deployment timestamp actually precede the first error?
- Missing evidence: the hypothesis claims X but no log/ticket/deployment record confirms it.
- Multi-hop plausibility: does the error propagation path match the dependency graph?

Return JSON (no markdown):
{
  "verified_hypotheses": [
    {
      "hypothesis_id": 1,
      "status": "confirmed | refuted | uncertain",
      "evidence_for": [
        "Log at 14:02 shows TokenValidationError immediately after v2.1.3 deploy at 13:55"
      ],
      "evidence_against": [
        "Ticket INC-042 incorrectly blames database latency; no DB logs support this"
      ],
      "final_confidence": 0.89,
      "conflicts_detected": [
        "Ticket says DB slowdown was root cause, but logs show auth errors before any DB errors"
      ]
    }
  ],
  "conflicting_signals": [
    "Ticket INC-042 and production logs disagree on which service failed first"
  ],
  "best_hypothesis_id": 1
}"""


class CriticAgent(BaseAgent):
    def run(self, context: dict[str, Any]) -> AgentResult:
        hyp_result = context.get("hypothesis")
        query: str = context.get("query", "")

        if not hyp_result or not hyp_result.success:
            return AgentResult(
                "critic", False,
                CriticResult([], [], None),
                error="No hypotheses available to verify",
            )

        hypotheses = hyp_result.data.hypotheses
        verification_evidence: list[str] = []

        # Per-hypothesis evidence gathering
        for h in hypotheses[:3]:
            answer, sources = self.ranked_rag_query(
                f"Is there evidence for or against: '{h.claim}'?"
            )
            top = sources[0].get("source_dataset", "?") if sources else "?"
            verification_evidence.append(
                f"[Source: {top}] Hypothesis {h.id}: {h.claim}\n"
                f"Evidence found: {answer}"
            )

        # Explicit conflict query
        conflict_ans, _ = self.ranked_rag_query(
            "Do tickets and production logs give conflicting explanations "
            f"for what caused the incident? {query}"
        )
        verification_evidence.append(f"Conflict check:\n{conflict_ans}")

        # Source-ranking check: does the higher-priority evidence support the hypothesis?
        ranking_ans, _ = self.ranked_rag_query(
            "What do the production logs (highest trust) say about the root cause "
            "compared to what incident tickets say?"
        )
        verification_evidence.append(f"Source ranking check:\n{ranking_ans}")

        hyp_lines = "\n".join(
            f"Hypothesis {h.id} (confidence={h.confidence}): {h.claim}"
            for h in hypotheses
        )
        evidence_text = "\n\n---\n\n".join(verification_evidence)
        data = self.ask_llm_json(
            _SYSTEM,
            f"Hypotheses to verify:\n{hyp_lines}\n\n"
            f"Verification evidence:\n{evidence_text}",
        )

        if not data:
            return AgentResult(
                "critic", False,
                CriticResult([], [], None),
                error="LLM returned no structured verification",
            )

        h_map = {h.id: h for h in hypotheses}
        verified: list[VerifiedHypothesis] = []

        for vh in data.get("verified_hypotheses", []):
            hid = vh.get("hypothesis_id", 1)
            hyp = h_map.get(hid) or (hypotheses[0] if hypotheses else None)
            if hyp is None:
                continue
            verified.append(VerifiedHypothesis(
                hypothesis=hyp,
                status=vh.get("status", "uncertain"),
                evidence_for=vh.get("evidence_for", []),
                evidence_against=vh.get("evidence_against", []),
                final_confidence=float(vh.get("final_confidence", hyp.confidence)),
                conflicts_detected=vh.get("conflicts_detected", []),
            ))

        best_id = data.get("best_hypothesis_id", 1)
        best = next((v for v in verified if v.hypothesis.id == best_id), verified[0] if verified else None)

        return AgentResult(
            "critic", True,
            CriticResult(
                verified_hypotheses=verified,
                conflicting_signals=data.get("conflicting_signals", []),
                best_hypothesis=best,
            ),
        )
