from __future__ import annotations

from typing import Any

from .base import BaseAgent
from .models import AffectedService, AgentResult, GraphAnalysisResult

_SYSTEM = """You are analyzing a service dependency graph to determine cascading failure blast radius.
Given dependency chains and impacted services, synthesize the critical propagation path.

Return JSON (no markdown):
{
  "critical_path": ["service-a", "service-b", "service-c"],
  "cascade_description": "auth-service → orders-service → gateway (via sync HTTP call)"
}"""


class GraphAgent(BaseAgent):
    def run(self, context: dict[str, Any]) -> AgentResult:
        plan = context.get("plan")
        log_result = context.get("log_analysis")
        query: str = context.get("query", "")

        services = self._collect_services(plan, log_result)

        if not self.graph:
            return self._rag_fallback(services, query)

        try:
            return self._run_neo4j(services, query)
        except Exception as exc:
            return self._rag_fallback(services, query, error=str(exc))

    # ── Neo4j path ────────────────────────────────────────────────────────────

    def _run_neo4j(self, services: list[str], query: str) -> AgentResult:
        affected: list[AffectedService] = []
        seen_ids: set[str] = set()
        dependency_chains: list[list[str]] = []

        # Seed services are root-cause candidates
        for svc in services:
            if svc not in seen_ids:
                affected.append(AffectedService(
                    service_id=svc, role="root_cause_candidate",
                    depth=0, has_circuit_breaker=False, has_fallback=False,
                ))
                seen_ids.add(svc)

        # Impact analysis: who is broken if this service fails?
        for svc in services[:3]:
            try:
                for rec in self.graph.impact_analysis(svc):
                    sid = rec.get("id", "")
                    if sid and sid not in seen_ids:
                        affected.append(AffectedService(
                            service_id=sid,
                            role="transitively_affected",
                            depth=int(rec.get("depth", 1)),
                            has_circuit_breaker=bool(rec.get("has_circuit_breaker", False)),
                            has_fallback=bool(rec.get("has_fallback", False)),
                            team=rec.get("team"),
                        ))
                        seen_ids.add(sid)
            except Exception:
                pass

            # Direct callers
            try:
                for rec in self.graph.direct_dependents(svc):
                    sid = rec.get("id", "")
                    if sid and sid not in seen_ids:
                        affected.append(AffectedService(
                            service_id=sid,
                            role="directly_affected",
                            depth=1,
                            has_circuit_breaker=bool(rec.get("circuit_breaker", False)),
                            has_fallback=bool(rec.get("fallback")) if rec.get("fallback") else False,
                        ))
                        seen_ids.add(sid)
            except Exception:
                pass

        # Dependency paths between pairs of failing services (multi-hop)
        for i, src in enumerate(services):
            for dst in services[i + 1:]:
                try:
                    path_records = self.graph.dependency_path(src, dst)
                    if path_records:
                        chain = [path_records[0]["from_node"]] + [r["to_node"] for r in path_records]
                        dependency_chains.append(list(dict.fromkeys(chain)))  # preserve order, dedup
                except Exception:
                    pass

        # Shared single-points-of-failure
        shared_deps: list[str] = []
        try:
            shared_deps = [r.get("id", "") for r in self.graph.find_shared_dependencies() if r.get("id")]
        except Exception:
            pass

        # Ask LLM to synthesise the critical path from what we found
        critical_path = services[:]
        if affected or dependency_chains:
            chains_txt = "\n".join(" → ".join(c) for c in dependency_chains[:5])
            affected_txt = "\n".join(
                f"{a.service_id} (role={a.role}, depth={a.depth}, cb={a.has_circuit_breaker})"
                for a in affected[:12]
            )
            llm_data = self.ask_llm_json(
                _SYSTEM,
                f"Services under investigation: {services}\n\n"
                f"Dependency chains:\n{chains_txt or 'none found'}\n\n"
                f"Impacted services:\n{affected_txt}",
            )
            critical_path = llm_data.get("critical_path", services)

        return AgentResult(
            "graph", True,
            GraphAnalysisResult(
                affected_services=affected,
                critical_path=critical_path,
                shared_dependencies=shared_deps,
                dependency_chains=dependency_chains,
                graph_available=True,
            ),
        )

    #RAG fallback
    def _rag_fallback(
        self, services: list[str], query: str, error: str | None = None
    ) -> AgentResult:
        svc_str = ", ".join(services) if services else "unknown"
        answer, _ = self.rag_query(
            f"What services depend on {svc_str}? "
            f"What is the blast radius if {svc_str} fails? {query}"
        )

        affected = [
            AffectedService(
                service_id=svc, role="root_cause_candidate",
                depth=0, has_circuit_breaker=False, has_fallback=False,
            )
            for svc in services
        ]
        return AgentResult(
            "graph", True,
            GraphAnalysisResult(
                affected_services=affected,
                critical_path=services,
                shared_dependencies=[],
                dependency_chains=[],
                graph_available=False,
            ),
            error=error,
        )

    # Helpers
    def _collect_services(self, plan, log_result) -> list[str]:
        seen: dict[str, None] = {}
        if plan:
            for svc in plan.focus_services:
                seen[svc] = None
        if log_result and log_result.success:
            for svc in log_result.data.affected_services:
                seen[svc] = None
        return list(seen)
