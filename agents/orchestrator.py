from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from dotenv import load_dotenv

# Path bootstrap
# agents/ lives one level below the project root.
# - project root must be in sys.path so `from graph.queries import …` resolves.
# - rag/ must be in sys.path so pipeline.py's bare `from ingestion.loaders import …`
#   resolves (rag/ has no __init__.py; it uses flat, non-prefixed imports).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAG_DIR = _PROJECT_ROOT / "rag"
for _p in (str(_PROJECT_ROOT), str(_RAG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from groq import Groq  # noqa: E402
from rag.pipeline import HybridRagPipeline  # noqa: E402

try:
    from graph.queries import GraphQueries as _GQ
    _GRAPH_MODULE_OK = True
except ImportError:
    _GRAPH_MODULE_OK = False

from .critic import CriticAgent  # noqa: E402
from .graph_agent import GraphAgent  # noqa: E402
from .hypothesis import HypothesisAgent  # noqa: E402
from .incident_memory_store import IncidentMemoryStore  # noqa: E402
from .log_analysis import LogAnalysisAgent  # noqa: E402
from .memory_agent import MemoryAgent  # noqa: E402
from .models import AgentResult, IncidentReport  # noqa: E402
from .planner import PlannerAgent  # noqa: E402
from .report import ReportGenerator  # noqa: E402
from .streaming import StreamEvent  # noqa: E402
from .timeline import TimelineAgent  # noqa: E402


@dataclass
class InvestigationResult:
    query: str
    report: IncidentReport | None
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


class Orchestrator:
    """
    Coordinates the 7-agent investigation pipeline.

    Execution order:
        1. Planner        → builds investigation plan
        2. [Parallel]     → LogAnalysis + Timeline + Graph  (evidence gathering)
        3. Hypothesis     → generates root-cause hypotheses from aggregated evidence
        4. Critic         → verifies hypotheses, surfaces conflicts
        5. ReportGenerator→ produces final structured report
    """

    def __init__(self) -> None:
        load_dotenv()
        print("  Loading RAG index…", flush=True)
        self._rag = self._init_rag()
        print("  Connecting to graph database…", flush=True)
        self._graph = self._init_graph()
        self._llm = Groq(api_key=os.environ["GROQ_API_KEY"])
        self._memory_store = IncidentMemoryStore()
        self._build_agents()

    # Initialisation

    def _init_rag(self) -> HybridRagPipeline:
        rag = HybridRagPipeline()
        rag.auto_load()  # loads pre-built FAISS + BM25, or builds from scratch
        return rag

    def _init_graph(self):
        if not _GRAPH_MODULE_OK:
            print("  [graph] langchain_neo4j not installed — graph queries disabled", flush=True)
            return None
        required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
        if not all(os.environ.get(k) for k in required):
            print("  [graph] NEO4J_* env vars not set — graph queries disabled", flush=True)
            return None
        try:
            gq = _GQ()
            print("  [graph] Connected to Neo4j.", flush=True)
            return gq
        except Exception as exc:
            print(f"  [graph] Could not connect to Neo4j ({exc}) — graph queries disabled", flush=True)
            return None

    def _build_agents(self) -> None:
        args = (self._rag, self._graph, self._llm)
        self._planner = PlannerAgent(*args)
        self._memory_agent = MemoryAgent(*args, memory_store=self._memory_store)
        self._log_analysis = LogAnalysisAgent(*args)
        self._timeline = TimelineAgent(*args)
        self._graph_agent = GraphAgent(*args)
        self._hypothesis = HypothesisAgent(*args)
        self._critic = CriticAgent(*args)
        self._report_gen = ReportGenerator(*args)

    # Public API

    def investigate(self, query: str) -> InvestigationResult:
        start = time.time()
        context: dict[str, Any] = {"query": query}
        results: dict[str, AgentResult] = {}
        errors: list[str] = []

        def safe_run(label: str, agent, ctx: dict) -> AgentResult:
            try:
                res = agent.run(ctx)
                if not res.success and res.error:
                    errors.append(f"[{label}] {res.error}")
                return res
            except Exception as exc:
                errors.append(f"[{label}] {exc}")
                from .models import AgentResult as AR
                return AR(label, False, None, error=str(exc))

        # Step 1: Plan 
        print("  [1/6] Planner…", flush=True)
        plan_res = safe_run("planner", self._planner, context)
        results["planner"] = plan_res
        if plan_res.success:
            context["plan"] = plan_res.data

        # Step 1b: Memory lookup 
        print("  [2/6] Incident Memory…", flush=True)
        mem_res = safe_run("memory", self._memory_agent, context)
        results["memory"] = mem_res
        context["memory"] = mem_res

        # Step 2: Parallel evidence gathering
        print("  [3/6] Gathering evidence (log analysis + timeline + graph)…", flush=True)
        evidence_agents = [
            ("log_analysis", self._log_analysis),
            ("timeline",     self._timeline),
            ("graph",        self._graph_agent),
        ]

        ctx_snapshot = dict(context)  # each agent gets the same immutable snapshot

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(safe_run, name, agent, ctx_snapshot): name
                for name, agent in evidence_agents
            }
            for fut in as_completed(futures):
                name = futures[fut]
                res = fut.result()
                results[name] = res
                context[name] = res  # make result available for downstream agents

        # Step 3: Hypothesis generation
        print("  [4/6] Generating hypotheses…", flush=True)
        hyp_res = safe_run("hypothesis", self._hypothesis, context)
        results["hypothesis"] = hyp_res
        context["hypothesis"] = hyp_res

        # Step 4: Critic 
        print("  [5/6] Verifying hypotheses…", flush=True)
        critic_res = safe_run("critic", self._critic, context)
        results["critic"] = critic_res
        context["critic"] = critic_res

        # Step 5: Report generation
        print("  [6/6] Generating report…", flush=True)
        report_res = safe_run("report", self._report_gen, context)
        results["report"] = report_res

        report = report_res.data if (report_res.success and report_res.data) else None

        if report is not None:
            try:
                self._memory_store.save_incident(report, query)
            except Exception as exc:
                errors.append(f"[memory_store] {exc}")

        total_ms = (time.time() - start) * 1000

        return InvestigationResult(
            query=query,
            report=report,
            agent_results=results,
            total_latency_ms=total_ms,
            errors=errors,
        )

    def stream_investigate(self, query: str) -> Generator[StreamEvent, None, None]:
        """
        Generator version of investigate() for real-time UI updates.

        Runs agents sequentially (not in parallel) so each completion can be
        streamed to the dashboard as it happens. Yields:
          StreamEvent("agent_start", name)        – before each agent runs
          StreamEvent("agent_done",  name, result) – after each agent finishes
          StreamEvent("complete", "orchestrator", InvestigationResult) – at the end
        """
        context: dict[str, Any] = {"query": query}
        results: dict[str, AgentResult] = {}
        errors: list[str] = []
        start = time.time()

        def safe_run(label: str, agent, ctx: dict) -> AgentResult:
            try:
                res = agent.run(ctx)
                if not res.success and res.error:
                    errors.append(f"[{label}] {res.error}")
                return res
            except Exception as exc:
                errors.append(f"[{label}] {exc}")
                return AgentResult(label, False, None, error=str(exc))

        pipeline = [
            ("planner",      self._planner),
            ("memory",       self._memory_agent),
            ("log_analysis", self._log_analysis),
            ("timeline",     self._timeline),
            ("graph",        self._graph_agent),
            ("hypothesis",   self._hypothesis),
            ("critic",       self._critic),
            ("report",       self._report_gen),
        ]

        for name, agent in pipeline:
            yield StreamEvent("agent_start", name)
            res = safe_run(name, agent, context)
            results[name] = res
            # Accumulate in context so downstream agents see prior results
            context[name] = res
            if name == "planner" and res.success:
                context["plan"] = res.data
            yield StreamEvent("agent_done", name, res, error=res.error)

        report_res = results.get("report")
        report = report_res.data if (report_res and report_res.success) else None

        # Persist this investigation to memory for future pattern recognition
        if report is not None:
            try:
                self._memory_store.save_incident(report, query)
            except Exception as exc:
                errors.append(f"[memory_store] {exc}")

        total_ms = (time.time() - start) * 1000
        final = InvestigationResult(
            query=query,
            report=report,
            agent_results=results,
            total_latency_ms=total_ms,
            errors=errors,
        )
        yield StreamEvent("complete", "orchestrator", final)

    @property
    def report_generator(self) -> ReportGenerator:
        """Expose report generator so the dashboard can call stream_narrative()."""
        return self._report_gen
