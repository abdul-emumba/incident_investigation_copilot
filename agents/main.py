#!/usr/bin/env python3
"""
Multi-Agent Incident Investigation CLI.

Usage:
    python -m agents.main "Why did auth-service fail on 2024-01-15?"
    python -m agents.main --interactive
"""
from __future__ import annotations

import argparse
import sys

from .models import IncidentReport, TimelineEvent
from .orchestrator import InvestigationResult, Orchestrator


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt_report(result: InvestigationResult) -> str:
    sep = "=" * 72
    thin = "-" * 48

    if not result.report:
        lines = [sep, "INCIDENT INVESTIGATION — FAILED", sep]
        lines.append("The report could not be generated. Errors encountered:")
        for e in result.errors:
            lines.append(f"  • {e}")
        return "\n".join(lines)

    r: IncidentReport = result.report
    lines: list[str] = [sep, "INCIDENT INVESTIGATION REPORT", sep]

    if r.incident_id:
        lines.append(f"Incident:  {r.incident_id}")
    lines.append(f"")

    lines.append(f"ROOT CAUSE\n{thin}")
    lines.append(r.root_cause)

    lines.append(f"\nSUMMARY\n{thin}")
    lines.append(r.investigation_summary)

    if r.timeline:
        lines.append(f"\nTIMELINE\n{thin}")
        for e in r.timeline[:20]:
            ts = e.timestamp.ljust(32) if e.timestamp != "unknown" else "(unknown time)".ljust(32)
            tag = f"[{e.event_type}]".ljust(16)
            svc = f"  ({e.service})" if e.service else ""
            lines.append(f"  {ts} {tag} {e.description}{svc}")

    if r.affected_services:
        lines.append(f"\nAFFECTED SERVICES\n{thin}")
        for svc in r.affected_services:
            lines.append(f"  • {svc}")

    if r.evidence:
        lines.append(f"\nEVIDENCE / CITATIONS\n{thin}")
        for ev in r.evidence:
            lines.append(f"  – {ev}")

    lines.append(f"\nCONFIDENCE SCORE: {r.confidence_score:.2f}")

    if r.conflicting_signals:
        lines.append(f"\nCONFLICTING SIGNALS\n{thin}")
        for cs in r.conflicting_signals:
            lines.append(f"  ⚠  {cs}")

    if r.recommended_actions:
        lines.append(f"\nRECOMMENDED ACTIONS\n{thin}")
        for i, act in enumerate(r.recommended_actions, 1):
            lines.append(f"  {i}. {act}")

    lines.append(f"\n{sep}")
    lines.append(f"Investigation completed in {result.total_latency_ms / 1000:.1f}s")
    if result.errors:
        lines.append(f"Non-fatal agent errors: {len(result.errors)}")
        for err in result.errors:
            lines.append(f"  ↳ {err}")
    lines.append(sep)
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Incident Investigation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agents.main "Why did auth-service fail on 2024-01-15 around 14:00?"
  python -m agents.main "Investigate INC-0042"
  python -m agents.main --interactive
""",
    )
    parser.add_argument("query", nargs="?", help="Investigation query")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.print_help()
        sys.exit(1)

    print("\nInitialising multi-agent investigation system…")
    orchestrator = Orchestrator()
    print("System ready.\n")

    if args.interactive:
        print("Multi-Agent Incident Investigator  (type 'exit' to quit)")
        print("-" * 60)
        while True:
            try:
                query = input("\nQuery> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break
            print()
            result = orchestrator.investigate(query)
            print(_fmt_report(result))
    else:
        result = orchestrator.investigate(args.query)
        print(_fmt_report(result))


if __name__ == "__main__":
    main()
