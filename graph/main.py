"""
Entry point: builds the Neo4j knowledge graph and runs demo queries.

Usage:
    source .env && python graph/main.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.builder import GraphBuilder
from graph.queries import GraphQueries


def _print(title: str, rows: list[dict]):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    if not rows:
        print("  (no results)")
        return
    print(json.dumps(rows, indent=2, default=str))


def main():
    print("Building knowledge graph from service_dependency_graph.json …")
    builder = GraphBuilder()
    builder.build(clear_first=True)

    q = GraphQueries()

    _print(
        "All nodes in the graph",
        q.list_services(),
    )

    _print(
        "Direct dependencies of 'orders'",
        q.direct_dependencies("orders"),
    )

    _print(
        "Transitive dependencies of 'orders'",
        q.transitive_dependencies("orders"),
    )

    _print(
        "Impact analysis: what breaks if 'redis' goes down?",
        q.impact_analysis("redis"),
    )

    _print(
        "Critical-path impact: hard failures if 'stripe' goes down",
        q.critical_impact_analysis("stripe"),
    )

    _print(
        "Dependency path: gateway → stripe",
        q.dependency_path("gateway", "stripe"),
    )

    _print(
        "Shared dependencies (single points of failure)",
        q.find_shared_dependencies(),
    )


if __name__ == "__main__":
    main()
