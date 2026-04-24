import json
from pathlib import Path
from typing import Any

_GRAPH_PATH = Path(__file__).parent.parent / "data" / "graph" / "service_dependency_graph.json"


def load_graph_data(path: Path = _GRAPH_PATH) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def parse_nodes(data: dict) -> list[dict]:
    nodes = []
    for node in data["nodes"]:
        owner = node.get("owner", {})
        record = {
            "id": node["id"],
            "type": node["type"],
            "display_name": node.get("display_name", node["id"]),
            "description": node.get("description", ""),
            "version": node.get("version", ""),
            "language": node.get("language", ""),
            "engine": node.get("engine", ""),
            "vendor": node.get("vendor", ""),
            "team": owner.get("team", "") if isinstance(owner, dict) else "",
            "team_lead": owner.get("lead", "") if isinstance(owner, dict) else "",
            "slack": owner.get("slack", "") if isinstance(owner, dict) else "",
            "on_call": owner.get("on_call", "") if isinstance(owner, dict) else "",
            "regions": json.dumps(node.get("regions", [])),
            "tags": json.dumps(node.get("tags", [])),
            "known_issues": json.dumps(node.get("known_issues", [])),
            "availability_target": (node.get("sla") or {}).get("availability_target", ""),
            "p99_latency_ms": (node.get("sla") or {}).get("p99_latency_ms", 0),
            "circuit_breaker": node.get("circuit_breaker", False),
            "replication": node.get("replication", ""),
            "repository": node.get("repository", ""),
        }
        nodes.append(record)
    return nodes


def parse_edges(data: dict) -> list[dict]:
    edges = []
    for edge in data["edges"]:
        record = {
            "from_id": edge["from"],
            "to_id": edge["to"],
            "type": edge.get("type", "sync"),
            "protocol": edge.get("protocol", ""),
            "description": edge.get("description", ""),
            "critical": edge.get("critical", False),
            "circuit_breaker": edge.get("circuit_breaker", False),
            "timeout_ms": edge.get("timeout_ms", 0),
            "known_issues": json.dumps(edge.get("known_issues", [])),
            "topics": json.dumps(edge.get("topics", [])),
            "fallback": edge.get("fallback", ""),
        }
        edges.append(record)
    return edges
