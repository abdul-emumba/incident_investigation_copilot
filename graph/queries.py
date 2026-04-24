import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

# Relationship pattern covering all dependency types
_ANY_DEP = "DEPENDS_ON_SYNC|DEPENDS_ON_ASYNC|PROXIES"


class GraphQueries:
    """
    Dependency traversal and impact analysis over the service knowledge graph.

    Traversal direction convention (matches edge direction in the graph):
      (A)-[:DEPENDS_ON_*]->(B)  means A depends on B.

    - upstream(service)   → nodes that `service` depends on (direct + transitive)
    - downstream(service) → nodes that depend on `service` (direct + transitive)
    - impact_analysis     → services impacted if a given node fails
    - critical_path       → path between two services via critical edges only
    """

    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
            refresh_schema=False,
        )

    # dependency traversal
    def direct_dependencies(self, service_id: str) -> list[dict]:
        """Services/datastores that `service_id` directly calls."""
        return self.graph.query(
            f"""
            MATCH (a {{id: $id}})-[r:{_ANY_DEP}]->(b)
            RETURN b.id AS id,
                   b.display_name AS name,
                   b.type AS type,
                   type(r) AS relationship,
                   r.critical AS critical,
                   r.protocol AS protocol
            ORDER BY r.critical DESC, b.id
            """,
            params={"id": service_id},
        )

    def transitive_dependencies(self, service_id: str, max_depth: int = 10) -> list[dict]:
        """All nodes reachable from `service_id` following dependency edges."""
        return self.graph.query(
            f"""
            MATCH path = (a {{id: $id}})-[:{_ANY_DEP}*1..{max_depth}]->(b)
            WHERE a <> b
            WITH b, min(length(path)) AS depth, collect(distinct type(relationships(path)[-1])) AS rel_types
            RETURN b.id AS id,
                   b.display_name AS name,
                   b.type AS type,
                   depth,
                   rel_types
            ORDER BY depth, b.id
            """,
            params={"id": service_id},
        )

    def dependency_path(self, from_id: str, to_id: str) -> list[dict]:
        """Shortest dependency path between two nodes."""
        return self.graph.query(
            f"""
            MATCH path = shortestPath(
                (a {{id: $from_id}})-[:{_ANY_DEP}*]->(b {{id: $to_id}})
            )
            UNWIND range(0, length(path)-1) AS i
            RETURN
                nodes(path)[i].id AS from_node,
                type(relationships(path)[i]) AS relationship,
                nodes(path)[i+1].id AS to_node,
                relationships(path)[i].critical AS critical
            """,
            params={"from_id": from_id, "to_id": to_id},
        )

    # impact analysis
    def direct_dependents(self, service_id: str) -> list[dict]:
        """Services that directly depend on `service_id` (callers)."""
        return self.graph.query(
            f"""
            MATCH (a)-[r:{_ANY_DEP}]->(b {{id: $id}})
            RETURN a.id AS id,
                   a.display_name AS name,
                   a.type AS type,
                   type(r) AS relationship,
                   r.critical AS critical,
                   r.circuit_breaker AS circuit_breaker,
                   r.fallback AS fallback
            ORDER BY r.critical DESC, a.id
            """,
            params={"id": service_id},
        )

    def impact_analysis(self, service_id: str, max_depth: int = 10) -> list[dict]:
        """
        All services impacted if `service_id` fails.
        Returns each impacted node with blast radius depth and whether
        any circuit breaker or fallback exists on the inbound edge.
        """
        return self.graph.query(
            f"""
            MATCH path = (b {{id: $id}})<-[:{_ANY_DEP}*1..{max_depth}]-(a)
            WHERE a <> b
            WITH a, min(length(path)) AS depth,
                 collect(distinct relationships(path)[-1].circuit_breaker) AS cb_list,
                 collect(distinct relationships(path)[-1].fallback) AS fallback_list,
                 collect(distinct relationships(path)[-1].critical) AS critical_list
            RETURN a.id AS id,
                   a.display_name AS name,
                   a.type AS type,
                   a.team AS team,
                   a.slack AS slack,
                   depth,
                   any(x IN critical_list WHERE x = true) AS on_critical_path,
                   any(x IN cb_list WHERE x = true) AS has_circuit_breaker,
                   any(x IN fallback_list WHERE x IS NOT NULL AND x <> '') AS has_fallback
            ORDER BY on_critical_path DESC, depth, a.id
            """,
            params={"id": service_id},
        )

    def critical_impact_analysis(self, service_id: str) -> list[dict]:
        """Impact analysis restricted to critical edges only (hard failures)."""
        return self.graph.query(
            f"""
            MATCH path = (b {{id: $id}})<-[r:{_ANY_DEP}*1..10]-(a)
            WHERE a <> b
              AND all(rel IN relationships(path) WHERE rel.critical = true)
            WITH a, min(length(path)) AS depth
            RETURN a.id AS id,
                   a.display_name AS name,
                   a.type AS type,
                   a.team AS team,
                   a.slack AS slack,
                   depth
            ORDER BY depth, a.id
            """,
            params={"id": service_id},
        )

    # utility queries
    def get_node(self, service_id: str) -> dict | None:
        results = self.graph.query(
            "MATCH (n {id: $id}) RETURN properties(n) AS props",
            params={"id": service_id},
        )
        return results[0]["props"] if results else None

    def list_services(self) -> list[dict]:
        return self.graph.query(
            """
            MATCH (n)
            RETURN n.id AS id, n.display_name AS name, n.type AS type, n.team AS team
            ORDER BY n.type, n.id
            """
        )

    def find_shared_dependencies(self) -> list[dict]:
        """Datastores/externals depended on by more than one service — single points of failure."""
        return self.graph.query(
            f"""
            MATCH (a)-[r:{_ANY_DEP}]->(b)
            WITH b, count(distinct a) AS dependent_count, collect(distinct a.id) AS dependents
            WHERE dependent_count > 1
            RETURN b.id AS id,
                   b.display_name AS name,
                   b.type AS type,
                   dependent_count,
                   dependents
            ORDER BY dependent_count DESC
            """
        )
