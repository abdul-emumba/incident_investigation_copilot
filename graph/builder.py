import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

from .loader import load_graph_data, parse_nodes, parse_edges

load_dotenv()

# Map edge type string -> Cypher relationship label
_EDGE_LABEL = {
    "sync": "DEPENDS_ON_SYNC",
    "async": "DEPENDS_ON_ASYNC",
    "proxy": "PROXIES",
}

# Node type -> Cypher label
_NODE_LABEL = {
    "service": "Service",
    "datastore": "Datastore",
    "external": "External",
}


class GraphBuilder:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
            refresh_schema=False,
        )

    def clear(self):
        self.graph.query("MATCH (n) DETACH DELETE n")

    def build(self, clear_first: bool = True):
        if clear_first:
            self.clear()

        data = load_graph_data()
        nodes = parse_nodes(data)
        edges = parse_edges(data)

        self._create_nodes(nodes)
        self._create_edges(edges)
        self._create_indexes()

        print(f"Graph built: {len(nodes)} nodes, {len(edges)} edges.")

    def _create_nodes(self, nodes: list[dict]):
        for node in nodes:
            label = _NODE_LABEL.get(node["type"], "Node")
            self.graph.query(
                f"""
                MERGE (n:{label} {{id: $id}})
                SET n += $props
                """,
                params={
                    "id": node["id"],
                    "props": {k: v for k, v in node.items() if k != "id"},
                },
            )

    def _create_edges(self, edges: list[dict]):
        for edge in edges:
            rel = _EDGE_LABEL.get(edge["type"], "DEPENDS_ON_SYNC")
            self.graph.query(
                f"""
                MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                MERGE (a)-[r:{rel}]->(b)
                SET r += $props
                """,
                params={
                    "from_id": edge["from_id"],
                    "to_id": edge["to_id"],
                    "props": {k: v for k, v in edge.items() if k not in ("from_id", "to_id", "type")},
                },
            )

    def _create_indexes(self):
        for label in ("Service", "Datastore", "External"):
            self.graph.query(
                f"CREATE INDEX {label.lower()}_id IF NOT EXISTS FOR (n:{label}) ON (n.id)"
            )
