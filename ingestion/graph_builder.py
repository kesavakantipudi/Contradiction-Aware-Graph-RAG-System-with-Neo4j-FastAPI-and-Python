from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List

from neo4j import GraphDatabase


class KnowledgeGraphBuilder:
    """Persist facts as first-class nodes connected to entity/attribute nodes."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def verify_connection(self) -> None:
        self.driver.verify_connectivity()

    def ingest_facts(self, facts: Iterable[Dict[str, Any]]) -> int:
        rows = []
        for fact in facts:
            entity = str(fact["entity"]).strip()
            attribute = str(fact["attribute"]).strip()
            source = str(fact["source"]).strip()
            timestamp = str(fact["timestamp"]).strip()
            fact_id = hashlib.sha256(
                json.dumps([entity, attribute, source, timestamp], ensure_ascii=True).encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "fact_id": fact_id,
                    "entity": entity,
                    "attribute": attribute,
                    "value": str(fact["value"]).strip(),
                    "source": source,
                    "timestamp": timestamp,
                    "confidence": float(fact["confidence"]),
                    "context_text": str(fact["context_text"]),
                }
            )

        if not rows:
            return 0

        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS "
                "FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE"
            ).consume()
            session.run(
                """
                UNWIND $facts AS row
                MERGE (e:Entity {name: row.entity})
                MERGE (a:Attribute {name: row.attribute})
                MERGE (e)-[:HAS_ATTRIBUTE]->(a)
                MERGE (f:Fact {fact_id: row.fact_id})
                SET f.value = row.value,
                    f.source = row.source,
                    f.timestamp = row.timestamp,
                    f.confidence = row.confidence,
                    f.context_text = row.context_text
                MERGE (a)-[:STATED_BY]->(f)
                """,
                facts=rows,
            ).consume()
        return len(rows)

    def list_targets(self) -> List[Dict[str, str]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (e:Entity)-[:HAS_ATTRIBUTE]->(a:Attribute)
                RETURN e.name AS entity, a.name AS attribute
                ORDER BY e.name, a.name
                """
            )
            return [{"entity": row["entity"], "attribute": row["attribute"]} for row in result]

    def fetch_facts_for(self, entity: str, attribute: str) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $entity})-[:HAS_ATTRIBUTE]->
                      (a:Attribute {name: $attribute})-[:STATED_BY]->(f:Fact)
                RETURN f.value AS value,
                       f.source AS source,
                       f.timestamp AS timestamp,
                       f.confidence AS confidence,
                       f.context_text AS context_text
                ORDER BY f.timestamp
                """,
                entity=entity,
                attribute=attribute,
            )
            return [dict(row) for row in result]

    def close(self) -> None:
        self.driver.close()
