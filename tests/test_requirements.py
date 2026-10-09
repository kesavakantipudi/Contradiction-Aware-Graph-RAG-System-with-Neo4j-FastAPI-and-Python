from __future__ import annotations

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from app.main import (
    GraphAskResponse,
    QuestionRequest,
    ask_route,
    classify_graph_facts,
    find_target,
    generate_baseline_answer,
    retrieve_baseline_facts,
)
from scripts.evaluate import evaluate
from scripts.generate_dataset import generate_synthetic_corpus


def sample_fact(
    value: str,
    source: str,
    timestamp: str,
    confidence: float = 0.9,
    entity: str = "database",
    attribute: str = "connection_timeout",
) -> dict:
    return {
        "entity": entity,
        "attribute": attribute,
        "value": value,
        "source": source,
        "timestamp": timestamp,
        "confidence": confidence,
        "context_text": f"The {entity} {attribute.replace('_', ' ')} is {value}.",
    }


class FakeGraph:
    def __init__(self, targets, facts_by_target):
        self.targets = targets
        self.facts_by_target = facts_by_target

    def list_targets(self):
        return self.targets

    def fetch_facts_for(self, entity, attribute):
        return self.facts_by_target.get((entity, attribute), [])


class RequirementTests(unittest.TestCase):
    def test_generated_corpus_contract_and_conflicts(self):
        facts = generate_synthetic_corpus()
        groups = defaultdict(list)
        for fact in facts:
            groups[(fact["entity"], fact["attribute"])].append(fact)

        conflicts = [rows for rows in groups.values() if len({row["value"] for row in rows}) > 1]
        self.assertGreaterEqual(len({fact["entity"] for fact in facts}), 10)
        self.assertGreaterEqual(len(facts), 30)
        self.assertLessEqual(len(facts), 50)
        self.assertGreaterEqual(len(conflicts), 8)
        for rows in conflicts:
            self.assertEqual(len({row["source"] for row in rows}), len(rows))
            self.assertEqual(len({row["timestamp"] for row in rows}), len(rows))
        for fact in facts:
            self.assertEqual(
                set(fact),
                {"entity", "attribute", "value", "source", "timestamp", "confidence", "context_text"},
            )
            self.assertGreaterEqual(fact["confidence"], 0.0)
            self.assertLessEqual(fact["confidence"], 1.0)

    def test_graph_target_matching_and_unknown_query(self):
        targets = [
            {"entity": "database", "attribute": "connection_timeout"},
            {"entity": "database", "attribute": "replication_mode"},
        ]
        self.assertEqual(
            find_target("What is the database connection timeout?", targets),
            ("database", "connection_timeout"),
        )
        self.assertIsNone(
            find_target("What is the quantum flux capacity of the hyperdrive server?", targets)
        )

    def test_graph_conflict_agreement_and_single_source_results(self):
        conflict = classify_graph_facts(
            "database",
            "connection_timeout",
            [
                sample_fact("30s", "runbook.pdf", "2022-06-02"),
                sample_fact("60s", "adr.pdf", "2024-02-18", 0.98),
            ],
        )
        self.assertEqual(conflict["agreement_status"], "conflict")
        self.assertEqual(len(conflict["sources"]), 2)
        self.assertIn("disagree", conflict["answer"])
        self.assertIn("newer", conflict["answer"])
        self.assertIn("prioritize", conflict["answer"])
        GraphAskResponse.model_validate(conflict)

        agreeing = classify_graph_facts(
            "database",
            "replication_mode",
            [
                sample_fact("async", "doc-a.md", "2023-01-01", attribute="replication_mode"),
                sample_fact("async", "doc-b.md", "2024-01-01", attribute="replication_mode"),
            ],
        )
        self.assertEqual(agreeing["agreement_status"], "agree")

        single = classify_graph_facts(
            "database",
            "replication_mode",
            [sample_fact("async", "doc.md", "2024-01-01", attribute="replication_mode")],
        )
        self.assertEqual(single["agreement_status"], "single-source")
        self.assertEqual(len(single["sources"]), 1)

        missing = classify_graph_facts("", "", [])
        self.assertEqual(missing["sources"], [])
        GraphAskResponse.model_validate(missing)

    def test_api_handler_uses_graph_targets_and_returns_strict_response(self):
        facts = [
            sample_fact("30s", "runbook.pdf", "2022-06-02"),
            sample_fact("60s", "adr.pdf", "2024-02-18"),
        ]
        graph = FakeGraph(
            [{"entity": "database", "attribute": "connection_timeout"}],
            {("database", "connection_timeout"): facts},
        )
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(graph_builder=graph)))
        result = ask_route(QuestionRequest(question="What is the database connection timeout?"), request)
        self.assertEqual(set(result), {"answer", "agreement_status", "sources"})
        self.assertEqual(result["agreement_status"], "conflict")
        GraphAskResponse.model_validate(result)

    def test_baseline_retrieves_top_k_vectors_and_silently_selects_one_fact(self):
        facts = [
            sample_fact("30s", "runbook.pdf", "2022-06-02"),
            sample_fact("60s", "adr.pdf", "2024-02-18"),
            sample_fact(
                "async",
                "db-replication.pdf",
                "2024-01-01",
                entity="database",
                attribute="replication_mode",
            ),
        ]
        retrieved = retrieve_baseline_facts("What is the database connection timeout?", facts)
        self.assertGreaterEqual(len(retrieved), 2)
        answer = generate_baseline_answer("What is the database connection timeout?", facts)
        self.assertTrue(any(value in answer for value in ("30s", "60s")))
        self.assertNotIn("conflict", answer.casefold())
        self.assertNotIn("disagree", answer.casefold())

    def test_evaluation_reports_recall_precision_and_silent_pick_rates(self):
        facts = [
            sample_fact("30s", "runbook.pdf", "2022-06-02"),
            sample_fact("60s", "adr.pdf", "2024-02-18"),
            sample_fact(
                "async",
                "replication.pdf",
                "2024-01-01",
                entity="database",
                attribute="replication_mode",
            ),
        ]

        def fake_post(_base_url, endpoint, question):
            if endpoint == "/api/baseline":
                return {"answer": "The answer is 60s." if "connection timeout" in question else "async."}
            if "connection timeout" in question:
                return {
                    "answer": "Sources disagree; prioritize the newer timestamp.",
                    "agreement_status": "conflict",
                    "sources": [
                        {"value": "30s", "source": "runbook.pdf", "timestamp": "2022-06-02"},
                        {"value": "60s", "source": "adr.pdf", "timestamp": "2024-02-18"},
                    ],
                }
            return {
                "answer": "Only one fact is recorded.",
                "agreement_status": "single-source",
                "sources": [
                    {"value": "async", "source": "replication.pdf", "timestamp": "2024-01-01"}
                ],
            }

        with patch("scripts.evaluate.post_question", side_effect=fake_post):
            metrics = evaluate("http://test", facts)
        self.assertEqual(metrics["total_questions_tested"], 2)
        self.assertEqual(metrics["graph_rag_conflict_recall"], 1.0)
        self.assertEqual(metrics["graph_rag_conflict_precision"], 1.0)
        self.assertEqual(metrics["baseline_silent_pick_rate"], 1.0)
        self.assertIsInstance(metrics["graph_rag_conflict_recall"], float)


if __name__ == "__main__":
    unittest.main()
