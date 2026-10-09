from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import load_dataset

RESULT_PATH = ROOT / "evaluation_results.json"


def build_evaluation_cases(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        groups[(fact["entity"], fact["attribute"])].append(fact)

    cases = []
    for (entity, attribute), group_facts in sorted(groups.items()):
        values = {str(fact["value"]).strip().casefold() for fact in group_facts}
        cases.append(
            {
                "question": f"What is the {attribute.replace('_', ' ')} for the {entity.replace('_', ' ')}?",
                "is_conflict": len(values) > 1,
                "entity": entity,
                "attribute": attribute,
                "values": values,
            }
        )
    return cases


def post_question(base_url: str, endpoint: str, question: str) -> Dict[str, Any]:
    payload = json.dumps({"question": question}).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{endpoint}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"{endpoint} returned a non-object JSON response")
    return decoded


def evaluate(base_url: str, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    cases = build_evaluation_cases(facts)
    if not cases or not any(case["is_conflict"] for case in cases):
        raise ValueError("Evaluation requires at least one conflicting entity/attribute group")

    conflict_true_positives = 0
    conflict_false_positives = 0
    conflict_false_negatives = 0
    baseline_silent_picks = 0
    conflict_question_count = 0
    example = None

    for case in cases:
        graph_response = post_question(base_url, "/api/ask", case["question"])
        baseline_response = post_question(base_url, "/api/baseline", case["question"])
        if set(graph_response) != {"answer", "agreement_status", "sources"}:
            raise ValueError("Graph endpoint did not return the required response fields")
        if not isinstance(graph_response["answer"], str) or not isinstance(graph_response["sources"], list):
            raise ValueError("Graph endpoint returned invalid answer or sources types")
        if graph_response["agreement_status"] not in {"agree", "conflict", "single-source"}:
            raise ValueError("Graph endpoint returned an unsupported agreement_status")
        if not isinstance(baseline_response.get("answer"), str):
            raise ValueError("Baseline endpoint did not return an answer string")

        actual_conflict = graph_response["agreement_status"] == "conflict"
        if case["is_conflict"]:
            conflict_question_count += 1
            if actual_conflict:
                conflict_true_positives += 1
            else:
                conflict_false_negatives += 1
            baseline_answer = baseline_response["answer"].casefold()
            if any(value in baseline_answer for value in case["values"]):
                baseline_silent_picks += 1
        elif actual_conflict:
            conflict_false_positives += 1

        if example is None and case["is_conflict"]:
            example = {
                "question": case["question"],
                "graph_response": graph_response,
                "baseline_response": baseline_response["answer"],
            }

    precision_denominator = conflict_true_positives + conflict_false_positives
    recall_denominator = conflict_true_positives + conflict_false_negatives
    return {
        "total_questions_tested": len(cases),
        "graph_rag_conflict_recall": round(conflict_true_positives / recall_denominator, 4),
        "graph_rag_conflict_precision": round(
            conflict_true_positives / precision_denominator if precision_denominator else 0.0, 4
        ),
        "baseline_silent_pick_rate": round(
            baseline_silent_picks / conflict_question_count if conflict_question_count else 0.0, 4
        ),
        "example_conflict_handled": example,
    }


def main() -> None:
    port = os.getenv("API_PORT", "8000")
    base_url = os.getenv("API_BASE_URL", f"http://localhost:{port}")
    try:
        metrics = evaluate(base_url, load_dataset())
    except (HTTPError, URLError, TimeoutError) as error:
        raise SystemExit(
            f"Could not reach the RAG API at {base_url}. Start the Compose stack and retry: {error}"
        ) from error
    RESULT_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
