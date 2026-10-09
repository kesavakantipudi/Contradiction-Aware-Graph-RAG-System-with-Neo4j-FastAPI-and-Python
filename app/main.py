from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "dataset.json"
BASELINE_TOP_K = 3
STOP_WORDS = {"a", "an", "are", "for", "is", "of", "the", "to", "what", "which"}
ATTRIBUTE_ALIASES = {
    "connection_timeout": ("database timeout", "connection timeout"),
    "api_timeout": ("api timeout", "request timeout"),
    "token_ttl": ("token ttl", "token lifetime"),
    "refresh_interval": ("refresh interval", "refresh cadence"),
    "retry_limit": ("retry limit", "retry count"),
    "max_connections": ("max connections", "maximum connections"),
    "replication_mode": ("replication mode",),
    "read_replica_count": ("read replica count", "read replicas"),
    "replication_factor": ("replication factor",),
    "latency_budget_ms": ("latency budget", "latency"),
    "memory_limit_gb": ("memory limit", "memory"),
    "ttl_seconds": ("ttl", "time to live"),
}
ENTITY_ALIASES = {
    "api_gateway": ("gateway",),
    "user_profile_store": ("profile store", "user profile"),
    "ml_inference_service": ("inference service", "ml inference"),
}

load_dotenv(ROOT / ".env")


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    source: str
    timestamp: str


class GraphAskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    agreement_status: Literal["agree", "conflict", "single-source"]
    sources: List[SourceEntry]


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def tokenize(value: str) -> List[str]:
    return [token for token in normalize_text(value).split() if token not in STOP_WORDS]


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def load_dataset() -> List[Dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Synthetic dataset is missing: {DATASET_PATH}")
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {DATASET_PATH}")
    return payload


def _as_target(target: Any) -> Tuple[str, str]:
    if isinstance(target, dict):
        return str(target["entity"]), str(target["attribute"])
    return str(target[0]), str(target[1])


def _phrase_in_question(phrase: str, normalized_question: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in f" {normalized_question} "


def find_target(question: str, targets: Iterable[Any]) -> Tuple[str, str] | None:
    normalized_question = normalize_text(question)
    matches: List[Tuple[int, str, str]] = []
    for target in targets:
        entity, attribute = _as_target(target)
        entity_phrases = (entity.replace("_", " "), *ENTITY_ALIASES.get(entity, ()))
        attribute_phrases = (attribute.replace("_", " "), *ATTRIBUTE_ALIASES.get(attribute, ()))
        entity_match = max(
            (len(tokenize(phrase)) for phrase in entity_phrases if _phrase_in_question(phrase, normalized_question)),
            default=0,
        )
        attribute_match = max(
            (len(tokenize(phrase)) for phrase in attribute_phrases if _phrase_in_question(phrase, normalized_question)),
            default=0,
        )
        if entity_match and attribute_match:
            matches.append((entity_match + attribute_match, entity, attribute))

    if not matches:
        return None
    best_score = max(match[0] for match in matches)
    best_targets = {(entity, attribute) for score, entity, attribute in matches if score == best_score}
    if len(best_targets) != 1:
        return None
    return next(iter(best_targets))


def classify_graph_facts(
    entity: str,
    attribute: str,
    facts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    ordered = sorted(facts, key=lambda item: parse_timestamp(item["timestamp"]))
    sources = [
        {
            "value": str(fact["value"]),
            "source": str(fact["source"]),
            "timestamp": str(fact["timestamp"]),
        }
        for fact in ordered
    ]

    if not ordered:
        return {
            "answer": "I could not find any facts for that entity or attribute in the knowledge graph.",
            "agreement_status": "single-source",
            "sources": [],
        }

    unique_values = {str(fact["value"]).strip().casefold() for fact in ordered}
    if len(ordered) == 1:
        fact = ordered[0]
        answer = f"Only one fact is recorded for {entity} {attribute}: {fact['value']} from {fact['source']}."
        return {"answer": answer, "agreement_status": "single-source", "sources": sources}

    if len(unique_values) == 1:
        value = ordered[-1]["value"]
        answer = f"The available facts agree on {entity} {attribute}: {value}."
        return {"answer": answer, "agreement_status": "agree", "sources": sources}

    newer = ordered[-1]
    answer = (
        f"The sources disagree about {entity} {attribute}; the graph records conflicting values. "
        f"The newer source, {newer['source']} ({newer['timestamp']}), reports {newer['value']} "
        f"with confidence {float(newer.get('confidence', 0.0)):.2f}. As a separate resolution "
        f"heuristic, prioritize the more recent timestamp (and compare confidence), but retain "
        f"the conflict rather than treating either value as undisputed."
    )
    return {"answer": answer, "agreement_status": "conflict", "sources": sources}


def build_tfidf_vectors(documents: Sequence[str]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    token_counts = [Counter(tokenize(document)) for document in documents]
    document_frequency: Counter[str] = Counter()
    for counts in token_counts:
        document_frequency.update(counts.keys())
    document_total = len(token_counts)
    inverse_document_frequency = {
        token: math.log((document_total + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }

    vectors: List[Dict[str, float]] = []
    for counts in token_counts:
        vector = {token: count * inverse_document_frequency[token] for token, count in counts.items()}
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        vectors.append({token: weight / norm for token, weight in vector.items()} if norm else {})
    return vectors, inverse_document_frequency


def retrieve_baseline_facts(
    question: str,
    facts: Sequence[Dict[str, Any]],
    top_k: int = BASELINE_TOP_K,
) -> List[Dict[str, Any]]:
    if not facts:
        return []
    documents = [
        " ".join(
            (
                str(fact.get("context_text", "")),
                str(fact.get("entity", "")).replace("_", " "),
                str(fact.get("attribute", "")).replace("_", " "),
                str(fact.get("value", "")),
            )
        )
        for fact in facts
    ]
    vectors, idf = build_tfidf_vectors(documents)
    query_counts = Counter(tokenize(question))
    query_vector = {token: count * idf[token] for token, count in query_counts.items() if token in idf}
    query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
    if query_norm == 0:
        return []
    query_vector = {token: weight / query_norm for token, weight in query_vector.items()}
    scored = [
        (sum(weight * vector.get(token, 0.0) for token, weight in query_vector.items()), index)
        for index, vector in enumerate(vectors)
    ]
    ranked = sorted(scored, key=lambda pair: (-pair[0], pair[1]))
    return [facts[index] for score, index in ranked[:top_k] if score > 0]


def generate_baseline_answer(question: str, facts: Sequence[Dict[str, Any]] | None = None) -> str:
    corpus = load_dataset() if facts is None else facts
    retrieved = retrieve_baseline_facts(question, corpus)
    if not retrieved:
        return "No matching facts were found in the baseline corpus."
    selected = retrieved[0]
    entity = str(selected["entity"]).replace("_", " ")
    attribute = str(selected["attribute"]).replace("_", " ")
    return f"The {entity} {attribute} is {selected['value']} according to {selected['source']}."


@asynccontextmanager
async def lifespan(application: FastAPI):
    from ingestion.graph_builder import KnowledgeGraphBuilder

    builder = KnowledgeGraphBuilder(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    builder.verify_connection()
    builder.ingest_facts(load_dataset())
    application.state.graph_builder = builder
    try:
        yield
    finally:
        builder.close()


app = FastAPI(title="Contradiction-Aware Graph RAG", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> Dict[str, str]:
    request.app.state.graph_builder.verify_connection()
    return {"status": "ok"}


@app.post("/api/baseline")
def baseline_route(request: QuestionRequest) -> Dict[str, str]:
    return {"answer": generate_baseline_answer(request.question)}


@app.post("/api/ask", response_model=GraphAskResponse)
def ask_route(body: QuestionRequest, request: Request) -> Dict[str, Any]:
    graph = request.app.state.graph_builder
    target = find_target(body.question, graph.list_targets())
    if target is None:
        return classify_graph_facts("", "", [])
    entity, attribute = target
    facts = graph.fetch_facts_for(entity, attribute)
    return classify_graph_facts(entity, attribute, facts)
