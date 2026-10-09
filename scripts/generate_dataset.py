from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel


class ProvenanceFact(BaseModel):
    entity: str
    attribute: str
    value: str
    source: str
    timestamp: str
    confidence: float
    context_text: str


def generate_synthetic_corpus() -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []

    def add_fact(entity: str, attribute: str, value: str, source: str, timestamp: str, confidence: float, context_text: str) -> None:
        facts.append(
            {
                "entity": entity,
                "attribute": attribute,
                "value": value,
                "source": source,
                "timestamp": timestamp,
                "confidence": confidence,
                "context_text": context_text,
            }
        )

    conflict_pairs = [
        ("database", "connection_timeout", [
            ("30s", "ops-runbook-2022.pdf", "2022-06-02", 0.9, "The database connection timeout for the primary database is set to 30 seconds in the legacy runbook."),
            ("60s", "architecture-adr-2024.pdf", "2024-02-18", 0.98, "The architecture ADR confirms that the database connection timeout was increased to 60 seconds."),
        ]),
        ("payments_service", "api_timeout", [
            ("20s", "payments-guide-v1.docx", "2021-11-14", 0.88, "The first payments runbook states the API timeout must not exceed 20 seconds."),
            ("45s", "payments-platform-update.pdf", "2024-03-12", 0.96, "The latest platform update raises the payments API timeout to 45 seconds."),
        ]),
        ("auth_service", "token_ttl", [
            ("60m", "security-baseline-2022.md", "2022-09-08", 0.91, "The security baseline defines a 60 minute authentication token TTL."),
            ("90m", "identity-access-policy-2024.pdf", "2024-05-26", 0.97, "The current identity access policy extends the token TTL to 90 minutes."),
        ]),
        ("cache_cluster", "eviction_policy", [
            ("lru", "cache-tuning-notes.txt", "2023-01-18", 0.86, "The initial cache tuning notes specify the eviction policy as least recently used."),
            ("ttl", "cache-rearchitecture-2024.md", "2024-01-30", 0.95, "The re-architecture document changes the eviction policy to time-to-live based expiry."),
        ]),
        ("search_index", "refresh_interval", [
            ("5m", "search-ops-checklist.pdf", "2022-12-06", 0.87, "The operations checklist expects the search index refresh interval to be every 5 minutes."),
            ("15m", "search-platform-roadmap.md", "2024-05-11", 0.94, "The product roadmap lowers the refresh cadence to 15 minutes for better stability."),
        ]),
        ("analytics_pipeline", "batch_window", [
            ("1m", "dashboard-legacy-spec.pdf", "2021-08-16", 0.89, "The legacy dashboard specification batches analytics events every minute."),
            ("3m", "analytics-modernization-adr.md", "2024-02-21", 0.96, "The analytics modernization ADR updates the batch window to 3 minutes."),
        ]),
        ("notification_service", "retry_limit", [
            ("3", "notification-runbook-v2.pdf", "2022-03-22", 0.9, "The notification runbook allows only three retries for failed messages."),
            ("5", "notification-resilience-2024.md", "2024-07-21", 0.97, "The resilience design review raises the retry limit to five."),
        ]),
        ("storage_broker", "max_connections", [
            ("200", "storage-runbook-2020.pdf", "2020-09-19", 0.83, "The original storage runbook caps the number of broker connections at 200."),
            ("350", "storage-capacity-plan-2024.md", "2024-04-09", 0.98, "The latest capacity plan increases the broker max connection limit to 350."),
        ]),
    ]

    for entity, attribute, variants in conflict_pairs:
        for value, source, timestamp, confidence, context in variants:
            add_fact(entity, attribute, value, source, timestamp, confidence, context)

    non_conflicting = [
        ("database", "replication_mode", "async", "database-ops-manual.pdf", "2024-01-10", 0.9, "The database replication mode remains asynchronous for the primary cluster."),
        ("database", "backup_window", "02:00-03:00 UTC", "backup-policy-2024.md", "2024-02-02", 0.92, "Nightly backups run from 02:00 to 03:00 UTC."),
        ("database", "read_replicas", "3", "capacity-plan-2024.md", "2024-03-05", 0.94, "The system maintains three read replicas for the primary database."),
        ("payments_service", "region", "us-east-1", "payments-ops-guide.pdf", "2024-04-10", 0.97, "The payments service is deployed in the us-east-1 region."),
        ("payments_service", "is_ha_enabled", "true", "payments-ha-checklist.md", "2024-07-01", 0.95, "The service is configured with high availability enabled."),
        ("auth_service", "sso_provider", "azure-ad", "identity-integration-doc.pdf", "2024-05-13", 0.96, "The application authenticates users through Azure AD."),
        ("auth_service", "mfa_required", "true", "security-policy-2024.md", "2024-05-20", 0.99, "Multi-factor authentication is required for all production users."),
        ("cache_cluster", "ttl_seconds", "1800", "cache-spec-v5.pdf", "2024-06-14", 0.95, "The cache entry TTL is set to 1800 seconds."),
        ("cache_cluster", "memory_limit_gb", "32", "platform-capacity-plan.md", "2024-04-07", 0.96, "The cluster capacity plan allocates 32 GB of memory."),
        ("search_index", "index_type", "vector", "search-architecture-doc.pdf", "2024-03-24", 0.94, "The search index is built using vector embeddings."),
        ("search_index", "shard_count", "6", "search-ops-benchmark.md", "2024-06-11", 0.9, "The search index uses six shards for load balancing."),
        ("analytics_pipeline", "data_source", "kafka", "analytics-integration-spec.md", "2024-03-18", 0.96, "The analytics pipeline consumes data from Kafka."),
        ("analytics_pipeline", "retention_days", "30", "retention-policy-2024.pdf", "2024-05-07", 0.93, "The analytics warehouse retains raw data for 30 days."),
        ("notification_service", "transport", "email", "notification-overview.pdf", "2024-04-22", 0.91, "The service primarily delivers notifications over email."),
        ("notification_service", "dead_letter_queue", "enabled", "resilience-design-v3.md", "2024-07-02", 0.97, "Dead-letter queues are enabled for failed notifications."),
        ("storage_broker", "storage_backend", "s3", "storage-architecture-doc.pdf", "2024-03-01", 0.94, "The broker stores payloads in S3."),
        ("storage_broker", "replication_factor", "3", "storage-capacity-plan.md", "2024-04-09", 0.93, "The storage broker replicates objects across three nodes."),
        ("job_scheduler", "cron_expression", "0 * * * *", "scheduler-runbook.pdf", "2024-02-09", 0.9, "The scheduler runs hourly jobs at minute 0."),
        ("job_scheduler", "max_retries", "4", "scheduler-resilience-plan.md", "2024-01-19", 0.92, "Job retries are capped at four attempts."),
        ("job_scheduler", "timezone", "UTC", "cloud-config-2024.md", "2024-05-02", 0.96, "All scheduled jobs are configured in UTC."),
        ("feature_flag_service", "rollout_strategy", "percentage", "feature-flag-policy.md", "2024-03-27", 0.9, "Feature flags use percentage-based rollouts."),
        ("feature_flag_service", "default_state", "off", "feature-flag-policy.md", "2024-03-27", 0.9, "New feature flags default to the off state."),
        ("user_profile_store", "read_replica_count", "2", "profile-service-spec.pdf", "2024-06-18", 0.94, "The profile store includes two read replicas."),
        ("user_profile_store", "cache_enabled", "true", "profile-service-spec.pdf", "2024-06-18", 0.95, "Caching is enabled for profile reads."),
        ("api_gateway", "rate_limit", "500rpm", "gateway-security-guide.pdf", "2024-05-17", 0.93, "The API gateway enforces a 500 requests per minute rate limit."),
        ("api_gateway", "tls_version", "1.3", "gateway-security-guide.pdf", "2024-05-17", 0.96, "The API gateway terminates TLS version 1.3."),
        ("message_bus", "retention_hours", "72", "messaging-policy.md", "2024-04-16", 0.92, "Messages are retained for 72 hours in the bus."),
        ("message_bus", "acks", "all", "messaging-policy.md", "2024-04-16", 0.94, "The bus requires all replicas to acknowledge a message."),
        ("ml_inference_service", "latency_budget_ms", "250", "inference-service-spec.pdf", "2024-04-30", 0.95, "The inference service keeps latency under 250 milliseconds."),
        ("ml_inference_service", "model_version", "v3.8", "inference-service-spec.pdf", "2024-04-30", 0.93, "The production inference model runs on version v3.8."),
    ]

    for entity, attribute, value, source, timestamp, confidence, context in non_conflicting:
        add_fact(entity, attribute, value, source, timestamp, confidence, context)

    for fact in facts:
        ProvenanceFact.model_validate(fact)

    return facts


def main() -> None:
    output_path = Path(__file__).resolve().parents[1] / "data" / "dataset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_corpus()
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2)
    print(f"Wrote {len(dataset)} facts to {output_path}")


if __name__ == "__main__":
    main()
