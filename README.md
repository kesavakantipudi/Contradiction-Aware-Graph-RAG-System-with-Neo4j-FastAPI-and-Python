# Contradiction-Aware Graph RAG

This project compares a top-K TF-IDF vector baseline with a Neo4j-backed Graph RAG API. The graph route retrieves all facts for the identified entity/attribute pair, then detects disagreement in deterministic Python code. Pydantic validates its structured response. Conflict resolution is reported separately as a heuristic and never hides conflicting sources.

## Run with Docker Compose

1. Copy `.env.example` to `.env` and set a local Neo4j password (at least 8 characters). `.env` is excluded from both Git and the Docker build context.
2. Start Neo4j and the API:
   `docker compose up -d --build`
3. Wait until both services are healthy:
   `docker compose ps`
4. Verify `http://localhost:7474` and `http://localhost:8000/health`.
5. Run the API-based evaluation:
   `python scripts/evaluate.py`

The evaluation writes `evaluation_results.json` at the project root and includes conflict recall, conflict precision, baseline silent-pick rate, and a worked conflicting response.

## API

- `GET /health`
- `POST /api/baseline` with `{"question":"What is the database connection timeout?"}`
- `POST /api/ask` with `{"question":"What is the database connection timeout?"}`

The Graph RAG response includes `answer`, `agreement_status` (`agree`, `conflict`, or `single-source`), and a `sources` list of `{value, source, timestamp}` objects.

## Regenerate and check the corpus

`python scripts/generate_dataset.py` writes `data/dataset.json`. The generated dataset contains 46 facts, 14 entities, and eight deliberately conflicting entity/attribute groups.

Run focused tests with:

`python -m unittest discover -s tests -v`

To stop the services without deleting Neo4j data, run `docker compose down`. Use `docker compose down -v` only if you intentionally want to erase the database volume.
