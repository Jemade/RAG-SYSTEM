# Engineering notes: RAG-SYSTEM

## Purpose and scope

Instrumented hybrid document retrieval and evaluation. This repository is an independently inspectable project; customer adoption, production scale and commercial readiness are not claimed without evidence.

## Request and data flow

Documents → chunks and embeddings → dense/BM25 candidates → reranking → answer/evidence checks → traces and evaluations.

## Implementation map

Primary implementation and review locations: `src/rag_system/core.py`, `src/rag_system/evaluation.py`, `src/rag_system/web.py`. Dependency manifests and `.github/workflows/` specify installation and automated checks. Read the source for exact contracts and data models.

## Local verification

From `.` in a configured virtual environment:

```sh
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

From the repository root, run `python scripts/repository_check.py` for documentation and tracked-file checks. CI evidence is available in [GitHub Actions](https://github.com/Jemade/RAG-SYSTEM/actions). Green hygiene checks alone do not mean application tests passed.

## Decisions and boundaries

Offline generation and embeddings demonstrate the pipeline. Semantic models require model downloads; a live LLM judge requires credentials and incurs provider cost. Dataset documents are synthetic.

Use the README's current run instructions and configuration examples. Keep provider credentials outside Git. Test changes against controlled fixtures before enabling external services. Health checks indicate process/service state, not end-to-end correctness.

## Review and operational evidence

[Review checklist](REVIEW_CHECKLIST.md) distinguishes repository evidence from outstanding human and deployment validation. Report measured workload, environment and method with any performance claim. Document incident fixes through reproducible issues and regression tests; do not invent user counts or peer reviews.

## Reuse and licensing

The root LICENSE describes the repository license. Third-party dependencies and assets retain their respective licenses.
