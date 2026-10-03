# Validation

## Checks performed

- Local unit/integration suite: **14 tests passed** on Python 3.12.
- Ruff lint and formatting checks passed; `pip check` found no broken requirements.
- Ingested all 12 sample documents into persistent Qdrant local storage.
- Ran all 60 questions using the deterministic offline configuration.
- Downloaded and loaded the actual MiniLM embedding model and MS MARCO cross-encoder, ingested the corpus, and ran all 60 questions using neural retrieval with the offline extractive generator.
- Exercised the Ragas 0.4.3 faithfulness adapter and open-answer correctness judge end to end through a mocked HTTP provider. This verifies adapter calls and structured response handling without claiming real model quality.
- Exercised the OpenAI generation path through mocked HTTP, including evidence serialization, cited quotes and token-usage logging. Provider clients are closed after generation.
- Verified the dashboard routes, trace list, trace detail and missing-trace response through FastAPI TestClient.
- Verified failed-query logging, invalid citations, stale-index replacement, failed-ingest recovery, strict exact match, source metric deduplication, failure triage and skipped-judge accounting.

Actual run artifacts are in [examples/reports](examples/reports). They explicitly identify retrieval mode, generator, judge status and scored coverage. The offline generator returns full evidence sentences and achieved **0/40 strict short-value exact matches** in both runs. This is a measured limitation of that baseline, not a successful answer-quality benchmark. The 20 open-answer cases were unscored because no live judge was configured.

## Not verified live

No paid OpenAI generation or judge evaluation was run: no API credential was configured for this project. The repository provides the full keyed commands. CI adapter tests use a mocked provider and do not establish real-model correctness, hallucination resistance or injection resistance.

A browser screenshot check could not run because the environment returned an invalid Chromium download. API/HTML route tests passed, but visual browser QA is not claimed.

This is a local single-user implementation. Public hosting, authentication, distributed scaling and real customer workload measurements are not part of the verified scope.
