import json

import pytest

from rag_system.core import Pipeline
from rag_system.evaluation import evaluate, exact_match, retrieval_metrics


@pytest.fixture
def pipeline(tmp_path):
    p = Pipeline(tmp_path)
    p.ingest("data/corpus")
    yield p
    p.close()


def test_retrieval_and_trace(pipeline):
    t = pipeline.query("Which team owns NS-731?")
    assert any(c["source"] == "catalog.md" for c in t["retrieved"])
    assert t["citation_integrity"] and t["has_evidence"]
    assert all(
        "dense_cosine" in c and "bm25" in c and "rerank_score" in c
        for c in t["candidates"]
    )
    assert pipeline.db.execute("SELECT count(*) FROM traces").fetchone()[0] == 1


def test_errors_are_logged(pipeline):
    with pytest.raises(ValueError):
        pipeline.query("")
    payload = json.loads(
        pipeline.db.execute("SELECT payload FROM traces").fetchone()[0]
    )
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"


def test_bad_citation_detected(pipeline, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "generate",
        lambda *a: (
            {
                "answer": "made up",
                "abstained": False,
                "citations": [{"chunk_id": "fake", "quote": "fake"}],
            },
            {},
        ),
    )
    t = pipeline.query("Backup retention?")
    assert not t["citation_integrity"]
    assert not t["has_evidence"]


def test_exact_match_not_substring():
    assert exact_match("  AES-256! ", ["aes256"]) == 1
    assert exact_match("The value is 5, or maybe 50.", ["5"]) == 0


def test_duplicate_sources_do_not_inflate_metrics():
    m = retrieval_metrics(
        [{"source": "a"}, {"source": "a"}, {"source": "b"}], ["a", "b"]
    )
    assert m["recall"] == 1
    assert m["mrr"] == 1
    assert 0 <= m["ndcg"] <= 1


def test_reingest_removes_stale_chunks(pipeline, tmp_path):
    docs = tmp_path / "new"
    docs.mkdir()
    (docs / "only.md").write_text("The new archive location is planet Mars.")
    pipeline.ingest(docs)
    t = pipeline.query("archive location")
    assert {c["source"] for c in t["retrieved"]} == {"only.md"}
    assert len(pipeline.vector.get_collections().collections) == 1


def test_dataset_references_exist():
    from pathlib import Path

    rows = [
        json.loads(line)
        for line in Path("data/evaluation.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 60
    assert len({r["id"] for r in rows}) == 60
    for r in rows:
        for source in r["sources"]:
            assert (Path("data/corpus") / source).exists()
    assert {r["split"] for r in rows} == {"dev", "holdout"}


def test_eval_does_not_count_skipped_judges(pipeline, tmp_path):
    report = evaluate(pipeline, "data/evaluation.jsonl", tmp_path / "report.json")
    assert report["count"] == 60
    assert report["correctness_scored"] == 40
    assert all(r["correctness"] is None for r in report["rows"] if r["kind"] == "open")
    assert not any("pipeline_error" in r["failures"] for r in report["rows"])


def test_manifest_survives_failed_ingestion(pipeline, monkeypatch):
    before = pipeline.manifest.read_text()
    monkeypatch.setattr(
        pipeline, "embed", lambda _: (_ for _ in ()).throw(RuntimeError("test failure"))
    )
    with pytest.raises(RuntimeError):
        pipeline.ingest("data/corpus")
    assert pipeline.manifest.read_text() == before
    assert len(pipeline.vector.get_collections().collections) == 1


def test_triage_links_observed_failures(pipeline, tmp_path):
    from rag_system.triage import export_failures

    report = tmp_path / "report.json"
    evaluate(pipeline, "data/evaluation.jsonl", report)
    target = tmp_path / "triage.jsonl"
    result = export_failures(pipeline.root, report, target)
    assert result["observed_failures"] > 0
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert all(
        r["trace_id"] and r["review_status"] == "needs_human_reference" for r in rows
    )


def test_readonly_dashboard(pipeline):
    from fastapi.testclient import TestClient

    from rag_system.web import create_app

    trace = pipeline.query("archive bucket")
    client = TestClient(create_app(pipeline.root))
    assert client.get("/").status_code == 200
    assert client.get("/api/traces").json()[0]["id"] == trace["id"]
    assert client.get("/api/traces/" + trace["id"]).json()["query"] == "archive bucket"
    assert client.get("/api/traces/unknown").status_code == 404


def test_semantic_reranker_is_used(pipeline):
    class FakeReranker:
        def predict(self, pairs):
            return [100.0 if "NS-731" in text else -100.0 for _, text in pairs]

    pipeline.reranker = FakeReranker()
    trace = pipeline.query("dataset owners")
    assert trace["retrieved"][0]["source"] == "catalog.md"
    assert trace["retrieved"][0]["rerank_score"] == 100.0


def test_generation_provider_and_usage_logged(pipeline, monkeypatch):
    import httpx
    import openai

    real_client = openai.OpenAI

    def respond(request):
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"
        evidence = json.loads(body["messages"][1]["content"])["evidence"][0]
        result = {
            "answer": "test answer",
            "abstained": False,
            "citations": [{"chunk_id": evidence["id"], "quote": evidence["text"][:30]}],
        }
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(result)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: real_client(
            api_key="test-only",
            http_client=httpx.Client(
                transport=httpx.MockTransport(respond), trust_env=False
            ),
            **kwargs,
        ),
    )
    trace = pipeline.query("backup retention", provider="gpt-4o-mini")
    assert trace["citation_integrity"]
    assert trace["usage"]["total_tokens"] == 15
