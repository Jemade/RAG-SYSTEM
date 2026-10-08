"""Authentication regression checks for the trace inspector."""

from fastapi.testclient import TestClient

from rag_system.web import create_app


def test_trace_api_requires_configured_token(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INSPECTOR_TOKEN", "test-secret-token")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/traces").status_code == 401
        assert client.get("/api/traces", headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = client.get(
            "/api/traces", headers={"Authorization": "Bearer test-secret-token"}
        )
        assert response.status_code == 200
        assert response.json() == []
        assert client.get("/api/traces/unknown").status_code == 401


def test_local_inspector_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_INSPECTOR_TOKEN", raising=False)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/traces").json() == []
