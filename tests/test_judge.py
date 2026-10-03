import asyncio
import json

import pytest


def test_ragas_and_open_judge_with_mock_provider(monkeypatch):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    pytest.importorskip("ragas")
    import httpx
    from openai import AsyncOpenAI

    from rag_system.evaluation import Judge

    def respond(request):
        body = request.content.decode()
        if "Evaluate answer correctness" in body:
            content = {"score": 1, "reason": "Matches reference"}
        elif "verdict" in body:
            content = {
                "statements": [
                    {
                        "statement": "Audit retention is 365 days.",
                        "reason": "Supported by context",
                        "verdict": 1,
                    }
                ]
            }
        else:
            content = {"statements": ["Audit retention is 365 days."]}
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
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    async def run():
        client = AsyncOpenAI(
            api_key="test-only",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(respond), trust_env=False
            ),
        )
        judge = Judge("gpt-4o-mini", client=client)
        try:
            return await judge.score(
                {
                    "kind": "open",
                    "question": "Audit retention?",
                    "answers": ["365 days"],
                    "rubric": "Must state 365 days",
                },
                {
                    "result": {
                        "answer": "Audit retention is 365 days.",
                        "abstained": False,
                    },
                    "retrieved": [{"text": "Audit retention is 365 days."}],
                },
            )
        finally:
            await client.close()

    faith, correctness = asyncio.run(run())
    assert faith == 1
    assert correctness["score"] == 1
