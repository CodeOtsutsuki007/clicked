"""Request size and value limits (issue #713).

Over-length or non-positive input must be rejected with 422 by validation,
before any OpenAI (or Weaviate) client is created.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from main import (
    MAX_SHORT_TEXT_LEN,
    MAX_TEXT_LEN,
    ProposalSummariseRequest,
    TransferAnalyseRequest,
    app,
)

client = TestClient(app)

_LONG_TEXT = "x" * (MAX_TEXT_LEN + 1)
_LONG_SHORT = "x" * (MAX_SHORT_TEXT_LEN + 1)

_CHAT = {"message": "hello", "conversation_id": "conv-1"}
_TRANSFER = {"amount": 100.0, "sender": "GABC", "recipient": "GDEF", "memo": "rent"}
_PROPOSAL = {"title": "Fund outreach", "description": "Grow the user base.", "amount": 500.0}
_INDEX = {"messageId": "msg-1", "conversationId": "conv-1", "senderId": "u-1", "content": "hi"}


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/chat", {**_CHAT, "message": _LONG_TEXT}),
        ("post", "/chat", {**_CHAT, "message": ""}),
        ("post", "/transfers/analyse", {**_TRANSFER, "memo": _LONG_SHORT}),
        ("post", "/transfers/analyse", {**_TRANSFER, "amount": 0}),
        ("post", "/transfers/analyse", {**_TRANSFER, "amount": -5.0}),
        ("post", "/proposals/summarise", {**_PROPOSAL, "title": _LONG_SHORT}),
        ("post", "/proposals/summarise", {**_PROPOSAL, "description": _LONG_TEXT}),
        ("post", "/proposals/summarise", {**_PROPOSAL, "description": ""}),
        ("post", "/proposals/summarise", {**_PROPOSAL, "amount": 0}),
        ("post", "/proposals/summarise", {**_PROPOSAL, "amount": -1.0}),
        ("post", "/index/message", {**_INDEX, "content": _LONG_TEXT}),
        ("post", "/index/message", {**_INDEX, "content": ""}),
    ],
)
def test_invalid_body_returns_422_without_calling_openai(method, path, payload):
    with (
        patch("main.OpenAI") as mock_openai,
        patch("main.weaviate.connect_to_local") as mock_weaviate,
    ):
        response = getattr(client, method)(path, json=payload)

    assert response.status_code == 422
    mock_openai.assert_not_called()
    mock_weaviate.assert_not_called()


@pytest.mark.parametrize("q", ["", _LONG_TEXT])
def test_invalid_search_query_returns_422_without_calling_openai(q):
    with (
        patch("main.OpenAI") as mock_openai,
        patch("main.weaviate.connect_to_local") as mock_weaviate,
    ):
        response = client.get("/search", params={"q": q, "conversationId": "conv-1"})

    assert response.status_code == 422
    mock_openai.assert_not_called()
    mock_weaviate.assert_not_called()


@pytest.mark.parametrize(
    ("request_model", "body"),
    [(TransferAnalyseRequest, _TRANSFER), (ProposalSummariseRequest, _PROPOSAL)],
)
@pytest.mark.parametrize("amount", [float("inf"), float("nan")])
def test_request_amount_must_be_finite(request_model, body, amount):
    with pytest.raises(ValidationError) as exc_info:
        request_model.model_validate({**body, "amount": amount})

    assert exc_info.value.errors()[0]["type"] == "finite_number"


def test_text_at_limit_is_accepted():
    with patch("main.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content = "ok"
        response = client.post("/chat", json={**_CHAT, "message": "x" * MAX_TEXT_LEN})

    assert response.status_code == 200
