"""Адаптер GigaChat на уровне HTTP: настоящий SDK разбирает подставные ответы.

В отличие от tests/test_providers.py, здесь не подменяется `chat()`: ответы SDK, OAuth и разбор
JSON выполняет сам SDK через `httpx.MockTransport`. Реальных токенов нет — только метки.
"""

import logging
import time
import traceback

import httpx
import pytest
from fastapi.testclient import TestClient
from gigachat.models import AccessToken

from app.config import Settings
from app.llm import (
    LLMAPIError,
    LLMAuthError,
    LLMInvalidResponse,
    LLMRateLimited,
    LLMUnavailable,
)
from app.main import app, get_llm
from app.providers.gigachat import GigaChatClient
from tests.conftest import PDF_PRODUCT

TOKEN = "TEST_ONLY_ACCESS_TOKEN_0123456789abcdef"
MESSAGES = [{"role": "user", "content": "привет"}]
GOOD_CHAT = {
    "choices": [
        {"message": {"role": "assistant", "content": "{}"}, "index": 0, "finish_reason": "stop"}
    ],
    "created": 1,
    "model": "GigaChat-2-Pro",
    "object": "chat.completion",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def make_client(chat=None, oauth=None) -> GigaChatClient:
    """`chat` и `oauth` — готовый `httpx.Response`-совместимый ответ: (status, json | bytes)."""
    settings = Settings(
        _env_file=None,
        gigachat_credentials="placeholder-not-a-real-key",
        gigachat_ca_bundle_file=None,
        llm_max_retries=0,  # без повторов SDK: тест не должен ждать
    )
    client = GigaChatClient(settings)

    def respond(spec):
        def handler(request: httpx.Request) -> httpx.Response:
            status, body = spec
            if isinstance(body, bytes):
                return httpx.Response(status, content=body, request=request)
            return httpx.Response(status, json=body, request=request)

        return handler

    giga = client._giga
    giga._client_instance = httpx.Client(
        transport=httpx.MockTransport(respond(chat or (200, GOOD_CHAT))),
        base_url="https://mock.invalid/api/v1",
    )
    giga._auth_client_instance = httpx.Client(
        transport=httpx.MockTransport(
            respond(oauth or (200, {"access_token": TOKEN, "expires_at": _in_an_hour()}))
        )
    )
    if oauth is None:  # токен уже получен: проверяем только чат
        giga._access_token = AccessToken(access_token=TOKEN, expires_at=_in_an_hour())
    return client


def _in_an_hour() -> int:
    return int((time.time() + 3600) * 1000)


def test_the_helper_works_with_a_valid_response():
    assert make_client().complete(MESSAGES).text == "{}"


# ---------------------------------------------------------------- разбор внешнего ответа


@pytest.mark.parametrize(
    ("scenario", "chat", "oauth"),
    [
        ("200 с невалидным JSON", (200, b"not json"), None),
        ("200 с пустым объектом", (200, {}), None),
        ("200 с массивом", (200, []), None),
        ("200 с неверными типами", (200, {"choices": "x", "usage": 1}), None),
        ("OAuth без expires_at", None, (200, {"access_token": TOKEN})),
        ("OAuth с невалидным JSON", None, (200, b"<html>")),
        ("OAuth с массивом", None, (200, [TOKEN])),
    ],
)
def test_malformed_provider_response_becomes_a_typed_error(scenario, chat, oauth):
    with pytest.raises(LLMInvalidResponse):
        make_client(chat, oauth).complete(MESSAGES)


@pytest.mark.parametrize(
    "oauth", [(200, {"access_token": TOKEN}), (200, [TOKEN])], ids=["no_expiry", "array"]
)
def test_malformed_oauth_never_leaks_token_fragments_into_logs_or_tracebacks(oauth, caplog):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(LLMInvalidResponse) as caught:
        make_client(None, oauth).complete(MESSAGES)
    shown = "".join(traceback.format_exception(caught.value)) + caplog.text + str(caught.value)
    assert "TEST_ONLY" not in shown and TOKEN[-12:] not in shown


def test_malformed_response_through_the_api_is_a_controlled_502_not_a_500(caplog):
    caplog.set_level(logging.DEBUG)
    app.dependency_overrides[get_llm] = lambda: make_client((200, b"not json"))
    try:
        r = TestClient(app, raise_server_exceptions=False).post("/generate", json=PDF_PRODUCT)
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 502 and r.json()["error"]["code"] == "llm_invalid_response"
    assert "Необработанная ошибка" not in caplog.text and "Traceback" not in caplog.text


def test_unexpected_errors_in_our_own_code_are_not_swallowed():
    """Перехватывается только граница с SDK: ошибка в нашем сообщении об ошибке не маскируется."""
    client = make_client()
    with pytest.raises(KeyError):
        client.complete([{"role": "user"}])  # нет content: ошибка вызывающего кода, не провайдера


# ---------------------------------------------------------------- классификация ошибок


@pytest.mark.parametrize("status", [400, 401, 403])
def test_oauth_rejection_is_an_authorization_error(status):
    with pytest.raises(LLMAuthError):
        make_client(None, (status, {"message": "invalid credentials"})).complete(MESSAGES)


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_oauth_server_failure_is_not_blamed_on_the_key(status):
    with pytest.raises(LLMUnavailable) as caught:
        make_client(None, (status, {"message": "down"})).complete(MESSAGES)
    assert "GIGACHAT_CREDENTIALS" not in caught.value.message  # не отправляем менять исправный ключ


def test_oauth_rate_limit_is_reported_as_rate_limit():
    with pytest.raises(LLMRateLimited):
        make_client(None, (429, {"message": "slow down"})).complete(MESSAGES)


def test_other_oauth_errors_are_api_errors():
    with pytest.raises(LLMAPIError) as caught:
        make_client(None, (404, {"message": "no such"})).complete(MESSAGES)
    assert not isinstance(caught.value, LLMAuthError)


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, LLMAuthError), (403, LLMAuthError), (429, LLMRateLimited), (500, LLMAPIError)],
)
def test_chat_errors_keep_their_classification(status, error):
    with pytest.raises(error):
        make_client((status, {"message": "x"})).complete(MESSAGES)


def test_error_log_contains_only_the_category_and_status(caplog):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(LLMUnavailable):
        make_client(None, (503, {"message": f"BODY_MARKER {TOKEN}"})).complete(MESSAGES)
    assert "TEST_ONLY" not in caplog.text and "BODY_MARKER" not in caplog.text
    assert "stage=oauth" in caplog.text and "status=503" in caplog.text
