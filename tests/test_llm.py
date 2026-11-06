"""Адаптер GigaChat: форма запроса и отображение ошибок SDK/HTTP в наши типизированные ошибки.

Здесь SDK подменяется. Это проверка нашей логики, а не доказательство работы настоящего
GigaChat — реальный контракт проверяется scripts/live_check.py.
"""

import ssl
from types import SimpleNamespace

import gigachat.exceptions as gx
import httpx
import pytest

from app.config import Settings
from app.llm import (
    LLMAPIError,
    LLMAuthError,
    LLMNotConfigured,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
)
from app.providers.gigachat import GigaChatClient

URL = "https://gigachat.example/api/v1/chat/completions"
SECRET = "SECRET-KEY-123"
MESSAGES = [{"role": "system", "content": "правила"}, {"role": "user", "content": "данные"}]


def settings(**kw) -> Settings:
    return Settings(_env_file=None, gigachat_credentials=SECRET, gigachat_ca_bundle_file=None, **kw)


def reply(content, model="GigaChat-2-Pro:2.0.1"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22),
    )


def client_with(chat):
    c = GigaChatClient(settings())
    c._giga.chat = chat
    return c


def resp_error(cls, status, headers=None, body=b"", url=URL):
    return cls(url, status, body, httpx.Headers(headers or {}))


def test_missing_credentials_is_a_configuration_error():
    with pytest.raises(LLMNotConfigured) as e:
        GigaChatClient(Settings(_env_file=None, gigachat_credentials=None))
    assert "GIGACHAT_CREDENTIALS" in str(e.value)


def test_missing_ca_bundle_file_is_a_configuration_error(tmp_path):
    with pytest.raises(LLMNotConfigured) as e:
        GigaChatClient(
            Settings(
                _env_file=None,
                gigachat_credentials=SECRET,
                gigachat_ca_bundle_file=tmp_path / "нет.crt",
            )
        )
    assert "нет.crt" in str(e.value) and SECRET not in str(e.value)


def test_request_has_model_messages_and_no_response_format():
    seen = []
    c = client_with(lambda chat: seen.append(chat) or reply("{}"))
    c.complete(MESSAGES)
    chat = seen[0]
    assert chat.model == "GigaChat-2-Pro" and chat.temperature == 0.0
    assert [(m.role, m.content) for m in chat.messages] == [
        ("system", "правила"),
        ("user", "данные"),
    ]
    # response_format на проверенном хосте не принуждает JSON (см. README) — не отправляем
    assert chat.response_format is None


def test_result_carries_text_actual_model_and_usage():
    r = client_with(lambda chat: reply('{"a": 1}')).complete(MESSAGES)
    assert (r.text, r.model, r.prompt_tokens, r.completion_tokens) == (
        '{"a": 1}',
        "GigaChat-2-Pro:2.0.1",
        11,
        22,
    )


def test_empty_content_becomes_empty_text_for_the_parser_to_reject():
    assert client_with(lambda chat: reply(None)).complete(MESSAGES).text == ""


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (resp_error(gx.AuthenticationError, 401), LLMAuthError),
        (resp_error(gx.ForbiddenError, 403), LLMAuthError),
        (resp_error(gx.RateLimitError, 429, {"Retry-After": "7"}), LLMRateLimited),
        (httpx.ReadTimeout("t"), LLMTimeout),
        (httpx.ConnectTimeout("t"), LLMTimeout),
        (httpx.ConnectError("нет сети"), LLMUnavailable),
        (resp_error(gx.ServerError, 503), LLMAPIError),
        (resp_error(gx.BadRequestError, 400), LLMAPIError),
        (resp_error(gx.ResponseError, 418), LLMAPIError),
        (gx.GigaChatException("прочее"), LLMAPIError),
    ],
)
def test_sdk_and_transport_errors_map_to_typed_errors(raised, expected):
    def boom(chat):
        raise raised

    with pytest.raises(expected):
        client_with(boom).complete(MESSAGES)


def test_rate_limit_keeps_retry_after_seconds():
    def boom(chat):
        raise resp_error(gx.RateLimitError, 429, {"Retry-After": "7"})

    with pytest.raises(LLMRateLimited) as e:
        client_with(boom).complete(MESSAGES)
    assert e.value.retry_after == 7


def test_tls_failure_points_to_ca_bundle_setting():
    def boom(chat):
        err = httpx.ConnectError("verify failed")
        err.__cause__ = ssl.SSLCertVerificationError("self-signed certificate in certificate chain")
        raise err

    with pytest.raises(LLMUnavailable) as e:
        client_with(boom).complete(MESSAGES)
    assert "GIGACHAT_CA_BUNDLE_FILE" in str(e.value)


def test_errors_never_leak_credentials_or_upstream_bodies():
    def boom(chat):
        raise resp_error(gx.AuthenticationError, 401, body=f"bad key {SECRET}".encode())

    with pytest.raises(LLMAuthError) as e:
        client_with(boom).complete(MESSAGES)
    assert SECRET not in str(e.value) and SECRET not in repr(e.value)


OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"


def test_any_error_at_the_token_step_means_a_credentials_problem():
    """Реальный API на неверный ключ отвечает на /oauth кодом 400, а не 401 (проверено вживую)."""

    def boom(chat):
        raise resp_error(gx.BadRequestError, 400, url=OAUTH_URL)

    with pytest.raises(LLMAuthError) as e:
        client_with(boom).complete(MESSAGES)
    assert "GIGACHAT_CREDENTIALS" in str(e.value)


def test_the_same_400_on_the_chat_endpoint_is_a_plain_api_error():
    def boom(chat):
        raise resp_error(gx.BadRequestError, 400)

    with pytest.raises(LLMAPIError):
        client_with(boom).complete(MESSAGES)
