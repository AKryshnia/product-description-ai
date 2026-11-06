"""Адаптер GigaChat — единственное место, которое знает про его SDK, авторизацию и TLS.

Отвечает за: OAuth-токен (делает SDK), перевод сообщений в формат SDK, вызов API, перевод
ошибок SDK/HTTP во внутренние `LLMError`. Параметры: GIGACHAT_* (доступ) и LLM_* (общие).
"""

import logging
import ssl

import gigachat.exceptions as gx
import httpx
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from app.config import Settings
from app.llm import (
    Capabilities,
    LLMAPIError,
    LLMAuthError,
    LLMInvalidResponse,
    LLMNotConfigured,
    LLMRateLimited,
    LLMResult,
    LLMTimeout,
    LLMUnavailable,
    LLMUnsupported,
    Message,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "GigaChat-2-Pro"

_AUTH_HINT = "GigaChat отклонил авторизацию. Проверьте GIGACHAT_CREDENTIALS и GIGACHAT_SCOPE."
_TLS_HINT = (
    "Не удалось проверить TLS-сертификат GigaChat. Подключите корневой сертификат Минцифры "
    "через GIGACHAT_CA_BUNDLE_FILE (см. README); отключать проверку нельзя."
)


def _caused_by_tls(exc: BaseException | None) -> bool:
    while exc is not None:
        if isinstance(exc, ssl.SSLError):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _retry_after(exc: gx.ResponseError) -> int | None:
    try:
        return int(exc.headers.get("Retry-After")) if exc.headers else None
    except (TypeError, ValueError):
        return None


def _log(exc: Exception) -> None:
    # Только тип и код: текст исключения SDK может содержать тело ответа.
    log.warning("GigaChat: %s (status=%s)", type(exc).__name__, getattr(exc, "status_code", None))


class GigaChatClient:
    # response_format (json_schema) проверен на реальном API: GigaChat-2-Pro, -2-Max и -Pro
    # вернули обычный текст, а не JSON. Поэтому возможность не заявляем: формат задаёт промпт,
    # а гарантию даёт разбор и проверка ответа на стороне приложения.
    capabilities = Capabilities(json_schema=False)

    def __init__(self, settings: Settings):
        creds = settings.gigachat_credentials
        if creds is None or not creds.get_secret_value().strip():
            raise LLMNotConfigured(
                "Не задан GIGACHAT_CREDENTIALS: укажите ключ авторизации GigaChat в .env."
            )
        ca = settings.gigachat_ca_bundle_file
        if ca is not None and not ca.is_file():
            raise LLMNotConfigured(f"Файл сертификата GIGACHAT_CA_BUNDLE_FILE не найден: {ca}")
        self.model = settings.llm_model or DEFAULT_MODEL
        self._settings = settings
        self._giga = GigaChat(
            credentials=creds.get_secret_value().strip(),
            scope=settings.gigachat_scope,
            base_url=settings.gigachat_base_url,
            ca_bundle_file=str(ca) if ca else None,
            verify_ssl_certs=True,  # всегда: см. app/config.py
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

    def complete(self, messages: list[Message], *, json_schema: dict | None = None) -> LLMResult:
        if json_schema is not None:
            raise LLMUnsupported("Адаптер GigaChat не использует JSON Schema (см. README).")
        s = self._settings
        chat = Chat(
            model=self.model,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            messages=[
                Messages(role=MessagesRole(m["role"]), content=m["content"]) for m in messages
            ],
        )
        try:
            resp = self._giga.chat(chat)
        except (gx.AuthenticationError, gx.ForbiddenError) as e:
            _log(e)
            raise LLMAuthError(_AUTH_HINT) from e
        except gx.RateLimitError as e:
            _log(e)
            raise LLMRateLimited(retry_after=_retry_after(e)) from e
        except gx.ResponseError as e:
            _log(e)
            if "/oauth" in str(e.url):
                # на этапе получения токена любая ошибка — это ключ или scope (неверный ключ
                # реальный API отклоняет кодом 400, а не 401)
                raise LLMAuthError(_AUTH_HINT) from e
            raise LLMAPIError(f"GigaChat вернул ошибку (HTTP {e.status_code}).") from e
        except httpx.TimeoutException as e:
            _log(e)
            raise LLMTimeout("GigaChat не ответил за отведённое время.") from e
        except httpx.TransportError as e:
            _log(e)
            raise LLMUnavailable(
                _TLS_HINT if _caused_by_tls(e) else "Не удалось подключиться к GigaChat."
            ) from e
        except gx.GigaChatException as e:
            _log(e)
            raise LLMAPIError("GigaChat вернул ошибку.") from e

        if not resp.choices:
            raise LLMInvalidResponse("GigaChat вернул ответ без вариантов генерации.")
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResult(
            text=choice.message.content or "",
            model=resp.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=choice.finish_reason,
        )

    def close(self) -> None:
        self._giga.close()
