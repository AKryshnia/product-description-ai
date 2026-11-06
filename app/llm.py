"""Универсальный интерфейс LLM — всё, что бизнес-логика знает о языковых моделях.

Здесь нет ни SDK, ни форматов конкретного провайдера. Чтобы подключить другого провайдера,
достаточно: реализовать `LLMClient` в app/providers/<имя>.py (перевод сообщений, вызов API,
отображение его ошибок в `LLMError`) и добавить запись в реестр app/providers/__init__.py.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass(frozen=True)
class Capabilities:
    """Что адаптер умеет сверх обычного чата. Поле добавляется, когда возникает реальная
    потребность; значение True должно быть проверено на реальном API, а не предположено."""

    json_schema: bool = False  # принуждает ответ соответствовать переданной JSON Schema


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str | None = None  # фактическая модель из ответа (может содержать версию)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    model: str  # идентификатор настроенной модели
    capabilities: Capabilities

    def complete(self, messages: list[Message], *, json_schema: dict | None = None) -> LLMResult:
        """Отправляет диалог, возвращает текст ответа.

        `json_schema` передаётся, только если `capabilities.json_schema` истинно; иначе адаптер
        вправе выбросить `LLMUnsupported`. Все сбои провайдера — только через `LLMError`."""
        ...


# ---------------------------------------------------------------- единые ошибки
# Общие для всех провайдеров. Сообщения — для пользователя API: без ключей и тел ответов.


class LLMError(Exception):
    code = "llm_error"
    http_status = 502
    default_message = "Ошибка при обращении к языковой модели."

    def __init__(self, message: str | None = None, *, retry_after: int | None = None):
        self.message = message or self.default_message
        self.retry_after = retry_after
        super().__init__(self.message)


class LLMNotConfigured(LLMError):
    code = "llm_not_configured"
    http_status = 503
    default_message = "Клиент LLM не настроен."


class LLMAuthError(LLMError):
    code = "llm_auth_failed"
    default_message = "Провайдер отклонил авторизацию. Проверьте ключ и параметры доступа в .env."


class LLMRateLimited(LLMError):
    code = "llm_rate_limited"
    http_status = 429
    default_message = "Превышен лимит запросов к провайдеру LLM. Повторите позже."


class LLMTimeout(LLMError):
    code = "llm_timeout"
    http_status = 504
    default_message = "Провайдер LLM не ответил за отведённое время."


class LLMUnavailable(LLMError):
    code = "llm_unavailable"
    default_message = "Не удалось подключиться к провайдеру LLM."


class LLMAPIError(LLMError):
    code = "llm_api_error"
    default_message = "Провайдер LLM вернул ошибку."


class LLMInvalidResponse(LLMError):
    """Ответ провайдера нельзя разобрать как ответ чата (не путать с плохим текстом модели)."""

    code = "llm_invalid_response"
    default_message = "Провайдер LLM вернул ответ неожиданной структуры."


class LLMUnsupported(LLMError):
    code = "llm_unsupported_capability"
    http_status = 501
    default_message = "Выбранный провайдер не поддерживает запрошенную возможность."
