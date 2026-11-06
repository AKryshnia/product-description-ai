"""FastAPI-приложение: POST /generate и GET /health. Swagger UI — на /docs."""

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.llm import Capabilities, LLMClient, LLMError, LLMNotConfigured
from app.providers import create_client
from app.schemas import ErrorBody, ErrorResponse, ProductCard, ProductInput
from app.service import generate_card

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


class _NotConfigured:
    """Заглушка вместо клиента без ключа: приложение стартует, а /generate отвечает 503.

    Так некорректные входные данные получают 422 независимо от настроек LLM, а Swagger и
    /health работают до того, как ключ задан.
    """

    model = ""
    capabilities = Capabilities()

    def __init__(self, reason: str):
        self.reason = reason

    def complete(self, messages, *, json_schema=None):
        raise LLMNotConfigured(self.reason)


@lru_cache
def get_llm() -> LLMClient:
    """Один клиент на процесс (токен провайдера переиспользуется SDK).
    Настройки читаются при первом обращении; после правки .env сервис нужно перезапустить."""
    try:
        return create_client(get_settings())
    except LLMNotConfigured as e:
        return _NotConfigured(e.message)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if get_llm.cache_info().currsize:
        close = getattr(get_llm(), "close", None)
        if close:
            close()


MAX_BODY_BYTES = 1_000_000  # 1 МБ: реальный запрос — единицы килобайт


class BodyLimitMiddleware:
    """Отклоняет слишком большое тело запроса, не дочитывая его (обычная и chunked-передача)."""

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            return await _too_large()(scope, receive, send)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:  # обработчик ошибок приложения вернёт 413
                    raise StarletteHTTPException(status_code=413)
            return message

        await self.app(scope, limited_receive, send)


def _too_large() -> JSONResponse:
    code, message = _HTTP_ERRORS[413]
    return _error(413, code, message)


app = FastAPI(
    title="AI-помощник для подготовки товарных описаний",
    version="0.1.0",
    description=(
        "Черновики карточек контрольно-измерительных приборов для B2B-каталога. "
        "Технические значения берутся только из входных данных; результат проходит "
        "программную проверку, которая эвристическая и достоверности не гарантирует: статус `ok` "
        "означает лишь «замечаний не найдено». Это черновик для контент-менеджера, который "
        "проверяют вручную перед публикацией."
    ),
    lifespan=lifespan,
)

app.add_middleware(BodyLimitMiddleware)

_ERRORS = {
    code: {"model": ErrorResponse, "description": text}
    for code, text in {
        400: "Некорректный JSON.",
        413: "Тело запроса больше 1 МБ.",
        422: "Данные не прошли проверку типов или обязательных полей.",
        429: "Превышен лимит запросов GigaChat (см. заголовок Retry-After).",
        502: "Ошибка авторизации или API GigaChat, недоступность сети.",
        503: "Не настроен ключ GigaChat.",
        504: "Тайм-аут GigaChat.",
    }.items()
}


def _error(status: int, code: str, message: str, details=None, headers=None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(
        status_code=status, content=body.model_dump(exclude_none=True), headers=headers
    )


_HTTP_ERRORS = {
    404: ("not_found", "Маршрут не найден."),
    405: ("method_not_allowed", "Метод не поддерживается для этого маршрута."),
    413: ("payload_too_large", "Размер запроса превышает допустимый (1 МБ)."),
}


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code, message = _HTTP_ERRORS.get(
        exc.status_code, ("http_error", "Запрос не может быть обработан.")
    )
    return _error(exc.status_code, code, message, headers=getattr(exc, "headers", None))


@app.exception_handler(LLMError)
async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    return _error(exc.http_status, exc.code, exc.message, headers=headers)


@app.exception_handler(RequestValidationError)
async def _invalid_input(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    if any(e["type"] == "json_invalid" for e in errors):
        return _error(400, "invalid_json", "Тело запроса не является корректным JSON.")
    details = []
    for e in errors:
        loc = e["loc"][1:] if e["loc"][:1] == ("body",) else e["loc"]
        # исходное значение (input) намеренно не возвращаем: оно может быть большим
        # или чувствительным
        details.append({"field": ".".join(str(p) for p in loc) or "body", "message": e["msg"]})
    return _error(422, "invalid_input", "Входные данные не прошли проверку.", details)


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
    log.exception("Необработанная ошибка")
    return _error(500, "internal_error", "Внутренняя ошибка сервиса.")


@app.post(
    "/generate",
    response_model=ProductCard,
    responses={
        **_ERRORS,
        200: {"content": {"text/markdown": {}}, "description": "JSON или Markdown"},
    },
    summary="Сформировать черновик карточки товара",
)
def generate(
    product: ProductInput,
    llm: Annotated[LLMClient, Depends(get_llm)],
    format: Annotated[
        Literal["json", "markdown"],
        Query(description="json — структура для интеграций; markdown — текст для копирования."),
    ] = "json",
):
    # обычная функция (не async): блокирующий вызов SDK идёт в пуле потоков
    card = generate_card(product, llm)
    if format == "markdown":
        return PlainTextResponse(card.markdown, media_type="text/markdown; charset=utf-8")
    return card


@app.get("/health", summary="Состояние сервиса")
def health(llm: Annotated[LLMClient, Depends(get_llm)]) -> dict:
    configured = not isinstance(llm, _NotConfigured)
    return {
        "status": "ok",
        "llm_configured": configured,
        "provider": get_settings().llm_provider,
        "model": llm.model if configured else None,
    }
