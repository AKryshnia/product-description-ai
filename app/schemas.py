"""Pydantic-схемы: входные данные, результат, замечания, ошибки API."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

NO_DATA = "данных нет"  # единая пометка для отсутствующих значений

MAX_SPECS = 100
MAX_NAME = 300
MAX_SPEC_VALUE = 2_000
MAX_DESCRIPTION = 20_000

_EXAMPLE_INPUT = {
    "name": "Манометр показывающий DM 02",
    "manufacturer": "Манотомь",
    "specs": {
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": "1,5",
        "присоединение": "G1/2",
        "диаметр корпуса": "100 мм",
    },
    "description": "Прибор предназначен для измерения избыточного давления...",
}


# ---------------------------------------------------------------- вход


class ProductInput(BaseModel):
    """Сведения об одном приборе. Единственный источник технических фактов."""

    # extra="allow": неизвестные поля не теряем молча — анализ полноты сообщит о них,
    # но как факты они не используются.
    model_config = ConfigDict(extra="allow", json_schema_extra={"examples": [_EXAMPLE_INPUT]})

    name: str = Field(max_length=MAX_NAME, description="Название товара (обязательно).")
    manufacturer: str | None = Field(default=None, max_length=MAX_NAME)
    specs: dict[str, str | int | float | None] | None = Field(
        default=None,
        description="Характеристики «название → значение». Значения лучше передавать строками.",
    )
    description: str | None = Field(
        default=None, max_length=MAX_DESCRIPTION, description="Текст или фрагмент паспорта."
    )

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("название товара не может быть пустым")
        return v

    @field_validator("specs", mode="before")
    @classmethod
    def _reject_bool_values(cls, v: Any) -> Any:
        # bool в Python — подкласс int, поэтому True молча стал бы «1»: отсекаем заранее.
        if isinstance(v, dict) and any(isinstance(x, bool) for x in v.values()):
            raise ValueError(
                "значение характеристики должно быть строкой или числом, не true/false"
            )
        return v

    @field_validator("specs")
    @classmethod
    def _normalize_specs(cls, v: dict | None) -> dict[str, str] | None:
        if v is None:
            return None
        out: dict[str, str] = {}
        for key, value in v.items():
            if not key.strip():
                raise ValueError("название характеристики не может быть пустым")
            text = "" if value is None else str(value)  # числа JSON -> текст; None -> пусто
            if len(text) > MAX_SPEC_VALUE:
                raise ValueError(f"значение характеристики «{key[:40]}» длиннее {MAX_SPEC_VALUE}")
            out[key] = text  # ключ и значение сохраняются без изменений
        if len(out) > MAX_SPECS:
            raise ValueError(f"слишком много характеристик (максимум {MAX_SPECS})")
        return out


# ---------------------------------------------------------------- замечания


class Severity(StrEnum):
    INFO = "info"  # справочно, на статус не влияет
    REVIEW = "review"  # нужна ручная проверка
    ERROR = "error"  # нарушение требований


class IssueCode(StrEnum):
    # анализ полноты входных данных
    MISSING_OPTIONAL = "missing_optional"
    EMPTY_VALUE = "empty_value"
    CONFLICT = "conflict"
    MANUAL_CHECK = "manual_check"
    INSUFFICIENT_DATA = "insufficient_data"
    # проверка результата
    LLM_OUTPUT_INVALID = "llm_output_invalid"
    TITLE_TOO_LONG = "title_too_long"
    TITLE_OFF_TOPIC = "title_off_topic"
    DESCRIPTION_LENGTH = "description_length"
    QUERIES_COUNT = "queries_count"
    QUERY_INVALID = "query_invalid"
    UNVERIFIED_NUMBER = "unverified_number"
    UNVERIFIED_UNIT = "unverified_unit"
    UNVERIFIED_DESIGNATION = "unverified_designation"
    UNIT_PAIR_UNCONFIRMED = "unit_pair_unconfirmed"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    BLOCK_WITHHELD = "block_withheld"
    REPAIR_FAILED = "repair_failed"


class Issue(BaseModel):
    code: IssueCode
    severity: Severity
    field: str | None = Field(default=None, description="Поле входа или блок результата.")
    message: str
    detail: str | None = Field(default=None, description="Фрагмент, вызвавший замечание.")


# ---------------------------------------------------------------- результат


class SpecItem(BaseModel):
    name: str
    value: str
    provided: bool = Field(default=True, description="False — значение не передано («данных нет»).")


def _collapse(text: str) -> str:
    """Пробельные последовательности -> один пробел. Длины считаем уже после этого."""
    return " ".join(text.split())


def _clean_items(items: list[str]) -> list[str]:
    return [c for c in (_collapse(i) for i in items) if c]


class Commercial(BaseModel):
    audience: list[str] = Field(description="Кому может подходить прибор.")
    tasks: list[str] = Field(description="Какие задачи решает.")
    scenarios: list[str] = Field(description="Сценарии применения, подтверждённые входом.")

    _clean = field_validator("audience", "tasks", "scenarios")(_clean_items)


class Draft(BaseModel):
    """Блоки, которые формирует LLM. Технические характеристики сюда намеренно не входят:
    поле `specs` в ответе модели игнорируется, список характеристик строит Python."""

    model_config = ConfigDict(extra="ignore")

    seo_title: str
    seo_description: str
    search_queries: list[str]
    commercial: Commercial

    @field_validator("seo_title", "seo_description")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = _collapse(v)
        if not v:
            raise ValueError("пустое значение")
        return v

    _clean_queries = field_validator("search_queries")(_clean_items)


class CardStatus(StrEnum):
    OK = "ok"  # все проверки пройдены
    NEEDS_REVIEW = "needs_review"  # есть пункты для ручной проверки
    PARTIAL = "partial"  # часть результата не сформирована или нарушает ограничения


class GenerationInfo(BaseModel):
    model: str | None = None
    llm_calls: int = 0
    repaired: bool = Field(default=False, description="Использована ли доп. генерация.")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProductCard(BaseModel):
    status: CardStatus
    name: str
    manufacturer: str | None
    seo_title: str | None
    seo_description: str | None
    specs: list[SpecItem] = Field(description="Копия входных характеристик; LLM их не меняет.")
    search_queries: list[str]
    commercial: Commercial | None
    missing_info: list[Issue] = Field(description="Что отсутствует или требует уточнения во входе.")
    warnings: list[Issue] = Field(description="Замечания программной проверки результата.")
    generation: GenerationInfo
    markdown: str = Field(default="", description="Тот же результат в Markdown для копирования.")


# ---------------------------------------------------------------- ошибки API


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict[str, str]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
