"""Общие фикстуры: образцы данных, «хороший» черновик и подставная LLM для тестов."""

import json
from dataclasses import dataclass, field

import pytest

from app.llm import Capabilities, LLMResult
from app.schemas import ProductInput

# Образец из тестового задания (PDF). Описание там обрезано многоточием — оставляем как есть.
PDF_PRODUCT = {
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

# Текст 800–1200 символов, в котором нет ни одного значения, отсутствующего в PDF_PRODUCT.
GOOD_DESCRIPTION = (
    "Манометр показывающий DM 02 производства «Манотомь» предназначен для измерения "
    "избыточного давления. Диапазон измерений прибора составляет 0–1,6 МПа, класс точности — 1,5. "
    "Для подключения к измерительной точке предусмотрено присоединение G1/2. Диаметр корпуса "
    "манометра — 100 мм, что следует учитывать при проектировании места установки и подготовке "
    "посадочного отверстия на щите или трубопроводе. Показывающий манометр применяется там, где "
    "требуется визуальный контроль давления по шкале. Перед выбором прибора сопоставьте диапазон "
    "измерений с ожидаемым давлением в системе и убедитесь, что тип присоединения G1/2 "
    "соответствует штуцеру на объекте. Технические сведения в этой карточке приведены по данным, "
    "предоставленным производителем и поставщиком, поэтому при заказе рекомендуется уточнить у "
    "менеджера комплектацию, срок поставки и наличие сопроводительной документации. Карточка носит "
    "справочный характер и подлежит проверке контент-менеджером перед публикацией на сайте."
)

GOOD_DRAFT = {
    "seo_title": "Манометр показывающий DM 02 Манотомь, 0–1,6 МПа, G1/2",
    "seo_description": GOOD_DESCRIPTION,
    "search_queries": [
        "манометр показывающий DM 02",
        "манометр Манотомь 0-1,6 МПа",
        "манометр G1/2 диаметр 100 мм",
    ],
    "commercial": {
        "audience": ["Специалисты, которым нужен прибор для измерения избыточного давления"],
        "tasks": ["Визуальный контроль избыточного давления в диапазоне 0–1,6 МПа"],
        "scenarios": [],
    },
}


def draft_json(**overrides) -> str:
    """JSON-ответ модели; overrides заменяют поля верхнего уровня."""
    return json.dumps({**GOOD_DRAFT, **overrides}, ensure_ascii=False)


@pytest.fixture
def pdf_product() -> ProductInput:
    return ProductInput.model_validate(PDF_PRODUCT)


@dataclass
class FakeLLM:
    """Подставная LLM: отдаёт заготовленные ответы по очереди и запоминает запросы.

    Элемент очереди — строка (ответ модели) или исключение (будет выброшено).
    """

    replies: list = field(default_factory=lambda: [draft_json()])
    model: str = "fake-model"
    capabilities: Capabilities = field(default_factory=Capabilities)
    calls: list = field(default_factory=list)
    schemas: list = field(default_factory=list)

    def complete(self, messages, *, json_schema=None):
        self.calls.append(messages)
        self.schemas.append(json_schema)
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(text=reply, model=self.model, prompt_tokens=10, completion_tokens=20)


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch):
    """Тесты не должны видеть реальный ключ из локального .env и не делиться кэшем настроек."""
    from app.config import get_settings
    from app.main import get_llm

    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "")
    get_settings.cache_clear()
    get_llm.cache_clear()
    yield
    get_settings.cache_clear()
    get_llm.cache_clear()
