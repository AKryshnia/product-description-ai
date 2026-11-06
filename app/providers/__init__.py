"""Реестр провайдеров LLM. Выбор — переменной окружения LLM_PROVIDER.

Чтобы добавить провайдера: написать app/providers/<имя>.py с классом, реализующим
`app.llm.LLMClient`, и добавить фабрику в PROVIDERS. Импорт SDK ленивый: SDK провайдера,
который не выбран, даже не должен быть установлен.
"""

from collections.abc import Callable

from app.config import Settings
from app.llm import LLMClient, LLMNotConfigured


def _gigachat(settings: Settings) -> LLMClient:
    from app.providers.gigachat import GigaChatClient

    return GigaChatClient(settings)


PROVIDERS: dict[str, Callable[[Settings], LLMClient]] = {"gigachat": _gigachat}


def create_client(settings: Settings) -> LLMClient:
    name = settings.llm_provider.strip().lower()
    factory = PROVIDERS.get(name)
    if factory is None:
        raise LLMNotConfigured(
            f"Неподдерживаемый LLM_PROVIDER «{settings.llm_provider}». "
            f"Доступны: {', '.join(sorted(PROVIDERS))}."
        )
    return factory(settings)
