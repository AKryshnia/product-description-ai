"""Настройки приложения: переменные окружения и файл .env (в git не попадает).

LLM_* — выбор провайдера и общие параметры генерации; GIGACHAT_* — доступ к GigaChat.
Проверка TLS-сертификатов намеренно НЕ настраивается: её нельзя отключить. Если сертификат
издателя GigaChat не доверен системой, подключается корневой сертификат Минцифры через
GIGACHAT_CA_BUNDLE_FILE (см. README).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # провайдер и общие параметры (не зависят от провайдера)
    llm_provider: str = "gigachat"
    llm_model: str | None = None  # None — модель по умолчанию выбранного провайдера
    llm_temperature: float = 0.0  # измерено: при 0.2 длина текста плавает, при 0.0 стабильна
    llm_max_tokens: int = 1500
    llm_timeout: float = 60.0
    llm_max_retries: int = 2

    # доступ к GigaChat
    gigachat_credentials: SecretStr | None = None
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_base_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_ca_bundle_file: Path | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
