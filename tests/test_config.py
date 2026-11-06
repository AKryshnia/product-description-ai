"""Конфигурация: значения по умолчанию, переопределение через окружение, секреты."""

from app.config import Settings


def test_defaults_are_safe_and_documented(monkeypatch):
    for name in (
        "GIGACHAT_CREDENTIALS",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "GIGACHAT_SCOPE",
        "GIGACHAT_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert s.gigachat_credentials is None
    assert s.llm_provider == "gigachat" and s.llm_model is None  # модель — умолчание провайдера
    assert s.gigachat_scope == "GIGACHAT_API_PERS"
    assert s.gigachat_base_url.startswith("https://")
    assert s.llm_max_retries == 2


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    monkeypatch.setenv("LLM_MODEL", "GigaChat-2-Max")
    monkeypatch.setenv("LLM_TIMEOUT", "15")
    s = Settings(_env_file=None)
    assert s.llm_model == "GigaChat-2-Max" and s.llm_timeout == 15


def test_credentials_never_appear_in_repr_or_dump(monkeypatch):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "TOP-SECRET-KEY")
    s = Settings(_env_file=None)
    assert s.gigachat_credentials.get_secret_value() == "TOP-SECRET-KEY"
    assert "TOP-SECRET-KEY" not in repr(s)
    assert "TOP-SECRET-KEY" not in str(s.model_dump())
    assert "TOP-SECRET-KEY" not in s.model_dump_json()


def test_tls_verification_is_not_configurable():
    # Требование безопасности: отключить проверку сертификатов настройкой нельзя.
    assert not any("verify" in name for name in Settings.model_fields)
