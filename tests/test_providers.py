"""Независимость бизнес-логики от провайдера LLM: интерфейс, выбор провайдера, возможности."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import (
    Capabilities,
    LLMClient,
    LLMError,
    LLMInvalidResponse,
    LLMNotConfigured,
    LLMResult,
    LLMUnsupported,
)
from app.main import app, get_llm
from app.providers import PROVIDERS, create_client
from app.schemas import ProductInput
from app.service import generate_card
from tests.conftest import GOOD_DRAFT, PDF_PRODUCT, FakeLLM, draft_json

ROOT = Path(__file__).resolve().parent.parent


def make_settings(**kw) -> Settings:
    return Settings(_env_file=None, gigachat_credentials="x", gigachat_ca_bundle_file=None, **kw)


# ---------------------------------------------------------------- выбор провайдера


def test_registry_knows_gigachat_and_nothing_else_is_assumed():
    assert set(PROVIDERS) == {"gigachat"}


def test_factory_builds_a_client_that_satisfies_the_interface():
    client = create_client(make_settings(llm_provider=" GigaChat "))  # регистр и пробелы не важны
    assert isinstance(client, LLMClient)
    assert client.model == "GigaChat-2-Pro"  # умолчание провайдера
    assert create_client(make_settings(llm_model="GigaChat-2-Max")).model == "GigaChat-2-Max"


def test_unknown_provider_is_a_controlled_configuration_error():
    with pytest.raises(LLMNotConfigured) as e:
        create_client(make_settings(llm_provider="skynet"))
    assert "skynet" in str(e.value) and "gigachat" in str(e.value)


def test_api_with_unknown_provider_returns_503_not_a_crash(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "skynet")
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/generate", json=PDF_PRODUCT)
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "llm_not_configured" and "skynet" in err["message"]
    assert client.get("/health").json()["llm_configured"] is False  # сервис при этом жив
    assert client.post("/generate", json={"name": ""}).status_code == 422  # вход проверяется первым


# ---------------------------------------------------------------- бизнес-логика не знает про SDK


def test_business_modules_do_not_import_the_gigachat_sdk():
    code = (
        "import sys, app.service, app.generator, app.validator, app.export, app.completeness, "
        "app.main\n"
        "bad = [m for m in sys.modules if m == 'gigachat' or m.startswith('gigachat.')]\n"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


class OtherStyleLLM:
    """Реализация «другого провайдера»: ни наследования, ни общего кода с FakeLLM."""

    model = "other-1"
    capabilities = Capabilities(json_schema=False)

    def __init__(self):
        self.seen = []

    def complete(self, messages, *, json_schema=None):
        self.seen.append(messages)
        # адаптер сам решает, как передать роли провайдеру (здесь — склеивает в один запрос)
        _ = "\n".join(m["content"] for m in messages)
        return LLMResult(text=draft_json(), model="other-1-2026")


@pytest.mark.parametrize("make_llm", [FakeLLM, OtherStyleLLM])
def test_generator_works_with_any_implementation_of_the_interface(make_llm):
    llm = make_llm()
    assert isinstance(llm, LLMClient)
    card = generate_card(_product(), llm)
    assert card.status.value == "ok" and card.seo_title == GOOD_DRAFT["seo_title"]


# ---------------------------------------------------------------- возможности провайдера


def _product():
    return ProductInput.model_validate(PDF_PRODUCT)


def test_schema_is_requested_only_from_providers_that_declare_support():
    plain, strict = FakeLLM(), FakeLLM(capabilities=Capabilities(json_schema=True))
    generate_card(_product(), plain)
    generate_card(_product(), strict)
    assert plain.schemas == [None]
    schema = strict.schemas[0]
    assert {"seo_title", "seo_description", "search_queries", "commercial"} <= set(
        schema["properties"]
    )


def test_prompt_does_not_depend_on_provider_capabilities():
    plain, strict = FakeLLM(), FakeLLM(capabilities=Capabilities(json_schema=True))
    generate_card(_product(), plain)
    generate_card(_product(), strict)
    assert plain.calls[0] == strict.calls[0]


# ---------------------------------------------------------------- единые ошибки


def test_new_provider_errors_reuse_the_existing_error_handling(monkeypatch):
    class ProviderQuotaExceeded(LLMError):  # ошибка будущего адаптера: достаточно наследоваться
        code = "llm_quota_exceeded"
        http_status = 429
        default_message = "Исчерпана квота провайдера."

    app.dependency_overrides[get_llm] = lambda: FakeLLM([ProviderQuotaExceeded()])
    try:
        r = TestClient(app, raise_server_exceptions=False).post("/generate", json=PDF_PRODUCT)
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 429 and r.json()["error"]["code"] == "llm_quota_exceeded"


def test_gigachat_adapter_refuses_what_it_cannot_do():
    client = create_client(make_settings())
    assert client.capabilities.json_schema is False  # проверено на реальном API, см. README
    with pytest.raises(LLMUnsupported):
        client.complete([{"role": "user", "content": "x"}], json_schema={"type": "object"})


def test_gigachat_adapter_rejects_a_response_without_choices():
    client = create_client(make_settings())
    client._giga.chat = lambda chat: SimpleNamespace(choices=[], usage=None, model="m")
    with pytest.raises(LLMInvalidResponse):
        client.complete([{"role": "user", "content": "x"}])
