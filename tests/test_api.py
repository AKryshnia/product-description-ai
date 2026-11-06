"""HTTP-контракт: форматы, коды и тела ошибок. LLM подменена; проверяется поведение API."""

import json

import pytest
from fastapi.testclient import TestClient

from app.llm import (
    LLMAPIError,
    LLMAuthError,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
)
from app.main import app, get_llm
from tests.conftest import GOOD_DESCRIPTION, PDF_PRODUCT, FakeLLM, draft_json

SECRET = "SECRET-KEY-123"


@pytest.fixture
def api():
    def make(llm=None):
        app.dependency_overrides[get_llm] = lambda: llm or FakeLLM()
        return TestClient(app, raise_server_exceptions=False)

    yield make
    app.dependency_overrides.clear()


def error_of(resp):
    body = resp.json()
    assert set(body) == {"error"} and {"code", "message"} <= set(body["error"])
    return body["error"]


def test_valid_input_returns_full_card_as_json(api):
    r = api().post("/generate", json=PDF_PRODUCT)
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["status"] == "ok"
    assert body["seo_description"] == GOOD_DESCRIPTION
    assert [s["value"] for s in body["specs"]] == list(PDF_PRODUCT["specs"].values())
    assert {
        "seo_title",
        "search_queries",
        "commercial",
        "missing_info",
        "warnings",
        "generation",
        "markdown",
    } <= set(body)
    assert SECRET not in r.text


def test_markdown_format_returns_copyable_text(api):
    r = api().post("/generate?format=markdown", json=PDF_PRODUCT)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
    assert r.text.startswith("# Манометр показывающий DM 02") and "## Характеристики" in r.text


def test_json_response_embeds_the_same_markdown(api):
    client = api()
    md = client.post("/generate?format=markdown", json=PDF_PRODUCT).text
    assert client.post("/generate", json=PDF_PRODUCT).json()["markdown"] == md


def test_unknown_format_is_rejected(api):
    r = api().post("/generate?format=xml", json=PDF_PRODUCT)
    assert r.status_code == 422 and error_of(r)["code"] == "invalid_input"


def test_malformed_json_is_400(api):
    r = api().post(
        "/generate", content='{"name": "Манометр",', headers={"content-type": "application/json"}
    )
    assert r.status_code == 400 and error_of(r)["code"] == "invalid_json"


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"name": 123}, "name"),
        ({"name": "Манометр", "specs": ["диапазон"]}, "specs"),
        ({"name": "Манометр", "specs": {"герметичен": True}}, "specs"),
        ({"name": "Манометр", "manufacturer": ["Манотомь"]}, "manufacturer"),
        ({"name": ""}, "name"),
        ({"name": "   "}, "name"),
        ({}, "name"),
    ],
)
def test_invalid_input_is_422_with_field_names_and_no_echo(api, payload, field):
    r = api().post("/generate", json=payload)
    assert r.status_code == 422
    err = error_of(r)
    assert err["code"] == "invalid_input" and field in [d["field"] for d in err["details"]]
    assert "input" not in json.dumps(err["details"], ensure_ascii=False)


def test_non_object_body_is_422(api):
    assert api().post("/generate", json=["не", "объект"]).status_code == 422


def test_minimal_input_is_processed_with_missing_info(api):
    r = api(FakeLLM([draft_json()])).post(
        "/generate", json={"name": "Манометр DM 02", "specs": {"присоединение": "G1/2"}}
    )
    assert r.status_code == 200
    fields = {i["field"] for i in r.json()["missing_info"]}
    assert {"manufacturer", "description"} <= fields


def test_insufficient_data_is_200_partial_without_calling_llm(api):
    llm = FakeLLM()
    r = api(llm).post("/generate", json={"name": "Манометр"})
    assert r.status_code == 200 and r.json()["status"] == "partial" and llm.calls == []


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMAuthError(), 502, "llm_auth_failed"),
        (LLMTimeout(), 504, "llm_timeout"),
        (LLMRateLimited(retry_after=7), 429, "llm_rate_limited"),
        (LLMUnavailable(), 502, "llm_unavailable"),
        (LLMAPIError("GigaChat вернул ошибку (HTTP 503)."), 502, "llm_api_error"),
    ],
)
def test_provider_failures_get_distinct_codes(api, error, status, code):
    r = api(FakeLLM([error])).post("/generate", json=PDF_PRODUCT)
    assert r.status_code == status and error_of(r)["code"] == code


def test_rate_limit_exposes_retry_after_header(api):
    r = api(FakeLLM([LLMRateLimited(retry_after=7)])).post("/generate", json=PDF_PRODUCT)
    assert r.headers["retry-after"] == "7"


def test_unexpected_errors_are_500_without_internals(api):
    r = api(FakeLLM([RuntimeError(f"утечка {SECRET} в трейсе")])).post(
        "/generate", json=PDF_PRODUCT
    )
    assert r.status_code == 500 and error_of(r)["code"] == "internal_error"
    assert SECRET not in r.text and "утечка" not in r.text


def test_missing_credentials_is_503_with_actionable_message(monkeypatch):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "")
    get_llm.cache_clear()
    from app.config import get_settings

    get_settings.cache_clear()
    r = TestClient(app, raise_server_exceptions=False).post("/generate", json=PDF_PRODUCT)
    assert r.status_code == 503
    err = error_of(r)
    assert err["code"] == "llm_not_configured" and "GIGACHAT_CREDENTIALS" in err["message"]


def test_invalid_input_wins_over_missing_credentials(monkeypatch):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "")
    get_llm.cache_clear()
    r = TestClient(app, raise_server_exceptions=False).post("/generate", json={"name": ""})
    assert r.status_code == 422


def test_health_and_openapi_are_available(api):
    client = api()
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["llm_configured"] is True
    spec = client.get("/openapi.json").json()
    assert "/generate" in spec["paths"]
    assert spec["components"]["schemas"]["ProductInput"]["examples"][0]["name"].startswith(
        "Манометр"
    )


def test_oversized_body_is_rejected_with_413_before_any_processing(api):
    llm = FakeLLM()
    big = json.dumps({"name": "Манометр", "description": "х" * 1_100_000})
    r = api(llm).post("/generate", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413 and error_of(r)["code"] == "payload_too_large"
    assert llm.calls == []


def test_oversized_streamed_body_without_content_length_is_rejected_too(api):
    chunks = (b"x" * 100_000 for _ in range(12))  # chunked: заголовка Content-Length нет
    r = api().post("/generate", content=chunks, headers={"content-type": "application/json"})
    assert r.status_code == 413 and error_of(r)["code"] == "payload_too_large"


def test_normal_sized_body_still_passes(api):
    assert api().post("/generate", json=PDF_PRODUCT).status_code == 200


def test_unknown_routes_use_the_same_error_envelope(api):
    r = api().get("/нет-такого-маршрута")
    assert r.status_code == 404 and error_of(r)["code"] == "not_found"
