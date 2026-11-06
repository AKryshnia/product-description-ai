"""Контракт входных данных: что принимаем, что отвергаем, что сохраняем без изменений."""

import pytest
from pydantic import ValidationError

from app.schemas import ProductInput


def make(**overrides):
    data = {"name": "Манометр", "manufacturer": "Манотомь", "specs": {"класс точности": "1,5"}}
    data.update(overrides)
    return ProductInput.model_validate(data)


def test_minimal_input_needs_only_name():
    p = ProductInput.model_validate({"name": "Манометр"})
    assert p.manufacturer is None and p.specs is None and p.description is None


@pytest.mark.parametrize("name", ["", "   ", "\n\t"])
def test_blank_name_is_rejected(name):
    with pytest.raises(ValidationError):
        make(name=name)


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": 123},
        {"manufacturer": ["Манотомь"]},
        {"specs": ["диапазон"]},
        {"specs": {"диапазон": ["0", "1"]}},
        {"specs": {"диапазон": {"min": 0}}},
        {"specs": {"герметичен": True}},  # bool — не значение характеристики
        {"description": 42},
    ],
)
def test_wrong_types_are_rejected(overrides):
    with pytest.raises(ValidationError):
        make(**overrides)


def test_spec_values_are_kept_verbatim_and_in_order():
    specs = {
        "диапазон измерений": "0–1,6 МПа",
        "класс точности": " 1,5 ",  # пробелы по краям сохраняем: значение не переписываем
        "присоединение": "G1/2",
    }
    p = make(specs=specs)
    assert list(p.specs.items()) == list(specs.items())


def test_numeric_spec_values_are_converted_to_text():
    p = make(specs={"класс точности": 1.5, "число каналов": 4})
    assert p.specs == {"класс точности": "1.5", "число каналов": "4"}


def test_empty_and_null_spec_values_are_allowed_for_later_reporting():
    p = make(specs={"питание": "", "интерфейс": None})
    assert p.specs == {"питание": "", "интерфейс": ""}


def test_blank_spec_name_is_rejected():
    with pytest.raises(ValidationError):
        make(specs={"  ": "значение"})


def test_unknown_fields_are_kept_aside_not_dropped_silently():
    p = make(sku="A-1")
    assert p.model_extra == {"sku": "A-1"}


def test_oversized_payloads_are_rejected():
    with pytest.raises(ValidationError):
        make(description="х" * 20_001)
    with pytest.raises(ValidationError):
        make(specs={f"п{i}": "1" for i in range(101)})
