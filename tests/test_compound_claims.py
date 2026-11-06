"""Составные технические утверждения: условие целиком, опция, диапазон, отрицание, граница.

Значение связано во входных данных с условием, опциональностью, границей или отрицанием. Если в
тексте сохранена только часть этой связи, значение не подтверждается: частичное совпадение — не
подтверждение. Верные формулировки того же рода обязаны оставаться без замечаний.

Наборы маленькие и размечены по смыслу входных данных. Это регрессионные проверки известных
классов ошибок, а не оценка точности на новых товарах. Каждый класс проверяется отдельно, чтобы
по имени упавшего теста было понятно, что именно перестало работать.
"""

import json

import pytest

from app.schemas import CardStatus, ProductInput, Severity
from app.schemas import IssueCode as C
from app.service import generate_card
from app.validator import build_facts, check_text
from tests.conftest import GOOD_DRAFT, PDF_PRODUCT, FakeLLM

TEMPERATURE_CONDITION = {"погрешность": "±0,2 °C (при температуре +23 °C и влажности 50 %)"}
TWO_RANGES = {"диапазоны срабатывания": "0–6 бар; 10–16 бар"}
OPTION = {"опции по заказу": "степень защиты IP67"}
OPTION_AMONG_BASE = {
    "базовая степень защиты": "IP54",
    "опции по заказу": "степень защиты IP67; кабель 7 м",
}
NEGATED = {"реле": "не поддерживается"}
UPPER_BOUND = {"масса": "не более 0,8 кг"}
# два исполнения с общим ограничением условия: влажность одинакова, различает температура
SHARED_CONDITION = {
    "погрешность": "±0,4 °C (при +18 °C и влажности 65 %); ±0,7 °C (при +28 °C и влажности 65 %)"
}
# опцию делает пометка в названии; слова «степень защиты» говорят только о самом параметре
OPTION_IN_NAME = {"питание": "24 В", "выход": "4–20 мА", "опциональная степень защиты": "IP68"}
BASIC_AND_OPTION = {"степень защиты": "IP54", "опциональная степень защиты": "IP68"}
# перечень без пометки опции: вариантом значение делает число пунктов, признак — название перечня
LISTED = {"диапазоны измерений": "0–1 бар; 0–10 бар; 0–100 бар"}


def findings(specs: dict, text: str):
    p = ProductInput(name="Измерительный прибор", specs=specs)
    issues = check_text(text, "seo_description", build_facts(p))
    return [i for i in issues if i.severity is not Severity.INFO]


# (описание случая, характеристики, верная формулировка)
FAITHFUL = [
    ("plain_range", {"давление": "0–16 бар"}, "Давление — 0–16 бар."),
    (
        "negative_range",
        {"температура эксплуатации": "-20…+60 °C", "температура хранения": "-40…+80 °C"},
        "Температура хранения — от -40 до +80 °C.",
    ),
    (
        "whole_condition",
        TEMPERATURE_CONDITION,
        "Погрешность ±0,2 °C при температуре +23 °C и влажности 50 %.",
    ),
    ("option_named", OPTION_AMONG_BASE, "Степень защиты IP67 доступна по заказу."),
    ("negation_kept", NEGATED, "Реле не поддерживается."),
    ("two_voltages", {"питание": "24 В", "испытательное напряжение": "48 В"}, "Питание — 24 В."),
    ("both_ranges", TWO_RANGES, "Диапазоны срабатывания — 0–6 бар и 10–16 бар."),
    (
        "signs_in_words",
        {"температура хранения": "от -37 до +83 °C"},
        "Температура хранения — от минус 37 до плюс 83 °C.",
    ),
    (
        "enumerated_rooms",
        {"места применения": "музеи, архивы"},
        "Применяется в различных помещениях: музеях и архивах.",
    ),
    (
        "neighbouring_values",
        {"питание": "12,5 В", "выходное напряжение": "3,3 В"},
        "Питание — 12,5 В; выходное напряжение — 3,3 В.",
    ),
    ("plain_protection", {"степень защиты": "IP67"}, "Степень защиты — IP67."),
    # те же классы, другие верные формулировки
    (
        "condition_reordered",
        TEMPERATURE_CONDITION,
        "При температуре +23 °C и влажности 50 % погрешность составляет ±0,2 °C.",
    ),
    (
        "condition_by_values",
        TEMPERATURE_CONDITION,
        "Погрешность ±0,2 °C при +23 °C и 50 %.",
    ),
    ("one_range_exactly", {"давление": "от 0 до 16 бар"}, "Давление — 0–16 бар."),
    ("option_named_by_key", OPTION, "Опционально доступна степень защиты IP67."),
    ("negation_other_form", NEGATED, "Прибор не поддерживает реле."),
    ("not_only_is_no_negation", {"назначение": "только измерение давления"}, "Не только давление."),
    ("bound_same_meaning", UPPER_BOUND, "Масса не превышает 0,8 кг."),
    (
        "bound_idiom_is_not_a_negation",  # «не превышает» — граница, а не отрицание слова источника
        {
            **UPPER_BOUND,
            "примечание": "для сред с температурой, превышающей 100 °C, не допускается",
        },
        "Масса не превышает 0,8 кг.",
    ),
    ("bound_up_to", UPPER_BOUND, "Масса — до 0,8 кг."),
    ("bound_lower", {"масса": "не менее 0,8 кг"}, "Масса — не менее 0,8 кг."),
    ("bound_strict", {"масса": "менее 0,8 кг"}, "Масса — менее 0,8 кг."),
    ("range_upper_end", {"частота": "от 5 до 25 Гц"}, "Частота — до 25 Гц."),
    ("range_lower_end", {"частота": "от 5 до 25 Гц"}, "Частота — не менее 5 Гц."),
    ("range_upper_end_as_maximum", {"частота": "5…25 Гц"}, "Частота — не более 25 Гц."),
    ("bound_more", {"масса": "более 0,8 кг"}, "Масса — более 0,8 кг."),
    # общее ограничение нескольких исполнений
    (
        "shared_condition_kept",
        SHARED_CONDITION,
        "Погрешность ±0,4 °C при +18 °C и влажности 65 %.",
    ),
    (
        "shared_condition_kept_in_other_variant",
        SHARED_CONDITION,
        "Погрешность ±0,7 °C при +28 °C и влажности 65 %.",
    ),
    (
        "variants_without_shared_condition",
        {
            "погрешность": (
                "±0,4 °C (при +18 °C и влажности 65 %); ±0,7 °C (при +28 °C и влажности 80 %)"
            )
        },
        "Погрешность ±0,7 °C при +28 °C и влажности 80 %.",
    ),
    # пометка опции в названии характеристики
    ("option_named_optionally", OPTION_IN_NAME, "Опционально доступна степень защиты IP68."),
    ("option_named_by_order", OPTION_IN_NAME, "Степень защиты IP68 — по заказу."),
    ("option_named_additionally", OPTION_IN_NAME, "Дополнительно доступна степень защиты IP68."),
    ("basic_values_beside_option", OPTION_IN_NAME, "Питание — 24 В, выход — 4–20 мА."),
    ("basic_value_named_like_option", BASIC_AND_OPTION, "Степень защиты — IP54."),
    ("basic_and_option_together", BASIC_AND_OPTION, "Степень защиты — IP54, по заказу — IP68."),
    ("listed_value_with_list_name", LISTED, "Один из диапазонов измерений — 0–10 бар."),
]

# (описание, характеристики, искажённая формулировка, код замечания)
DISTORTED = [
    (
        "condition_part_dropped",
        TEMPERATURE_CONDITION,
        "Погрешность ±0,2 °C при влажности 50 %.",
        C.CONDITION_DROPPED,
    ),
    (
        "condition_other_part_dropped",
        TEMPERATURE_CONDITION,
        "Погрешность ±0,2 °C при температуре +23 °C.",
        C.CONDITION_DROPPED,
    ),
    ("negation_removed", NEGATED, "Реле поддерживается.", C.UNSUPPORTED_CLAIM),
    ("negation_added", {"реле": "поддерживается"}, "Реле не поддерживается.", C.UNSUPPORTED_CLAIM),
    (
        "ranges_merged",
        TWO_RANGES,
        "Диапазон срабатывания — 0–16 бар.",
        C.COMBINATION_UNCONFIRMED,
    ),
    (
        "range_from_two_characteristics",
        {"рабочий диапазон": "10–16 бар", "диапазон калибровки": "0–6 бар"},
        "Рабочий диапазон — 0–16 бар.",
        C.COMBINATION_UNCONFIRMED,
    ),
    (
        "single_option_as_basic",
        OPTION,
        "Степень защиты — IP67.",
        C.CONDITION_DROPPED,
    ),
    ("upper_bound_became_lower", UPPER_BOUND, "Масса — не менее 0,8 кг.", C.BOUND_MISMATCH),
    ("upper_bound_became_strict", UPPER_BOUND, "Масса — менее 0,8 кг.", C.BOUND_MISMATCH),
    (
        "lower_bound_became_upper",
        {"масса": "не менее 0,8 кг"},
        "Масса — до 0,8 кг.",
        C.BOUND_MISMATCH,
    ),
    ("bound_dropped", UPPER_BOUND, "Масса — 0,8 кг.", C.BOUND_MISMATCH),
    (
        "range_lower_end_as_maximum",
        {"частота": "от 5 до 25 Гц"},
        "Частота — не более 5 Гц.",
        C.BOUND_MISMATCH,
    ),
    (
        "range_upper_end_as_minimum",
        {"частота": "от 5 до 25 Гц"},
        "Частота — от 25 Гц.",
        C.BOUND_MISMATCH,
    ),
    # общее ограничение нескольких исполнений обязательно, хотя исполнения не различает
    (
        "shared_condition_dropped",
        SHARED_CONDITION,
        "Погрешность ±0,4 °C при +18 °C.",
        C.CONDITION_DROPPED,
    ),
    (
        "shared_condition_dropped_in_other_variant",
        SHARED_CONDITION,
        "Погрешность ±0,7 °C при +28 °C.",
        C.CONDITION_DROPPED,
    ),
    (
        "distinct_condition_dropped_shared_kept",
        SHARED_CONDITION,
        "Погрешность ±0,4 °C при влажности 65 %.",
        C.CONDITION_DROPPED,
    ),
    (
        "variant_condition_swapped",
        SHARED_CONDITION,
        "Погрешность ±0,4 °C при +28 °C и влажности 65 %.",
        C.CONDITION_DROPPED,
    ),
    # название параметра не заменяет пометку опции
    ("option_marker_dropped", OPTION_IN_NAME, "Степень защиты — IP68.", C.CONDITION_DROPPED),
    (
        "option_given_as_basic_beside_same_basic",
        BASIC_AND_OPTION,
        "Степень защиты — IP68.",
        C.CONDITION_DROPPED,
    ),
    (
        "single_order_option_as_basic",
        {"степень защиты по заказу": "IP68"},
        "Степень защиты — IP68.",
        C.CONDITION_DROPPED,
    ),
    (
        "single_additional_option_as_basic",
        {"дополнительная степень защиты": "IP68"},
        "Степень защиты — IP68.",
        C.CONDITION_DROPPED,
    ),
    ("listed_value_without_list_name", LISTED, "Рабочее давление — 0–10 бар.", C.CONDITION_DROPPED),
]


@pytest.mark.parametrize(("case", "specs", "text"), FAITHFUL, ids=[c[0] for c in FAITHFUL])
def test_faithful_statement_has_no_findings(case, specs, text):
    assert findings(specs, text) == []


@pytest.mark.parametrize(
    ("case", "specs", "text", "code"), DISTORTED, ids=[c[0] for c in DISTORTED]
)
def test_distorted_statement_is_reported_with_its_reason(case, specs, text, code):
    found = findings(specs, text)
    assert code in {i.code for i in found}, [(i.code.value, i.message) for i in found]
    # неопределённость и расхождение отправляются на проверку человеку, а не выдаются за факт
    assert all(i.severity is Severity.REVIEW for i in found if i.code is code)


# ---------------------------------------------------------------- полная карточка


def _card_with(specs: dict, sentence: str, replies_after=()):
    product = ProductInput.model_validate(
        {**PDF_PRODUCT, "specs": {**PDF_PRODUCT["specs"], **specs}}
    )
    draft = {**GOOD_DRAFT, "seo_description": GOOD_DRAFT["seo_description"] + " " + sentence}
    assert 800 <= len(draft["seo_description"]) <= 1200
    llm = FakeLLM([json.dumps(draft, ensure_ascii=False), *replies_after])
    return generate_card(product, llm), llm


FULL_CARD_CASES = [
    c
    for c in DISTORTED
    if c[0]
    in {
        "condition_part_dropped",
        "negation_removed",
        "ranges_merged",
        "single_option_as_basic",
        "upper_bound_became_lower",
        "shared_condition_dropped",
        "option_marker_dropped",
    }
]


@pytest.mark.parametrize(
    ("case", "specs", "text", "code"), FULL_CARD_CASES, ids=[c[0] for c in FULL_CARD_CASES]
)
def test_distorted_statement_never_makes_a_full_card_ok(case, specs, text, code):
    card, llm = _card_with(specs, text)
    assert card.status is CardStatus.NEEDS_REVIEW
    assert code in {w.code for w in card.warnings}
    assert len(llm.calls) == card.generation.llm_calls == 2  # причина передана в корректировку
    assert text.rstrip(".") in card.seo_description  # текст сохранён, решает человек


@pytest.mark.parametrize(
    ("case", "specs", "text", "code"), FULL_CARD_CASES, ids=[c[0] for c in FULL_CARD_CASES]
)
def test_repair_that_restores_the_relation_gives_ok(case, specs, text, code):
    card, llm = _card_with(specs, text, replies_after=[json.dumps(GOOD_DRAFT, ensure_ascii=False)])
    assert card.status is CardStatus.OK and card.generation.repaired
    feedback = llm.calls[1][-1]["content"]
    messages = {i.message for i in findings(specs, text) if i.code is code}
    assert messages and all(m in feedback for m in messages)  # модель получила причину


@pytest.mark.parametrize(("case", "specs", "text"), FAITHFUL, ids=[c[0] for c in FAITHFUL])
def test_faithful_statement_in_a_full_card_costs_no_repair(case, specs, text):
    card, llm = _card_with(specs, text)
    assert card.status is CardStatus.OK and len(llm.calls) == 1
