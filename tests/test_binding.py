"""Привязка значений к своей характеристике, исполнению и условию .

Случаи C01–C08 (верно) и B01–B17 (искажено) образуют контрольный набор: разметка сделана
по смыслу входных данных. Это регрессионные проверки известных классов ошибок, а не
оценка точности проверки на новых данных.
"""

import json
from pathlib import Path

import pytest

from app.schemas import Draft, ProductInput, Severity
from app.schemas import IssueCode as C
from app.service import generate_card
from app.validator import build_facts, check_text, validate_draft
from tests.conftest import FakeLLM

ROOT = Path(__file__).resolve().parent.parent


def product(name: str) -> ProductInput:
    return ProductInput.model_validate_json((ROOT / f"examples/inputs/{name}.json").read_text())


def base_draft(name: str) -> dict:
    return json.loads((Path(__file__).parent / f"data/{name}.draft.json").read_text())


def findings(name: str, text: str, block: str = "seo_description"):
    issues = check_text(text, block, build_facts(product(name)))
    return [i for i in issues if i.severity is not Severity.INFO]


# (id, входной файл, текст). Верные формулировки не должны давать замечаний.
CLEAN = [
    (
        "C01",
        "mp3-u",
        "Температура окружающей среды от минус 50 до плюс 60 °С в металлическом корпусе.",
    ),
    ("C02", "testo-608-h1", "Погрешность измерения температуры ±0,5 °C при +25 °C."),
    ("C03", "ifn-300", "Рабочий диапазон напряжений при измерении петли — 180–450 В."),
    ("C04", "vit-1", "Термометрическая жидкость — органическая жидкость."),
    ("C05", "mp3-u", "Базовая степень защиты IP40, по заказу IP54."),
    (
        "C06",
        "testo-608-h1",
        "Прибор применяют в различных помещениях: музеях, библиотеках и серверных комнатах.",
    ),
    ("C07", "ifn-300", "Пределы измерения сопротивления постоянному току — 0,01–999 Ом."),
    (
        "C08",
        "mp3-u",
        "Назначение: измерение избыточного давления неагрессивных, некристаллизующихся "
        "жидкостей, пара и газа.",
    ),
    # дополнительные верные формулировки того же рода (перефразирование, порядок, запись знака)
    ("C09", "mp3-u", "В корпусе из полистирола температура окружающей среды — от −30 до +60 °C."),
    (
        "C10",
        "testo-608-h1",
        "Температура хранения — от −40 до +70 °C, рабочая температура 0…+50 °C.",
    ),
    ("C11", "ifn-300", "Диапазон вычисления ожидаемого тока КЗ «фаза-нуль» — 0–22 кА."),
    ("C12", "ifn-300", "Ток КЗ «фаза-фаза» вычисляется в диапазоне 0–38 кА."),
]

# (id, входной файл, искажённый текст, допустимые коды замечаний — достаточно любого из них)
BROKEN = [
    (
        "B01",
        "mp3-u",
        "Температура окружающей среды от минус 50 до плюс 60 °С в корпусе из полистирола.",
        {C.VALUE_CONTEXT_MISMATCH, C.CONDITION_DROPPED},
    ),
    (
        "B02",
        "mp3-u",
        "Степень защиты базового исполнения — IP54.",
        {C.CONDITION_DROPPED, C.VALUE_CONTEXT_MISMATCH},
    ),
    ("B03", "mp3-u", "Корпус изготовлен из нержавеющей стали.", {C.CONDITION_DROPPED}),
    ("B04", "mp3-u", "Манометр предназначен для агрессивных жидкостей.", {C.UNSUPPORTED_CLAIM}),
    (
        "B05",
        "testo-608-h1",
        "Погрешность измерения температуры — ±0,5 °C во всём диапазоне.",
        {C.CONDITION_DROPPED},
    ),
    (
        "B06",
        "testo-608-h1",
        "Рабочая температура от −40 до +70 °C.",
        {C.VALUE_CONTEXT_MISMATCH},
    ),
    (
        "B07",
        "testo-608-h1",
        "Температура хранения от +40 до +70 °C.",
        {C.UNVERIFIED_NUMBER, C.UNIT_PAIR_UNCONFIRMED},
    ),
    (
        "B08",
        "testo-608-h1",
        "Для измерения температуры требуется три батареи.",
        {C.VALUE_CONTEXT_MISMATCH, C.UNVERIFIED_NUMBER},
    ),
    (
        "B09",
        "ifn-300",
        "Диапазон вычисления тока КЗ «фаза-нуль» — 0–38 кА.",
        {C.VALUE_CONTEXT_MISMATCH},
    ),
    (
        "B10",
        "ifn-300",
        "Рабочий диапазон напряжений при измерении петли — 10,0–450,0 В.",
        {C.VALUE_CONTEXT_MISMATCH},
    ),
    ("B11", "vit-1", "Корпус изготовлен из латуни.", {C.UNSUPPORTED_CLAIM}),
    ("B12", "vit-1", "Прибор работает без обслуживания.", {C.UNSUPPORTED_CLAIM}),
    (
        "B13",
        "testo-608-h1",
        "Погрешность температуры — ±0,5 градусов Фаренгейта.",
        {C.UNVERIFIED_UNIT},
    ),
    ("B14", "testo-608-h1", "Напряжение питания — 12 В.", {C.UNVERIFIED_NUMBER}),
    ("B15", "mp3-u", "Степень защиты базового исполнения — IP68.", {C.UNVERIFIED_NUMBER}),
    ("B16", "vit-1", "Диапазон измерения температуры — 0–25 бар.", {C.UNVERIFIED_UNIT}),
    (
        "B17",
        "vit-1",
        "Прибор имеет сертификат для пищевой промышленности.",
        {C.UNSUPPORTED_CLAIM},
    ),
]


@pytest.mark.parametrize(("ident", "name", "text"), CLEAN, ids=[c[0] for c in CLEAN])
def test_correct_statements_are_not_flagged(ident, name, text):
    assert findings(name, text) == []


@pytest.mark.parametrize(("ident", "name", "text", "codes"), BROKEN, ids=[b[0] for b in BROKEN])
def test_distorted_statements_never_pass_silently(ident, name, text, codes):
    found = findings(name, text)
    assert found, f"{ident}: искажение прошло без замечаний"
    assert {i.code for i in found} & codes


# ----------------------------------------------------------- через весь сервис (статус)


def card_with(name: str, old: str, new: str, *, replies: int = 2):
    draft = base_draft(name)
    assert old in draft["seo_description"]
    draft["seo_description"] = draft["seo_description"].replace(old, new, 1)
    llm = FakeLLM([json.dumps(draft, ensure_ascii=False)] * replies)
    return generate_card(product(name), llm), llm


def test_stored_real_drafts_are_the_baseline_for_the_mutations_below():
    # реальные ответы GigaChat (examples/outputs), каждый — до мутации
    for name in ("mp3-u", "ifn-300", "vit-1", "testo-608-h1"):
        d = Draft.model_validate(base_draft(name))
        issues = validate_draft(d, build_facts(product(name)))
        assert not [i for i in issues if i.severity is Severity.ERROR], (name, issues)


def test_material_swapped_between_conditions_is_not_ok():
    """Минимальное воспроизведение: диапазон металлического корпуса приписан полистиролу."""
    card, _ = card_with("mp3-u", "в металлическом корпусе", "в корпусе из полистирола")
    assert card.status.value != "ok"
    assert {w.code for w in card.warnings} & {C.VALUE_CONTEXT_MISMATCH, C.CONDITION_DROPPED}
    # нарушение найдено в тексте, а таблица характеристик осталась исходной
    assert [(s.name, s.value) for s in card.specs] == list(product("mp3-u").specs.items())


def test_option_presented_as_base_is_not_ok():
    card, _ = card_with("mp3-u", "степень защиты — IP40", "степень защиты — IP54")
    assert card.status.value != "ok"


def test_non_aggressive_turned_into_aggressive_is_not_ok():
    card, _ = card_with(
        "mp3-u", "неагрессивных, некристаллизующихся", "агрессивных, кристаллизующихся"
    )
    assert card.status.value != "ok"
    assert C.UNSUPPORTED_CLAIM in {w.code for w in card.warnings}


def test_the_repair_attempt_is_still_limited_to_one():
    card, llm = card_with("mp3-u", "в металлическом корпусе", "в корпусе из полистирола")
    assert len(llm.calls) == 2 and card.generation.llm_calls == 2


# ---------------------------------------------------------------- устройство проверки


def test_signs_are_part_of_the_value():
    """Раньше «−40» и «+40» были одним числом 40."""
    f = build_facts(product("testo-608-h1"))
    flipped = check_text("Температура хранения от +40 до +70 °C.", "seo_description", f)
    assert any(i.code in {C.UNVERIFIED_NUMBER, C.UNIT_PAIR_UNCONFIRMED} for i in flipped)
    same = check_text("Температура хранения от минус 40 до плюс 70 °C.", "seo_description", f)
    assert same == []


def test_a_value_taken_from_a_list_must_name_the_list():
    """Значение из перечня опций нельзя назвать характеристикой без слов «по заказу»."""
    found = findings("mp3-u", "Степень защиты — IP54.")
    assert [i.code for i in found] == [C.CONDITION_DROPPED]
    assert findings("mp3-u", "Степень защиты IP54 доступна по заказу.") == []


def test_dropping_a_condition_in_parentheses_or_after_pri_is_reported():
    assert [i.code for i in findings("testo-608-h1", "Погрешность температуры ±0,5 °C.")] == [
        C.CONDITION_DROPPED
    ]
    assert [i.code for i in findings("mp3-u", "Среда с температурой до плюс 150 °С.")] == [
        C.CONDITION_DROPPED
    ]


def test_a_number_that_exists_only_in_unrelated_context_points_to_the_competing_value():
    (issue,) = findings("ifn-300", "Диапазон вычисления тока КЗ «фаза-нуль» — 0–38 кА.")
    assert issue.code is C.VALUE_CONTEXT_MISMATCH and issue.severity is Severity.REVIEW
    assert "фаза-фаза" in issue.message  # менеджеру показано, к какой характеристике относится 0–38


def test_unit_names_written_in_words_are_checked_like_symbols():
    f = build_facts(product("testo-608-h1"))
    assert check_text("±0,5 градусов Цельсия при +25 °C", "seo_description", f) == []
    found = check_text("±0,5 градусов Фаренгейта", "seo_description", f)
    assert C.UNVERIFIED_UNIT in {i.code for i in found}


def test_numerals_in_words_are_numbers():
    f = build_facts(product("testo-608-h1"))
    found = check_text("Питание от двенадцати вольт.", "seo_description", f)
    assert {i.code for i in found} & {C.UNVERIFIED_NUMBER, C.VALUE_CONTEXT_MISMATCH}


def test_compound_words_tell_phase_zero_from_phase_phase():
    found = findings("ifn-300", "Диапазон вычисления тока КЗ — 0–22 кА для петли фаза-фаза.")
    assert [i.code for i in found] == [C.VALUE_CONTEXT_MISMATCH]
    assert findings("ifn-300", "Диапазон вычисления тока КЗ — 0–22 кА для петли фаза-нуль.") == []


def test_a_condition_written_in_the_name_of_a_characteristic_must_be_kept():
    # ключ: «ток в измерительной цепи при сопротивлении не более 10 Ом»
    assert [i.code for i in findings("ifn-300", "Ток в измерительной цепи — не менее 200 мА.")] == [
        C.CONDITION_DROPPED
    ]
    assert (
        findings(
            "ifn-300",
            "Ток в измерительной цепи при сопротивлении не более 10 Ом — не менее 200 мА.",
        )
        == []
    )


def test_designation_from_the_product_name_is_not_compared_with_characteristics():
    # «H1» и «ВИТ-1» — часть названия, соседние слова описывают характеристики
    assert (
        findings("testo-608-h1", "Термогигрометр testo 608-H1: влажность и температура воздуха")
        == []
    )
    assert findings("vit-1", "Гигрометр психрометрический ВИТ-1: диапазон влажности 20–90 %.") == []


def test_negation_cannot_be_dropped_or_added():
    assert findings("mp3-u", "Измерение давления неагрессивных жидкостей.") == []
    assert [
        i.code for i in findings("mp3-u", "Измерение давления кристаллизующихся жидкостей.")
    ] == [C.UNSUPPORTED_CLAIM]
