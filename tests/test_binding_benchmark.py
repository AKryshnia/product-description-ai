"""Бенчмарк привязки значений: ложные срабатывания на верных фразах и обнаружение подмен.

Корпус — 32 верные формулировки по четырём каталожным товарам (составлены автором проверки по
смыслу входных данных). Подмена генерируется автоматически: значение в фразе заменяется на другое
значение той же размерности из этого же товара. Доля замеченных подмен — не точность проверки на
новых данных, а регрессионная метрика и описание границы метода: если соседние слова подходят
сразу к нескольким характеристикам («температура»), подмену по контексту не различить.
Запуск с таблицей: pytest tests/test_binding_benchmark.py -s
"""

import re
from pathlib import Path

import pytest

from app.facts import chains
from app.schemas import ProductInput, Severity
from app.validator import build_facts, check_text

ROOT = Path(__file__).resolve().parent.parent

CORRECT = [
    ("mp3-u", "Манометр МП3-У имеет корпус диаметром 100 мм."),
    ("mp3-u", "Класс точности — 1,5; степень защиты — IP40."),
    ("mp3-u", "Резьба штуцера — M20x1.5-8g."),
    (
        "mp3-u",
        "Допускается эксплуатация при температуре окружающей среды от −50 до +60 °C "
        "в металлическом корпусе.",
    ),
    ("mp3-u", "Виброзащита класса L3: от 5 до 25 Гц с амплитудой 0,1 мм."),
    ("mp3-u", "Масса прибора — не более 0,7 кг, срок службы — не менее 10 лет."),
    ("mp3-u", "Межповерочный интервал — 2 года."),
    ("mp3-u", "По заказу доступны класс точности 1,0 и степень защиты IP54."),
    (
        "mp3-u",
        "Опции по карте заказа: обезжиривание рабочей полости, пломбировка, экспортное исполнение.",
    ),
    ("mp3-u", "Климатическое исполнение — У2 и Т2."),
    (
        "mp3-u",
        "Для сред с температурой до плюс 150 °С рекомендовано применение охладительных устройств.",
    ),
    ("mp3-u", "Прибор имеет радиальный штуцер без фланца и резьбу M20x1.5-8g."),
    ("mp3-u", "В исполнении по заказу предусмотрена табличка из нержавеющей стали."),
    (
        "testo-608-h1",
        "Диапазон измерения относительной влажности — 10–95 %, "
        "погрешность ±3 % в диапазоне 10–95 %.",
    ),
    ("testo-608-h1", "Разрешение по влажности составляет 0,1 %."),
    ("testo-608-h1", "Температура хранения: от −40 до +70 °C."),
    ("testo-608-h1", "Питание — батарея 9 В."),
    ("testo-608-h1", "Габариты прибора — 111 × 90 × 40 мм, масса — 168 г."),
    ("testo-608-h1", "Корпус выполнен из ABS."),
    ("testo-608-h1", "Рабочая температура: 0…+50 °C."),
    ("testo-608-h1", "Термогигрометр testo 608-H1: влажность 10–95 %, температура 0…+50 °C."),
    ("ifn-300", "Диапазон частот 45–65 Гц."),
    ("ifn-300", "Память — до 10 000 измерений."),
    ("ifn-300", "Питание от аккумулятора Ni-MH 6 В или от пяти элементов типа AA."),
    ("ifn-300", "Ток в измерительной цепи при сопротивлении не более 10 Ом — не менее 200 мА."),
    ("ifn-300", "Диапазон напряжения при измерении петли — от 180 до 450 В."),
    (
        "ifn-300",
        "Измеряет действующее значение переменного напряжения в диапазоне 10,0–450,0 В.",
    ),
    (
        "ifn-300",
        "Прибор измеряет полное сопротивление петли в диапазонах 0,01–9,99 Ом, 10,0–99,9 Ом "
        "и 100–300 Ом.",
    ),
    ("vit-1", "Диапазон измерения относительной влажности — 20–90 %."),
    ("vit-1", "Цена деления шкалы — 0,2 °C."),
    ("vit-1", "Габариты — 295 × 120 × 50 мм."),
    ("vit-1", "Измеряет температуру воздуха от 0 до 25 °C."),
]

MIN_SWAPS_NOTICED = 0.70  # зафиксировано при разработке: 38 из 51 (75 %)


def product(name: str) -> ProductInput:
    return ProductInput.model_validate_json((ROOT / f"examples/inputs/{name}.json").read_text())


def findings(name: str, text: str):
    issues = check_text(text, "seo_description", build_facts(product(name)))
    return [i for i in issues if i.severity is not Severity.INFO]


def swaps():
    """(товар, фраза с подменённым значением): то же размерность, другое значение товара."""
    for name, text in CORRECT:
        pool = {}
        for value in product(name).specs.values():
            for part in re.split(r"[;\n]", value):
                for ch in chains(part):
                    if ch.unit:
                        pool.setdefault(ch.unit, {})[tuple(ch.numbers)] = ch.fragment
        for ch in chains(text):
            for numbers, fragment in pool.get(ch.unit, {}).items() if ch.unit else ():
                if numbers != tuple(ch.numbers) and not set(numbers) <= set(ch.numbers):
                    mutated = text.replace(ch.fragment, fragment, 1)
                    if mutated != text:
                        yield name, mutated


@pytest.mark.parametrize(("name", "text"), CORRECT)
def test_correct_paraphrases_are_not_flagged(name, text):
    assert findings(name, text) == []


def test_substituted_values_are_noticed_at_the_recorded_rate(capsys):
    cases = list(swaps())
    noticed = [bool(findings(name, text)) for name, text in cases]
    rate = sum(noticed) / len(noticed)
    with capsys.disabled():
        print(
            f"\nПРИВЯЗКА: верных фраз с замечаниями 0/{len(CORRECT)}; подменённых значений "
            f"замечено {sum(noticed)}/{len(noticed)} = {rate:.0%}"
        )
    assert len(cases) >= 45  # корпус не выродился
    assert rate >= MIN_SWAPS_NOTICED
