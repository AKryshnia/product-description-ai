"""Бенчмарк проверки достоверности: насколько валидатор ловит выдуманные значения.

Размеченный корпус коротких фрагментов к образцу из ТЗ (PDF_PRODUCT). Считаем:
 * ложные срабатывания на корректных фрагментах (должно быть 0);
 * полноту (recall) на фрагментах с выдумкой — ВКЛЮЧАЯ заведомо пропускаемые случаи,
   чтобы цифра была честной, а не подогнанной под реализацию.

Корпус составлен автором валидатора, поэтому это регрессионный бенчмарк и документ о слепых
зонах, а не независимая оценка. Для живого сравнения см. scripts/live_check.py.
Запуск с таблицей: pytest tests/test_validator_benchmark.py -s
"""

import pytest

from app.schemas import IssueCode as C
from app.schemas import ProductInput, Severity
from app.validator import build_facts, check_text
from tests.conftest import PDF_PRODUCT

CLEAN = [
    "Манометр показывающий DM 02 с диапазоном измерений 0–1,6 МПа",
    "Класс точности 1,5, присоединение G1/2, диаметр корпуса 100 мм.",
    "Диапазон от 0 до 1,6 МПа",
    "Корпус диаметром 100 мм",
    "манометр DM02 G1/2",
    "показывающий манометр Манотомь",
    "Прибор предназначен для измерения избыточного давления.",
    "Манометр с резьбой G 1/2",
    "Предел 1,60 МПа",
    "диапазон 0 - 1,6 МПа",
    "Купить манометр DM 02",
    "манометр 100 mm",
    "Манометр DM02 для измерения избыточного давления",
    "штуцер G1/2",
    "манометр Манотомь",
]

# (фрагмент, ожидаемый код, ловим ли мы его). known_miss=False — осознанная слепая зона.
FABRICATED = [
    ("рабочая температура до 150 °C", C.UNVERIFIED_NUMBER, True),
    ("диапазон 0–16 бар", C.UNVERIFIED_NUMBER, True),
    ("рабочее давление до 25 МПа", C.UNVERIFIED_NUMBER, True),
    ("диаметр корпуса 160 мм", C.UNVERIFIED_NUMBER, True),
    ("присоединение G3/4", C.UNVERIFIED_NUMBER, True),
    ("погрешность 0,5 %", C.UNVERIFIED_NUMBER, True),
    ("степень защиты IP65", C.UNVERIFIED_NUMBER, True),
    ("срок службы 10 лет", C.UNVERIFIED_NUMBER, True),
    ("1,6 МПа — это 16 бар", C.UNVERIFIED_NUMBER, True),  # пересчёт единиц = новое число
    ("Диапазон 0–1,6 бар", C.UNVERIFIED_UNIT, True),
    ("корпус 1,6 мм", C.UNIT_PAIR_UNCONFIRMED, True),
    ("класс точности 1,0", C.UNVERIFIED_NUMBER, True),  # 1 есть в G1/2, но дробного 1,0 нет
    ("Надёжный манометр", C.UNSUPPORTED_CLAIM, True),
    ("уникальный прибор", C.UNSUPPORTED_CLAIM, True),
    ("для нефтегазовой отрасли", C.UNSUPPORTED_CLAIM, True),
    ("применяется в энергетике и ЖКХ", C.UNSUPPORTED_CLAIM, True),
    # наблюдалось у реальной модели: выдуманные места и процессы применения, транслитерация
    ("для производственных и складских помещений", C.UNSUPPORTED_CLAIM, True),
    ("контроль в технологических процессах", C.UNSUPPORTED_CLAIM, True),
    ("для предприятий, зданий и сооружений", C.UNSUPPORTED_CLAIM, True),
    ("манометр ДМ02", C.UNVERIFIED_DESIGNATION, True),
    # наблюдалось в «ok»-карточке реальной модели: оценки, расшифровка обозначений, «преимущества»
    ("обеспечивая высокую точность", C.UNSUPPORTED_CLAIM, True),
    ("что делает его устойчивым к воздействию пыли и влаги", C.UNSUPPORTED_CLAIM, True),
    ("пластиковый корпус облегчает монтаж", C.UNSUPPORTED_CLAIM, True),
    ("широкий спектр температур", C.UNSUPPORTED_CLAIM, True),
    ("обеспечивающими стабильную работу прибора", C.UNSUPPORTED_CLAIM, True),
    ("в условиях повышенной влажности и загрязнения", C.UNSUPPORTED_CLAIM, True),
    ("что делает устройство автономным", C.UNSUPPORTED_CLAIM, True),
    ("позволяет использовать прибор", C.UNSUPPORTED_CLAIM, True),
    ("обеспечивая необходимую точность показаний", C.UNSUPPORTED_CLAIM, True),
    ("для удобства монтажа предусмотрено присоединение", C.UNSUPPORTED_CLAIM, True),
    ("продолжительное время работы без замены", C.UNSUPPORTED_CLAIM, True),
    ("использовать в различных условиях эксплуатации", C.UNSUPPORTED_CLAIM, True),
    ("реле давления Demoprodukt", C.UNVERIFIED_DESIGNATION, True),
    ("с интерфейсом Bluetooth", C.UNVERIFIED_DESIGNATION, True),
    ("присоединение соответствует стандартам", C.UNSUPPORTED_CLAIM, True),
    ("имеет сертификат соответствия", C.UNSUPPORTED_CLAIM, True),
    ("изготовлен по ГОСТ", C.UNSUPPORTED_CLAIM, True),
    # --- известные слепые зоны (документируем, а не скрываем) ---
    ("точнее в 2 раза", C.UNVERIFIED_NUMBER, False),  # «2» есть в «DM 02»
    ("рабочая температура до ста градусов", C.UNVERIFIED_NUMBER, False),  # число прописью
    (
        "выдерживает перегрузку без поломок",
        C.UNSUPPORTED_CLAIM,
        False,
    ),  # смысловая выдумка без числа
]

MIN_RECALL = 37 / 40  # зафиксированное значение: падение означает регресс проверки


@pytest.fixture(scope="module")
def facts():
    return build_facts(ProductInput.model_validate(PDF_PRODUCT))


def findings(text, facts):
    return [
        i for i in check_text(text, "seo_description", facts) if i.severity is not Severity.INFO
    ]


@pytest.mark.parametrize("text", CLEAN)
def test_no_false_positives_on_clean_fragments(text, facts):
    assert findings(text, facts) == []


@pytest.mark.parametrize(("text", "code", "catchable"), [c for c in FABRICATED if c[2]])
def test_fabrications_are_caught_with_expected_code(text, code, catchable, facts):
    assert code in [i.code for i in findings(text, facts)]


def test_benchmark_summary(facts, capsys):
    fp = sum(bool(findings(t, facts)) for t in CLEAN)
    caught = [bool(findings(t, facts)) for t, _, _ in FABRICATED]
    recall = sum(caught) / len(caught)
    with capsys.disabled():
        print(
            f"\nВАЛИДАТОР: ложных срабатываний {fp}/{len(CLEAN)}, "
            f"полнота {sum(caught)}/{len(caught)} = {recall:.0%} "
            f"(пропущено: {[t for (t, _, _), c in zip(FABRICATED, caught, strict=True) if not c]})"
        )
    assert fp == 0
    assert recall >= MIN_RECALL - 1e-9
