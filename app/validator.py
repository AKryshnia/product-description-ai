"""Программная проверка результата генерации.

Три исхода для каждого наблюдения: подтверждено (замечаний нет), нарушение (ERROR) и
«нельзя определить уверенно» (REVIEW — на ручную проверку). Проверка эвристическая и защиту от
выдумок НЕ гарантирует: она ловит числа, единицы и типовые рекламные/отраслевые утверждения,
которых нет во входных данных; смысловые домыслы без чисел остаются на ручной проверке.

Подсчёт символов: len() строки Python (кодовые точки Юникода) после схлопывания пробельных
последовательностей в один пробел (это делает `Draft`); пробелы и знаки препинания учитываются.
"""

import re
from dataclasses import dataclass

from app.facts import chains, drop_instruction_sentences, standalone_units
from app.schemas import Draft, Issue, IssueCode, ProductInput, Severity

TITLE_MAX = 70
DESCRIPTION_MIN, DESCRIPTION_MAX = 800, 1200
QUERIES_MIN, QUERIES_MAX = 3, 5
QUERY_MAX = 100

# Основы слов (в нижнем регистре, «ё» -> «е»). Утверждение допустимо, только если такая основа
# есть и во входных данных. Список эвристический и намеренно короткий.
_PROMO = (
    "лучш",
    "идеальн",
    "уникальн",
    "высококачествен",
    "безупречн",
    "надежн",
    "долговечн",
    "высокоточн",
    "инновацион",
    "передов",
    "ведущ",
    "непревзойд",
    "премиальн",
    "гарантир",
    "максимальн",
    "абсолютн",
    "лидер",
    "эксклюзив",
    "революцион",
    "превосход",
    "оптимальн",
    "высок",
    "широк",
    "удобн",
    "облегча",
    "стабильн",
    "эффективн",
    "комфортн",
    "быстр",
    "позволя",
    "делает",
    "повыша",
    "автоном",
    "обеспечи",
    "удобств",
    "продолжительн",
    "длительн",
    "различн",
)
_INDUSTRIES = (
    "нефтегаз",
    "нефтян",
    "нефтехим",
    "энергетик",
    "жкх",
    "теплосет",
    "теплоснабж",
    "водоснабж",
    "химическ",
    "пищев",
    "фармацевт",
    "металлург",
    "судостро",
    "авиацион",
    "атомн",
    "горнодобыв",
    "машиностро",
    "агропром",
    "котельн",
    "промышленн",
    "производствен",
    "складск",
    "предприят",
    r"технологическ\w* процесс",  # фраза, а не основа: см. тест про технологическую черту
    "мониторинг",
    "здани",
    "сооружен",
    "стандарт",
    "сертифик",
    "сертифиц",
    "гост",
)
_CONDITIONS = (
    "взрывоопасн",
    "агрессивн",
    "вибраци",
    "морозост",
    "влагозащ",
    "пылезащ",
    "ударопроч",
    "износостойк",
    "коррози",
    "всепогодн",
    "экстремальн",
    "устойчив",
    "пыл",
    "влаг",
    "защищ",
    "влажн",
    "загрязн",
)
_CLAIM_STEMS = _PROMO + _INDUSTRIES + _CONDITIONS


def _fold(text: str) -> str:
    return text.casefold().replace("ё", "е")  # длина строки не меняется


def _squash(text: str) -> str:
    return re.sub(r"[\s\-–—]", "", _fold(text))


# обозначение модели/типа: буквы + цифра(ы), например «DM02», «RS-485», «G1/2», «IP65»
_DESIGNATION = re.compile(r"(?<!\w)[A-Za-zА-Яа-яЁё]{1,8}-?\d[\w./-]*")
# слова латиницей (от 4 букв): названия, протоколы, бренды — должны встречаться во входных данных
_LATIN_WORD = re.compile(r"(?<![\w])[A-Za-z]{4,}(?![\w])")


@dataclass(frozen=True)
class Facts:
    """Всё, что можно утверждать на основании входных данных."""

    numbers: frozenset[str]  # все числа (нормализованные)
    fractional: frozenset[str]  # числа, записанные во входе с десятичным разделителем
    pairs: frozenset[tuple[str, str]]  # (число, единица), стоящие рядом во входе
    units: frozenset[str]  # единицы, упомянутые во входе
    text: str  # корпус входного текста в нижнем регистре
    squashed: str  # он же без пробелов и дефисов — для сравнения обозначений
    name_stems: tuple[str, ...]  # основы слов названия — для проверки заголовка


def build_facts(product: ProductInput) -> Facts:
    description = drop_instruction_sentences(product.description or "")
    parts = [product.name, product.manufacturer or "", description]
    for key, value in (product.specs or {}).items():
        parts += [key, value]
    corpus = "\n".join(parts)  # перевод строки не даёт числам разных полей склеиться

    numbers, fractional, pairs, units = set(), set(), set(), set()
    for ch in chains(corpus):
        for n, frac in zip(ch.numbers, ch.fractional, strict=True):
            numbers.add(n)
            if frac:
                fractional.add(n)
            if ch.unit:
                pairs.add((n, ch.unit))
                units.add(ch.unit)
    units |= standalone_units(corpus)
    stems = tuple(w[:5] for w in re.findall(r"[a-zа-яё0-9]{3,}", _fold(product.name)))
    return Facts(
        frozenset(numbers),
        frozenset(fractional),
        frozenset(pairs),
        frozenset(units),
        _fold(corpus),
        _squash(corpus),
        stems,
    )


def _issue(code, severity, field, message, detail=None) -> Issue:
    return Issue(code=code, severity=severity, field=field, message=message, detail=detail)


def validate_draft(draft: Draft, facts: Facts) -> list[Issue]:
    """Все замечания к черновику; пустой список — проверки пройдены."""
    issues = _check_format(draft)
    issues += _check_title_topic(draft.seo_title, facts)
    for block, text in _text_blocks(draft):
        issues += check_text(text, block, facts)
    return issues


def _text_blocks(draft: Draft):
    yield "seo_title", draft.seo_title
    yield "seo_description", draft.seo_description
    for q in draft.search_queries:
        yield "search_queries", q
    c = draft.commercial
    for item in (*c.audience, *c.tasks, *c.scenarios):
        yield "commercial", item


def check_text(text: str, block: str, facts: Facts) -> list[Issue]:
    """Числа, единицы и неподтверждённые утверждения в одном фрагменте текста."""
    return (
        _check_numbers(text, block, facts)
        + _check_designations(text, block, facts)
        + _check_claims(text, block, facts)
    )


def _check_numbers(text: str, block: str, facts: Facts) -> list[Issue]:
    found: list[Issue] = []
    seen: set[str] = set()
    for ch in chains(text):
        if ch.fragment in seen:
            continue
        seen.add(ch.fragment)
        number_issue = False
        for n, frac in zip(ch.numbers, ch.fractional, strict=True):
            if n not in facts.numbers:
                number_issue = True
                found.append(
                    _issue(
                        IssueCode.UNVERIFIED_NUMBER,
                        Severity.ERROR,
                        block,
                        f"Значение «{n}» отсутствует во входных данных.",
                        ch.fragment,
                    )
                )
            elif frac and n not in facts.fractional:
                # «1,0» при входном «1»: возможно, выдуманное значение (например, класс точности)
                found.append(
                    _issue(
                        IssueCode.UNVERIFIED_NUMBER,
                        Severity.REVIEW,
                        block,
                        f"Значение «{n}» записано как десятичное, а во входных данных "
                        "такой записи нет. "
                        "Проверьте вручную.",
                        ch.fragment,
                    )
                )
        if ch.unit and not number_issue:
            if ch.unit not in facts.units:
                found.append(
                    _issue(
                        IssueCode.UNVERIFIED_UNIT,
                        Severity.ERROR,
                        block,
                        f"Единица «{ch.unit}» не встречается во входных данных "
                        "(возможен пересчёт единиц или выдуманное значение).",
                        ch.fragment,
                    )
                )
            elif any((n, ch.unit) not in facts.pairs for n in ch.numbers):
                found.append(
                    _issue(
                        IssueCode.UNIT_PAIR_UNCONFIRMED,
                        Severity.REVIEW,
                        block,
                        f"Сочетание значения и единицы «{ch.unit}» не найдено во входных данных "
                        "в таком виде. Проверьте вручную.",
                        ch.fragment,
                    )
                )
    return found


def _check_designations(text: str, block: str, facts: Facts) -> list[Issue]:
    """Обозначения должны совпадать с входными (модель не должна транслитерировать ДЕМО -> DEMO).
    Пробелы и дефисы не различаются: «RS485» и «RS-485» — одна запись."""
    found, seen = [], set()
    designations = list(_DESIGNATION.finditer(text))
    covered = [m.span() for m in designations]
    # латинское слово, уже вошедшее в найденное обозначение («DEMO» в «DEMO-100»), не дублируем
    words = [
        m
        for m in _LATIN_WORD.finditer(text)
        if not any(a <= m.start() and m.end() <= b for a, b in covered)
    ]
    for m in (*designations, *words):
        token = m.group(0).rstrip(".,/-")
        if token in seen or _squash(token) in facts.squashed:
            continue
        seen.add(token)
        found.append(
            _issue(
                IssueCode.UNVERIFIED_DESIGNATION,
                Severity.REVIEW,
                block,
                f"Обозначение или название «{token}» не совпадает ни с одной записью во входных "
                "данных (возможна транслитерация или выдумка). Проверьте вручную.",
                token,
            )
        )
    return found


def _check_claims(text: str, block: str, facts: Facts) -> list[Issue]:
    folded = _fold(text)
    found = []
    for stem in _CLAIM_STEMS:
        m = re.search(rf"\w*{stem}\w*", folded)
        if m and not re.search(stem, facts.text):  # stem — основа или фразовый шаблон
            found.append(
                _issue(
                    IssueCode.UNSUPPORTED_CLAIM,
                    Severity.REVIEW,
                    block,
                    "Формулировка не подтверждена входными данными (рекламное утверждение, отрасль "
                    "или условие эксплуатации). Уберите или подтвердите источником.",
                    text[m.start() : m.end()],
                )
            )
    return found


def _check_title_topic(title: str, facts: Facts) -> list[Issue]:
    if facts.name_stems and not any(s in _fold(title) for s in facts.name_stems):
        return [
            _issue(
                IssueCode.TITLE_OFF_TOPIC,
                Severity.REVIEW,
                "seo_title",
                "В заголовке нет слов из названия товара — проверьте соответствие.",
            )
        ]
    return []


def _length_message(n: int) -> str:
    if n < DESCRIPTION_MIN:
        return (
            f"SEO-описание слишком короткое: {n} символов при минимуме {DESCRIPTION_MIN}; "
            f"нужно добавить не менее {DESCRIPTION_MIN - n}. Если исходных данных объективно "
            "мало, дополните их: объём не набирается домыслами."
        )
    return (
        f"SEO-описание слишком длинное: {n} символов при максимуме {DESCRIPTION_MAX}; "
        f"нужно сократить не менее чем на {n - DESCRIPTION_MAX}."
    )


def _check_format(d: Draft) -> list[Issue]:
    found = []
    if len(d.seo_title) > TITLE_MAX:
        found.append(
            _issue(
                IssueCode.TITLE_TOO_LONG,
                Severity.ERROR,
                "seo_title",
                f"SEO-заголовок слишком длинный: {len(d.seo_title)} символов при максимуме "
                f"{TITLE_MAX}; сократите не менее чем на {len(d.seo_title) - TITLE_MAX}.",
            )
        )
    n = len(d.seo_description)
    if not DESCRIPTION_MIN <= n <= DESCRIPTION_MAX:
        found.append(
            _issue(
                IssueCode.DESCRIPTION_LENGTH,
                Severity.ERROR,
                "seo_description",
                _length_message(n),
            )
        )
    queries = d.search_queries
    if not QUERIES_MIN <= len(queries) <= QUERIES_MAX:
        found.append(
            _issue(
                IssueCode.QUERIES_COUNT,
                Severity.ERROR,
                "search_queries",
                f"Поисковых запросов: {len(queries)}, требуется {QUERIES_MIN}–{QUERIES_MAX}.",
            )
        )
    seen: set[str] = set()
    for q in queries:
        if q.casefold() in seen:
            found.append(
                _issue(
                    IssueCode.QUERY_INVALID,
                    Severity.ERROR,
                    "search_queries",
                    "Повторяющийся поисковый запрос.",
                    q,
                )
            )
        elif len(q) > QUERY_MAX:
            found.append(
                _issue(
                    IssueCode.QUERY_INVALID,
                    Severity.ERROR,
                    "search_queries",
                    f"Запрос длиннее {QUERY_MAX} символов.",
                    q[:60] + "…",
                )
            )
        seen.add(q.casefold())
    return found
