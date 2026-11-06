"""Программная проверка результата генерации.

Три исхода для каждого наблюдения: подтверждено (замечаний нет), нарушение (ERROR) и
«нельзя определить уверенно» (REVIEW — на ручную проверку). Проверка эвристическая и защиту от
выдумок НЕ гарантирует: она ловит числа, единицы и типовые рекламные/отраслевые утверждения,
которых нет во входных данных; смысловые домыслы без чисел остаются на ручной проверке.

Подсчёт символов: len() строки Python (кодовые точки Юникода) после схлопывания пробельных
последовательностей в один пробел (это делает `Draft`); пробелы и знаки препинания учитываются.

Числа, единицы и обозначения проверяются в два шага: «есть ли такое значение во входе»
(этот модуль) и «к той ли характеристике, исполнению и условию оно отнесено» (app/binding.py).
"""

import re
from dataclasses import dataclass, field

from app.binding import (
    DESIGNATION,
    Place,
    build_places,
    check_bindings,
    conditional_words,
    squash,
    stems,
)
from app.facts import (
    UNIT_SPELLINGS,
    chains,
    drop_instruction_sentences,
    fold,
    standalone_units,
)
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
    # не слово «различные» само по себе («в различных помещениях: музеях, библиотеках» — перечень
    # из источника), а обобщение об условиях и областях применения
    r"различн\w* (?:услови|режим|отрасл|област|сфер|задач)",
)
_CLAIM_STEMS = _PROMO + _INDUSTRIES + _CONDITIONS
# сведения о сертификации и стандартах подтверждает только структурированная характеристика:
# упоминание в свободном описании (в том числе внедрённой команде) для этого недостаточно
_SENSITIVE = frozenset({"сертифик", "сертифиц", "гост", "стандарт"})


# слова латиницей (от 3 букв; единицы вроде kPa не в счёт): названия, протоколы, бренды —
# должны встречаться во входных данных
_LATIN_WORD = re.compile(r"(?<![\w])[A-Za-z]{3,}(?![\w])")


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
    trusted_text: str = ""  # то же без свободного описания: названия и характеристики
    words: frozenset[str] = frozenset()  # основы всех слов входа («из …», «без …»)
    polar: dict[str, frozenset[bool]] = field(default_factory=dict)  # основа -> с «не-» / без
    places: tuple[Place, ...] = ()  # фрагменты входа со значениями (см. app/binding.py)
    conditional: dict[str, frozenset] = field(default_factory=dict)  # слова только из перечней


def _split_negation(word: str) -> tuple[str, bool]:
    """'неагрессивных' -> ('агрес', True); 'агрессивных' -> ('агрес', False)."""
    if word.startswith("не") and len(word) - 2 >= 6:
        return word[2:7], True
    return word[:5], False


def build_facts(product: ProductInput) -> Facts:
    description = drop_instruction_sentences(product.description or "")
    trusted = [product.name, product.manufacturer or ""]
    for key, value in (product.specs or {}).items():
        trusted += [key, drop_instruction_sentences(value)]
    corpus = "\n".join([*trusted, description])  # перевод строки не даёт числам склеиться

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
    folded = fold(corpus)
    polar: dict[str, set[bool]] = {}
    for w in re.findall(r"[а-яе]{6,}", folded):
        stem, negated = _split_negation(w)
        polar.setdefault(stem, set()).add(negated)
    places = build_places(product, description)
    return Facts(
        frozenset(numbers),
        frozenset(fractional),
        frozenset(pairs),
        frozenset(units),
        folded,
        squash(corpus),
        tuple(w[:5] for w in re.findall(r"[a-zа-яё0-9]{3,}", fold(product.name))),
        fold("\n".join(trusted)),
        frozenset(w[:5] for w in re.findall(r"[а-яе]{4,}", folded)),
        {k: frozenset(v) for k, v in polar.items()},
        places,
        conditional_words(places),
    )


def _issue(code, severity, field, message, detail=None) -> Issue:
    return Issue(code=code, severity=severity, field=field, message=message, detail=detail)


def validate_draft(draft: Draft, facts: Facts) -> list[Issue]:
    """Все замечания к черновику; пустой список — проверки пройдены."""
    issues = _check_format(draft)
    issues += _check_commercial(draft)
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
        + check_bindings(text, block, facts)
        + _check_glitches(text, block)
        + _check_claims(text, block, facts)
        + _check_polarity(text, block, facts)
        + _check_phrases(text, block, facts)
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
                opposite = n[1:] if n.startswith("-") else "-" + n
                hint = (
                    " Во входных данных такое число есть, но с противоположным знаком."
                    if opposite in facts.numbers
                    else ""
                )
                found.append(
                    _issue(
                        IssueCode.UNVERIFIED_NUMBER,
                        # число, записанное словом, — утверждение о количестве: на проверку
                        Severity.REVIEW if ch.spelled else Severity.ERROR,
                        block,
                        f"Значение «{n}» отсутствует во входных данных.{hint}",
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


_REPEAT = re.compile(r"(?<!\S)(\S{1,15})[ \t]+\1(?!\S)", re.IGNORECASE)


def _check_glitches(text: str, block: str) -> list[Issue]:
    """Подряд идущие одинаковые токены («% %», «для для») — типичный сбой генерации."""
    return [
        _issue(
            IssueCode.TEXT_GLITCH,
            Severity.REVIEW,
            block,
            f"В тексте подряд повторяется «{m.group(0)}» — похоже на сбой генерации. Исправьте.",
            m.group(0),
        )
        for m in _REPEAT.finditer(text)
    ]


def _check_designations(text: str, block: str, facts: Facts) -> list[Issue]:
    """Обозначения должны совпадать с входными (модель не должна транслитерировать ДЕМО -> DEMO).
    Пробелы и дефисы не различаются: «RS485» и «RS-485» — одна запись."""
    found, seen = [], set()
    designations = list(DESIGNATION.finditer(text))
    covered = [m.span() for m in designations]
    # латинское слово, уже вошедшее в найденное обозначение («DEMO» в «DEMO-100»), не дублируем
    words = [
        m
        for m in _LATIN_WORD.finditer(text)
        if not any(a <= m.start() and m.end() <= b for a, b in covered)
        and m.group(0) not in UNIT_SPELLINGS
    ]
    for m in (*designations, *words):
        token = m.group(0).rstrip(".,/-")
        if token in seen or squash(token) in facts.squashed:
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
    folded = fold(text)
    found = []
    for stem in _CLAIM_STEMS:
        m = re.search(rf"\w*{stem}\w*", folded)
        if not m:
            continue
        sensitive = stem in _SENSITIVE
        # stem — основа или фразовый шаблон
        if re.search(stem, facts.trusted_text if sensitive else facts.text):
            continue
        only_free_text = sensitive and re.search(stem, facts.text)
        found.append(
            _issue(
                IssueCode.UNSUPPORTED_CLAIM,
                Severity.REVIEW,
                block,
                "Сведение о сертификации или стандарте есть только в свободном описании, а не в "
                "характеристиках. Подтвердите его документом."
                if only_free_text
                else "Формулировка не подтверждена входными данными (рекламное утверждение, "
                "отрасль или условие эксплуатации). Уберите или подтвердите источником.",
                text[m.start() : m.end()],
            )
        )
    return found


def _check_polarity(text: str, block: str, facts: Facts) -> list[Issue]:
    """«Неагрессивных» во входе не подтверждает «агрессивных»: слово с противоположным смыслом
    (с «не-» или без него) не должно появляться, если в источнике оно только в другой форме."""
    found, seen = [], set()
    for m in re.finditer(r"[А-Яа-яЁё]{6,}", text):
        word = fold(m.group())
        stem, negated = _split_negation(word)
        known = facts.polar.get(stem)
        if known and negated not in known and stem not in seen:
            seen.add(stem)
            found.append(
                _issue(
                    IssueCode.UNSUPPORTED_CLAIM,
                    Severity.REVIEW,
                    block,
                    f"Слово «{m.group()}» во входных данных встречается только с противоположным "
                    f"смыслом ({'без' if negated else 'с'} отрицания «не-»): возможна замена "
                    "свойства на противоположное. Проверьте вручную.",
                    m.group(),
                )
            )
    return found


# «изготовлен из …», «корпус из …» и «без …»: материал и отсутствие обслуживания или функций —
# утверждения о приборе, которые нужно найти во входе
_MADE_OF = re.compile(
    r"(?:изготовл|выполнен|сделан|состо|собран|покрыт|материал|корпус)\w*\s+(?:\S+\s+){0,2}?"
    r"из\s+([а-яе]{4,})"
)
_WITHOUT = re.compile(r"\bбез\s+([а-яе]{4,})")


def _check_phrases(text: str, block: str, facts: Facts) -> list[Issue]:
    folded = fold(text)
    found = []
    for pattern, lead in ((_MADE_OF, "из"), (_WITHOUT, "без")):
        for m in pattern.finditer(folded):
            word = m.group(1)
            shown = text[m.start(1) : m.end(1)]
            keys = facts.conditional.get(word[:5])
            if keys and not keys & stems(text):
                found.append(
                    _issue(
                        IssueCode.CONDITION_DROPPED,
                        Severity.REVIEW,
                        block,
                        f"Слово «{shown}» во входных данных стоит только среди вариантов или опций "
                        "перечня, а название перечня (например, «по заказу») в тексте не указано: "
                        "материал или вариант могли выдать за основную характеристику. "
                        "Проверьте вручную.",
                        f"{lead} {shown}",
                    )
                )
            elif word[:5] not in facts.words:
                found.append(
                    _issue(
                        IssueCode.UNSUPPORTED_CLAIM,
                        Severity.REVIEW,
                        block,
                        f"После «{lead}» названо «{shown}», такого слова во входных данных нет: "
                        "материал или отсутствие свойства не подтверждены. Проверьте вручную.",
                        f"{lead} {shown}",
                    )
                )
    return found


def _check_title_topic(title: str, facts: Facts) -> list[Issue]:
    if facts.name_stems and not any(s in fold(title) for s in facts.name_stems):
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


# заглушки вместо содержания: «—», «данных нет», «не указано»
_PLACEHOLDER = re.compile(r"(?:данных нет|нет данных|не указан\w*|не определен\w*|н/д|n/a)", re.I)


def _meaningful(item: str) -> bool:
    return bool(re.search(r"[A-Za-zА-Яа-яЁё]{3,}", item)) and not _PLACEHOLDER.fullmatch(
        item.strip(" .—–-")
    )


def _check_commercial(d: Draft) -> list[Issue]:
    """Аудитория и задачи обязательны; сценарии могут быть пустыми, если во входе их нет."""
    c = d.commercial
    empty = [
        label
        for label, items in (("audience (кому подходит)", c.audience), ("tasks (задачи)", c.tasks))
        if not any(_meaningful(i) for i in items)
    ]
    if not empty:
        return []
    return [
        _issue(
            IssueCode.COMMERCIAL_EMPTY,
            Severity.ERROR,
            "commercial",
            "В коммерческом блоке нет содержательных аудитории и задач: пусты "
            f"{', '.join(empty)}. Заполните их по назначению прибора из данных "
            "(например: «Специалисты, которым требуется измерение <величина из данных>»); "
            "не выдумывай отрасли, категории покупателей и задачи, которых нет во входных "
            "данных. «scenarios» может остаться пустым.",
        )
    ]


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
