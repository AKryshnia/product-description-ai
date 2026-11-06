"""Привязка значений в тексте к характеристикам входных данных.

Раньше значение считалось подтверждённым, если такое число встречается во входе где угодно.
Поэтому число, взятое у другой характеристики, другого исполнения или другого условия, проходило
проверку. Теперь для каждого числа, единицы и обозначения в тексте программа ищет «место» входа
(фрагмент значения характеристики), где стоит то же значение, и сверяет контекст:

 * УСЛОВИЕ — если у значения во входе есть условие («(в металлическом корпусе)», «при +25 °C»),
   оно должно быть названо рядом; среди вариантов одной характеристики сравниваются только
   отличительные слова условия (без общего «корпус»);
 * ПЕРЕЧЕНЬ — значение из списка вариантов или опций («…; …; …») нельзя выдать за основное:
   в тексте должно быть название списка («опции по карте заказа», «варианты исполнения»);
 * СОСЕДИ — если соседние слова описывают другую характеристику с другим значением («рабочая
   температура» вместо «температура хранения», «фаза-нуль» вместо «фаза-фаза»), это противоречие;
 * ИСТОЧНИК — значение, найденное только в свободном описании, подтверждённым не считается.

Проверка эвристическая и работает по основам слов, без разбора смысла: она находит
несоответствие контекста, но не доказывает, что смысл сохранён.
"""

import re
from bisect import bisect
from dataclasses import dataclass, replace

from app.facts import chains, drop_instruction_sentences, fold, normalize_text, sentence_spans
from app.schemas import Issue, IssueCode, Severity

# обозначение модели/типа: буквы + цифра(ы), например «DM02», «RS-485», «G1/2», «IP65»
DESIGNATION = re.compile(r"(?<!\w)[A-Za-zА-Яа-яЁё]{1,8}-?\d[\w./-]*")
_WORD = re.compile(r"[a-zа-яё]{4,}")
_PAREN = re.compile(r"\(([^()]{2,120})\)")
_PRI = re.compile(r"\bпри\b")
# «перечень» — три и более варианта через «;» (опции, исполнения). Два варианта одной величины
# («в металлическом корпусе» / «в корпусе из полистирола») различает условие, а не название.
LIST_MIN = 3


def squash(text: str) -> str:
    return re.sub(r"[\s\-–—]", "", fold(text))


_COMPOUND = re.compile(r"[a-zа-яё]{3,}-[a-zа-яё]{3,}")


def stems(text: str) -> frozenset[str]:
    """Основы слов (первые 5 букв) от 4 букв и составные слова целиком («фаза-нуль»): «метки», по
    которым сравнивается контекст. Составные слова различают «фаза-нуль» и «фаза-фаза»."""
    folded = fold(text)
    return frozenset([w[:5] for w in _WORD.findall(folded)] + _COMPOUND.findall(folded))


@dataclass(frozen=True)
class Found:
    """Значение в тексте: число (с единицей или без) либо обозначение."""

    kind: str  # "num" | "des"
    numbers: tuple[str, ...]
    unit: str | None
    code: str  # у обозначения — запись без пробелов и дефисов в нижнем регистре
    fragment: str
    start: int
    end: int
    spelled: bool = False


def scan(norm: str) -> list[Found]:
    """Числа и обозначения в нормализованном тексте. Число внутри обозначения («65» в «IP65»)
    отдельным значением не считается."""
    designations = [
        Found(
            "des", (), None, squash(m.group().rstrip(".,/-")), m.group().rstrip(".,/-"), *m.span()
        )
        for m in DESIGNATION.finditer(norm)
    ]
    found = list(designations)
    for ch in chains(norm):
        if not any(d.start <= ch.start and ch.end <= d.end for d in designations):
            found.append(
                Found(
                    "num", tuple(ch.numbers), ch.unit, "", ch.fragment, ch.start, ch.end, ch.spelled
                )
            )
    return sorted(found, key=lambda f: f.start)


@dataclass(frozen=True)
class Token:
    found: Found
    qualifier: str = ""  # текст условия: «в металлическом корпусе», «при +25 °C»
    need: frozenset = frozenset()  # отличительные основы и числа условия (см. Place)


@dataclass(frozen=True)
class Place:
    """Одно место входа: фрагмент значения характеристики (между «;»), предложение описания или
    название товара."""

    trusted: bool  # характеристики и название — да; свободный текст описания — нет
    key: str  # название характеристики ("" у названия и описания)
    label: str  # короткая подпись для сообщений
    marks: frozenset  # основы слов названия характеристики и фрагмента: «о чём речь»
    key_marks: frozenset  # основы слов названия характеристики
    words: frozenset  # основы слов самого фрагмента (без названия)
    listed: bool  # значение — перечень из трёх и более вариантов через «;»
    tokens: tuple[Token, ...] = ()


def _qualifier(part: str, found: list[Found]) -> dict[int, str]:
    """Условия значений фрагмента: скобки сразу после значения или хвост «при …»."""
    cond: dict[int, str] = {}
    pri = _PRI.search(part)
    for i, f in enumerate(found):
        if f.kind != "num" or not f.unit:
            continue  # у обозначений («L3 (…)») скобки — определение, а не условие
        paren = _PAREN.search(part, f.end)
        near = (
            paren
            and not re.search(r"\d", part[f.end : paren.start()])
            and (paren.start() - f.end <= 40)
        )
        if near:
            cond[i] = paren.group(1).strip()
        elif pri and f.start < pri.start():
            cond[i] = part[pri.start() :].strip(" .,;")
    return cond


def _need(qualifier: str) -> frozenset:
    nums = {n for c in chains(qualifier) for n in c.numbers}
    return stems(qualifier) | nums


def _place(key: str, part: str, *, trusted: bool, listed: bool = False) -> Place:
    norm_part = normalize_text(part)
    found = scan(norm_part)
    cond = _qualifier(norm_part, found) if trusted else {}
    # «ток … при сопротивлении не более 10 Ом»: «при …» в названии — тоже условие значения
    in_key = _PRI.search(key)
    key_condition = key[in_key.start() :].strip() if in_key and trusted else ""
    tokens = []
    for i, f in enumerate(found):
        qualifier = cond.get(i) or (key_condition if f.kind == "num" and f.unit else "")
        tokens.append(Token(f, qualifier, _need(qualifier) if qualifier else frozenset()))
    tokens += [Token(f) for f in scan(normalize_text(key))]  # значения в названии: «не более 10 Ом»
    shown = f"{key}: {part}" if key else part
    return Place(
        trusted,
        key,
        shown if len(shown) <= 90 else shown[:87] + "…",
        stems(key) | stems(part),
        stems(key),
        stems(part),
        listed,
        tuple(tokens),
    )


def build_places(product, description: str) -> tuple[Place, ...]:
    """Места входа: название и производитель, фрагменты характеристик, предложения описания.
    `description` — уже без предложений, похожих на команды."""
    places = [
        _place("", t, trusted=True) for t in (product.name, product.manufacturer or "") if t.strip()
    ]
    for key, value in (product.specs or {}).items():
        value = drop_instruction_sentences(value)  # команда не подтверждает свои значения
        parts = [p.strip() for p in re.split(r"[;\n]", value) if p.strip()]
        group = [_place(key, p, trusted=True, listed=len(parts) >= LIST_MIN) for p in parts]
        places += _distinguish(group)
    for a, b in sentence_spans(description):
        places.append(_place("", description[a:b], trusted=False))
    return tuple(places)


def _distinguish(group: list[Place]) -> list[Place]:
    """Среди вариантов одной характеристики у значения отличительны только те слова условия,
    которых нет у других вариантов («металл», а не общее «корпус»)."""
    out = []
    for place in group:
        others = [t for p in group if p is not place for t in p.tokens if t.need]
        tokens = []
        for t in place.tokens:
            rest = set().union(*(o.need for o in others if o.found.numbers != t.found.numbers))
            tokens.append(replace(t, need=frozenset((t.need - rest) or t.need)))
        out.append(replace(place, tokens=tuple(tokens)))
    return out


def conditional_words(places: tuple[Place, ...]) -> dict[str, frozenset]:
    """Слова, которые во входе стоят ТОЛЬКО среди вариантов перечня (не в названиях, не в обычных
    значениях, не в описании): основа -> название перечня. Материал («из нержавеющей стали»),
    названный в тексте без названия перечня, мог быть взят у опции и выдан за основное."""
    listed: dict[str, set] = {}
    plain: set[str] = set()
    for p in places:
        plain |= p.key_marks
        if p.listed:
            for w in p.words:
                listed.setdefault(w, set()).add(p.key_marks)
        else:
            plain |= p.words
    return {
        w: frozenset().union(*keys) for w, keys in listed.items() if w not in plain and len(w) >= 5
    }


# ---------------------------------------------------------------- проверка текста


def _matches(gen: Found, src: Found) -> bool:
    if gen.kind == "des" or src.kind == "des":
        return gen.kind == src.kind and gen.code == src.code
    if gen.unit and gen.unit != src.unit:
        return False
    n = len(gen.numbers)
    return any(src.numbers[i : i + n] == gen.numbers for i in range(len(src.numbers) - n + 1))


def _issue(code, field, message, detail, severity=Severity.REVIEW) -> Issue:
    return Issue(code=code, severity=severity, field=field, message=message, detail=detail)


def _context(norm: str, a: int, b: int, found: list[Found]):
    """Слова предложения: все (broad), слова собственной части каждого значения (own) и они же
    вместе со словами частей без значений (near). Части предложения разделены «, ; ( )». Слово
    относится к ближайшему значению своей части; при равном расстоянии — к следующему (подпись
    стоит перед значением: «диаметр 100 мм»). Часть без значений отдаёт слова ближайшему
    значению предложения (условие в скобках: «60 °С (в металлическом корпусе)»)."""
    breaks = [i for i in range(a, b) if norm[i] in ",;()"]
    clauses = [bisect(breaks, f.start) for f in found]
    broad: set[str] = set()
    own: list[set[str]] = [set() for _ in found]
    near: list[set[str]] = [set() for _ in found]
    folded = fold(norm)
    marks = [(m.start(), m.end(), m.group()[:5]) for m in _WORD.finditer(folded, a, b)]
    marks += [(m.start(), m.end(), m.group()) for m in _COMPOUND.finditer(folded, a, b)]
    for start, end, stem in marks:
        broad.add(stem)
        if not found:
            continue
        same = [i for i, c in enumerate(clauses) if c == bisect(breaks, start)]
        i = min(
            same or range(len(found)),
            key=lambda i: (
                max(0, found[i].start - end, start - found[i].end),
                found[i].start < start,
            ),
        )
        near[i].add(stem)
        if same:
            own[i].add(stem)
    return broad, own, near


def check_bindings(text: str, field: str, facts) -> list[Issue]:
    """Контекст каждого значения и слов из перечней. Возвращает замечания к тексту."""
    norm = normalize_text(text)
    issues: list[Issue] = []
    seen: set[tuple] = set()

    def add(issue: Issue) -> None:
        key = (issue.code, issue.detail)
        if key not in seen:
            seen.add(key)
            issues.append(issue)

    everything = scan(norm)
    for a, b in sentence_spans(norm):
        found = [f for f in everything if a <= f.start < b]
        broad, own, near = _context(norm, a, b, found)
        sentence_numbers = {n for f in found for n in f.numbers}
        for f, mine, wide in zip(found, own, near, strict=True):
            problem = _judge(f, mine, wide, broad, sentence_numbers, facts, field)
            if problem:
                add(problem)
    return issues


def _judge(f: Found, own, near, broad, sentence_numbers, facts, field) -> Issue | None:
    """Замечание к одному значению или None, если хотя бы одно место входа ему соответствует."""
    candidates = [(p, t) for p in facts.places for t in p.tokens if _matches(f, t.found)]
    if not candidates:
        return None  # значения нет во входе — это находят проверки чисел, единиц и обозначений
    trusted = [c for c in candidates if c[0].trusted]
    if not trusted:
        return _issue(
            IssueCode.NARRATIVE_ONLY_VALUE,
            field,
            f"Значение «{f.fragment}» найдено только в свободном описании товара, а не в "
            "характеристиках. Описание не считается подтверждением: сверьте значение с "
            "паспортом.",
            f.fragment,
        )
    if f.kind == "des" and any(not p.key for p, _ in trusted):
        return None  # обозначение из самого названия товара («testo 608-H1») — не характеристика
    first = None
    for place, token in sorted(trusted, key=lambda c: -len(broad & c[0].marks)):
        problem = _problem(
            f, own, near, broad, sentence_numbers, field, facts, place, token, candidates
        )
        if problem is None:
            return None
        first = first or problem
    return first


def _problem(f, own, near, broad, sentence_numbers, field, facts, place, token, candidates):
    matching = {id(p) for p, _ in candidates}
    mine = len(own & place.marks)
    best = None
    for other in facts.places:
        if not other.trusted or not other.key or id(other) in matching or not other.tokens:
            continue
        gain = own & other.marks
        if len(gain) > mine and gain - place.marks and (best is None or len(gain) > best[0]):
            best = (len(gain), other)
    if best:
        other = best[1]
        return _issue(
            IssueCode.VALUE_CONTEXT_MISMATCH,
            field,
            f"Значение «{f.fragment}» во входных данных относится к «{place.label}», а соседние "
            f"слова описывают «{other.label}». Проверьте, к какой характеристике оно относится.",
            f.fragment,
        )
    if token.need and not token.need & (near | sentence_numbers):
        return _issue(
            IssueCode.CONDITION_DROPPED,
            field,
            f"Во входных данных значение дано с условием «{token.qualifier}», а в тексте условие "
            "рядом не указано: значение могло быть обобщено на все варианты. Проверьте вручную.",
            f.fragment,
        )
    if place.listed and place.key_marks and not broad & place.key_marks:
        return _issue(
            IssueCode.CONDITION_DROPPED,
            field,
            f"Значение взято из перечня вариантов «{place.label}», а название перечня в тексте не "
            "указано: вариант (опцию, исполнение) могли выдать за основную характеристику. "
            "Проверьте вручную.",
            f.fragment,
        )
    return None
