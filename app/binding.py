"""Привязка значений в тексте к характеристикам входных данных.

Раньше значение считалось подтверждённым, если такое число встречается во входе где угодно.
Поэтому число, взятое у другой характеристики, другого исполнения или другого условия, проходило
проверку. Теперь для каждого числа, единицы и обозначения в тексте программа ищет «место» входа
(фрагмент значения характеристики), где стоит то же значение, и сверяет, сохранена ли связь:

 * УСЛОВИЕ — если у значения во входе есть условие («(в металлическом корпусе)», «при +23 °C и
   влажности 50 %»), в тексте должно быть названо КАЖДОЕ ограничение условия, а не любое одно;
   среди вариантов одной характеристики сравниваются только отличительные слова условия;
 * ОПЦИЯ — значение из списка вариантов или из опций («опции по заказу») нельзя выдать за
   основное: в тексте должно быть название списка. Опцией характеристику делает её название или
   число вариантов от трёх, а не одно только число пунктов;
 * ГРАНИЦА — «не более 0,8 кг» и «не менее 0,8 кг» — разные утверждения; граница значения
   («не более», «не менее», «менее», «более», «до», «от») должна сохраниться;
 * СОСЕДИ — если соседние слова описывают другую характеристику с другим значением («рабочая
   температура» вместо «температура хранения», «фаза-нуль» вместо «фаза-фаза»), это противоречие;
 * ЦЕЛОСТНОСТЬ — диапазон или перечень значений должен быть записан во входе целиком: «0–16 бар»
   не подтверждается тем, что есть «0–6 бар» и «10–16 бар», хотя оба числа во входе встречаются;
 * ИСТОЧНИК — значение, найденное только в свободном описании, подтверждённым не считается.

У проверки три исхода: подтверждено (None), расхождение (замечание о найденной причине) и
«нельзя подтвердить» (замечание о неопределённости). Отсутствие найденного расхождения не
доказательство верности, поэтому «не подтверждено» не превращается в молчаливый успех.
Проверка эвристическая и работает по основам слов, без разбора смысла.
"""

import re
from bisect import bisect
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, NamedTuple

from app.facts import chains, drop_instruction_sentences, fold, normalize_text, sentence_spans
from app.schemas import Issue, IssueCode, Severity

if TYPE_CHECKING:
    from app.validator import Facts

# обозначение модели/типа: буквы + цифра(ы), например «DM02», «RS-485», «G1/2», «IP65»
DESIGNATION = re.compile(r"(?<!\w)[A-Za-zА-Яа-яЁё]{1,8}-?\d[\w./-]*")
_WORD = re.compile(r"[a-zа-яё]{4,}")
_COMPOUND = re.compile(r"[a-zа-яё]{3,}-[a-zа-яё]{3,}")
_PAREN = re.compile(r"\(([^()]{2,120})\)")
_PRI = re.compile(r"\bпри\b")
# отдельные ограничения условия: «при +23 °C и влажности 50 %» -> два; запятая десятичной не делит
_AND = re.compile(r"\s+и\s+|;\s*|,\s+")
# вариантов от трёх через «;» — перечень. Два варианта одной величины («в металлическом корпусе» /
# «в корпусе из полистирола») различает условие, а не название.
LIST_MIN = 3
# название характеристики говорит, что значение — вариант или опция, сколько бы пунктов ни было
_OPTION_KEY = re.compile(r"опци|по заказу|под заказ|дополнительн|вариант")
# основы слов, которыми сам текст называет значение вариантом или опцией («опционально»)
_OPTION_STEMS = frozenset({"опции", "опция", "опцио", "допол", "вариа"})

# граница перед числом: порядок важен («не более» раньше «более»)
_BOUNDS = (
    ("<=", re.compile(r"(?:^|\W)(?:не\s+(?:более|выше|превыша\w+)|до|макс\w*\.?)\s*$")),
    (">=", re.compile(r"(?:^|\W)(?:не\s+(?:менее|ниже)|от|мин\w*\.?)\s*$")),
    ("<", re.compile(r"(?:^|\W)(?:менее|меньше|ниже)\s*$")),
    (">", re.compile(r"(?:^|\W)(?:более|больше|выше|свыше|превыша\w+)\s*$")),
)
_BOUND_WORDS = {"<=": "не более (до)", ">=": "не менее (от)", "<": "менее", ">": "более"}


def squash(text: str) -> str:
    return re.sub(r"[\s\-–—]", "", fold(text))


def stems(text: str) -> frozenset[str]:
    """Основы слов (первые 5 букв) от 4 букв и составные слова целиком («фаза-нуль»): «метки», по
    которым сравнивается контекст. Составные слова различают «фаза-нуль» и «фаза-фаза»."""
    folded = fold(text)
    return frozenset([w[:5] for w in _WORD.findall(folded)] + _COMPOUND.findall(folded))


@dataclass(frozen=True)
class Found:
    """Значение в тексте: число (с единицей или без) либо обозначение."""

    kind: Literal["num", "des"]
    numbers: tuple[str, ...]
    unit: str | None
    code: str  # у обозначения — запись без пробелов и дефисов в нижнем регистре
    fragment: str
    start: int
    end: int
    spelled: bool = False
    bound: str = ""  # граница перед одиночным числом: "<=", ">=", "<", ">" или ""


def _bound(norm: str, start: int) -> str:
    before = fold(norm[max(0, start - 20) : start])
    return next((sign for sign, rx in _BOUNDS if rx.search(before)), "")


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
            bound = _bound(norm, ch.start) if len(ch.numbers) == 1 else ""
            found.append(
                Found(
                    "num",
                    tuple(ch.numbers),
                    ch.unit,
                    "",
                    ch.fragment,
                    ch.start,
                    ch.end,
                    ch.spelled,
                    bound,
                )
            )
    return sorted(found, key=lambda f: f.start)


@dataclass(frozen=True)
class Token:
    found: Found
    qualifier: str = ""  # текст условия: «в металлическом корпусе», «при +25 °C»
    # отличительные основы и числа каждого ограничения условия (см. _need и _distinguish)
    need: tuple[frozenset[str], ...] = ()


@dataclass(frozen=True)
class Place:
    """Одно место входа: фрагмент значения характеристики (между «;»), предложение описания или
    название товара."""

    trusted: bool  # характеристики и название — да; свободный текст описания — нет
    key: str  # название характеристики ("" у названия и описания)
    label: str  # короткая подпись для сообщений
    marks: frozenset[str]  # основы слов названия характеристики и фрагмента: «о чём речь»
    key_marks: frozenset[str]  # основы слов названия характеристики
    words: frozenset[str]  # основы слов самого фрагмента (без названия)
    optional: bool  # значение — вариант или опция, а не основная характеристика
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


def _need(qualifier: str) -> tuple[frozenset[str], ...]:
    """По одному набору основ и чисел на каждое ограничение условия."""
    parts = _AND.split(qualifier) if qualifier else []
    needs = [stems(p) | {n for c in chains(p) for n in c.numbers} for p in parts]
    return tuple(n for n in needs if n)


def _place(key: str, part: str, *, trusted: bool, optional: bool = False) -> Place:
    norm_part = normalize_text(part)
    found = scan(norm_part)
    cond = _qualifier(norm_part, found) if trusted else {}
    # «ток … при сопротивлении не более 10 Ом»: «при …» в названии — тоже условие значения
    in_key = _PRI.search(key)
    key_condition = key[in_key.start() :].strip() if in_key and trusted else ""
    tokens = []
    for i, f in enumerate(found):
        qualifier = cond.get(i) or (key_condition if f.kind == "num" and f.unit else "")
        tokens.append(Token(f, qualifier, _need(qualifier)))
    tokens += [Token(f) for f in scan(normalize_text(key))]  # значения в названии: «не более 10 Ом»
    shown = f"{key}: {part}" if key else part
    return Place(
        trusted,
        key,
        shown if len(shown) <= 90 else shown[:87] + "…",
        stems(key) | stems(part),
        stems(key),
        stems(part),
        optional,
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
        optional = len(parts) >= LIST_MIN or bool(_OPTION_KEY.search(fold(key)))
        group = [_place(key, p, trusted=True, optional=optional) for p in parts]
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
            rest = set().union(
                *(a for o in others if o.found.numbers != t.found.numbers for a in o.need)
            )
            kept = tuple(a - rest for a in t.need if a - rest)
            tokens.append(replace(t, need=kept or t.need))
        out.append(replace(place, tokens=tuple(tokens)))
    return out


def conditional_words(places: tuple[Place, ...]) -> dict[str, frozenset[str]]:
    """Слова, которые во входе стоят ТОЛЬКО среди вариантов или опций (не в названиях, не в
    обычных значениях, не в описании): основа -> название списка. Материал («из нержавеющей
    стали»), названный в тексте без названия списка, мог быть взят у опции и выдан за основной."""
    optional: dict[str, set] = {}
    plain: set[str] = set()
    for p in places:
        plain |= p.key_marks
        if p.optional:
            for w in p.words:
                optional.setdefault(w, set()).add(p.key_marks)
        else:
            plain |= p.words
    return {
        w: frozenset().union(*keys)
        for w, keys in optional.items()
        if w not in plain and len(w) >= 5
    }


# ---------------------------------------------------------------- проверка текста


def _matches(gen: Found, src: Found) -> bool:
    if gen.kind == "des" or src.kind == "des":
        return gen.kind == src.kind and gen.code == src.code
    if gen.unit and gen.unit != src.unit:
        return False
    n = len(gen.numbers)
    return any(src.numbers[i : i + n] == gen.numbers for i in range(len(src.numbers) - n + 1))


def _issue(code: IssueCode, field: str, message: str, detail: str) -> Issue:
    return Issue(code=code, severity=Severity.REVIEW, field=field, message=message, detail=detail)


class Context(NamedTuple):
    """Слова (основы) и числа вокруг одного значения текста."""

    own: frozenset[str]  # слова его части предложения, ближайшие к нему
    near: frozenset[str]  # они же и слова частей без значений (условие в скобках)
    sentence: frozenset[str]  # слова всего предложения
    numbers: frozenset[str]  # числа всего предложения


def _contexts(norm: str, a: int, b: int, found: list[Found]) -> list[Context]:
    """Контексты значений предложения. Части предложения разделены «, ; ( )». Слово относится к
    ближайшему значению своей части; при равном расстоянии — к следующему (подпись стоит перед
    значением: «диаметр 100 мм»). Часть без значений отдаёт слова ближайшему значению
    предложения (условие в скобках: «60 °С (в металлическом корпусе)»)."""
    breaks = [i for i in range(a, b) if norm[i] in ",;()"]
    clauses = [bisect(breaks, f.start) for f in found]
    sentence: set[str] = set()
    own: list[set[str]] = [set() for _ in found]
    near: list[set[str]] = [set() for _ in found]
    folded = fold(norm)
    marks = [(m.start(), m.end(), m.group()[:5]) for m in _WORD.finditer(folded, a, b)]
    marks += [(m.start(), m.end(), m.group()) for m in _COMPOUND.finditer(folded, a, b)]
    for start, end, stem in marks:
        sentence.add(stem)
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
    numbers = frozenset(n for f in found for n in f.numbers)
    return [
        Context(frozenset(o), frozenset(n), frozenset(sentence), numbers)
        for o, n in zip(own, near, strict=True)
    ]


def check_bindings(text: str, field: str, facts: "Facts") -> list[Issue]:
    """Связь каждого значения текста с местом входа. Возвращает замечания к тексту."""
    norm = normalize_text(text)
    issues: dict[tuple, Issue] = {}
    everything = scan(norm)
    for a, b in sentence_spans(norm):
        found = [f for f in everything if a <= f.start < b]
        for f, ctx in zip(found, _contexts(norm, a, b, found), strict=True):
            problem = _judge(f, ctx, field, facts)
            if problem:
                issues.setdefault((problem.code, problem.detail), problem)
    return list(issues.values())


def _judge(f: Found, ctx: Context, field: str, facts: "Facts") -> Issue | None:
    """None — значение подтверждено хотя бы одним местом входа; иначе замечание с причиной."""
    candidates = [(p, t) for p in facts.places for t in p.tokens if _matches(f, t.found)]
    if not candidates:
        return _unconfirmed(f, field, facts)
    trusted = [c for c in candidates if c[0].trusted]
    if not trusted:
        return _narrative_only(f, field)
    if f.kind == "des" and any(not p.key for p, _ in trusted):
        return None  # обозначение из самого названия товара («testo 608-H1») — не характеристика
    rivals = _rivals(facts, {id(p) for p, _ in candidates})
    problems = []
    for place, token in sorted(trusted, key=lambda c: -len(ctx.sentence & c[0].marks)):
        problem = _problem(f, ctx, place, token, rivals, field)
        if problem is None:
            return None
        problems.append(problem)
    return problems[0]


def _rivals(facts: "Facts", matching: set[int]) -> list[Place]:
    """Другие характеристики со значениями: с ними можно спутать место, где значение найдено."""
    return [p for p in facts.places if p.trusted and p.key and p.tokens and id(p) not in matching]


def _narrative_only(f: Found, field: str) -> Issue:
    return _issue(
        IssueCode.NARRATIVE_ONLY_VALUE,
        field,
        f"Значение «{f.fragment}» найдено только в свободном описании товара, а не в "
        "характеристиках. Описание не считается подтверждением: сверьте значение с паспортом.",
        f.fragment,
    )


def _unconfirmed(f: Found, field: str, facts: "Facts") -> Issue | None:
    """Значения нет целиком ни в одном месте входа. Если нет и отдельных его чисел с единицей, это
    уже находят проверки чисел и единиц (None). Если они есть по отдельности, но не вместе
    («0–6» и «10–16» вместо «0–16»), подтверждения нет: диапазон могли составить из разных мест."""
    if f.kind != "num" or len(f.numbers) < 2:
        return None
    if not all((n, f.unit) in facts.pairs if f.unit else n in facts.numbers for n in f.numbers):
        return None
    return _issue(
        IssueCode.COMBINATION_UNCONFIRMED,
        field,
        f"Диапазон или перечень «{f.fragment}» не записан во входных данных целиком: отдельные "
        "значения там есть, но в разных записях. Проверьте вручную, не составлен ли он из разных "
        "диапазонов или характеристик.",
        f.fragment,
    )


def _problem(
    f: Found, ctx: Context, place: Place, token: Token, rivals: list[Place], field: str
) -> Issue | None:
    """Замечание, если значение не соответствует именно этому месту входа."""
    return (
        _neighbour(f, ctx, place, rivals, field)
        or _bound_mismatch(f, token, field)
        or _condition_dropped(f, ctx, token, field)
        or _option_as_basic(f, ctx, place, field)
    )


def _neighbour(f: Found, ctx: Context, place: Place, rivals: list[Place], field: str):
    mine = len(ctx.own & place.marks)
    best = None
    for other in rivals:
        gain = ctx.own & other.marks
        if len(gain) > mine and gain - place.marks and (best is None or len(gain) > best[0]):
            best = (len(gain), other)
    if not best:
        return None
    return _issue(
        IssueCode.VALUE_CONTEXT_MISMATCH,
        field,
        f"Значение «{f.fragment}» во входных данных относится к «{place.label}», а соседние "
        f"слова описывают «{best[1].label}». Проверьте, к какой характеристике оно относится.",
        f.fragment,
    )


def _expected_bound(src: Found, f: Found) -> str:
    """Граница, которую для значения текста задаёт запись входа; "" — сверять нечего. У одиночного
    числа это его граница, у конца диапазона «от 5 до 25»: первое число — нижняя, последнее —
    верхняя граница (сверяется, только если в тексте граница названа)."""
    if len(src.numbers) == 1:
        return src.bound
    if len(f.numbers) == 1 and f.bound:
        return {src.numbers[0]: ">=", src.numbers[-1]: "<="}.get(f.numbers[0], "")
    return ""


def _bound_mismatch(f: Found, token: Token, field: str) -> Issue | None:
    want = _expected_bound(token.found, f)
    if not want or f.bound == want:
        return None
    return _issue(
        IssueCode.BOUND_MISMATCH,
        field,
        f"Во входных данных значение «{f.fragment}» дано с границей «{_BOUND_WORDS[want]}», а в "
        "тексте граница другая или не указана: «не более» нельзя заменять на «не менее», а "
        "границу — точным значением. Проверьте вручную.",
        f.fragment,
    )


def _condition_dropped(f: Found, ctx: Context, token: Token, field: str) -> Issue | None:
    if all(part & (ctx.near | ctx.numbers) for part in token.need):
        return None
    return _issue(
        IssueCode.CONDITION_DROPPED,
        field,
        f"Во входных данных значение дано с условием «{token.qualifier}», а в тексте оно указано "
        "не полностью: значение могло быть обобщено на все варианты. Проверьте вручную.",
        f.fragment,
    )


def _option_as_basic(f: Found, ctx: Context, place: Place, field: str) -> Issue | None:
    if not (place.optional and place.key_marks) or ctx.sentence & (place.key_marks | _OPTION_STEMS):
        return None
    return _issue(
        IssueCode.CONDITION_DROPPED,
        field,
        f"Значение взято из списка вариантов или опций «{place.label}», а название списка в "
        "тексте не указано: вариант (опцию, исполнение) могли выдать за основную характеристику. "
        "Проверьте вручную.",
        f.fragment,
    )
