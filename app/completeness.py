"""Анализ полноты и согласованности входных данных (до обращения к LLM).

Принцип: об отсутствии характеристики сообщаем, только если её применимость к прибору
известна из самих данных — то есть ключ передан, но значение пусто. «Типовых» характеристик
не придумываем и обязательными не объявляем.
"""

import re
from dataclasses import dataclass

from app.facts import INJECTION, chains
from app.schemas import Issue, IssueCode, ProductInput, Severity

# между названием характеристики и числом в описании допускаем только «служебные» слова
_FILLER = re.compile(r"[\s:—–\-=,]*(?:(?:от|до|составляет|равен|равна|не более|не менее)\s+)*")
_RANGE_DASH = re.compile(r"\d+(?:[.,]\d+)?\s*[–—-]\s*\d+(?:[.,]\d+)?")
_SIGNS = set("+−-±")


@dataclass
class Completeness:
    issues: list[Issue]
    can_generate: bool


def _norm_key(key: str) -> str:
    return re.sub(r"[\s:]+", " ", key.casefold()).strip(" :")


def _issue(code, severity, field, message, detail=None) -> Issue:
    return Issue(code=code, severity=severity, field=field, message=message, detail=detail)


def analyze(product: ProductInput) -> Completeness:
    issues: list[Issue] = []
    specs = product.specs or {}
    description = (product.description or "").strip()
    usable = {k: v for k, v in specs.items() if v.strip()}

    if not (product.manufacturer or "").strip():
        issues.append(
            _issue(
                IssueCode.MISSING_OPTIONAL,
                Severity.INFO,
                "manufacturer",
                "Производитель не указан — в карточке он упоминаться не будет.",
            )
        )
    if not specs:
        issues.append(
            _issue(
                IssueCode.MISSING_OPTIONAL,
                Severity.INFO,
                "specs",
                "Характеристики не переданы — структурированный список будет пустым.",
            )
        )
    if not description:
        issues.append(
            _issue(
                IssueCode.MISSING_OPTIONAL,
                Severity.INFO,
                "description",
                "Текстовое описание (паспорт) не передано — факты берутся только из характеристик.",
            )
        )
    elif description.endswith(("...", "…")):
        issues.append(
            _issue(
                IssueCode.MANUAL_CHECK,
                Severity.INFO,
                "description",
                "Описание заканчивается многоточием — вероятно, это фрагмент паспорта; "
                "сведения за его пределами не используются.",
            )
        )

    for key, value in specs.items():
        if not value.strip():
            issues.append(
                _issue(
                    IssueCode.EMPTY_VALUE,
                    Severity.REVIEW,
                    f"specs.{key}",
                    f"Характеристика «{key}» передана без значения: данных нет. Уточните значение.",
                )
            )

    issues += _conflicts(usable, description)
    issues += _suspicious_text(product, specs)

    for extra in product.model_extra or {}:
        issues.append(
            _issue(
                IssueCode.MANUAL_CHECK,
                Severity.INFO,
                extra,
                f"Неизвестное поле «{extra}» проигнорировано: "
                "как источник фактов оно не используется.",
            )
        )

    can_generate = bool(usable) or bool(description)
    if not can_generate:
        issues.append(
            _issue(
                IssueCode.INSUFFICIENT_DATA,
                Severity.ERROR,
                None,
                "Недостаточно сведений: нет ни значений характеристик, ни описания. "
                "Текстовые блоки не сформированы, чтобы не домысливать характеристики.",
            )
        )
    return Completeness(issues=issues, can_generate=can_generate)


def _conflicts(usable: dict[str, str], description: str) -> list[Issue]:
    found: list[Issue] = []

    groups: dict[str, list[str]] = {}
    for key in usable:
        groups.setdefault(_norm_key(key), []).append(key)
    for keys in groups.values():
        if len({usable[k].strip() for k in keys}) > 1:
            shown = "; ".join(f"«{k}» = «{usable[k]}»" for k in keys)
            found.append(
                _issue(
                    IssueCode.CONFLICT,
                    Severity.REVIEW,
                    f"specs.{keys[0]}",
                    f"Одна характеристика передана с разными значениями: {shown}.",
                )
            )

    for key, value in usable.items():
        for ch in chains(value):
            # только однозначный случай: два неотрицательных числа через тире и единица
            if (
                ch.unit
                and len(ch.numbers) == 2
                and _RANGE_DASH.fullmatch(ch.raw)
                and ch.before not in _SIGNS  # set, не str: "" in "abc" == True
                and float(ch.numbers[0]) > float(ch.numbers[1])
            ):
                found.append(
                    _issue(
                        IssueCode.CONFLICT,
                        Severity.REVIEW,
                        f"specs.{key}",
                        f"Диапазон «{value}» задан в обратном порядке (начало больше конца).",
                    )
                )

    desc = re.sub(r"\s+", " ", description.casefold())
    for key, value in usable.items():
        spec_numbers = {n for ch in chains(value) for n in ch.numbers}
        key_norm = _norm_key(key)
        if not spec_numbers or len(key_norm) < 4:
            continue
        start = desc.find(key_norm)
        while start != -1:
            tail = desc[start + len(key_norm) : start + len(key_norm) + 60]
            rest = tail[_FILLER.match(tail).end() :]
            ch = next(iter(chains(rest)), None)
            if ch and rest[:1].isdigit() and not set(ch.numbers) <= spec_numbers:
                found.append(
                    _issue(
                        IssueCode.CONFLICT,
                        Severity.REVIEW,
                        f"specs.{key}",
                        f"В описании для «{key}» указано иное значение, чем в характеристиках "
                        f"(«{value}»). Проверьте, какое верно.",
                        detail=rest[: len(ch.raw) + 8].strip(),
                    )
                )
                break
            start = desc.find(key_norm, start + 1)
    return found


def _suspicious_text(product: ProductInput, specs: dict[str, str]) -> list[Issue]:
    """Фрагменты, похожие на инструкции модели. Они не выполняются — только сообщаем."""
    texts = {
        "name": product.name,
        "manufacturer": product.manufacturer,
        "description": product.description,
    }
    texts |= {f"specs.{k}": v for k, v in specs.items()}
    found = []
    for field, text in texts.items():
        m = INJECTION.search(text or "")
        if m:
            found.append(
                _issue(
                    IssueCode.MANUAL_CHECK,
                    Severity.REVIEW,
                    field,
                    "В тексте обнаружен фрагмент, похожий на инструкцию для ИИ. Он рассматривается "
                    "как данные и не выполняется; проверьте исходный текст.",
                    detail=m.group(0),
                )
            )
    return found
