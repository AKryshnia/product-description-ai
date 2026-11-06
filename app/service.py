"""Оркестрация: анализ полноты -> генерация -> проверка -> (одна) корректировка -> карточка.

Решения, которые принимаются здесь, а не в промпте:
 * технические характеристики копируются из входных данных, ответ модели на них не влияет;
 * при недостатке сведений (нет ни значений, ни описания) модель не вызывается;
 * корректирующая генерация — не больше одной;
 * блок с неподтверждённым техническим значением не выдаётся, остальное сохраняется.
"""

import logging
from dataclasses import dataclass

from app.completeness import analyze
from app.export import to_markdown
from app.generator import (
    DraftError,
    build_messages,
    draft_json_schema,
    parse_draft,
    repair_messages,
)
from app.llm import LLMClient, LLMError, LLMResult
from app.schemas import (
    NO_DATA,
    CardStatus,
    Draft,
    GenerationInfo,
    Issue,
    IssueCode,
    ProductCard,
    ProductInput,
    Severity,
    SpecItem,
)
from app.validator import DESCRIPTION_MAX, DESCRIPTION_MIN, Facts, build_facts, validate_draft

log = logging.getLogger(__name__)

# нарушения такого рода — недостоверные технические сведения: блок с ними не выдаётся
_FACT_ERRORS = {IssueCode.UNVERIFIED_NUMBER, IssueCode.UNVERIFIED_UNIT}


@dataclass
class _Attempt:
    result: LLMResult
    draft: Draft | None  # None — ответ не удалось разобрать
    issues: list[Issue]

    @property
    def score(self) -> tuple[int, int, int, int]:
        """Чем меньше, тем лучше: (нет черновика, нарушений, пунктов на проверку, отклонение
        длины описания от допустимого диапазона)."""
        errors = sum(i.severity is Severity.ERROR for i in self.issues)
        reviews = sum(i.severity is Severity.REVIEW for i in self.issues)
        n = len(self.draft.seo_description) if self.draft else 0
        gap = max(0, DESCRIPTION_MIN - n, n - DESCRIPTION_MAX)
        return (self.draft is None, errors, reviews, gap)


def _attempt(llm: LLMClient, messages: list, facts: Facts) -> _Attempt:
    schema = draft_json_schema() if llm.capabilities.json_schema else None
    result = llm.complete(messages, json_schema=schema)
    try:
        draft = parse_draft(result.text)
    except DraftError as e:
        issue = Issue(
            code=IssueCode.LLM_OUTPUT_INVALID,
            severity=Severity.ERROR,
            message=f"Ответ модели не соответствует требуемой схеме: {e}.",
        )
        return _Attempt(result, None, [issue])
    return _Attempt(result, draft, validate_draft(draft, facts))


def _sum(values: list[int | None]) -> int | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def generate_card(product: ProductInput, llm: LLMClient) -> ProductCard:
    completeness = analyze(product)
    specs = [
        SpecItem(name=k, value=v if v.strip() else NO_DATA, provided=bool(v.strip()))
        for k, v in (product.specs or {}).items()
    ]
    shared = {
        "name": product.name,
        "manufacturer": (product.manufacturer or "").strip() or None,
        "specs": specs,
        "missing_info": completeness.issues,
    }

    if not completeness.can_generate:  # нечего проверять и не из чего писать: LLM не зовём
        return _finish(
            ProductCard(
                status=CardStatus.PARTIAL,
                seo_title=None,
                seo_description=None,
                search_queries=[],
                commercial=None,
                warnings=[],
                generation=GenerationInfo(),
                **shared,
            )
        )

    facts = build_facts(product)
    messages = build_messages(product)
    first = _attempt(llm, messages, facts)
    final, attempts, extra = first, [first], []

    if first.issues:  # одна, и только одна, корректирующая генерация
        try:
            second = _attempt(
                llm, repair_messages(messages, first.result.text, first.issues), facts
            )
        except LLMError as e:
            if first.draft is None:
                raise  # исправлять нечего и показать нечего
            extra.append(
                Issue(
                    code=IssueCode.REPAIR_FAILED,
                    severity=Severity.REVIEW,
                    message=(
                        f"Корректирующая генерация не удалась ({e.message}); "
                        "показан первый вариант."
                    ),
                )
            )
        else:
            attempts.append(second)
            if second.score < first.score:
                final = second

    draft, issues = final.draft, [*final.issues, *extra]
    withheld = {
        i.field
        for i in final.issues
        if i.severity is Severity.ERROR and i.code in _FACT_ERRORS and i.field
    }
    for block in sorted(withheld):
        issues.append(
            Issue(
                code=IssueCode.BLOCK_WITHHELD,
                severity=Severity.ERROR,
                field=block,
                message=(
                    f"Блок «{block}» не выдан: в нём найдены неподтверждённые технические значения."
                ),
            )
        )

    if draft is None or withheld or any(i.severity is Severity.ERROR for i in issues):
        status = CardStatus.PARTIAL
    elif any(i.severity is Severity.REVIEW for i in [*issues, *completeness.issues]):
        status = CardStatus.NEEDS_REVIEW
    else:
        status = CardStatus.OK

    card = ProductCard(
        status=status,
        seo_title=None if draft is None or "seo_title" in withheld else draft.seo_title,
        seo_description=(
            None if draft is None or "seo_description" in withheld else draft.seo_description
        ),
        search_queries=[]
        if draft is None or "search_queries" in withheld
        else draft.search_queries,
        commercial=None if draft is None or "commercial" in withheld else draft.commercial,
        warnings=issues,
        generation=GenerationInfo(
            model=final.result.model,
            llm_calls=len(attempts),
            repaired=final is not first,
            prompt_tokens=_sum([a.result.prompt_tokens for a in attempts]),
            completion_tokens=_sum([a.result.completion_tokens for a in attempts]),
        ),
        **shared,
    )
    log.info(
        "card status=%s calls=%d repaired=%s", card.status.value, len(attempts), final is not first
    )
    return _finish(card)


def _finish(card: ProductCard) -> ProductCard:
    card.markdown = to_markdown(card)
    return card
