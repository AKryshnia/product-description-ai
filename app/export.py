"""Экспорт карточки в Markdown для копирования контент-менеджером."""

from app.schemas import CardStatus, Issue, ProductCard
from app.validator import DESCRIPTION_MAX, DESCRIPTION_MIN, TITLE_MAX

_STATUS_TEXT = {
    CardStatus.OK: "черновик готов к редактированию",
    CardStatus.NEEDS_REVIEW: "черновик, требуется ручная проверка (см. «Замечания» ниже)",
    CardStatus.PARTIAL: (
        "частичный результат: часть блоков не сформирована или нарушает ограничения "
        "(см. «Замечания» ниже)"
    ),
}
_NOT_BUILT = "_Блок не сформирован._"


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", r"\|")


def _bullets(items: list[str], empty: str) -> list[str]:
    return [f"- {i}" for i in items] or [empty]


def _issues(issues: list[Issue], empty: str) -> list[str]:
    lines = []
    for i in issues:
        line = f"- **{i.field or 'общее'}** ({i.severity.value}): {i.message}"
        if i.detail:
            line += f" Фрагмент: «{i.detail}»"
        lines.append(line)
    return lines or [empty]


def to_markdown(card: ProductCard) -> str:
    out = [
        f"# {card.name}",
        "",
        f"**Производитель:** {card.manufacturer or 'данных нет'}  ",
        f"**Статус:** {_STATUS_TEXT[card.status]}",
        "",
        "## SEO-заголовок",
    ]
    if card.seo_title is None:
        out.append(_NOT_BUILT)
    else:
        out += [card.seo_title, "", f"_Длина: {len(card.seo_title)} из {TITLE_MAX} симв._"]

    out += ["", "## SEO-описание"]
    if card.seo_description is None:
        out.append(_NOT_BUILT)
    else:
        n = len(card.seo_description)
        out += [
            card.seo_description,
            "",
            f"_Длина: {n} симв. (норма {DESCRIPTION_MIN}–{DESCRIPTION_MAX})_",
        ]

    out += ["", "## Характеристики"]
    if card.specs:
        out += ["| Характеристика | Значение |", "| --- | --- |"]
        out += [f"| {_cell(s.name)} | {_cell(s.value)} |" for s in card.specs]
    else:
        out.append("Характеристики не переданы.")

    out += ["", "## Поисковые запросы"]
    out += _bullets(card.search_queries, _NOT_BUILT)

    out += ["", "## Коммерческий блок"]
    if card.commercial is None:
        out.append(_NOT_BUILT)
    else:
        c = card.commercial
        out += ["**Кому подходит**", *_bullets(c.audience, "- —"), ""]
        out += ["**Какие задачи решает**", *_bullets(c.tasks, "- —"), ""]
        out += [
            "**Сценарии применения**",
            *_bullets(c.scenarios, "- — (в исходных данных не подтверждены)"),
        ]

    out += ["", "## Недостающие сведения", *_issues(card.missing_info, "Недостающих сведений нет.")]
    out += ["", "## Замечания проверки", *_issues(card.warnings, "Замечаний нет.")]
    return "\n".join(out) + "\n"
