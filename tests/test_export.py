"""Markdown-экспорт: структура, экранирование, явные пометки отсутствующего."""

from app.export import to_markdown
from app.schemas import (
    CardStatus,
    Commercial,
    GenerationInfo,
    Issue,
    IssueCode,
    ProductCard,
    Severity,
    SpecItem,
)


def card(**overrides) -> ProductCard:
    base = dict(
        status=CardStatus.OK,
        name="Манометр DM 02",
        manufacturer="Манотомь",
        seo_title="Манометр DM 02, 0–1,6 МПа",
        seo_description="Описание прибора.",
        specs=[
            SpecItem(name="диапазон измерений", value="0–1,6 МПа"),
            SpecItem(name="класс точности", value="1,5"),
        ],
        search_queries=["манометр DM 02", "манометр 0-1,6 МПа", "манометр G1/2"],
        commercial=Commercial(audience=["Инженеры КИП"], tasks=["Контроль давления"], scenarios=[]),
        missing_info=[],
        warnings=[],
        generation=GenerationInfo(model="m", llm_calls=1),
    )
    return ProductCard(**{**base, **overrides})


def test_sections_appear_in_a_stable_order():
    md = to_markdown(card())
    heads = [line for line in md.splitlines() if line.startswith("## ")]
    assert heads == [
        "## SEO-заголовок",
        "## SEO-описание",
        "## Характеристики",
        "## Поисковые запросы",
        "## Коммерческий блок",
        "## Недостающие сведения",
        "## Замечания проверки",
    ]
    assert md.startswith("# Манометр DM 02")


def test_character_counts_are_shown_for_title_and_description():
    md = to_markdown(card(seo_title="Манометр", seo_description="Описание."))
    assert "Длина: 8 из 70 симв." in md
    assert "Длина: 9 симв. (норма 800–1200)" in md


def test_specs_table_is_well_formed_and_verbatim():
    md = to_markdown(card())
    table = [line for line in md.splitlines() if line.startswith("|")]
    assert table[0] == "| Характеристика | Значение |" and set(table[1]) <= set("|- ")
    assert table[2:] == ["| диапазон измерений | 0–1,6 МПа |", "| класс точности | 1,5 |"]


def test_table_cells_escape_pipes_and_newlines():
    c = card(specs=[SpecItem(name="a|b", value="x | y\nz")])
    row = [line for line in to_markdown(c).splitlines() if line.startswith("| a")][0]
    assert row == r"| a\|b | x \| y z |"


def test_missing_values_are_marked_explicitly():
    c = card(specs=[SpecItem(name="питание", value="данных нет", provided=False)])
    assert "| питание | данных нет |" in to_markdown(c)


def test_withheld_blocks_are_flagged_not_silently_dropped():
    c = card(
        status=CardStatus.PARTIAL,
        seo_description=None,
        search_queries=[],
        commercial=None,
        warnings=[
            Issue(
                code=IssueCode.BLOCK_WITHHELD,
                severity=Severity.ERROR,
                field="seo_description",
                message="Блок не выдан.",
            )
        ],
    )
    md = to_markdown(c)
    assert md.count("не сформирован") >= 3
    assert "Блок не выдан." in md and "частичный результат" in md.lower()


def test_status_line_tells_the_editor_what_to_do():
    assert "ручная проверка" in to_markdown(card(status=CardStatus.NEEDS_REVIEW)).lower()
    assert "черновик" in to_markdown(card()).lower()


def test_issues_are_listed_with_severity_field_and_fragment():
    c = card(
        missing_info=[
            Issue(
                code=IssueCode.MISSING_OPTIONAL,
                severity=Severity.INFO,
                field="manufacturer",
                message="Производитель не указан.",
            )
        ],
        warnings=[
            Issue(
                code=IssueCode.UNVERIFIED_NUMBER,
                severity=Severity.ERROR,
                field="seo_title",
                message="Значение «25» отсутствует.",
                detail="25 МПа",
            )
        ],
    )
    md = to_markdown(c)
    assert "- **manufacturer** (info): Производитель не указан." in md
    assert "- **seo_title** (error): Значение «25» отсутствует. Фрагмент: «25 МПа»" in md


def test_empty_sections_say_so():
    md = to_markdown(card())
    assert md.count("—") >= 2 or md.count("нет") >= 2
    assert "Сценарии" in md
