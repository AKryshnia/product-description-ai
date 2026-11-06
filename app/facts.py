"""Числа и единицы измерения в тексте.

Общая основа анализа полноты (поиск противоречий) и валидатора результата (поиск чисел и
единиц, которых нет во входных данных). Регулярные выражения — сознательно простые; всё,
что они не могут определить уверенно, отдаётся на ручную проверку.
"""

import re
from decimal import Decimal
from typing import NamedTuple

# каноническая единица -> допустимые написания (кириллица и латиница, регистр важен)
_UNIT_VARIANTS: dict[str, list[str]] = {
    "Па": ["Па", "Pa"],
    "кПа": ["кПа", "kPa"],
    "МПа": ["МПа", "MPa"],
    "ГПа": ["ГПа", "GPa"],
    "бар": ["бар", "bar"],
    "мбар": ["мбар", "mbar"],
    "атм": ["атм", "atm"],
    "кгс/см2": ["кгс/см²", "кгс/см2"],
    "мкм": ["мкм", "µm"],
    "мм": ["мм", "mm"],
    "см": ["см", "cm"],
    "м": ["м", "m"],
    "км": ["км", "km"],
    "°C": ["°C", "°С"],  # латинская и кириллическая «С»
    "°F": ["°F"],
    "мВ": ["мВ", "mV"],
    "кВ": ["кВ", "kV"],
    "В": ["В", "V"],
    "мА": ["мА", "mA"],
    "А": ["А", "A"],
    "кВт": ["кВт", "kW"],
    "Вт": ["Вт", "W"],
    "кГц": ["кГц", "kHz"],
    "МГц": ["МГц", "MHz"],
    "Гц": ["Гц", "Hz"],
    "кОм": ["кОм", "kΩ"],
    "Ом": ["Ом", "Ω", "Ohm"],
    "кг": ["кг", "kg"],
    "г": ["г", "g"],
    "мс": ["мс", "ms"],
    "с": ["сек", "s"],  # одиночная «с» не берётся: это предлог («02 с диапазоном»)
    "мин": ["мин", "min"],
    "ч": ["ч", "h"],
    "мл": ["мл", "ml"],
    "л": ["л", "l"],
    "м3": ["м³", "м3", "m³", "m3"],
    "л/мин": ["л/мин", "l/min"],
    "м3/ч": ["м³/ч", "м3/ч", "m³/h", "m3/h"],
    "%": ["%"],
    "дБ": ["дБ", "dB"],
    "об/мин": ["об/мин", "rpm"],
}
_CANON = {v: canon for canon, variants in _UNIT_VARIANTS.items() for v in variants}
UNIT_SPELLINGS = frozenset(_CANON)  # «kPa», «mm» и т. п. — это единицы, а не посторонние слова

_LETTER = "A-Za-zА-Яа-яЁё"
_NUM = r"\d+(?:[.,]\d+)?"
# пробелы внутри цепочки — только пробел/таб: перевод строки разделяет поля и не склеивает числа
_SEP = r"[ \t]*(?:[–—−\-…]|\.\.\.|до)[ \t]*[+±−-]?[ \t]*"
_UNIT = "|".join(re.escape(v) for v in sorted(_CANON, key=len, reverse=True))
# число или цепочка чисел-диапазона («0–1,6», «от 0 до 1,6»); единица относится ко всей цепочке
_CHAIN = re.compile(
    rf"(?<!\d)(?P<chain>{_NUM}(?:{_SEP}{_NUM})*)"
    rf"(?:[ \t]*(?P<unit>{_UNIT})(?![{_LETTER}]))?"
)
# «1 200» -> «1200»; не склеиваем после «/», «.», «,» и числа с нуля («0 100 200»)
_THOUSANDS = re.compile(r"(?<![\d/.,])(?!0)\d{1,3}(?: \d{3})+(?!\d)")
# единицы из нескольких букв, которые считаем «упомянутыми» и без числа рядом
_MULTI_LETTER = "|".join(re.escape(v) for v in sorted(_CANON, key=len, reverse=True) if len(v) > 1)
_STANDALONE = re.compile(
    rf"(?<![{_LETTER}])({_MULTI_LETTER})"
    rf"(?![{_LETTER}])"
)


# Фразы, похожие на команды для ИИ. Такой текст — данные: он не выполняется и фактом не считается.
INJECTION = re.compile(
    "|".join(
        [
            r"игнорир\w*\s+(?:все\s+|любые\s+|предыдущ\w*\s+|прежн\w*\s+)*(?:инструкц|правил|указани|промпт)",
            r"забудь\s+(?:все|предыдущ|прежн|инструкц)",
            r"(?:системн\w*|скрыт\w*)\s+(?:промпт|инструкц)",
            r"system\s*prompt",
            r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)",
            r"disregard\s+(?:all\s+|the\s+)?(?:previous|above|instructions)",
            r"ты\s+(?:теперь|больше\s+не)\s",
            r"you\s+are\s+now\b",
            r"(?:раскрой|покажи|выведи)\s+(?:свои\s+|эти\s+|системн\w*\s+)?(?:инструкц|промпт)",
            # команды «добавь/укажи … в описание/карточку»: адресованы тексту, а не читателю
            r"\b(?:добавь|добавьте|укажи|укажите|впиши|впишите|включи|включите|упомяни|упомяните"
            r"|дополни|дополните|вставь|вставьте)\b[^.!?\n]{0,80}?(?:описани|карточк|заголов|характеристик|ответ\b|текст)",
            r"\b(?:add|insert|include|mention|write)\b[^.!?\n]{0,80}?"
            r"\b(?:description|card|title|text|specs?|characteristics?|answer)\b",
            r"\b(?:отвечай|ответь|верни)\s+(?:только|всегда|вместо)\b",
        ]
    ),
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"(?<=[.!?…])\s+|\n+")


def drop_instruction_sentences(text: str) -> str:
    """Убирает предложения-«инструкции», чтобы числа из них не считались подтверждёнными."""
    parts = _SENTENCES.split(text)
    kept = [x for x in parts if not INJECTION.search(x)]
    return text if len(kept) == len(parts) else " ".join(kept)


class Chain(NamedTuple):
    numbers: list[str]  # нормализованные числа цепочки
    fractional: list[bool]  # записано ли число с десятичным разделителем («1,0» — да, «1» — нет)
    unit: str | None  # каноническая единица или None
    raw: str  # исходный фрагмент цепочки (без единицы)
    fragment: str  # фрагмент вместе с единицей — для сообщений
    before: str  # символ перед цепочкой ("" в начале текста)
    start: int = 0  # позиции фрагмента (с единицей) в нормализованном тексте
    end: int = 0


def norm_number(s: str) -> str:
    """'1,50' -> '1.5', '100' -> '100', '02' -> '2': сравнение по значению, не по записи."""
    return format(Decimal(s.replace(",", ".")).normalize(), "f")


def normalize_text(text: str) -> str:
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    return _THOUSANDS.sub(lambda m: m.group(0).replace(" ", ""), text)


def chains(text: str) -> list[Chain]:
    text = normalize_text(text)
    found = []
    for m in _CHAIN.finditer(text):
        unit = _CANON[m.group("unit")] if m.group("unit") else None
        written = re.findall(_NUM, m.group("chain"))
        before = text[m.start() - 1] if m.start() else ""
        found.append(
            Chain(
                [norm_number(n) for n in written],
                [bool(re.search(r"[.,]", n)) for n in written],
                unit,
                m.group("chain"),
                m.group(0).strip(),
                before,
                m.start(),
                m.end(),
            )
        )
    return found


def conditions(text: str) -> list[tuple[tuple[str, ...], str, str]]:
    """Значения с единицей, за которыми в скобках стоит условие: «до плюс 60 °С (в металлическом
    корпусе)» -> (('60',), '°C', 'в металлическом корпусе'). Обозначения без единицы («L3 (…)»)
    не в счёт: скобки там — определение, а не условие."""
    norm = normalize_text(text)
    found = []
    for ch in chains(norm):
        tail = re.match(r"\s*\(([^()]{3,80})\)", norm[ch.end :])
        if ch.unit and tail:
            found.append((tuple(ch.numbers), ch.unit, tail.group(1).strip()))
    return found


def measurements(text: str) -> list[tuple[str, str | None]]:
    """Все числа текста с единицей (или None): [('0','МПа'), ('1.6','МПа')]."""
    return [(n, c.unit) for c in chains(text) for n in c.numbers]


def numbers(text: str) -> list[str]:
    return [n for n, _ in measurements(text)]


def standalone_units(text: str) -> set[str]:
    """Единицы из нескольких букв, упомянутые целым словом, даже без числа («давление в МПа»)."""
    return {_CANON[m.group(1)] for m in _STANDALONE.finditer(text)}
