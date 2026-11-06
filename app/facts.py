"""Числа и единицы измерения в тексте.

Общая основа анализа полноты (поиск противоречий) и валидатора результата (поиск чисел и
единиц, которых нет во входных данных). Регулярные выражения — сознательно простые; всё,
что они не могут определить уверенно, отдаётся на ручную проверку.

Значение — это число СО ЗНАКОМ: «−40» и «+40» разные значения. Знак читается и из символов
(«−», «-», «+», «±»), и из слов («минус 50», «плюс 60»). Единицы, записанные словами после числа
(«градусов Цельсия»), и числительные словами («три батареи») приводятся к тому же виду, чтобы
их нельзя было использовать в обход проверки.
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

# единицы словами (только после числа): «0,5 градусов Фаренгейта» -> «0,5 °F»
_WORD_UNITS = {
    "°C": r"градус\w*[ \t]+цельси\w*",
    "°F": r"градус\w*[ \t]+фаренгейт\w*",
    "МПа": r"мегапаскал\w*",
    "кПа": r"килопаскал\w*",
    "Па": r"паскал\w*",
    "мбар": r"миллибар\w*",
    "мм": r"миллиметр\w*",
    "см": r"сантиметр\w*",
    "м": r"метр(?:а|ов|ы|у|ам|ами|ах)?",
    "кг": r"килограмм\w*",
    "г": r"грамм(?:а|ов|ы|у|ам)?",
    "мВ": r"милливольт\w*",
    "кВ": r"киловольт\w*",
    "В": r"вольт(?:а|ов|ы|у|ам)?",
    "мА": r"миллиампер\w*",
    "А": r"ампер(?:а|ов|ы|у|ам)?",
    "кВт": r"киловатт\w*",
    "Вт": r"ватт(?:а|ов|ы|у|ам)?",
    "кГц": r"килогерц\w*",
    "МГц": r"мегагерц\w*",
    "Гц": r"герц(?:а|ев|ы|у|ам)?",
    "кОм": r"килоом\w*",
    "Ом": r"ом(?:а|ов|ы|у|ам)?",
    "%": r"процент\w*",
    "мс": r"миллисекунд\w*",
    "с": r"секунд\w*",
    "мин": r"минут\w*",
    "ч": r"час(?:а|ов|ы|у|ам)?",
    "мл": r"миллилитр\w*",
    "л": r"литр\w*",
}
_WORD_UNIT_RES = [(canon, re.compile(rx, re.I)) for canon, rx in _WORD_UNITS.items()]


def _canon_unit(text: str) -> str:
    return _CANON.get(text) or next(c for c, rx in _WORD_UNIT_RES if rx.fullmatch(text))


# числительные словами (все падежи): значение -> шаблон. «Один» не берём: чаще это не количество.
_NUMERALS = {
    2: r"дв(?:а|е|ух|ум|умя)",
    3: r"тр(?:и|[её]х|ем|емя)",
    4: r"четыр(?:е|[её]х|ем|ьмя)",
    8: r"восем(?:ь|и|ью)|восьм(?:и|ью)",
    40: r"сорок(?:а|у)?",
    100: r"сто|ста",
    **{
        value: rf"{stem}(?:ь|и|ью)"
        for value, stem in {
            5: "пят",
            6: "шест",
            7: "сем",
            9: "девят",
            10: "десят",
            11: "одиннадцат",
            12: "двенадцат",
            13: "тринадцат",
            14: "четырнадцат",
            15: "пятнадцат",
            16: "шестнадцат",
            17: "семнадцат",
            18: "восемнадцат",
            19: "девятнадцат",
            20: "двадцат",
            30: "тридцат",
        }.items()
    },
}
_NUMERAL_RES = [(v, re.compile(rx, re.I)) for v, rx in _NUMERALS.items()]
_WORD_NUMBER = "|".join(f"(?:{rx})" for rx in _NUMERALS.values())

_LETTER = "A-Za-zА-Яа-яЁё"
_NUM = r"\d+(?:[.,]\d+)?"
_NUMBER = rf"(?:(?i:(?<!\w)(?:{_WORD_NUMBER})(?!\w))|{_NUM})"
# знак: символ или слово; минус внутри слова («ИФН-300», «Ni-MH») знаком не считается
_SIGN = r"(?:(?i:минус|плюс)[ \t]+|[+±−-][ \t]?)"
# пробелы внутри цепочки — только пробел/таб: перевод строки разделяет поля и не склеивает числа
_SEP = r"[ \t]*(?:[–—−\-…]|\.\.\.|до)[ \t]*"
_UNIT = "|".join(
    [rf"(?i:{rx})" for rx in _WORD_UNITS.values()]
    + [re.escape(v) for v in sorted(_CANON, key=len, reverse=True)]
)
# число или цепочка чисел-диапазона («0–1,6», «от минус 50 до плюс 60»); единица относится ко
# всей цепочке
_CHAIN = re.compile(
    rf"(?<!\d)(?P<chain>(?:(?<!\w){_SIGN})?{_NUMBER}(?:{_SEP}(?:{_SIGN})?{_NUMBER})*)"
    rf"(?:[ \t]*(?P<unit>{_UNIT})(?![{_LETTER}]))?"
)
_FIRST = re.compile(rf"(?P<sign>{_SIGN})?(?P<num>{_NUMBER})")
_NEXT = re.compile(rf"{_SEP}(?P<sign>{_SIGN})?(?P<num>{_NUMBER})")
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
# Общий признак: предложение адресовано не читателю, а результату или редактору («итоговая
# карточка должна…», «редактору: …»). Паспорт прибора так не пишет. Список формулировок команд
# неполон по природе, поэтому защита этим не ограничена: значения, найденные только в свободном
# описании, в любом случае не считаются подтверждёнными (см. app/binding.py).
_ADDRESSED_TO_OUTPUT = re.compile(
    r"(?:итогов|финальн|выходн|готов|публикуем|сгенерирован)\w*\s+"
    r"(?:карточк|текст|описани|ответ|заголов|результат)\w*"
    r"|\bредактор\w*|\b(?:json|seo|сео)\b|\bответ\w*\s+модел\w*|\b(?:промпт|нейросет)\w*",
    re.IGNORECASE,
)
_OBLIGATION = re.compile(
    r"\b(?:долж\w*|необходимо|нужно|следует|обязательно|требуется|надо|нельзя"
    r"|замени\w*|заменить|укажи\w*|указать|сообщ\w*|включи\w*|включить|добав\w*|вставь\w*"
    r"|вписать)\b",
    re.IGNORECASE,
)
# конец предложения; следующее начинается с заглавной, цифры или кавычки («отн. влажности» —
# не конец)
_SENTENCES = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ«\"\d])|\n+")


def fold(text: str) -> str:
    """Регистр и «ё» не различаются; длина строки не меняется."""
    return text.casefold().replace("ё", "е")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENTENCES.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def find_instruction(text: str) -> str | None:
    """Первый фрагмент, похожий на команду для ИИ (или None)."""
    for a, b in sentence_spans(text):
        sentence = text[a:b]
        m = INJECTION.search(sentence)
        if m:
            return m.group(0)
        if _ADDRESSED_TO_OUTPUT.search(sentence) and _OBLIGATION.search(sentence):
            return sentence.strip()[:120]
    return None


def is_instruction(sentence: str) -> bool:
    return find_instruction(sentence) is not None


def drop_instruction_sentences(text: str) -> str:
    """Убирает предложения-«инструкции», чтобы числа из них не считались подтверждёнными."""
    spans = sentence_spans(text)
    kept = [text[a:b] for a, b in spans if find_instruction(text[a:b]) is None]
    return text if len(kept) == len(spans) else " ".join(kept)


class Chain(NamedTuple):
    numbers: list[str]  # нормализованные числа цепочки, со знаком: «-50», «60»
    fractional: list[bool]  # записано ли число с десятичным разделителем («1,0» — да, «1» — нет)
    unit: str | None  # каноническая единица или None
    raw: str  # исходный фрагмент цепочки (без единицы)
    fragment: str  # фрагмент вместе с единицей — для сообщений
    before: str  # символ перед цепочкой ("" в начале текста)
    start: int = 0  # позиции фрагмента (с единицей) в нормализованном тексте
    end: int = 0
    spelled: bool = False  # хотя бы одно число записано словом («три»)


def norm_number(s: str) -> str:
    """'1,50' -> '1.5', '100' -> '100', '02' -> '2': сравнение по значению, не по записи."""
    return format(Decimal(s.replace(",", ".")).normalize(), "f")


def _value(sign: str | None, token: str) -> tuple[str, bool, bool]:
    """(число со знаком, десятичная ли запись, записано ли словом)."""
    spelled = not token[0].isdigit()
    if spelled:
        value = str(next(v for v, rx in _NUMERAL_RES if rx.fullmatch(token)))
    else:
        value = norm_number(token)
    negative = bool(sign) and sign.strip().casefold() in {"-", "−", "минус"}
    return (
        ("-" + value if negative and value != "0" else value),
        bool(re.search(r"[.,]", token)),
        spelled,
    )


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    return _THOUSANDS.sub(lambda m: m.group(0).replace(" ", ""), text)


def chains(text: str) -> list[Chain]:
    text = normalize_text(text)
    found = []
    for m in _CHAIN.finditer(text):
        raw = m.group("chain")
        parsed = [_FIRST.match(raw)]
        while parsed[-1].end() < len(raw):
            parsed.append(_NEXT.match(raw, parsed[-1].end()))
        values = [_value(p.group("sign"), p.group("num")) for p in parsed]
        found.append(
            Chain(
                [v[0] for v in values],
                [v[1] for v in values],
                _canon_unit(m.group("unit")) if m.group("unit") else None,
                raw,
                m.group(0).strip(),
                text[m.start() - 1] if m.start() else "",
                m.start(),
                m.end(),
                any(v[2] for v in values),
            )
        )
    return found


def measurements(text: str) -> list[tuple[str, str | None]]:
    """Все числа текста с единицей (или None): [('0','МПа'), ('1.6','МПа')]."""
    return [(n, c.unit) for c in chains(text) for n in c.numbers]


def numbers(text: str) -> list[str]:
    return [n for n, _ in measurements(text)]


def standalone_units(text: str) -> set[str]:
    """Единицы из нескольких букв, упомянутые целым словом, даже без числа («давление в МПа»)."""
    return {_CANON[m.group(1)] for m in _STANDALONE.finditer(text)}
