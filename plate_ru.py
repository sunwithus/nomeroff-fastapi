# -*- coding: utf-8 -*-
"""
Формат российского номера: маска, валидация кода региона и декодирование по маске.

Отделено от main.py, чтобы одни и те же правила использовали и OCR-эндпоинты,
и инструменты (стенд, экспорт датасета).
"""
from __future__ import annotations

import re

# Буквы, которые физически бывают на номерах РФ (совпадают по виду с латиницей).
RU_LETTERS = "АВЕКМНОРСТУХ"

_LATIN_TO_CYRILLIC = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
})

RU_CIVILIAN = re.compile(rf"^[{RU_LETTERS}]\d{{3}}[{RU_LETTERS}]{{2}}\d{{2,3}}$")
RU_MILITARY = re.compile(rf"^\d{{4}}[{RU_LETTERS}]{{2}}\d{{2,3}}$")

# Маски: 'L' — буква из RU_LETTERS, 'D' — цифра, '?' — необязательная цифра региона.
MASK_CIVILIAN = "LDDDLLDD?"
MASK_MILITARY = "DDDDLLDD?"

# Коды регионов РФ. Двузначные 01..99 плюс реально выданные трёхзначные серии.
# Список сознательно закрытый: он и отсекает фантомы вида ...356 / ...107 / ...135.
_REGION_2 = {f"{i:02d}" for i in range(1, 100)}
_REGION_3 = {
    "102", "103", "104", "105", "106", "109", "111", "113", "116", "118",
    "121", "123", "124", "125", "126", "128", "130", "134", "136", "138",
    "142", "150", "152", "154", "156", "158", "159", "161", "163", "164",
    "173", "174", "177", "178", "186", "190", "196", "197", "199",
    "702", "716", "725", "750", "754", "763", "777", "790", "793", "797",
    "799", "977",
}
VALID_REGIONS = _REGION_2 | _REGION_3

# Пары глифов, которые CTC-голова путает чаще всего. Используются, когда
# декодирование по маске должно заменить символ неподходящего класса.
DIGIT_FOR_LETTER = {
    "О": "0", "О".lower(): "0",
    "В": "8", "З": "3", "Е": "6", "Т": "7", "А": "4",
    "С": "5", "Н": "4", "У": "4", "Х": "8", "К": "8", "М": "4", "Р": "9",
}
LETTER_FOR_DIGIT = {
    "0": "О", "8": "В", "3": "З", "6": "Е", "7": "Т", "4": "А",
    "5": "С", "1": "Т", "2": "Т", "9": "Р",
}


def normalize_plate(text: str) -> str:
    """Убрать разделители, привести к верхнему регистру, латиницу — в кириллицу."""
    if not text:
        return ""
    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]", "", text)
    return cleaned.upper().translate(_LATIN_TO_CYRILLIC)


def is_civilian(plate: str) -> bool:
    return bool(RU_CIVILIAN.match(plate or ""))


def is_military(plate: str) -> bool:
    return bool(RU_MILITARY.match(plate or ""))


def region_of(plate: str) -> str:
    """Код региона (2–3 цифры в конце) или '' если формат не распознан."""
    if is_civilian(plate):
        return plate[6:]
    if is_military(plate):
        return plate[6:]
    return ""


def region_is_valid(plate: str) -> bool:
    region = region_of(plate)
    return bool(region) and region in VALID_REGIONS


def looks_like_ru_plate(text: str, *, check_region: bool = False) -> bool:
    """Валидный формат РФ. check_region=True дополнительно сверяет код региона."""
    plate = normalize_plate(text)
    if not (is_civilian(plate) or is_military(plate)):
        return False
    return region_is_valid(plate) if check_region else True


def _fit_mask(plate: str, mask: str) -> str | None:
    """
    Подогнать текст под маску, заменяя символы неверного класса на самый похожий
    глиф нужного класса (О->0, 8->В и т.д.). Возвращает None, если длина не подходит
    или подмена невозможна.
    """
    required = mask.replace("?", "")
    optional = mask.count("?")
    if not (len(required) <= len(plate) <= len(required) + optional):
        return None

    out = []
    for i, ch in enumerate(plate):
        kind = mask[i] if i < len(mask) else "D"
        if kind == "?":
            kind = "D"
        if kind == "D":
            if ch.isdigit():
                out.append(ch)
                continue
            swap = DIGIT_FOR_LETTER.get(ch)
            if swap is None:
                return None
            out.append(swap)
        else:  # 'L'
            if ch in RU_LETTERS:
                out.append(ch)
                continue
            swap = LETTER_FOR_DIGIT.get(ch)
            if swap is None or swap not in RU_LETTERS:
                return None
            out.append(swap)
    return "".join(out)


# Буквы, в которые CTC-голова превращает первую цифру военного номера на
# негативе. Пары по форме глифа, а не по частоте в выборке.
_LETTER_TO_DIGIT_LOOKALIKE = {
    "О": "0", "В": "8", "Е": "6", "Т": "7",
    "А": "4", "С": "5", "У": "9", "Р": "9",
}

# Серии военных номеров — сдвоенные буквы; без этого признака подмена
# первой цифры ломала бы обычные гражданские номера.
_MILITARY_SERIES = {"СС", "ВВ", "КК", "ММ", "ТТ", "НН", "ЕЕ", "АА"}


def try_military_from_civilian_lookalike(text: str) -> str | None:
    """
    На негативе военный 9036СС45 читается как гражданский У036СС45: первая
    цифра принята за похожую букву. Возвращаем подмену только если глиф
    действительно похож и серия сдвоенная — иначе молча портим номер.
    """
    plate = normalize_plate(text)
    if not is_civilian(plate) or len(plate) < 8:
        return None
    if plate[4:6] not in _MILITARY_SERIES:
        return None
    digit = _LETTER_TO_DIGIT_LOOKALIKE.get(plate[0])
    if digit is None:
        return None
    candidate = digit + plate[1:]
    return candidate if is_military(candidate) and region_is_valid(candidate) else None


def decode_constrained(text: str, *, require_region: bool = True) -> tuple[str, str] | None:
    """
    Декодировать чтение OCR по маскам формата РФ.

    Возвращает (plate, kind) где kind = 'civilian' | 'military', либо None,
    если ни одна маска не подходит (или код региона не существует).
    Порядок предпочтения: точное совпадение маски > подгонка глифов.
    """
    plate = normalize_plate(text)
    if not plate:
        return None

    candidates: list[tuple[int, str, str]] = []
    for mask, kind, exact in (
        (MASK_CIVILIAN, "civilian", is_civilian(plate)),
        (MASK_MILITARY, "military", is_military(plate)),
    ):
        if exact:
            candidates.append((0, plate, kind))
            continue
        fitted = _fit_mask(plate, mask)
        if fitted:
            edits = sum(1 for a, b in zip(plate, fitted) if a != b)
            candidates.append((edits, fitted, kind))

    # OCR иногда клеит лишнюю букву перед военным номером: Е9036СС45 -> 9036СС45
    if len(plate) in (9, 10) and plate[0] in RU_LETTERS:
        tail = plate[1:]
        if is_military(tail):
            candidates.append((1, tail, "military"))

    candidates.sort(key=lambda c: c[0])
    for _, cand, kind in candidates:
        if not require_region or region_is_valid(cand):
            return cand, kind
    return None


def vote_plate(readings: list[tuple[str, list[float]]]) -> tuple[str, float, int] | None:
    """
    Голосование по позициям символов между несколькими чтениями одного номера.

    Args:
        readings: список (текст, вероятности по символам). Пустые вероятности
                  трактуются как вес 0.5 (нейтральный голос).

    Returns:
        (plate, confidence, votes) — победивший номер, средняя уверенность
        победивших глифов и число чтений, участвовавших в голосовании.
        None, если чтений нет.
    """
    usable = [(normalize_plate(t), p) for t, p in readings if normalize_plate(t)]
    if not usable:
        return None

    # Голосуем внутри самой популярной длины: смешивать А123ВС45 и А123ВС125 нельзя.
    by_len: dict[int, list[tuple[str, list[float]]]] = {}
    for text, probs in usable:
        by_len.setdefault(len(text), []).append((text, probs))
    # предпочитаем более длинный вариант при равном числе голосов (полный регион)
    best_len = max(by_len, key=lambda k: (len(by_len[k]), k))
    group = by_len[best_len]

    plate_chars: list[str] = []
    plate_confs: list[float] = []
    for pos in range(best_len):
        weights: dict[str, float] = {}
        for text, probs in group:
            ch = text[pos]
            w = probs[pos] if pos < len(probs) else 0.5
            weights[ch] = weights.get(ch, 0.0) + float(w)
        winner = max(weights, key=lambda c: (weights[c], c))
        plate_chars.append(winner)
        plate_confs.append(weights[winner] / sum(weights.values()))

    voted = "".join(plate_chars)
    conf = sum(plate_confs) / len(plate_confs) if plate_confs else 0.0
    return voted, conf, len(group)
