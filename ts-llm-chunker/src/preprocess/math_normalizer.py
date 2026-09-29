"""Сборка разорванных display-формул и склейка разорванных параграфов."""

from __future__ import annotations

import re

from src.common.logging_config import get_logger

log = get_logger(__name__)

# Display-формула: строка заканчивается на $$  (может начинаться с текста)
_LINE_ENDS_WITH_DISPLAY = re.compile(r"\$\$\s*$")

# Начало display-формулы: строка начинается с $$=  (продолжение предыдущей формулы)
_DISPLAY_CONTINUATION = re.compile(r"^\$\$\s*=")

# Standalone equation number: строка вида (3) или (A.1)
_STANDALONE_EQ_NUM = re.compile(r"^\((\d+|[A-Z]\.\d+)\)\s*$")


def merge_broken_display_math(text: str) -> str:
    """Объединяет разорванные display-формулы.

    Паттерн: $$expr1$$\\n\\n$$= expr2 (N)$$
    Результат: $$expr1 = expr2 \\qquad (N)$$

    Также обрабатывает standalone номер формулы:
    $$expr$$\\n(3)\\n -> $$expr \\tag{3}$$
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    merged = 0

    while i < len(lines):
        line = lines[i]

        # Проверяем: текущая строка заканчивается на $$, а через пустую строку идёт continuation
        if _LINE_ENDS_WITH_DISPLAY.search(line):
            # Собираем все пустые строки после текущей
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1

            # Следующая непустая — continuation $$= ... $$?
            if j < len(lines) and _DISPLAY_CONTINUATION.match(lines[j].strip()):
                # Склеиваем: убираем $$ на конце первой и $$ на начале второй
                first = line.strip().rstrip("$").rstrip()
                second = lines[j].strip().lstrip("$").lstrip()
                merged_line = f"$${first} {second}$$"
                result.append(merged_line)
                i = j + 1
                merged += 1
                continue

            # Следующая непустая — standalone номер формулы (N)?
            if j < len(lines):
                eq_m = _STANDALONE_EQ_NUM.match(lines[j].strip())
                if eq_m:
                    num = eq_m.group(1)
                    formula = line.strip().rstrip("$").rstrip()
                    merged_line = f"$${formula} \\tag{{{num}}}$$"
                    result.append(merged_line)
                    i = j + 1
                    merged += 1
                    continue

        result.append(line)
        i += 1

    if merged:
        log.info("Merged %d broken display formula(s)", merged)
    return "\n".join(result)


def join_broken_paragraphs(text: str) -> str:
    """Склеивает ложные разрывы параграфов.

    Условия для склейки:
    - Предыдущая строка НЕ заканчивается завершающей пунктуацией (.?!:)
    - Следующая строка начинается с маленькой буквы
    Это означает, что предложение было разорвано OCR посередине.
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    fixed = 0

    # Пунктуация, завершающая предложение (после неё разрыв — нормальный)
    _SENTENCE_END = re.compile(r"[.?!:;)\]]\s*$")

    while i < len(lines):
        line = lines[i]

        if (
            i + 2 < len(lines)
            and line.strip()
            and not line.strip().startswith("#")
            and not line.strip().startswith("|")
            and not line.strip().startswith("$$")
            and not line.strip().endswith("$$")
            and not _SENTENCE_END.search(line.strip())  # предложение НЕ завершено
            and lines[i + 1].strip() == ""
            and lines[i + 2].strip()
            and lines[i + 2].strip()[0].islower()
            and not lines[i + 2].strip().startswith("*")
        ):
            result.append(line.rstrip() + " " + lines[i + 2].strip())
            i += 3
            fixed += 1
            continue

        result.append(line)
        i += 1

    if fixed:
        log.info("Joined %d broken paragraph(s)", fixed)
    return "\n".join(result)


def normalize_math(text: str) -> str:
    """Применяет все нормализации формул и параграфов."""
    text = merge_broken_display_math(text)
    text = join_broken_paragraphs(text)
    return text
