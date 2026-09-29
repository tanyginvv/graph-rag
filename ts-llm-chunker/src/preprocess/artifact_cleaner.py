"""Чистка OCR-артефактов: image placeholders, email-адреса, аффилиации, пустые строки таблиц."""

from __future__ import annotations

import re

from src.common.logging_config import get_logger

log = get_logger(__name__)

# --- Image placeholders ---
# Паттерн: ![](_page_N_Figure_M.jpeg) или аналогичный — на отдельной строке.
_IMAGE_PLACEHOLDER = re.compile(r"^!\[.*?\]\(_page_\d+.*?\)\s*$")

# --- Email-адреса ---
_EMAIL = re.compile(r"<[^>]*@[^>]+>")

# --- Аффилиация + Correspondence блок (встроен в конец абзаца) ---
# Ловим от маркера типа "School of", "Department of", "University" до конца абзаца,
# если после этого идёт Correspondence / email.
_AFFILIATION_TAIL = re.compile(
    r"\s*(School of|Department of|Faculty of)"
    r".*?"                       # текст аффилиации (не жадный, через точки)
    r"Correspondence to:.*$",    # маркер переписки до конца СТРОКИ (не файла)
    re.MULTILINE,
)

# Менее точный fallback: хвост с "Correspondence to:"
_CORRESPONDENCE_TAIL = re.compile(
    r"\s*Correspondence to:.*$",
    re.MULTILINE,
)

# --- Пустые строки таблиц: | | | | | ---
_EMPTY_TABLE_ROW = re.compile(r"^\|(?:\s*\|)+\s*$")


def remove_image_placeholders(text: str) -> str:
    """Удаляет строки вида `![](_page_X_Figure_Y.jpeg)`."""
    lines = text.split("\n")
    result = []
    removed = 0
    for line in lines:
        if _IMAGE_PLACEHOLDER.match(line.strip()):
            removed += 1
            continue
        result.append(line)
    if removed:
        log.info("Removed %d image placeholder(s)", removed)
    return "\n".join(result)


def remove_empty_table_rows(text: str) -> str:
    """Удаляет строки таблиц, состоящие только из пустых ячеек."""
    lines = text.split("\n")
    result = []
    removed = 0
    for line in lines:
        if _EMPTY_TABLE_ROW.match(line.strip()):
            removed += 1
            continue
        result.append(line)
    if removed:
        log.info("Removed %d empty table row(s)", removed)
    return "\n".join(result)


def remove_affiliation_tails(text: str) -> str:
    """Удаляет хвосты аффилиаций + Correspondence, вклеенные OCR в конец абзаца."""
    new_text = _AFFILIATION_TAIL.sub("", text)
    if new_text == text:
        # Пробуем fallback — только Correspondence
        new_text = _CORRESPONDENCE_TAIL.sub("", text)
    if new_text != text:
        log.info("Removed affiliation/correspondence tail")
    return new_text


def remove_inline_emails(text: str) -> str:
    """Удаляет email-адреса в угловых скобках из тела текста."""
    new_text = _EMAIL.sub("", text)
    if new_text != text:
        log.info("Removed inline email addresses")
    return new_text


def clean_artifacts(
    text: str,
    *,
    remove_images: bool = True,
    remove_emails: bool = True,
) -> str:
    """Применяет все чистки артефактов последовательно."""
    if remove_images:
        text = remove_image_placeholders(text)
    text = remove_empty_table_rows(text)
    text = remove_affiliation_tails(text)
    if remove_emails:
        text = remove_inline_emails(text)
    return text
