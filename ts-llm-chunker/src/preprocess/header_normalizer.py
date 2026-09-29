"""Нормализация заголовков: разделение склеенных, удаление строки авторов после заголовка."""

from __future__ import annotations

import re

from src.common.logging_config import get_logger

log = get_logger(__name__)

# Паттерн склеенного заголовка: текст перед `# Heading` на одной строке.
# Пример: "Dongjin Song # Abstract" -> две строки.
_GLUED_HEADING = re.compile(
    r"^(.+?)\s+(#{1,6}\s+.+)$"
)

# Строка авторов — идёт сразу после заголовка первого уровня (название статьи).
# Обычно это просто имена через пробел, без цифр, без спецсимволов кроме запятых.
_AUTHOR_LINE = re.compile(
    r"^[A-Z][a-z]+(?: [A-Z][a-z]+)+(?: {2,}[A-Z][a-z]+(?: [A-Z][a-z]+)+)+$"
)


def split_glued_headings(text: str) -> str:
    """Разделяет строки вида `текст # Heading` на две отдельные строки."""
    lines = text.split("\n")
    result: list[str] = []
    fixed = 0
    for line in lines:
        m = _GLUED_HEADING.match(line)
        if m:
            before = m.group(1).strip()
            heading = m.group(2).strip()
            # Проверяем, что heading — реально заголовок, а не формула с #
            if heading.startswith("#"):
                result.append(before)
                result.append("")
                result.append(heading)
                fixed += 1
                continue
        result.append(line)
    if fixed:
        log.info("Split %d glued heading(s)", fixed)
    return "\n".join(result)


def remove_author_line(text: str) -> str:
    """Удаляет строку с перечнем авторов (идёт после заголовка `# Title`).

    Авторы уже есть в YAML frontmatter, дублировать их в теле не нужно.
    """
    lines = text.split("\n")
    result: list[str] = []
    prev_was_title = False

    for line in lines:
        stripped = line.strip()

        if prev_was_title and stripped == "":
            result.append(line)
            continue

        if prev_was_title and _AUTHOR_LINE.match(stripped):
            log.info("Removed author line: %s", stripped[:80])
            prev_was_title = False
            continue

        # Детектим заголовок первого уровня (# Title)
        prev_was_title = stripped.startswith("# ") and not stripped.startswith("## ")
        result.append(line)

    return "\n".join(result)


def normalize_headers(text: str) -> str:
    """Применяет все нормализации заголовков."""
    text = split_glued_headings(text)
    text = remove_author_line(text)
    return text
