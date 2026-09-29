"""Удаление секций References и Impact Statement из MD."""

from __future__ import annotations

import re

from src.common.logging_config import get_logger

log = get_logger(__name__)

# Паттерны заголовков секций, которые нужно удалить (case-insensitive).
_REMOVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^#{1,3}\s+References\s*$", re.IGNORECASE),
    re.compile(r"^#{1,3}\s+Impact\s+Statement\s*$", re.IGNORECASE),
]


def _is_heading(line: str) -> bool:
    """Строка начинается с `#` (MD-заголовок)."""
    return bool(line.lstrip().startswith("#"))


def _heading_level(line: str) -> int:
    """Возвращает уровень MD-заголовка (1..6) или 0 если не заголовок."""
    m = re.match(r"^(#{1,6})\s", line)
    return len(m.group(1)) if m else 0


def _should_remove(line: str, remove_references: bool, remove_impact: bool) -> bool:
    """Проверяет, является ли строка заголовком удаляемой секции."""
    stripped = line.strip()
    if remove_references and _REMOVE_PATTERNS[0].match(stripped):
        return True
    if remove_impact and _REMOVE_PATTERNS[1].match(stripped):
        return True
    return False


def remove_sections(
    text: str,
    *,
    remove_references: bool = True,
    remove_impact_statement: bool = True,
) -> str:
    """Удаляет секции References и/или Impact Statement целиком.

    Удаляется всё от заголовка удаляемой секции до следующего заголовка
    того же или более высокого уровня (включительно заголовок, не включая следующий).
    """
    lines = text.split("\n")
    result: list[str] = []
    skip = False
    skip_level = 0

    for line in lines:
        if skip:
            # Встретили заголовок того же или более высокого уровня — прекращаем пропуск
            lvl = _heading_level(line)
            if lvl > 0 and lvl <= skip_level:
                skip = False
                # Проверяем — может, этот заголовок тоже надо удалить
                if _should_remove(line, remove_references, remove_impact_statement):
                    skip = True
                    skip_level = lvl
                    log.info("Removing section: %s", line.strip())
                    continue
                result.append(line)
            # иначе пропускаем строку (внутри удаляемой секции)
            continue

        # Проверяем, не начало ли удаляемой секции
        if _is_heading(line) and _should_remove(
            line, remove_references, remove_impact_statement
        ):
            lvl = _heading_level(line)
            skip = True
            skip_level = lvl
            log.info("Removing section: %s", line.strip())
            continue

        result.append(line)

    return "\n".join(result)
