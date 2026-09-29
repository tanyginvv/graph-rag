"""Парсинг блочной структуры нормализованного MD.

Превращает плоский MD-текст в список типизированных блоков (MDBlock):
heading, paragraph, table, display_math, figure_caption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.common.logging_config import get_logger

log = get_logger(__name__)

BlockType = Literal["heading", "paragraph", "table", "display_math", "figure_caption"]


@dataclass
class MDBlock:
    """Один логический блок MD-документа."""

    type: BlockType
    content: str
    heading_level: int | None = None
    heading_number: str | None = None  # "3.2", "A.1", None для ненумерованных
    start_line: int = 0
    end_line: int = 0


# --- Регулярные выражения ---

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_HEADING_NUMBER = re.compile(
    r"^(\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)*\.?)\s+"
)
_FIGURE_CAPTION = re.compile(r"^Figure\s+\d+", re.IGNORECASE)
_TABLE_CAPTION = re.compile(r"^Table\s+\d+", re.IGNORECASE)
_TABLE_ROW = re.compile(r"^\|")
_TABLE_SEPARATOR = re.compile(r"^\|[\s\-:|]+\|")
_DISPLAY_MATH_START = re.compile(r"^\$\$")


def _extract_heading_number(text: str) -> str | None:
    """Извлекает номер заголовка: '3.2. Time Series' -> '3.2'."""
    m = _HEADING_NUMBER.match(text)
    return m.group(1).rstrip(".") if m else None


def parse_blocks(text: str) -> list[MDBlock]:
    """Парсит MD-текст в список типизированных блоков.

    Обработка:
    - Заголовки: `# ...`, `## ...` и т.д.
    - Таблицы: от первой строки `|...` до окончания блока `|...`, включая caption.
    - Display math: `$$...$$` (может быть на одной строке или на нескольких).
    - Figure captions: строки начинающиеся с `Figure N`.
    - Всё остальное: paragraph.

    Frontmatter (---...---) автоматически пропускается.
    """
    lines = text.split("\n")
    blocks: list[MDBlock] = []
    i = 0
    n = len(lines)

    # Пропуск YAML frontmatter
    if i < n and lines[i].strip() == "---":
        i += 1
        while i < n and lines[i].strip() != "---":
            i += 1
        if i < n:
            i += 1  # skip closing ---

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Пропуск пустых строк
        if not stripped:
            i += 1
            continue

        # --- Heading ---
        hm = _HEADING.match(stripped)
        if hm:
            level = len(hm.group(1))
            title = hm.group(2).strip()
            blocks.append(MDBlock(
                type="heading",
                content=stripped,
                heading_level=level,
                heading_number=_extract_heading_number(title),
                start_line=i,
                end_line=i,
            ))
            i += 1
            continue

        # --- Display math (строка начинается с $$) ---
        if _DISPLAY_MATH_START.match(stripped):
            start = i
            # Если $$ на одной строке ($$...$$)
            if stripped.endswith("$$") and len(stripped) > 4:
                blocks.append(MDBlock(
                    type="display_math",
                    content=stripped,
                    start_line=start,
                    end_line=i,
                ))
                i += 1
                continue
            # Многострочная display math: ищем закрывающий $$
            math_lines = [line]
            i += 1
            while i < n:
                math_lines.append(lines[i])
                if lines[i].strip().endswith("$$"):
                    break
                i += 1
            blocks.append(MDBlock(
                type="display_math",
                content="\n".join(math_lines),
                start_line=start,
                end_line=i,
            ))
            i += 1
            continue

        # --- Table ---
        if _TABLE_ROW.match(stripped):
            start = i
            table_lines = []
            while i < n and _TABLE_ROW.match(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1
            content = "\n".join(table_lines)
            # Ищем table caption перед таблицей (предыдущий блок может быть caption)
            # или после таблицы
            blocks.append(MDBlock(
                type="table",
                content=content,
                start_line=start,
                end_line=i - 1,
            ))
            continue

        # --- Figure caption ---
        if _FIGURE_CAPTION.match(stripped):
            # Собираем многострочный caption
            start = i
            cap_lines = [line]
            i += 1
            while i < n and lines[i].strip() and not _HEADING.match(lines[i].strip()):
                # Caption может продолжаться на следующих строках
                if lines[i].strip() == "" or _TABLE_ROW.match(lines[i].strip()):
                    break
                cap_lines.append(lines[i])
                i += 1
            blocks.append(MDBlock(
                type="figure_caption",
                content="\n".join(cap_lines),
                start_line=start,
                end_line=i - 1,
            ))
            continue

        # --- Table caption (перед таблицей) ---
        if _TABLE_CAPTION.match(stripped):
            start = i
            cap_lines = [line]
            i += 1
            # Собираем caption до начала таблицы или пустой строки
            while i < n and lines[i].strip() and not _TABLE_ROW.match(lines[i].strip()):
                cap_lines.append(lines[i])
                i += 1
            # Если дальше идёт таблица — прикрепляем caption к ней
            if i < n and _TABLE_ROW.match(lines[i].strip()):
                table_start = i
                table_lines = list(cap_lines)
                while i < n and _TABLE_ROW.match(lines[i].strip()):
                    table_lines.append(lines[i])
                    i += 1
                blocks.append(MDBlock(
                    type="table",
                    content="\n".join(table_lines),
                    start_line=start,
                    end_line=i - 1,
                ))
            else:
                # caption без таблицы — просто параграф
                blocks.append(MDBlock(
                    type="paragraph",
                    content="\n".join(cap_lines),
                    start_line=start,
                    end_line=i - 1,
                ))
            continue

        # --- Paragraph (по умолчанию) ---
        start = i
        para_lines = [line]
        i += 1
        while i < n and lines[i].strip():
            next_stripped = lines[i].strip()
            # Прерываем, если встретили начало другого блока
            if (
                _HEADING.match(next_stripped)
                or _DISPLAY_MATH_START.match(next_stripped)
                or _TABLE_ROW.match(next_stripped)
                or _FIGURE_CAPTION.match(next_stripped)
                or _TABLE_CAPTION.match(next_stripped)
            ):
                break
            para_lines.append(lines[i])
            i += 1
        blocks.append(MDBlock(
            type="paragraph",
            content="\n".join(para_lines),
            start_line=start,
            end_line=i - 1,
        ))

    log.info(
        "Parsed %d blocks: %s",
        len(blocks),
        {bt: sum(1 for b in blocks if b.type == bt) for bt in ["heading", "paragraph", "table", "display_math", "figure_caption"]},
    )
    return blocks
