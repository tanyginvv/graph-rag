"""Сборка / нормализация YAML frontmatter в MD-файле.

Если frontmatter уже есть — оставляет как есть (он был заполнен при конвертации).
Если нет — пытается извлечь метаданные из тела документа и создать frontmatter.
"""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from src.common.logging_config import get_logger

log = get_logger(__name__)

# Паттерн для вычленения arxiv_id из имени файла: 2403_05798.md -> 2403.05798
_ARXIV_FROM_FILENAME = re.compile(r"(\d{4})[._](\d{4,5})")


def article_id_from_filename(filename: str) -> str:
    """Формирует article_id из имени файла.

    '2403_05798.md' -> 'arxiv_2403_05798'
    '2603.29756.md' -> 'arxiv_2603_29756'  (точки нормализуются в подчёркивания)
    """
    stem = Path(filename).stem
    normalized = stem.replace(".", "_")
    return f"arxiv_{normalized}"


def arxiv_id_from_filename(filename: str) -> str | None:
    """Извлекает arxiv_id вида '2403.05798' из имени файла."""
    m = _ARXIV_FROM_FILENAME.search(Path(filename).stem)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def ensure_frontmatter(text: str, *, filename: str = "") -> str:
    """Гарантирует наличие YAML frontmatter с обязательными полями.

    Если frontmatter уже есть — проверяет, что article_id и arxiv_id заполнены
    (по возможности дополняет из имени файла). Если нет — создаёт минимальный.
    """
    post = frontmatter.loads(text)

    # Дополняем пропущенные поля
    if not post.get("article_id") and filename:
        post["article_id"] = article_id_from_filename(filename)

    if not post.get("arxiv_id") and filename:
        aid = arxiv_id_from_filename(filename)
        if aid:
            post["arxiv_id"] = aid

    # Если title не задан — попробуем вытащить из первого заголовка `# ...`
    if not post.get("title"):
        for line in post.content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                post["title"] = stripped[2:].strip()
                break

    result = frontmatter.dumps(post)
    log.info("Frontmatter fields: %s", list(post.metadata.keys()))
    return result
