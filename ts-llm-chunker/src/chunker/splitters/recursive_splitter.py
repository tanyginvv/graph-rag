"""Recursive splitter: разбивает текст секции на чанки по подзаголовкам -> абзацам -> предложениям.

Атомарные блоки (таблицы, display math) не разрезаются.
Overlap применяется только внутри одной секции.
"""

from __future__ import annotations

import re

from src.chunker.structure_parser import MDBlock
from src.chunker.token_counter import count_tokens
from src.common.logging_config import get_logger

log = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _token_len(text: str, encoding: str) -> int:
    return count_tokens(text, encoding)


def _split_by_paragraphs(text: str) -> list[str]:
    """Разбивает текст на абзацы по двойному переносу строки."""
    parts = re.split(r"\n\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_sentences(text: str) -> list[str]:
    """Разбивает абзац на предложения."""
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _merge_small_chunks(
    pieces: list[str],
    max_size: int,
    encoding: str,
) -> list[str]:
    """Объединяет мелкие куски до max_size."""
    if not pieces:
        return []
    merged: list[str] = []
    current = pieces[0]

    for piece in pieces[1:]:
        candidate = current + "\n\n" + piece
        if _token_len(candidate, encoding) <= max_size:
            current = candidate
        else:
            merged.append(current)
            current = piece
    merged.append(current)
    return merged


def _split_oversized_piece(text: str, max_size: int, encoding: str) -> list[str]:
    """Cut a long sentence at word boundaries, with a character fallback."""
    pieces: list[str] = []
    remaining = text.strip()
    while _token_len(remaining, encoding) > max_size:
        low, high = 1, len(remaining)
        cut = 0
        while low <= high:
            middle = (low + high) // 2
            if _token_len(remaining[:middle], encoding) <= max_size:
                cut = middle
                low = middle + 1
            else:
                high = middle - 1
        if cut == 0:
            raise ValueError("max_chunk_size cannot fit one character")
        space = remaining.rfind(" ", 0, cut + 1)
        if space >= cut // 2:
            cut = space
        piece = remaining[:cut].strip()
        if not piece:
            raise ValueError("Cannot split oversized paragraph")
        pieces.append(piece)
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _split_large_table(table: str, max_size: int, encoding: str) -> list[str]:
    """Keep the table header and fit complete rows into bounded table fragments."""
    if _token_len(table, encoding) <= max_size:
        return [table]
    lines = table.splitlines()
    if len(lines) < 3 or not re.match(r"^\s*\|?[\s:|-]*---", lines[1]):
        log.warning("Oversized table has no recognizable header; keeping it intact")
        return [table]
    header = lines[:2]
    fragments: list[str] = []
    current = header.copy()
    for row in lines[2:]:
        candidate = "\n".join([*current, row])
        if _token_len(candidate, encoding) <= max_size:
            current.append(row)
            continue
        if len(current) > len(header):
            fragments.append("\n".join(current))
            current = header.copy()
        if _token_len("\n".join([*header, row]), encoding) > max_size:
            log.warning("One table row exceeds max_chunk_size; keeping row intact")
            fragments.append("\n".join([*header, row]))
        else:
            current.append(row)
    if len(current) > len(header):
        fragments.append("\n".join(current))
    return fragments


def _add_overlap(
    chunks: list[str],
    overlap_tokens: int,
    encoding: str,
    max_chunk_size: int,
    atomic_indices: set[int],
) -> list[str]:
    """Добавляет overlap — копирует конец предыдущего чанка в начало следующего."""
    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        if i in atomic_indices or i - 1 in atomic_indices:
            result.append(chunks[i])
            continue
        prev = chunks[i - 1]
        # Берём последние N токенов из предыдущего чанка
        sentences = _split_by_sentences(prev)
        overlap_text = ""
        for sent in reversed(sentences):
            candidate = sent if not overlap_text else sent + " " + overlap_text
            if (
                _token_len(candidate, encoding) > overlap_tokens
                or _token_len(candidate + "\n\n" + chunks[i], encoding) > max_chunk_size
            ):
                break
            overlap_text = candidate

        if overlap_text:
            result.append(overlap_text + "\n\n" + chunks[i])
        else:
            result.append(chunks[i])

    return result


def recursive_split(
    blocks: list[MDBlock],
    *,
    max_chunk_size: int = 600,
    min_chunk_size: int = 150,
    overlap_tokens: int = 80,
    encoding: str = "cl100k_base",
) -> list[str]:
    """Разбивает список блоков секции на текстовые чанки.

    Логика:
    1. Атомарные блоки (table, display_math) — всегда отдельный чанк.
    2. Остальные блоки объединяются в текст и разбиваются:
       - по абзацам
       - если абзац > max_size, по предложениям
    3. Мелкие куски объединяются до max_size.
    4. Overlap добавляется между чанками.
    """
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must not be negative")
    chunks: list[str] = []
    atomic_indices: set[int] = set()
    text_buffer: list[str] = []

    def _flush_buffer() -> None:
        if not text_buffer:
            return
        full_text = "\n\n".join(text_buffer)
        text_buffer.clear()

        paragraphs = _split_by_paragraphs(full_text)
        pieces: list[str] = []
        for para in paragraphs:
            if _token_len(para, encoding) <= max_chunk_size:
                pieces.append(para)
            else:
                # Разбиваем по предложениям
                sentences = _split_by_sentences(para)
                for sentence in sentences:
                    pieces.extend(_split_oversized_piece(sentence, max_chunk_size, encoding))

        merged = _merge_small_chunks(pieces, max_chunk_size, encoding)
        chunks.extend(merged)

    for block in blocks:
        if block.type in ("table", "display_math"):
            # Сначала сбрасываем буфер текста
            _flush_buffer()
            fragments = (
                _split_large_table(block.content, max_chunk_size, encoding)
                if block.type == "table" else [block.content]
            )
            for fragment in fragments:
                atomic_indices.add(len(chunks))
                chunks.append(fragment)
                if _token_len(fragment, encoding) > max_chunk_size:
                    log.warning("Atomic %s exceeds max_chunk_size", block.type)
        elif block.type == "heading":
            # Заголовки подсекций не создают новый чанк, а включаются в поток
            text_buffer.append(block.content)
        else:
            # paragraph, figure_caption
            text_buffer.append(block.content)

    _flush_buffer()

    # Добавляем overlap
    chunks = _add_overlap(chunks, overlap_tokens, encoding, max_chunk_size, atomic_indices)

    return chunks
