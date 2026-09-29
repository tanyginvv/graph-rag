"""Главная оркестрация чанкинга: MD -> list[Chunk] -> JSON.

Пайплайн:
  clean_md
    -> parse_blocks (list[MDBlock])
    -> group_by_section (heading -> blocks до следующего heading)
    -> for each section:
        mask_formulas в блоках
        recursive_split с сохранением атомарности таблиц/формул
        unmask_formulas
    -> emit list[Chunk]
"""

from __future__ import annotations

from typing import Any

import frontmatter

from src.chunker.math_masker import mask_formulas, unmask_formulas
from src.chunker.section_mapper import classify_heading
from src.chunker.splitters.recursive_splitter import recursive_split
from src.chunker.structure_parser import MDBlock, parse_blocks
from src.chunker.token_counter import count_tokens
from src.common.logging_config import get_logger
from src.common.schemas import ArticleMetadata, Chunk, SectionLabel

log = get_logger(__name__)


def _extract_metadata(text: str, filename: str = "") -> ArticleMetadata:
    """Извлекает метаданные из YAML frontmatter."""
    post = frontmatter.loads(text)
    meta = post.metadata

    from src.preprocess.frontmatter_builder import article_id_from_filename

    article_id = meta.get("article_id", "")
    if not article_id and filename:
        article_id = article_id_from_filename(filename)

    arxiv_id = meta.get("arxiv_id")
    if arxiv_id is not None:
        arxiv_id = str(arxiv_id)

    return ArticleMetadata(
        article_id=article_id,
        title=meta.get("title", ""),
        authors=meta.get("authors", []),
        year=meta.get("year"),
        arxiv_id=arxiv_id,
        venue=meta.get("venue"),
    )


def _detect_top_heading_level(blocks: list[MDBlock]) -> int:
    """Определяет фактический верхний уровень заголовков основного текста.

    Эвристика: берём уровень первого заголовка с level >= 2.
    Первый заголовок после title (#) — это всегда первая секция документа
    (Abstract или Introduction), по нему определяется конвенция статьи.

    Стандартные arXiv-статьи: ## (level 2). IEEE: #### (level 4).
    """
    for b in blocks:
        if b.type == "heading" and b.heading_level is not None and b.heading_level >= 2:
            return b.heading_level
    return 2


def _has_sub_headings(blocks: list[MDBlock], top_level: int) -> bool:
    """Проверяет, есть ли в документе подзаголовки глубже top_level.

    Если да — иерархия многоуровневая (arXiv-стиль: ## + ###).
    Если нет — плоская (IEEE-стиль: все ####), нужен pattern-based подход.
    """
    return any(
        b.type == "heading" and b.heading_level is not None and b.heading_level > top_level
        for b in blocks
    )


# Секции основного текста: используются для определения boundary в плоской иерархии.
# Appendix-секции не включены — они обнаруживаются через headings уровнем выше top_level.
_MAIN_BODY_LABELS: frozenset[str] = frozenset({
    "Abstract", "Introduction", "RelatedWork", "Preliminaries",
    "Methodology", "Experiments", "Results", "Discussion", "Conclusion",
})


def _is_section_boundary(block: MDBlock, top_level: int, flat_hierarchy: bool) -> bool:
    """Определяет, начинает ли заголовок новую секцию.

    Для многоуровневой иерархии: любой heading на уровне <= top_level.
    Для плоской иерархии (IEEE): heading выше top_level всегда boundary;
    heading на top_level — только если classify_heading даёт label основного текста.
    Это позволяет не разбивать подсекции (B. In Modality Adaptation),
    которые ложно матчатся под Appendix-паттерны.
    """
    if block.type != "heading" or block.heading_level is None:
        return False

    # Заголовок уровнем выше (меньше числом) top_level — всегда boundary
    # (напр. ## Supplementary в документе где top_level=4)
    if block.heading_level < top_level:
        return True

    if block.heading_level == top_level:
        if not flat_hierarchy:
            # Многоуровневая иерархия — все headings на top_level = boundary
            return True
        # Плоская иерархия — только main body labels
        return classify_heading(block.content) in _MAIN_BODY_LABELS

    return False


def _group_by_section(
    blocks: list[MDBlock],
) -> list[tuple[SectionLabel, list[MDBlock]]]:
    """Группирует блоки по секциям.

    Адаптивно определяет уровень top-level заголовков и стратегию разбиения:
    - Многоуровневая иерархия (## + ###): разбивает по heading level.
    - Плоская иерархия (все ####, IEEE): разбивает по pattern-matching,
      чтобы отличить секции (I. INTRODUCTION) от подсекций (A. Background).
    """
    top_level = _detect_top_heading_level(blocks)
    flat = not _has_sub_headings(blocks, top_level)
    log.debug("Detected top heading level: %d, flat hierarchy: %s", top_level, flat)

    sections: list[tuple[SectionLabel, list[MDBlock]]] = []
    current_label: SectionLabel = "Unknown"
    current_blocks: list[MDBlock] = []

    for block in blocks:
        if _is_section_boundary(block, top_level, flat):
            if current_blocks:
                sections.append((current_label, current_blocks))
            current_label = classify_heading(block.content)
            current_blocks = [block]
        else:
            current_blocks.append(block)

    # Последняя секция
    if current_blocks:
        sections.append((current_label, current_blocks))

    return sections


def chunk_article(
    text: str,
    *,
    filename: str = "",
    max_chunk_size: int = 600,
    min_chunk_size: int = 150,
    overlap_tokens: int = 80,
    encoding: str = "cl100k_base",
) -> tuple[ArticleMetadata, list[Chunk]]:
    """Чанкит одну статью (нормализованный MD) в список Chunk-ов.

    Returns:
        (metadata, chunks)
    """
    metadata = _extract_metadata(text, filename)
    article_id = metadata.article_id

    # 1. Парсим структуру
    blocks = parse_blocks(text)

    # 2. Группируем по секциям
    sections = _group_by_section(blocks)
    log.info("Found %d section(s): %s", len(sections), [s[0] for s in sections])

    # 3. Для каждой секции: mask -> split -> unmask -> emit chunks
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for section_label, section_blocks in sections:
        # Маскируем формулы во всех блоках секции
        masked_blocks: list[MDBlock] = []
        full_mapping: dict[str, str] = {}

        for block in section_blocks:
            masked_content, mapping = mask_formulas(block.content)
            full_mapping.update(mapping)
            masked_blocks.append(MDBlock(
                type=block.type,
                content=masked_content,
                heading_level=block.heading_level,
                heading_number=block.heading_number,
                start_line=block.start_line,
                end_line=block.end_line,
            ))

        # Recursive split
        raw_chunks = recursive_split(
            masked_blocks,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            overlap_tokens=overlap_tokens,
            encoding=encoding,
        )

        # Unmask и создаём Chunk-объекты
        for raw_text in raw_chunks:
            unmasked = unmask_formulas(raw_text, full_mapping)
            token_count = count_tokens(unmasked, encoding)

            chunk = Chunk(
                chunk_id=f"{article_id}_chunk_{chunk_index:03d}",
                article_id=article_id,
                section=section_label,
                index=chunk_index,
                text=unmasked,
                token_count=token_count,
            )
            all_chunks.append(chunk)
            chunk_index += 1

    log.info(
        "Chunked '%s': %d chunks, total tokens: %d",
        article_id,
        len(all_chunks),
        sum(c.token_count for c in all_chunks),
    )
    return metadata, all_chunks
