"""Оркестратор препроцессинга: применяет все чистки в правильном порядке.

Пайплайн:
  raw_md
    -> header_normalizer  (склеенные заголовки, авторы)
    -> artifact_cleaner   (image placeholders, email, аффилиации, пустые строки таблиц)
    -> math_normalizer    (разорванные display-формулы, ложные разрывы параграфов)
    -> section_remover    (References, Impact Statement)
    -> frontmatter_builder (YAML метаданные)
    -> clean_md
"""

from __future__ import annotations

from src.common.logging_config import get_logger
from src.preprocess.artifact_cleaner import clean_artifacts
from src.preprocess.frontmatter_builder import ensure_frontmatter
from src.preprocess.header_normalizer import normalize_headers
from src.preprocess.math_normalizer import normalize_math
from src.preprocess.section_remover import remove_sections

log = get_logger(__name__)


def clean_md(
    raw_text: str,
    *,
    filename: str = "",
    remove_references: bool = True,
    remove_impact_statement: bool = True,
    remove_images: bool = True,
    remove_emails: bool = True,
) -> str:
    """Полный пайплайн очистки сырого MD."""
    text = raw_text

    log.info("=== Preprocessing: %s ===", filename or "<unknown>")

    text = normalize_headers(text)
    text = clean_artifacts(text, remove_images=remove_images, remove_emails=remove_emails)
    text = normalize_math(text)
    text = remove_sections(
        text,
        remove_references=remove_references,
        remove_impact_statement=remove_impact_statement,
    )
    text = ensure_frontmatter(text, filename=filename)

    # Убираем trailing whitespace и лишние пустые строки в конце
    lines = text.split("\n")
    text = "\n".join(line.rstrip() for line in lines).rstrip() + "\n"

    log.info("=== Done: %s ===", filename or "<unknown>")
    return text
