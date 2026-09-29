from __future__ import annotations

from collections.abc import Mapping, Sequence


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def build_section_path(section_hierarchy: Sequence[str]) -> str:
    return " > ".join(item.strip() for item in section_hierarchy if item.strip())


def build_retrieval_text(chunk: Mapping[str, object]) -> str:
    article_meta = chunk.get("article_meta") or {}
    if not isinstance(article_meta, Mapping):
        article_meta = {}

    section_hierarchy = _as_string_list(chunk.get("section_hierarchy"))
    keywords = _as_string_list(chunk.get("keywords"))
    question_types = _as_string_list(chunk.get("question_types"))
    title = str(article_meta.get("title") or "").strip()
    summary = str(chunk.get("summary") or "").strip()
    section_path = build_section_path(section_hierarchy)

    return "\n".join(
        [
            f"Title: {title}",
            f"Section: {section_path}",
            f"Summary: {summary}",
            f"Question Types: {', '.join(question_types)}",
            f"Keywords: {', '.join(keywords)}",
        ]
    )


def build_sparse_text(chunk: Mapping[str, object]) -> str:
    article_meta = chunk.get("article_meta") or {}
    if not isinstance(article_meta, Mapping):
        article_meta = {}

    section_hierarchy = _as_string_list(chunk.get("section_hierarchy"))
    keywords = _as_string_list(chunk.get("keywords"))
    question_types = _as_string_list(chunk.get("question_types"))
    title = str(article_meta.get("title") or "").strip()
    summary = str(chunk.get("summary") or "").strip()
    text = str(chunk.get("text") or "").strip()
    section_path = build_section_path(section_hierarchy)

    return "\n".join(
        [
            f"Title: {title}",
            f"Section: {section_path}",
            f"Summary: {summary}",
            f"Question Types: {', '.join(question_types)}",
            f"Keywords: {', '.join(keywords)}",
            f"Text: {text}",
        ]
    )
