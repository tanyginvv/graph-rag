from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chunks import ALLOWED_QUESTION_TYPES, QUESTION_TYPE_CHOICES


@dataclass(frozen=True)
class FilterSpec:
    article_id: str | None = None
    arxiv_id: str | None = None
    section: str | None = None
    chunk_type: str | None = None
    chunk_types: tuple[str, ...] = field(default_factory=tuple)
    keywords: tuple[str, ...] = field(default_factory=tuple)
    question_types: tuple[str, ...] = field(default_factory=tuple)
    year_gte: int | None = None
    year_lte: int | None = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.article_id,
                self.arxiv_id,
                self.section,
                self.chunk_type,
                self.chunk_types,
                self.keywords,
                self.question_types,
                self.year_gte is not None,
                self.year_lte is not None,
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "arxiv_id": self.arxiv_id,
            "section": self.section,
            "chunk_type": self.chunk_type,
            "chunk_types": list(self.chunk_types),
            "keywords": list(self.keywords),
            "question_types": list(self.question_types),
            "year_gte": self.year_gte,
            "year_lte": self.year_lte,
        }


def _parse_multi_values(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    parsed: list[str] = []
    for value in values:
        parsed.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(parsed)


def parse_keywords(values: list[str] | None) -> tuple[str, ...]:
    return _parse_multi_values(values)


def parse_question_types(values: list[str] | None) -> tuple[str, ...]:
    question_types = _parse_multi_values(values)
    invalid = [value for value in question_types if value not in ALLOWED_QUESTION_TYPES]
    if invalid:
        raise ValueError(
            f"Unsupported question type values: {invalid}. "
            f"Allowed values: {list(QUESTION_TYPE_CHOICES)}"
        )
    return tuple(dict.fromkeys(question_types))
