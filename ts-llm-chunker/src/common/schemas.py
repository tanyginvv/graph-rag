"""Pydantic-схемы — контракт между всеми модулями пайплайна."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SectionLabel = Literal[
    "Abstract",
    "Introduction",
    "RelatedWork",
    "Preliminaries",
    "Methodology",
    "Experiments",
    "Results",
    "Discussion",
    "Conclusion",
    "Appendix_Implementation",
    "Appendix_Baselines",
    "Appendix_Datasets",
    "Appendix_Metrics",
    "Appendix_Results",
    "Appendix_Other",
    "Unknown",
]


class ArticleMetadata(BaseModel):
    """Метаданные статьи — собираются препроцессором, пишутся в YAML frontmatter."""

    article_id: str = Field(..., description="Стабильный id, например 'arxiv_2403_05798'")
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    arxiv_id: str | None = None
    venue: str | None = None

    @field_validator("article_id")
    @classmethod
    def _id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("article_id must be non-empty")
        return v


class Chunk(BaseModel):
    """Финальный чанк — соответствует атрибутам онтологии GraphRAG."""

    chunk_id: str = Field(..., description="Например 'arxiv_2403_05798_chunk_007'")
    article_id: str
    section: SectionLabel
    index: int = Field(..., ge=0, description="Порядковый номер в статье, с нуля")
    text: str = Field(..., min_length=1)
    token_count: int = Field(..., ge=0)


class ChunksDocument(BaseModel):
    """Обёртка над коллекцией чанков всех статей — пишется в dic_chunks.json."""

    chunker_version: str
    config_hash: str
    chunks: list[Chunk]
