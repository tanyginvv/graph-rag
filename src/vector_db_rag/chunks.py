from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import ALLOWED_CHUNK_TYPES
from .text_fields import build_retrieval_text, build_sparse_text

QUESTION_TYPE_CHOICES = (
    "\u0410",
    "\u0410+",
    "\u0412",
    "\u0417",
    "\u0413",
    "\u0411",
    "\u0414",
    "\u0415",
    "\u0418",
    "\u0416",
    "\u041a",
    "\u041b",
    "\u041c",
    "\u0421\u043f\u0435\u0446",
)
ALLOWED_QUESTION_TYPES = frozenset(QUESTION_TYPE_CHOICES)


class ArticleMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str
    authors: list[str]
    year: int
    arxiv_id: str

    @field_validator("title")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("arxiv_id")
    @classmethod
    def clean_arxiv_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("authors")
    @classmethod
    def non_empty_authors(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must not be empty")
        return value


class Chunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    chunk_id: str
    article_id: str
    article_meta: ArticleMeta
    section: str
    section_hierarchy: list[str]
    chunk_type: str
    index: int
    text: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    question_types: list[str] = Field(default_factory=list)
    token_count: int

    @field_validator("chunk_id", "article_id", "section", "text", "summary")
    @classmethod
    def required_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("section_hierarchy")
    @classmethod
    def required_section_hierarchy(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("chunk_type")
    @classmethod
    def allowed_chunk_type(cls, value: str) -> str:
        if value not in ALLOWED_CHUNK_TYPES:
            raise ValueError(f"must be one of {sorted(ALLOWED_CHUNK_TYPES)}")
        return value

    @field_validator("index")
    @classmethod
    def non_negative_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative")
        return value

    @field_validator("token_count")
    @classmethod
    def positive_token_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("keywords")
    @classmethod
    def clean_keywords(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @field_validator("question_types")
    @classmethod
    def clean_question_types(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        invalid = [item for item in cleaned if item not in ALLOWED_QUESTION_TYPES]
        if invalid:
            raise ValueError(
                f"must contain only supported values: {list(QUESTION_TYPE_CHOICES)}"
            )
        return list(dict.fromkeys(cleaned))

    def to_payload(
        self,
        *,
        chunker_version: str,
        config_hash: str,
        include_sparse_text: bool = False,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["retrieval_text"] = build_retrieval_text(payload)
        if include_sparse_text:
            payload["sparse_text"] = build_sparse_text(payload)
        payload["chunker_version"] = chunker_version
        payload["config_hash"] = config_hash
        if extra_metadata:
            payload.update(extra_metadata)
        return payload


class ChunkFile(BaseModel):
    chunker_version: str
    config_hash: str
    chunks: list[Chunk]

    @field_validator("chunker_version", "config_hash")
    @classmethod
    def required_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def unique_keys(self) -> "ChunkFile":
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            duplicates = _duplicates(chunk_ids)
            raise ValueError(f"duplicate chunk_id values: {duplicates}")

        article_index_keys = [(chunk.article_id, chunk.index) for chunk in self.chunks]
        if len(article_index_keys) != len(set(article_index_keys)):
            duplicates = _duplicates(article_index_keys)
            raise ValueError(f"duplicate article_id+index values: {duplicates}")

        return self


def _duplicates(values: list[Any]) -> list[Any]:
    counts = Counter(values)
    return [value for value, count in counts.items() if count > 1]


def load_chunk_file(path: str | Path) -> ChunkFile:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return ChunkFile.model_validate(data)


def build_validation_report(chunk_file: ChunkFile) -> dict[str, Any]:
    by_article = Counter(chunk.article_id for chunk in chunk_file.chunks)
    by_section = Counter(chunk.section for chunk in chunk_file.chunks)
    by_chunk_type = Counter(chunk.chunk_type for chunk in chunk_file.chunks)

    return {
        "chunker_version": chunk_file.chunker_version,
        "config_hash": chunk_file.config_hash,
        "chunks": len(chunk_file.chunks),
        "articles": len(by_article),
        "by_article": dict(sorted(by_article.items())),
        "by_section": dict(sorted(by_section.items())),
        "by_chunk_type": dict(sorted(by_chunk_type.items())),
    }
