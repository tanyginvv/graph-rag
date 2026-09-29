"""Convert a basic chunker export into a minimal, traceable GraphRAG corpus."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .chunks import ArticleMeta, Chunk, ChunkFile


class BasicChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    article_id: str = Field(min_length=1)
    section: str = Field(min_length=1)
    index: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)


class BasicChunkFile(BaseModel):
    chunker_version: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    chunks: list[BasicChunk]


def _read_article_metadata(directory: Path) -> dict[str, ArticleMeta]:
    metadata: dict[str, ArticleMeta] = {}
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            continue
        parts = text.split("---\n", 2)
        if len(parts) < 3:
            continue
        data = yaml.safe_load(parts[1])
        if not isinstance(data, dict) or not data.get("article_id"):
            continue
        article_id = str(data["article_id"])
        if article_id in metadata:
            raise ValueError(f"Duplicate article_id in Markdown: {article_id}")
        metadata[article_id] = ArticleMeta.model_validate(
            {
                "title": data.get("title"),
                "authors": data.get("authors"),
                "year": data.get("year"),
                "arxiv_id": str(data.get("arxiv_id") or ""),
            }
        )
    return metadata


def _extractive_summary(text: str, *, max_chars: int = 280) -> str:
    without_headings = re.sub(r"(?m)^#{1,6}\s*", "", text)
    return " ".join(without_headings.split())[:max_chars].strip()


def _chunk_type(text: str) -> str:
    if re.search(r"(?m)^\s*\|.+\|\s*$", text):
        return "table_with_prose" if len(text.splitlines()) > 8 else "table"
    if "$$" in text or "\\[" in text:
        return "formula_heavy"
    return "prose"


def adapt_basic_chunks(chunks_path: str | Path, articles_dir: str | Path) -> ChunkFile:
    """Copy source text and metadata; leave semantic tags empty rather than invent them."""
    source = BasicChunkFile.model_validate_json(Path(chunks_path).read_text(encoding="utf-8"))
    articles = _read_article_metadata(Path(articles_dir))
    missing = sorted({chunk.article_id for chunk in source.chunks} - articles.keys())
    if missing:
        raise ValueError(f"Missing Markdown metadata for article_id: {', '.join(missing)}")

    converted = [
        Chunk(
            chunk_id=chunk.chunk_id,
            article_id=chunk.article_id,
            article_meta=articles[chunk.article_id],
            section=chunk.section,
            section_hierarchy=[chunk.section],
            chunk_type=_chunk_type(chunk.text),
            index=chunk.index,
            text=chunk.text,
            summary=_extractive_summary(chunk.text),
            keywords=[],
            question_types=[],
            token_count=chunk.token_count,
        )
        for chunk in source.chunks
    ]
    fingerprint = hashlib.sha256(
        json.dumps(
            [source.config_hash, [chunk.model_dump(mode="json") for chunk in converted]],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return ChunkFile(
        chunker_version=f"basic-adapter-1/{source.chunker_version}",
        config_hash=fingerprint,
        chunks=converted,
    )


def write_adapted_corpus(corpus: ChunkFile, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = corpus.model_dump(mode="json")
    data["provenance"] = {
        "adapter": "basic-extractive-1",
        "semantic_enrichment": False,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
