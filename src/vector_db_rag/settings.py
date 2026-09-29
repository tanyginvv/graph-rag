from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .constants import (
    DEFAULT_COLLECTION,
    DENSE_DIM,
    DENSE_DISTANCE,
    DENSE_INPUT_FIELD,
    DENSE_MODEL,
    DENSE_VECTOR_NAME,
    SPARSE_INPUT_FIELD,
    SPARSE_MODEL,
    SPARSE_MODIFIER,
    SPARSE_VECTOR_NAME,
)


@dataclass(frozen=True)
class PathsConfig:
    chunks: Path = Path("chunks/dic_chunks.json")
    embedding_cache: Path = Path("data/cache/embeddings")
    sparse_cache: Path = Path("data/cache/sparse_bm25.json")
    manifest: Path = Path("data/index_manifest.json")


@dataclass(frozen=True)
class QdrantConfig:
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = DEFAULT_COLLECTION


@dataclass(frozen=True)
class DenseConfig:
    provider: str = "openai"
    model: str = DENSE_MODEL
    dim: int = DENSE_DIM
    vector_name: str = DENSE_VECTOR_NAME
    input_field: str = DENSE_INPUT_FIELD
    distance: str = DENSE_DISTANCE
    batch_size: int = 64


@dataclass(frozen=True)
class SparseConfig:
    model: str = SPARSE_MODEL
    vector_name: str = SPARSE_VECTOR_NAME
    input_field: str = SPARSE_INPUT_FIELD
    modifier: str = SPARSE_MODIFIER
    batch_size: int = 64


@dataclass(frozen=True)
class RetrievalConfig:
    default_profile: str = "dense_only"
    default_limit: int = 10
    prefetch_limit: int = 80


@dataclass(frozen=True)
class GraphConfig:
    enabled: bool = True
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "change-me"
    database: str = "neo4j"
    batch_size: int = 200


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    api_key: str | None = None
    model: str = "deepseek-v4-flash"
    thinking_enabled: bool = False
    reasoning_effort: str = "high"
    temperature: float = 0.1
    max_tokens: int = 1200
    timeout_seconds: int = 120
    max_context_chunks: int = 5
    max_chars_per_chunk: int = 1800


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    qdrant: QdrantConfig
    dense: DenseConfig
    sparse: SparseConfig
    retrieval: RetrievalConfig
    graph: GraphConfig
    llm: LLMConfig


def load_config(path: str | Path = "config/retrieval.yaml") -> AppConfig:
    load_dotenv()
    config_path = Path(path)
    data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config must be a mapping: {config_path}")
            data = loaded

    paths_data = data.get("paths") or {}
    qdrant_data = data.get("qdrant") or {}
    dense_data = data.get("dense") or {}
    sparse_data = data.get("sparse") or {}
    retrieval_data = data.get("retrieval") or {}
    graph_data = data.get("graph") or {}
    llm_data = data.get("llm") or {}

    qdrant_api_key_env = qdrant_data.get("api_key_env", "QDRANT_API_KEY")
    qdrant_api_key = os.getenv(qdrant_api_key_env) or os.getenv("QDRANT_API_KEY") or None
    graph_password_env = graph_data.get("password_env", "NEO4J_PASSWORD")
    graph_password = os.getenv(graph_password_env) or os.getenv("NEO4J_PASSWORD") or "change-me"
    llm_api_key_env = llm_data.get("api_key_env", "DEEPSEEK_API_KEY")
    llm_api_key = os.getenv(llm_api_key_env) or os.getenv("DEEPSEEK_API_KEY") or None

    return AppConfig(
        paths=PathsConfig(
            chunks=Path(paths_data.get("chunks", "chunks/dic_chunks.json")),
            embedding_cache=Path(paths_data.get("embedding_cache", "data/cache/embeddings")),
            sparse_cache=Path(paths_data.get("sparse_cache", "data/cache/sparse_bm25.json")),
            manifest=Path(paths_data.get("manifest", "data/index_manifest.json")),
        ),
        qdrant=QdrantConfig(
            url=os.getenv("QDRANT_URL") or qdrant_data.get("url", "http://localhost:6333"),
            api_key=qdrant_api_key,
            collection=qdrant_data.get("collection", DEFAULT_COLLECTION),
        ),
        dense=DenseConfig(
            provider=dense_data.get("provider", "openai"),
            model=dense_data.get("model", DENSE_MODEL),
            dim=int(dense_data.get("dim", DENSE_DIM)),
            vector_name=dense_data.get("vector_name", DENSE_VECTOR_NAME),
            input_field=dense_data.get("input_field", DENSE_INPUT_FIELD),
            distance=dense_data.get("distance", DENSE_DISTANCE),
            batch_size=int(dense_data.get("batch_size", 64)),
        ),
        sparse=SparseConfig(
            model=sparse_data.get("model", SPARSE_MODEL),
            vector_name=sparse_data.get("vector_name", SPARSE_VECTOR_NAME),
            input_field=sparse_data.get("input_field", SPARSE_INPUT_FIELD),
            modifier=sparse_data.get("modifier", SPARSE_MODIFIER),
            batch_size=int(sparse_data.get("batch_size", 64)),
        ),
        retrieval=RetrievalConfig(
            default_profile=retrieval_data.get("default_profile", "dense_only"),
            default_limit=int(retrieval_data.get("default_limit", 10)),
            prefetch_limit=int(retrieval_data.get("prefetch_limit", 80)),
        ),
        graph=GraphConfig(
            enabled=_as_bool(os.getenv("NEO4J_ENABLED"), graph_data.get("enabled", True)),
            uri=os.getenv("NEO4J_URI") or graph_data.get("uri", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USERNAME") or graph_data.get("user", "neo4j"),
            password=graph_password,
            database=os.getenv("NEO4J_DATABASE") or graph_data.get("database", "neo4j"),
            batch_size=int(graph_data.get("batch_size", 200)),
        ),
        llm=LLMConfig(
            provider=os.getenv("LLM_PROVIDER") or llm_data.get("provider", "deepseek"),
            base_url=os.getenv("LLM_BASE_URL") or llm_data.get("base_url", "https://api.deepseek.com"),
            api_key=llm_api_key,
            model=os.getenv("LLM_MODEL") or llm_data.get("model", "deepseek-v4-flash"),
            thinking_enabled=_as_bool(
                os.getenv("LLM_THINKING_ENABLED"),
                llm_data.get("thinking_enabled", False),
            ),
            reasoning_effort=os.getenv("LLM_REASONING_EFFORT") or llm_data.get("reasoning_effort", "high"),
            temperature=float(os.getenv("LLM_TEMPERATURE") or llm_data.get("temperature", 0.1)),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS") or llm_data.get("max_tokens", 1200)),
            timeout_seconds=int(
                os.getenv("LLM_TIMEOUT_SECONDS") or llm_data.get("timeout_seconds", 120)
            ),
            max_context_chunks=int(
                os.getenv("LLM_MAX_CONTEXT_CHUNKS") or llm_data.get("max_context_chunks", 5)
            ),
            max_chars_per_chunk=int(
                os.getenv("LLM_MAX_CHARS_PER_CHUNK") or llm_data.get("max_chars_per_chunk", 1800)
            ),
        ),
    )


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")
