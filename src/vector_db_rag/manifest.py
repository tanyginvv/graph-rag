from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .constants import (
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


def build_index_manifest(
    *,
    collection: str,
    chunker_version: str,
    config_hash: str,
    chunk_count: int,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_version": __version__,
        "collection": collection,
        "chunker_version": chunker_version,
        "config_hash": config_hash,
        "chunk_count": chunk_count,
        "dense": {
            "provider": "openai",
            "model": DENSE_MODEL,
            "dim": DENSE_DIM,
            "vector_name": DENSE_VECTOR_NAME,
            "input_field": DENSE_INPUT_FIELD,
            "distance": DENSE_DISTANCE,
        },
        "sparse": {
            "model": SPARSE_MODEL,
            "vector_name": SPARSE_VECTOR_NAME,
            "input_field": SPARSE_INPUT_FIELD,
            "modifier": SPARSE_MODIFIER,
        },
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

