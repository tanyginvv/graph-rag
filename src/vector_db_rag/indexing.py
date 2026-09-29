from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .chunks import ChunkFile
from .constants import (
    DENSE_DIM,
    DENSE_INPUT_FIELD,
    DENSE_MODEL,
    DENSE_VECTOR_NAME,
    SPARSE_INPUT_FIELD,
    SPARSE_MODEL,
    SPARSE_MODIFIER,
    SPARSE_VECTOR_NAME,
)
from .embeddings import OpenAIEmbeddingClient
from .ids import point_id_from_chunk_id
from .manifest import build_index_manifest, write_manifest
from .qdrant_store import update_sparse_vectors, upsert_points
from .sparse import BM25SparseEncoder, save_sparse_encoder
from .text_fields import build_sparse_text


def iter_dense_points(
    chunk_file: ChunkFile,
    *,
    embedder: OpenAIEmbeddingClient,
    batch_size: int = 64,
) -> Iterator[dict[str, Any]]:
    extra_metadata = {
        "embedding_vector_name": DENSE_VECTOR_NAME,
        "embedding_provider": "openai",
        "embedding_model": DENSE_MODEL,
        "embedding_dim": DENSE_DIM,
        "embedding_input_field": DENSE_INPUT_FIELD,
    }
    chunks = chunk_file.chunks
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        payloads = [
            chunk.to_payload(
                chunker_version=chunk_file.chunker_version,
                config_hash=chunk_file.config_hash,
                include_sparse_text=True,
                extra_metadata=extra_metadata,
            )
            for chunk in batch
        ]
        texts = [payload["retrieval_text"] for payload in payloads]
        vectors = embedder.embed_texts(texts)
        for chunk, payload, vector in zip(batch, payloads, vectors, strict=True):
            yield {
                "id": point_id_from_chunk_id(chunk.chunk_id),
                "vector": {DENSE_VECTOR_NAME: vector},
                "payload": payload,
            }


def index_dense_chunks(
    *,
    client: Any,
    collection: str,
    chunk_file: ChunkFile,
    embedder: OpenAIEmbeddingClient,
    batch_size: int,
    manifest_path: str,
) -> dict[str, Any]:
    points = iter_dense_points(chunk_file, embedder=embedder, batch_size=batch_size)
    upsert_points(client, collection=collection, points=points, batch_size=batch_size)
    manifest = build_index_manifest(
        collection=collection,
        chunker_version=chunk_file.chunker_version,
        config_hash=chunk_file.config_hash,
        chunk_count=len(chunk_file.chunks),
    )
    write_manifest(manifest_path, manifest)
    return manifest


def iter_sparse_points(
    chunk_file: ChunkFile,
    *,
    encoder: BM25SparseEncoder,
) -> Iterator[dict[str, Any]]:
    sparse_metadata = {
        "sparse_vector_name": SPARSE_VECTOR_NAME,
        "sparse_model": SPARSE_MODEL,
        "sparse_input_field": SPARSE_INPUT_FIELD,
        "sparse_modifier": SPARSE_MODIFIER,
        "hybrid_fusion": "RRF",
    }
    for chunk in chunk_file.chunks:
        chunk_dict = chunk.model_dump(mode="json")
        sparse_text = build_sparse_text(chunk_dict)
        sparse_vector = encoder.encode_document(sparse_text)
        yield {
            "id": point_id_from_chunk_id(chunk.chunk_id),
            "sparse_vector": sparse_vector,
            "payload": {
                "sparse_text": sparse_text,
                **sparse_metadata,
            },
            "full_payload": chunk.to_payload(
                chunker_version=chunk_file.chunker_version,
                config_hash=chunk_file.config_hash,
                include_sparse_text=True,
                extra_metadata=sparse_metadata,
            ),
        }


def fit_sparse_encoder(chunk_file: ChunkFile) -> BM25SparseEncoder:
    sparse_texts = [
        build_sparse_text(chunk.model_dump(mode="json"))
        for chunk in chunk_file.chunks
    ]
    return BM25SparseEncoder.fit(sparse_texts)


def index_sparse_chunks(
    *,
    client: Any,
    collection: str,
    chunk_file: ChunkFile,
    sparse_cache_path: str,
    batch_size: int,
    manifest_path: str,
) -> dict[str, Any]:
    encoder = fit_sparse_encoder(chunk_file)
    save_sparse_encoder(sparse_cache_path, encoder)
    points = iter_sparse_points(chunk_file, encoder=encoder)
    update_sparse_vectors(
        client,
        collection=collection,
        points=points,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        batch_size=batch_size,
    )
    manifest = build_index_manifest(
        collection=collection,
        chunker_version=chunk_file.chunker_version,
        config_hash=chunk_file.config_hash,
        chunk_count=len(chunk_file.chunks),
    )
    write_manifest(manifest_path, manifest)
    return manifest
