"""Qdrant's in-process mode checks the sparse-only startup path."""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from vector_db_rag.chunks import ChunkFile, load_chunk_file
from vector_db_rag.ids import point_id_from_chunk_id
from vector_db_rag.indexing import fit_sparse_encoder, index_dense_chunks, index_sparse_chunks
from vector_db_rag.qdrant_store import create_or_recreate_collection, sparse_query


CORPUS = Path(__file__).resolve().parents[1] / "chunks 2" / "docs" / "dic_chunks.json"


def test_sparse_index_and_query_on_empty_collection(tmp_path):
    source = load_chunk_file(CORPUS)
    corpus = ChunkFile(
        chunker_version=source.chunker_version,
        config_hash=source.config_hash,
        chunks=source.chunks[:5],
    )
    client = QdrantClient(":memory:")
    create_or_recreate_collection(client, collection="test_chunks")
    index_sparse_chunks(
        client=client,
        collection="test_chunks",
        chunk_file=corpus,
        sparse_cache_path=str(tmp_path / "sparse.json"),
        batch_size=2,
        manifest_path=str(tmp_path / "manifest.json"),
    )
    assert client.count(collection_name="test_chunks", exact=True).count == 5
    encoder = fit_sparse_encoder(corpus)
    results = sparse_query(
        client,
        collection="test_chunks",
        sparse_vector=encoder.encode_query("Time-LLM reprogramming"),
        limit=3,
    )
    assert results
    assert results[0]["payload"]["chunk_id"]


class _FakeEmbedder:
    def embed_texts(self, texts):
        return [[1.0] + [0.0] * 3071 for _ in texts]


@pytest.mark.parametrize("sparse_first", [True, False])
def test_dense_and_sparse_vectors_survive_either_indexing_order(tmp_path, sparse_first):
    source = load_chunk_file(CORPUS)
    corpus = ChunkFile(
        chunker_version=source.chunker_version,
        config_hash=source.config_hash,
        chunks=source.chunks[:2],
    )
    client = QdrantClient(":memory:")
    create_or_recreate_collection(client, collection="test_chunks")

    def index_sparse():
        index_sparse_chunks(
            client=client, collection="test_chunks", chunk_file=corpus,
            sparse_cache_path=str(tmp_path / "sparse.json"), batch_size=2,
            manifest_path=str(tmp_path / "manifest.json"),
        )

    def index_dense():
        index_dense_chunks(
            client=client, collection="test_chunks", chunk_file=corpus,
            embedder=_FakeEmbedder(), batch_size=2,
            manifest_path=str(tmp_path / "manifest.json"),
        )

    if sparse_first:
        index_sparse()
        index_dense()
    else:
        index_dense()
        index_sparse()
    record = client.retrieve(
        collection_name="test_chunks",
        ids=[point_id_from_chunk_id(corpus.chunks[0].chunk_id)],
        with_vectors=True,
    )[0]
    assert "dense_retrieval" in record.vector
    assert "sparse_bm25" in record.vector
