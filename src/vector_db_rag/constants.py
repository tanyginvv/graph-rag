from __future__ import annotations

ALLOWED_CHUNK_TYPES = {"prose", "formula_heavy", "table_with_prose", "table"}

DENSE_VECTOR_NAME = "dense_retrieval"
DENSE_MODEL = "text-embedding-3-large"
DENSE_DIM = 3072
DENSE_INPUT_FIELD = "retrieval_text"
DENSE_DISTANCE = "Cosine"

SPARSE_VECTOR_NAME = "sparse_bm25"
SPARSE_MODEL = "Qdrant/bm25"
SPARSE_INPUT_FIELD = "sparse_text"
SPARSE_MODIFIER = "IDF"

DEFAULT_COLLECTION = "chunks"

