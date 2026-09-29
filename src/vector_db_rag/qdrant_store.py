from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .constants import (
    DENSE_DIM,
    DENSE_VECTOR_NAME,
    SPARSE_MODIFIER,
    SPARSE_VECTOR_NAME,
)
from .filters import FilterSpec


class MissingQdrantClientError(RuntimeError):
    pass


def _import_qdrant() -> Any:
    try:
        import qdrant_client
        from qdrant_client import models
    except ImportError as exc:
        raise MissingQdrantClientError(
            "qdrant-client is not installed. Install project dependencies first: "
            "pip install -e ."
        ) from exc
    return qdrant_client, models


def make_client(*, url: str, api_key: str | None = None) -> Any:
    qdrant_client, _ = _import_qdrant()
    return qdrant_client.QdrantClient(url=url, api_key=api_key)


def check_qdrant(client: Any) -> dict[str, Any]:
    collections = client.get_collections()
    names = [collection.name for collection in collections.collections]
    return {"collections": names}


def create_or_recreate_collection(
    client: Any,
    *,
    collection: str,
    recreate: bool = False,
    dense_vector_name: str = DENSE_VECTOR_NAME,
    dense_dim: int = DENSE_DIM,
    sparse_vector_name: str = SPARSE_VECTOR_NAME,
) -> None:
    _, models = _import_qdrant()
    exists = client.collection_exists(collection)
    if exists and not recreate:
        return
    if exists and recreate:
        client.delete_collection(collection)

    vectors_config = {
        dense_vector_name: models.VectorParams(
            size=dense_dim,
            distance=models.Distance.COSINE,
        )
    }
    sparse_vectors_config = {
        sparse_vector_name: models.SparseVectorParams(
            modifier=models.Modifier.IDF
            if SPARSE_MODIFIER.upper() == "IDF"
            else None
        )
    }
    client.create_collection(
        collection_name=collection,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )


def ensure_payload_indexes(client: Any, *, collection: str) -> None:
    _, models = _import_qdrant()
    fields = [
        ("chunk_id", models.PayloadSchemaType.KEYWORD),
        ("article_id", models.PayloadSchemaType.KEYWORD),
        ("article_meta.year", models.PayloadSchemaType.INTEGER),
        ("article_meta.arxiv_id", models.PayloadSchemaType.KEYWORD),
        ("section", models.PayloadSchemaType.KEYWORD),
        ("section_hierarchy", models.PayloadSchemaType.KEYWORD),
        ("chunk_type", models.PayloadSchemaType.KEYWORD),
        ("question_types", models.PayloadSchemaType.KEYWORD),
        ("keywords", models.PayloadSchemaType.KEYWORD),
        ("index", models.PayloadSchemaType.INTEGER),
    ]
    for field_name, field_schema in fields:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already" not in message and "exist" not in message:
                raise


def upsert_points(
    client: Any,
    *,
    collection: str,
    points: Iterable[dict[str, Any]],
    batch_size: int = 64,
) -> None:
    batch: list[dict[str, Any]] = []
    for point in points:
        batch.append(point)
        if len(batch) >= batch_size:
            _write_vector_batch(client, collection=collection, batch=batch)
            batch = []
    if batch:
        _write_vector_batch(client, collection=collection, batch=batch)


def update_sparse_vectors(
    client: Any,
    *,
    collection: str,
    points: Iterable[dict[str, Any]],
    sparse_vector_name: str = SPARSE_VECTOR_NAME,
    batch_size: int = 64,
) -> None:
    batch: list[dict[str, Any]] = []
    for point in points:
        batch.append(
            {
                "id": point["id"],
                "vector": {sparse_vector_name: _sparse_vector_model(point["sparse_vector"])},
                "payload": point["full_payload"],
                "update_payload": point["payload"],
            }
        )
        if len(batch) >= batch_size:
            _write_vector_batch(client, collection=collection, batch=batch)
            batch = []
    if batch:
        _write_vector_batch(client, collection=collection, batch=batch)


def _write_vector_batch(
    client: Any,
    *,
    collection: str,
    batch: list[dict[str, Any]],
) -> None:
    """Create missing points and update vectors on existing points without erasing other vectors."""
    _, models = _import_qdrant()
    existing = {
        str(record.id)
        for record in client.retrieve(
            collection_name=collection,
            ids=[point["id"] for point in batch],
            with_payload=False,
            with_vectors=False,
        )
    }
    new_points = [
        models.PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
        for point in batch
        if str(point["id"]) not in existing
    ]
    if new_points:
        client.upsert(collection_name=collection, points=new_points, wait=True)
    updates = [
        models.PointVectors(id=point["id"], vector=point["vector"])
        for point in batch
        if str(point["id"]) in existing
    ]
    if updates:
        client.update_vectors(collection_name=collection, points=updates, wait=True)
    for point in batch:
        if str(point["id"]) not in existing:
            continue
        client.set_payload(
            collection_name=collection,
            payload=point.get("update_payload", point["payload"]),
            points=[point["id"]],
            wait=True,
        )


def qdrant_filter_from_spec(spec: FilterSpec | None) -> Any | None:
    if spec is None or spec.is_empty():
        return None

    _, models = _import_qdrant()
    must: list[Any] = []
    if spec.article_id:
        must.append(
            models.FieldCondition(
                key="article_id",
                match=models.MatchValue(value=spec.article_id),
            )
        )
    if spec.arxiv_id:
        must.append(
            models.FieldCondition(
                key="article_meta.arxiv_id",
                match=models.MatchValue(value=spec.arxiv_id),
            )
        )
    if spec.section:
        must.append(
            models.FieldCondition(
                key="section",
                match=models.MatchValue(value=spec.section),
            )
        )
    if spec.chunk_type:
        must.append(
            models.FieldCondition(
                key="chunk_type",
                match=models.MatchValue(value=spec.chunk_type),
            )
        )
    if spec.chunk_types:
        must.append(
            models.FieldCondition(
                key="chunk_type",
                match=models.MatchAny(any=list(spec.chunk_types)),
            )
        )
    if spec.question_types:
        must.append(
            models.FieldCondition(
                key="question_types",
                match=models.MatchAny(any=list(spec.question_types)),
            )
        )
    for keyword in spec.keywords:
        must.append(
            models.FieldCondition(
                key="keywords",
                match=models.MatchValue(value=keyword),
            )
        )
    if spec.year_gte is not None or spec.year_lte is not None:
        must.append(
            models.FieldCondition(
                key="article_meta.year",
                range=models.Range(gte=spec.year_gte, lte=spec.year_lte),
            )
        )
    return models.Filter(must=must)


def fetch_collection_chunks(
    client: Any,
    *,
    collection: str,
    limit: int = 256,
) -> dict[str, str]:
    chunk_map: dict[str, str] = {}
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            with_payload=["chunk_id", "article_id"],
            with_vectors=False,
            limit=limit,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            article_id = payload.get("article_id")
            if chunk_id:
                chunk_map[str(chunk_id)] = str(article_id or "")
        if offset is None:
            break
    return chunk_map


def dense_query(
    client: Any,
    *,
    collection: str,
    vector: list[float],
    vector_name: str = DENSE_VECTOR_NAME,
    limit: int = 10,
    filter_spec: FilterSpec | None = None,
) -> list[dict[str, Any]]:
    query_filter = qdrant_filter_from_spec(filter_spec)
    try:
        response = client.query_points(
            collection_name=collection,
            query=vector,
            using=vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        points = response.points
    except (AttributeError, TypeError):
        points = client.search(
            collection_name=collection,
            query_vector=(vector_name, vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

    return [
        {
            "id": str(point.id),
            "score": float(point.score),
            "payload": point.payload or {},
        }
        for point in points
    ]


def _sparse_vector_model(sparse_vector: Any) -> Any:
    _, models = _import_qdrant()
    if hasattr(sparse_vector, "indices") and hasattr(sparse_vector, "values"):
        return models.SparseVector(indices=sparse_vector.indices, values=sparse_vector.values)
    if isinstance(sparse_vector, dict):
        return models.SparseVector(
            indices=sparse_vector["indices"],
            values=sparse_vector["values"],
        )
    raise TypeError(f"Unsupported sparse vector type: {type(sparse_vector)!r}")


def sparse_query(
    client: Any,
    *,
    collection: str,
    sparse_vector: Any,
    vector_name: str = SPARSE_VECTOR_NAME,
    limit: int = 10,
    filter_spec: FilterSpec | None = None,
) -> list[dict[str, Any]]:
    query_filter = qdrant_filter_from_spec(filter_spec)
    response = client.query_points(
        collection_name=collection,
        query=_sparse_vector_model(sparse_vector),
        using=vector_name,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return _format_points(response.points)


def hybrid_query(
    client: Any,
    *,
    collection: str,
    dense_vector: list[float],
    sparse_vector: Any,
    dense_vector_name: str = DENSE_VECTOR_NAME,
    sparse_vector_name: str = SPARSE_VECTOR_NAME,
    prefetch_limit: int = 80,
    limit: int = 30,
    filter_spec: FilterSpec | None = None,
) -> list[dict[str, Any]]:
    _, models = _import_qdrant()
    query_filter = qdrant_filter_from_spec(filter_spec)
    response = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=dense_vector_name,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=_sparse_vector_model(sparse_vector),
                using=sparse_vector_name,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return _format_points(response.points)


def _format_points(points: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": str(point.id),
            "score": float(point.score),
            "payload": point.payload or {},
        }
        for point in points
    ]
