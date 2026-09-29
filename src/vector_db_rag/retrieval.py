from __future__ import annotations

from dataclasses import replace
from typing import Any

from .constants import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from .embeddings import OpenAIEmbeddingClient
from .filters import FilterSpec
from .qdrant_store import dense_query, hybrid_query, sparse_query
from .sparse import BM25SparseEncoder


def search_dense_only(
    *,
    client: Any,
    collection: str,
    query: str,
    embedder: OpenAIEmbeddingClient,
    limit: int = 10,
    filter_spec: FilterSpec | None = None,
) -> list[dict[str, Any]]:
    query_vector = embedder.embed_text(query)
    return dense_query(
        client,
        collection=collection,
        vector=query_vector,
        vector_name=DENSE_VECTOR_NAME,
        limit=limit,
        filter_spec=filter_spec,
    )


def search_sparse_only(
    *,
    client: Any,
    collection: str,
    query: str,
    sparse_encoder: BM25SparseEncoder,
    limit: int = 10,
    filter_spec: FilterSpec | None = None,
) -> list[dict[str, Any]]:
    sparse_vector = sparse_encoder.encode_query(query)
    return sparse_query(
        client,
        collection=collection,
        sparse_vector=sparse_vector,
        vector_name=SPARSE_VECTOR_NAME,
        limit=limit,
        filter_spec=filter_spec,
    )


def search_hybrid_default(
    *,
    client: Any,
    collection: str,
    query: str,
    embedder: OpenAIEmbeddingClient,
    sparse_encoder: BM25SparseEncoder,
    limit: int = 30,
    prefetch_limit: int = 80,
    filter_spec: FilterSpec | None = None,
) -> list[dict[str, Any]]:
    dense_vector = embedder.embed_text(query)
    sparse_vector = sparse_encoder.encode_query(query)
    return hybrid_query(
        client,
        collection=collection,
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        dense_vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        prefetch_limit=prefetch_limit,
        limit=limit,
        filter_spec=filter_spec,
    )


def search_by_profile(
    *,
    profile: str,
    client: Any,
    collection: str,
    query: str,
    embedder: OpenAIEmbeddingClient | None,
    sparse_encoder: BM25SparseEncoder | None,
    limit: int,
    prefetch_limit: int,
    filter_spec: FilterSpec | None = None,
) -> list[dict[str, Any]]:
    if profile == "dense_only":
        if embedder is None:
            raise ValueError("dense_only requires an embedding client.")
        return search_dense_only(
            client=client,
            collection=collection,
            query=query,
            embedder=embedder,
            limit=limit,
            filter_spec=filter_spec,
        )
    if profile == "sparse_only":
        if sparse_encoder is None:
            raise ValueError("sparse_only requires a sparse encoder.")
        return search_sparse_only(
            client=client,
            collection=collection,
            query=query,
            sparse_encoder=sparse_encoder,
            limit=limit,
            filter_spec=filter_spec,
        )
    if profile in {"hybrid_default", "entity_exact"}:
        if embedder is None or sparse_encoder is None:
            raise ValueError(f"{profile} requires embedding and sparse clients.")
        return search_hybrid_default(
            client=client,
            collection=collection,
            query=query,
            embedder=embedder,
            sparse_encoder=sparse_encoder,
            limit=limit,
            prefetch_limit=prefetch_limit,
            filter_spec=filter_spec,
        )
    if profile == "table_numeric":
        if embedder is None or sparse_encoder is None:
            raise ValueError("table_numeric requires embedding and sparse clients.")
        table_filter = filter_spec or FilterSpec()
        if not table_filter.chunk_type and not table_filter.chunk_types:
            table_filter = replace(
                table_filter,
                chunk_types=("table", "table_with_prose", "formula_heavy"),
            )
        return search_hybrid_default(
            client=client,
            collection=collection,
            query=query,
            embedder=embedder,
            sparse_encoder=sparse_encoder,
            limit=limit,
            prefetch_limit=prefetch_limit,
            filter_spec=table_filter,
        )
    raise ValueError(f"Unknown retrieval profile: {profile}")


def attach_graph_context(
    results: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    graph_by_chunk_id = {
        str(row["chunk_id"]): row
        for row in graph_rows
        if row.get("chunk_id")
    }
    enriched_results: list[dict[str, Any]] = []
    for result in results:
        payload = result.get("payload") or {}
        chunk_id = payload.get("chunk_id")
        if not chunk_id or chunk_id not in graph_by_chunk_id:
            enriched_results.append(result)
            continue
        enriched_result = dict(result)
        enriched_result["graph_context"] = graph_by_chunk_id[str(chunk_id)]
        enriched_results.append(enriched_result)
    return enriched_results


def format_result(result: dict[str, Any], *, preview_chars: int = 240) -> str:
    payload = result.get("payload") or {}
    article_meta = payload.get("article_meta") or {}
    section_hierarchy = payload.get("section_hierarchy") or []
    section_path = " > ".join(str(item) for item in section_hierarchy)
    summary = str(payload.get("summary") or "")
    preview = summary[:preview_chars].replace("\n", " ")
    if len(summary) > preview_chars:
        preview += "..."
    lines = [
        f"score: {result.get('score'):.4f}",
        f"chunk_id: {payload.get('chunk_id')}",
        f"article: {article_meta.get('title')} ({article_meta.get('year')})",
        f"section: {section_path}",
        f"summary: {preview}",
    ]
    graph_summary = format_graph_summary(result.get("graph_context"))
    if graph_summary:
        lines.append(f"graph: {graph_summary}")
    return "\n".join(lines)


def format_graph_summary(graph_context: Any, *, max_groups: int = 4, max_names_per_group: int = 3) -> str:
    entity_summary = _format_entity_summary(
        graph_context,
        max_groups=max_groups,
        max_names_per_group=max_names_per_group,
    )
    relation_summary = _format_relation_summary(
        graph_context,
        max_groups=max_groups,
        max_names_per_group=max_names_per_group,
    )
    if entity_summary and relation_summary:
        return f"{entity_summary}; relations: {relation_summary}"
    return entity_summary or relation_summary


def _format_entity_summary(graph_context: Any, *, max_groups: int, max_names_per_group: int) -> str:
    if not isinstance(graph_context, dict):
        return ""
    entities = graph_context.get("entities")
    if not isinstance(entities, list) or not entities:
        return ""
    grouped: dict[str, list[str]] = {}
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        label = str(entity.get("label") or "Entity")
        name = str(entity.get("name") or "").strip()
        if not name:
            continue
        grouped.setdefault(label, [])
        if name not in grouped[label]:
            grouped[label].append(name)
    if not grouped:
        return ""
    parts: list[str] = []
    for label in sorted(grouped)[:max_groups]:
        names = grouped[label]
        visible_names = names[:max_names_per_group]
        suffix = ""
        if len(names) > max_names_per_group:
            suffix = f" (+{len(names) - max_names_per_group})"
        parts.append(f"{label}: {', '.join(visible_names)}{suffix}")
    if len(grouped) > max_groups:
        parts.append(f"+{len(grouped) - max_groups} classes")
    return " | ".join(parts)


def _format_relation_summary(graph_context: Any, *, max_groups: int, max_names_per_group: int) -> str:
    if not isinstance(graph_context, dict):
        return ""
    relations = graph_context.get("relations")
    if not isinstance(relations, list) or not relations:
        return ""
    grouped: dict[str, list[str]] = {}
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        relation_type = str(relation.get("type") or "").strip()
        target_name = str(relation.get("target_name") or "").strip()
        if not relation_type or not target_name:
            continue
        grouped.setdefault(relation_type, [])
        if target_name not in grouped[relation_type]:
            grouped[relation_type].append(target_name)
    if not grouped:
        return ""
    parts: list[str] = []
    for relation_type in sorted(grouped)[:max_groups]:
        names = grouped[relation_type]
        visible_names = names[:max_names_per_group]
        suffix = ""
        if len(names) > max_names_per_group:
            suffix = f" (+{len(names) - max_names_per_group})"
        parts.append(f"{relation_type}: {', '.join(visible_names)}{suffix}")
    if len(grouped) > max_groups:
        parts.append(f"+{len(grouped) - max_groups} relation types")
    return " | ".join(parts)
