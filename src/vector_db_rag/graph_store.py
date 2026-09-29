from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

from .graph_entities import FIRST_STAGE_ENTITY_LABELS
from .graph_payloads import GraphImportBundle


class MissingNeo4jDriverError(RuntimeError):
    pass


def _import_neo4j() -> Any:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise MissingNeo4jDriverError(
            "neo4j is not installed. Install project dependencies first: pip install -e ."
        ) from exc
    return GraphDatabase


def make_driver(*, uri: str, user: str, password: str) -> Any:
    graph_database = _import_neo4j()
    return graph_database.driver(uri, auth=(user, password))


def ensure_graph_enabled(enabled: bool) -> None:
    if not enabled:
        raise ValueError("Graph layer is disabled in config. Set graph.enabled=true or NEO4J_ENABLED=true.")


def check_neo4j(
    driver: Any,
    *,
    database: str,
    qdrant_chunks: dict[str, str] | None = None,
) -> dict[str, Any]:
    driver.verify_connectivity()
    labels = {
        row["label"]: row["count"]
        for row in _run_read_query(
            driver,
            database=database,
            query=(
                "MATCH (n) "
                "UNWIND labels(n) AS label "
                "RETURN label, count(*) AS count "
                "ORDER BY label"
            ),
        )
    }
    relationships = {
        row["type"]: row["count"]
        for row in _run_read_query(
            driver,
            database=database,
            query="MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type",
        )
    }
    orphan_chunks = _estimate_orphan_chunks(labels, relationships)
    if orphan_chunks is None:
        orphan_chunks = _count_query(
            driver,
            database=database,
            query="MATCH (c:Chunk) WHERE NOT EXISTS { MATCH (:Article)-[:CONTAINS]->(c) } RETURN count(c) AS count",
        )
    orphan_articles = _estimate_orphan_articles(labels, relationships)
    if orphan_articles is None:
        orphan_articles = _count_query(
            driver,
            database=database,
            query="MATCH (a:Article) WHERE NOT EXISTS { MATCH (a)-[:CONTAINS]->(:Chunk) } RETURN count(a) AS count",
        )
    chunks_missing_question_types = 0
    if labels.get("Chunk", 0):
        chunks_missing_question_types = _count_query(
            driver,
            database=database,
            query=(
                "MATCH (c:Chunk) "
                "WHERE c.question_types IS NULL OR size(c.question_types) = 0 "
                "RETURN count(c) AS count"
            ),
        )
    entities_without_mentions = _estimate_entities_without_mentions(labels, relationships)
    if entities_without_mentions is None:
        entities_without_mentions = _count_query(
            driver,
            database=database,
            query=(
                "MATCH (n) "
                "WHERE any(label IN labels(n) WHERE label IN $labels) "
                "AND NOT EXISTS { MATCH (:Chunk)-[:MENTIONS]->(n) } "
                "RETURN count(n) AS count"
            ),
            labels=list(FIRST_STAGE_ENTITY_LABELS),
        )
    report: dict[str, Any] = {
        "database": database,
        "node_labels": labels,
        "relationship_types": relationships,
        "orphan_chunks": orphan_chunks,
        "orphan_articles": orphan_articles,
        "entities_without_mentions": entities_without_mentions,
        "chunks_missing_question_types": chunks_missing_question_types,
    }
    if qdrant_chunks is not None:
        report["qdrant_consistency"] = compare_graph_to_qdrant(
            graph_chunks=list_graph_chunks(driver, database=database),
            qdrant_chunks=qdrant_chunks,
        )
    return report


def _estimate_orphan_chunks(
    labels: dict[str, int],
    relationships: dict[str, int],
) -> int | None:
    chunk_count = int(labels.get("Chunk", 0))
    if chunk_count == 0:
        return 0
    if relationships.get("CONTAINS", 0) == 0:
        return chunk_count
    return None


def _estimate_orphan_articles(
    labels: dict[str, int],
    relationships: dict[str, int],
) -> int | None:
    article_count = int(labels.get("Article", 0))
    if article_count == 0:
        return 0
    if relationships.get("CONTAINS", 0) == 0:
        return article_count
    return None


def _estimate_entities_without_mentions(
    labels: dict[str, int],
    relationships: dict[str, int],
) -> int | None:
    entity_count = sum(int(labels.get(label, 0)) for label in FIRST_STAGE_ENTITY_LABELS)
    if entity_count == 0:
        return 0
    if labels.get("Chunk", 0) == 0:
        return entity_count
    if relationships.get("MENTIONS", 0) == 0:
        return entity_count
    return None


def index_graph_bundle(
    driver: Any,
    *,
    database: str,
    bundle: GraphImportBundle,
    batch_size: int,
) -> dict[str, Any]:
    article_rows_by_id = {row["article_id"]: row for row in bundle.article_rows}
    article_chunk_rows = [
        {**article_rows_by_id[chunk["article_id"]], **chunk}
        for chunk in bundle.chunk_rows
    ]
    cleanup_report = _delete_keyword_mentions(
        driver,
        database=database,
        chunk_ids=[row["chunk_id"] for row in bundle.chunk_rows],
        batch_size=batch_size,
    )
    cleanup_report.update(
        _delete_projected_relations(
            driver,
            database=database,
            article_ids=[row["article_id"] for row in bundle.article_rows],
            batch_size=batch_size,
        )
    )
    _write_batches(
        driver,
        database=database,
        rows=article_chunk_rows,
        batch_size=batch_size,
        query="""
UNWIND $rows AS row
MERGE (a:Article {article_id: row.article_id})
SET a.title = row.title,
    a.authors = row.authors,
    a.year = row.year,
    a.arxiv_id = row.arxiv_id
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.article_id = row.article_id,
    c.section = row.section,
    c.section_hierarchy = row.section_hierarchy,
    c.index = row.index,
    c.chunk_type = row.chunk_type,
    c.question_types = row.question_types,
    c.token_count = row.token_count,
    c.summary = row.summary
MERGE (a)-[:CONTAINS]->(c)
""",
    )
    for label, entity_rows in bundle.entity_rows_by_label.items():
        if not entity_rows:
            continue
        _write_batches(
            driver,
            database=database,
            rows=entity_rows,
            batch_size=batch_size,
            query=f"""
UNWIND $rows AS row
MERGE (e:{label} {{id: row.id}})
SET e.name = row.name,
    e.source = row.source
""",
        )
    if bundle.mention_rows:
        _write_batches(
            driver,
            database=database,
            rows=bundle.mention_rows,
            batch_size=batch_size,
            query="""
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
MATCH (e {id: row.entity_id})
MERGE (c)-[r:MENTIONS]->(e)
SET r.article_id = row.article_id,
    r.source_chunk_id = row.chunk_id,
    r.extraction_source = row.extraction_source,
    r.question_types = row.question_types,
    r.keyword = row.keyword
""",
        )
    cleanup_report["orphan_keyword_entities_deleted"] = _delete_orphan_keyword_entities(
        driver,
        database=database,
    )
    _write_relation_batches(
        driver,
        database=database,
        relation_rows_by_type=bundle.relation_rows_by_type,
        batch_size=batch_size,
    )
    report = bundle.build_report()
    report.update({"database": database, "batch_size": batch_size, **cleanup_report})
    return report


def list_graph_chunks(driver: Any, *, database: str) -> dict[str, str]:
    rows = _run_read_query(
        driver,
        database=database,
        query="MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id, c.article_id AS article_id",
    )
    return {
        str(row["chunk_id"]): str(row["article_id"])
        for row in rows
        if row.get("chunk_id")
    }


def compare_graph_to_qdrant(
    *,
    graph_chunks: dict[str, str],
    qdrant_chunks: dict[str, str],
) -> dict[str, Any]:
    graph_ids = set(graph_chunks)
    qdrant_ids = set(qdrant_chunks)
    missing_in_graph = sorted(qdrant_ids - graph_ids)
    missing_in_qdrant = sorted(graph_ids - qdrant_ids)
    article_mismatches = sorted(
        chunk_id
        for chunk_id in (graph_ids & qdrant_ids)
        if graph_chunks[chunk_id] != qdrant_chunks[chunk_id]
    )
    return {
        "graph_chunks": len(graph_chunks),
        "qdrant_chunks": len(qdrant_chunks),
        "missing_in_graph": missing_in_graph[:20],
        "missing_in_graph_count": len(missing_in_graph),
        "missing_in_qdrant": missing_in_qdrant[:20],
        "missing_in_qdrant_count": len(missing_in_qdrant),
        "article_id_mismatches": article_mismatches[:20],
        "article_id_mismatches_count": len(article_mismatches),
    }


def expand_chunks(
    driver: Any,
    *,
    database: str,
    chunk_ids: list[str],
) -> list[dict[str, Any]]:
    chunk_rows = _run_read_query(
        driver,
        database=database,
        query="""
MATCH (c:Chunk)
WHERE c.chunk_id IN $chunk_ids
OPTIONAL MATCH (c)-[:MENTIONS]->(e)
WITH c, collect(
    DISTINCT CASE
        WHEN e IS NULL THEN NULL
        ELSE {
            entity_id: e.id,
            name: e.name,
            label: head(labels(e))
        }
    END
) AS raw_entities
RETURN c.chunk_id AS chunk_id,
       c.article_id AS article_id,
       c.section AS section,
       c.question_types AS question_types,
       [entity IN raw_entities WHERE entity IS NOT NULL] AS entities
ORDER BY c.article_id, c.index
""",
        chunk_ids=chunk_ids,
    )
    relation_rows = _run_read_query(
        driver,
        database=database,
        query="""
MATCH (c:Chunk)
WHERE c.chunk_id IN $chunk_ids
MATCH (a:Article {article_id: c.article_id})-[r]->(target)
WHERE r.extraction_source = 'article_projection'
  AND (
      r.evidence_chunk_ids IS NULL
      OR size(r.evidence_chunk_ids) = 0
      OR c.chunk_id IN r.evidence_chunk_ids
  )
RETURN c.chunk_id AS chunk_id,
       collect(
           DISTINCT {
               type: type(r),
                source_label: 'Article',
                source_name: coalesce(a.title, a.article_id),
                target_label: head(labels(target)),
                target_name: coalesce(target.name, target.title, target.id),
                evidence_count: coalesce(r.evidence_count, 0)
            }
       ) AS article_relations
""",
        chunk_ids=chunk_ids,
    )
    entity_relation_rows = _run_read_query(
        driver,
        database=database,
        query="""
MATCH (c:Chunk)-[:MENTIONS]->(source)-[r]->(target)
WHERE c.chunk_id IN $chunk_ids
  AND r.extraction_source = 'article_projection'
  AND r.article_id = c.article_id
  AND (
      r.evidence_chunk_ids IS NULL
      OR size(r.evidence_chunk_ids) = 0
      OR c.chunk_id IN r.evidence_chunk_ids
  )
RETURN c.chunk_id AS chunk_id,
       collect(
           DISTINCT {
               type: type(r),
               source_label: head(labels(source)),
               source_name: coalesce(source.name, source.title, source.id),
                target_label: head(labels(target)),
                target_name: coalesce(target.name, target.title, target.id),
                evidence_count: coalesce(r.evidence_count, 0)
            }
       ) AS entity_relations
""",
        chunk_ids=chunk_ids,
    )
    incoming_entity_relation_rows = _run_read_query(
        driver,
        database=database,
        query="""
MATCH (c:Chunk)-[:MENTIONS]->(target)<-[r]-(source)
WHERE c.chunk_id IN $chunk_ids
  AND r.extraction_source = 'article_projection'
  AND r.article_id = c.article_id
  AND (
      r.evidence_chunk_ids IS NULL
      OR size(r.evidence_chunk_ids) = 0
      OR c.chunk_id IN r.evidence_chunk_ids
  )
RETURN c.chunk_id AS chunk_id,
       collect(
           DISTINCT {
               type: type(r),
               source_label: head(labels(source)),
               source_name: coalesce(source.name, source.title, source.id),
                target_label: head(labels(target)),
                target_name: coalesce(target.name, target.title, target.id),
                evidence_count: coalesce(r.evidence_count, 0)
            }
       ) AS incoming_entity_relations
""",
        chunk_ids=chunk_ids,
    )
    relations_by_chunk_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relation_rows:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            continue
        relations_by_chunk_id[chunk_id].extend(_clean_relations(row.get("article_relations")))
    for row in entity_relation_rows:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            continue
        relations_by_chunk_id[chunk_id].extend(_clean_relations(row.get("entity_relations")))
    for row in incoming_entity_relation_rows:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            continue
        relations_by_chunk_id[chunk_id].extend(_clean_relations(row.get("incoming_entity_relations")))

    enriched_rows: list[dict[str, Any]] = []
    for row in chunk_rows:
        chunk_row = dict(row)
        relations = relations_by_chunk_id.get(str(chunk_row["chunk_id"]), [])
        if relations:
            chunk_row["relations"] = _dedupe_relations(relations)
        enriched_rows.append(chunk_row)
    return enriched_rows


def _write_batches(
    driver: Any,
    *,
    database: str,
    rows: list[dict[str, Any]],
    batch_size: int,
    query: str,
) -> None:
    with driver.session(database=database) as session:
        for batch in _iter_batches(rows, batch_size):
            session.run(query, rows=batch).consume()


def _run_read_query(
    driver: Any,
    *,
    database: str,
    query: str,
    **params: Any,
) -> list[dict[str, Any]]:
    with driver.session(database=database) as session:
        result = session.run(query, **params)
        return [dict(record) for record in result]


def _count_query(
    driver: Any,
    *,
    database: str,
    query: str,
    **params: Any,
) -> int:
    rows = _run_read_query(driver, database=database, query=query, **params)
    return int(rows[0]["count"]) if rows else 0


def _iter_batches(rows: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_relation_batches(
    driver: Any,
    *,
    database: str,
    relation_rows_by_type: dict[str, list[dict[str, Any]]],
    batch_size: int,
) -> None:
    grouped_rows: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for relation_type, rows in relation_rows_by_type.items():
        for row in rows:
            grouped_rows[
                (
                    relation_type,
                    str(row["source_label"]),
                    str(row["source_key"]),
                    str(row["target_label"]),
                    str(row["target_key"]),
                )
            ].append(row)
    for (relation_type, source_label, source_key, target_label, target_key), rows in grouped_rows.items():
        _write_batches(
            driver,
            database=database,
            rows=rows,
            batch_size=batch_size,
            query=f"""
UNWIND $rows AS row
MATCH (source:{source_label} {{{source_key}: row.source_value}})
MATCH (target:{target_label} {{{target_key}: row.target_value}})
MERGE (source)-[r:{relation_type} {{article_id: row.article_id}}]->(target)
SET r.extraction_source = row.extraction_source,
    r.evidence_chunk_ids = row.evidence_chunk_ids,
    r.evidence_sections = row.evidence_sections,
    r.evidence_question_types = row.evidence_question_types,
    r.evidence_count = row.evidence_count
""",
        )


def _delete_keyword_mentions(
    driver: Any,
    *,
    database: str,
    chunk_ids: list[str],
    batch_size: int,
) -> dict[str, int]:
    deleted = 0
    with driver.session(database=database) as session:
        for batch in _iter_batches(
            [{"chunk_id": chunk_id} for chunk_id in chunk_ids],
            batch_size,
        ):
            summary = session.run(
                """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})-[r:MENTIONS]->()
WHERE r.extraction_source = 'keywords'
DELETE r
""",
                rows=batch,
            ).consume()
            deleted += int(summary.counters.relationships_deleted)
    return {"previous_keyword_mentions_deleted": deleted}


def _delete_projected_relations(
    driver: Any,
    *,
    database: str,
    article_ids: list[str],
    batch_size: int,
) -> dict[str, int]:
    deleted = 0
    with driver.session(database=database) as session:
        for batch in _iter_batches(
            [{"article_id": article_id} for article_id in article_ids],
            batch_size,
        ):
            summary = session.run(
                """
UNWIND $rows AS row
MATCH ()-[r]->()
WHERE r.extraction_source = 'article_projection'
  AND r.article_id = row.article_id
DELETE r
""",
                rows=batch,
            ).consume()
            deleted += int(summary.counters.relationships_deleted)
    return {"previous_projected_relations_deleted": deleted}


def _delete_orphan_keyword_entities(
    driver: Any,
    *,
    database: str,
) -> int:
    deleted = 0
    with driver.session(database=database) as session:
        for label in FIRST_STAGE_ENTITY_LABELS:
            summary = session.run(
                f"""
MATCH (e:{label})
WHERE e.source = 'keywords'
  AND NOT EXISTS {{ MATCH (e)--() }}
DELETE e
""",
            ).consume()
            deleted += int(summary.counters.nodes_deleted)
    return deleted


def _clean_relations(raw_relations: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_relations, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for relation in raw_relations:
        if not isinstance(relation, dict):
            continue
        target_name = str(relation.get("target_name") or "").strip()
        if not target_name:
            continue
        cleaned.append(
            {
                "type": str(relation.get("type") or ""),
                "source_label": str(relation.get("source_label") or ""),
                "source_name": str(relation.get("source_name") or "").strip(),
                "target_label": str(relation.get("target_label") or ""),
                "target_name": target_name,
                "evidence_count": int(relation.get("evidence_count") or 0),
            }
        )
    return cleaned


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for relation in relations:
        key = (
            relation["type"],
            relation["source_label"],
            relation["source_name"],
            relation["target_label"],
            relation["target_name"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(relation)
    return sorted(
        deduped,
        key=lambda relation: (
            relation["type"],
            relation["source_label"],
            relation["source_name"],
            relation["target_label"],
            relation["target_name"],
        ),
    )
