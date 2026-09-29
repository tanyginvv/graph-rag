from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .chunks import ChunkFile
from .graph_entities import FIRST_STAGE_ENTITY_LABELS, build_entity_mention

DERIVED_RELATION_TYPES = (
    "PROPOSES",
    "EVALUATES",
    "SURVEYS",
    "ADDRESSES",
    "CONDUCTS",
    "HAS_ARCHITECTURE",
    "HAS_TUNING",
    "USES_TECHNIQUE",
    "USES_PREPROCESSING",
    "HAS_PIPELINE",
    "FOR_TASK",
    "ON_DATASET",
    "USES_BENCHMARK",
    "HAS_RESULT",
    "MEASURED_BY",
    "OF_METHOD",
    "BELONGS_TO",
)

METHOD_SECTION_NAMES = frozenset({"Methodology"})
EVALUATION_SECTION_NAMES = frozenset({"Experiments", "Appendix"})


@dataclass(frozen=True)
class GraphImportBundle:
    article_rows: list[dict[str, Any]]
    chunk_rows: list[dict[str, Any]]
    entity_rows_by_label: dict[str, list[dict[str, Any]]]
    mention_rows: list[dict[str, Any]]
    relation_rows_by_type: dict[str, list[dict[str, Any]]]
    unresolved_keywords: list[dict[str, str]]

    def build_report(self) -> dict[str, Any]:
        unresolved_counter = Counter(item["keyword"] for item in self.unresolved_keywords)
        return {
            "articles": len(self.article_rows),
            "chunks": len(self.chunk_rows),
            "entity_counts": {
                label: len(rows)
                for label, rows in self.entity_rows_by_label.items()
                if rows
            },
            "mentions": len(self.mention_rows),
            "relation_counts": {
                relation_type: len(rows)
                for relation_type, rows in self.relation_rows_by_type.items()
                if rows
            },
            "unresolved_keywords": len(self.unresolved_keywords),
            "unresolved_keywords_top": [
                {"keyword": keyword, "count": count}
                for keyword, count in unresolved_counter.most_common(20)
            ],
            "unresolved_keyword_examples": self.unresolved_keywords[:20],
        }


def build_graph_import_bundle(chunk_file: ChunkFile) -> GraphImportBundle:
    article_rows_by_id: dict[str, dict[str, Any]] = {}
    entity_rows_by_label: dict[str, dict[str, dict[str, Any]]] = {
        label: {} for label in FIRST_STAGE_ENTITY_LABELS
    }
    chunk_contexts_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mention_keys: set[tuple[str, str]] = set()
    mention_rows: list[dict[str, Any]] = []
    unresolved_keywords: list[dict[str, str]] = []
    chunk_rows: list[dict[str, Any]] = []

    for chunk in chunk_file.chunks:
        article_rows_by_id.setdefault(
            chunk.article_id,
            {
                "article_id": chunk.article_id,
                "title": chunk.article_meta.title,
                "authors": chunk.article_meta.authors,
                "year": chunk.article_meta.year,
                "arxiv_id": chunk.article_meta.arxiv_id,
            },
        )
        chunk_rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "article_id": chunk.article_id,
                "section": chunk.section,
                "section_hierarchy": chunk.section_hierarchy,
                "index": chunk.index,
                "chunk_type": chunk.chunk_type,
                "question_types": chunk.question_types,
                "token_count": chunk.token_count,
                "summary": chunk.summary,
            }
        )
        chunk_context = {
            "chunk_id": chunk.chunk_id,
            "article_id": chunk.article_id,
            "section": chunk.section,
            "question_types": list(chunk.question_types),
            "entity_ids_by_label": {},
        }
        chunk_contexts_by_article[chunk.article_id].append(chunk_context)
        for keyword in chunk.keywords:
            mention = build_entity_mention(keyword)
            if mention is None:
                unresolved_keywords.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "article_id": chunk.article_id,
                        "keyword": keyword,
                    }
                )
                continue
            entity_rows_by_label[mention.label].setdefault(
                mention.entity_id,
                {
                    "id": mention.entity_id,
                    "name": mention.name,
                    "source": mention.source,
                },
            )
            mention_key = (chunk.chunk_id, mention.entity_id)
            if mention_key in mention_keys:
                continue
            mention_keys.add(mention_key)
            mention_row = {
                "chunk_id": chunk.chunk_id,
                "article_id": chunk.article_id,
                "entity_id": mention.entity_id,
                "entity_label": mention.label,
                "entity_name": mention.name,
                "extraction_source": mention.source,
                "question_types": chunk.question_types,
                "keyword": keyword,
            }
            mention_rows.append(mention_row)
            entity_ids = chunk_context["entity_ids_by_label"].setdefault(mention.label, [])
            if mention.entity_id not in entity_ids:
                entity_ids.append(mention.entity_id)

    relation_rows_by_type = _build_relation_rows_by_type(
        article_rows_by_id=article_rows_by_id,
        chunk_contexts_by_article=chunk_contexts_by_article,
    )

    return GraphImportBundle(
        article_rows=_sorted_rows(article_rows_by_id.values(), "article_id"),
        chunk_rows=_sorted_rows(chunk_rows, "chunk_id"),
        entity_rows_by_label={
            label: _sorted_rows(rows.values(), "id")
            for label, rows in entity_rows_by_label.items()
        },
        mention_rows=_sorted_rows(mention_rows, "chunk_id", "entity_id"),
        relation_rows_by_type={
            relation_type: _sorted_rows(
                rows,
                "article_id",
                "source_label",
                "source_value",
                "target_label",
                "target_value",
            )
            for relation_type, rows in relation_rows_by_type.items()
        },
        unresolved_keywords=_sorted_rows(unresolved_keywords, "chunk_id", "keyword"),
    )


def _sorted_rows(rows: Any, *keys: str) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: tuple(str(row.get(key, "")) for key in keys),
    )


def _build_relation_rows_by_type(
    *,
    article_rows_by_id: dict[str, dict[str, Any]],
    chunk_contexts_by_article: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    relation_rows_by_type: dict[str, list[dict[str, Any]]] = {
        relation_type: [] for relation_type in DERIVED_RELATION_TYPES
    }
    relation_row_lookup: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for article_id, article in article_rows_by_id.items():
        chunk_contexts = chunk_contexts_by_article.get(article_id, [])
        if not chunk_contexts:
            continue
        ids_by_label = _group_entity_ids_by_label(chunk_contexts)
        primary_method_ids = _infer_primary_method_ids(
            article_id=article_id,
            article_title=str(article["title"]),
            method_ids=ids_by_label.get("Method", []),
        )
        is_survey_article = "survey" in article_id or "survey" in str(article["title"]).lower()

        for method_id in primary_method_ids:
            _append_relation(
                relation_rows_by_type,
                relation_row_lookup,
                relation_type="PROPOSES",
                article_id=article_id,
                source_label="Article",
                source_key="article_id",
                source_value=article_id,
                target_label="Method",
                target_key="id",
                target_value=method_id,
            )
        for chunk_context in chunk_contexts:
            labels_in_chunk = chunk_context["entity_ids_by_label"]
            evidence = _relation_evidence(chunk_context)

            for task_id in labels_in_chunk.get("Task", []):
                _append_relation(
                    relation_rows_by_type,
                    relation_row_lookup,
                    relation_type="ADDRESSES",
                    article_id=article_id,
                    source_label="Article",
                    source_key="article_id",
                    source_value=article_id,
                    target_label="Task",
                    target_key="id",
                    target_value=task_id,
                    **evidence,
                )
            for experiment_id in labels_in_chunk.get("Experiment", []):
                _append_relation(
                    relation_rows_by_type,
                    relation_row_lookup,
                    relation_type="CONDUCTS",
                    article_id=article_id,
                    source_label="Article",
                    source_key="article_id",
                    source_value=article_id,
                    target_label="Experiment",
                    target_key="id",
                    target_value=experiment_id,
                    **evidence,
                )
            if is_survey_article:
                for category_id in labels_in_chunk.get("TaxonomyCategory", []):
                    _append_relation(
                        relation_rows_by_type,
                        relation_row_lookup,
                        relation_type="SURVEYS",
                        article_id=article_id,
                        source_label="Article",
                        source_key="article_id",
                        source_value=article_id,
                        target_label="TaxonomyCategory",
                        target_key="id",
                        target_value=category_id,
                        **evidence,
                    )
            if _chunk_supports_evaluation_projection(chunk_context):
                for method_id in labels_in_chunk.get("Method", []):
                    if method_id in primary_method_ids:
                        continue
                    _append_relation(
                        relation_rows_by_type,
                        relation_row_lookup,
                        relation_type="EVALUATES",
                        article_id=article_id,
                        source_label="Article",
                        source_key="article_id",
                        source_value=article_id,
                        target_label="Method",
                        target_key="id",
                        target_value=method_id,
                        **evidence,
                    )
            if _chunk_supports_method_projection(chunk_context):
                for method_id in primary_method_ids:
                    _append_article_scope_relations(
                        relation_rows_by_type,
                        relation_row_lookup,
                        article_id=article_id,
                        source_label="Method",
                        source_value=method_id,
                        source_key="id",
                        relation_type="HAS_ARCHITECTURE",
                        target_label="Architecture",
                        target_values=labels_in_chunk.get("Architecture", []),
                        **evidence,
                    )
                    _append_article_scope_relations(
                        relation_rows_by_type,
                        relation_row_lookup,
                        article_id=article_id,
                        source_label="Method",
                        source_value=method_id,
                        source_key="id",
                        relation_type="HAS_TUNING",
                        target_label="TuningStrategy",
                        target_values=labels_in_chunk.get("TuningStrategy", []),
                        **evidence,
                    )
                    _append_article_scope_relations(
                        relation_rows_by_type,
                        relation_row_lookup,
                        article_id=article_id,
                        source_label="Method",
                        source_value=method_id,
                        source_key="id",
                        relation_type="USES_TECHNIQUE",
                        target_label="Technique",
                        target_values=labels_in_chunk.get("Technique", []),
                        **evidence,
                    )
                    _append_article_scope_relations(
                        relation_rows_by_type,
                        relation_row_lookup,
                        article_id=article_id,
                        source_label="Method",
                        source_value=method_id,
                        source_key="id",
                        relation_type="USES_PREPROCESSING",
                        target_label="DataPreprocessing",
                        target_values=labels_in_chunk.get("DataPreprocessing", []),
                        **evidence,
                    )
                    _append_article_scope_relations(
                        relation_rows_by_type,
                        relation_row_lookup,
                        article_id=article_id,
                        source_label="Method",
                        source_value=method_id,
                        source_key="id",
                        relation_type="HAS_PIPELINE",
                        target_label="Pipeline",
                        target_values=labels_in_chunk.get("Pipeline", []),
                        **evidence,
                    )
                    _append_article_scope_relations(
                        relation_rows_by_type,
                        relation_row_lookup,
                        article_id=article_id,
                        source_label="Method",
                        source_value=method_id,
                        source_key="id",
                        relation_type="BELONGS_TO",
                        target_label="TaxonomyCategory",
                        target_values=labels_in_chunk.get("TaxonomyCategory", []),
                        **evidence,
                    )
            for experiment_id in labels_in_chunk.get("Experiment", []):
                _append_article_scope_relations(
                    relation_rows_by_type,
                    relation_row_lookup,
                    article_id=article_id,
                    source_label="Experiment",
                    source_value=experiment_id,
                    source_key="id",
                    relation_type="FOR_TASK",
                    target_label="Task",
                    target_values=labels_in_chunk.get("Task", []),
                    **evidence,
                )
                _append_article_scope_relations(
                    relation_rows_by_type,
                    relation_row_lookup,
                    article_id=article_id,
                    source_label="Experiment",
                    source_value=experiment_id,
                    source_key="id",
                    relation_type="ON_DATASET",
                    target_label="Dataset",
                    target_values=labels_in_chunk.get("Dataset", []),
                    **evidence,
                )
                _append_article_scope_relations(
                    relation_rows_by_type,
                    relation_row_lookup,
                    article_id=article_id,
                    source_label="Experiment",
                    source_value=experiment_id,
                    source_key="id",
                    relation_type="USES_BENCHMARK",
                    target_label="Benchmark",
                    target_values=labels_in_chunk.get("Benchmark", []),
                    **evidence,
                )
                _append_article_scope_relations(
                    relation_rows_by_type,
                    relation_row_lookup,
                    article_id=article_id,
                    source_label="Experiment",
                    source_value=experiment_id,
                    source_key="id",
                    relation_type="HAS_RESULT",
                    target_label="ExperimentResult",
                    target_values=labels_in_chunk.get("ExperimentResult", []),
                    **evidence,
                )
            for result_id in labels_in_chunk.get("ExperimentResult", []):
                _append_article_scope_relations(
                    relation_rows_by_type,
                    relation_row_lookup,
                    article_id=article_id,
                    source_label="ExperimentResult",
                    source_value=result_id,
                    source_key="id",
                    relation_type="MEASURED_BY",
                    target_label="Metric",
                    target_values=labels_in_chunk.get("Metric", []),
                    **evidence,
                )
                _append_article_scope_relations(
                    relation_rows_by_type,
                    relation_row_lookup,
                    article_id=article_id,
                    source_label="ExperimentResult",
                    source_value=result_id,
                    source_key="id",
                    relation_type="OF_METHOD",
                    target_label="Method",
                    target_values=primary_method_ids,
                    **evidence,
                )
            for dataset_id in labels_in_chunk.get("Dataset", []):
                _append_article_scope_relations(
                    relation_rows_by_type,
                    relation_row_lookup,
                    article_id=article_id,
                    source_label="Dataset",
                    source_value=dataset_id,
                    source_key="id",
                    relation_type="BELONGS_TO",
                    target_label="Domain",
                    target_values=labels_in_chunk.get("Domain", []),
                    **evidence,
                )
    return relation_rows_by_type


def _group_entity_ids_by_label(chunk_contexts: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for chunk_context in chunk_contexts:
        for label, entity_ids in chunk_context["entity_ids_by_label"].items():
            grouped.setdefault(str(label), [])
            for entity_id in entity_ids:
                entity_id = str(entity_id)
                if entity_id not in grouped[str(label)]:
                    grouped[str(label)].append(entity_id)
    return grouped


def _infer_primary_method_ids(
    *,
    article_id: str,
    article_title: str,
    method_ids: list[str],
) -> list[str]:
    if not method_ids:
        return []
    exact_id = f"Method:{article_id}"
    if exact_id in method_ids:
        return [exact_id]
    if "survey" in article_id or "survey" in article_title.lower():
        return []
    normalized_title = _normalize_for_match(article_title)
    title_matches = [
        method_id for method_id in method_ids
        if _normalize_for_match(method_id.removeprefix("Method:")) in normalized_title
    ]
    if len(title_matches) == 1:
        return title_matches
    if len(method_ids) == 1:
        return list(method_ids)
    return []


def _append_article_scope_relations(
    relation_rows_by_type: dict[str, list[dict[str, Any]]],
    relation_row_lookup: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
    *,
    article_id: str,
    source_label: str,
    source_key: str,
    source_value: str,
    relation_type: str,
    target_label: str,
    target_values: list[str],
    evidence_chunk_id: str | None = None,
    evidence_section: str | None = None,
    evidence_question_types: list[str] | None = None,
) -> None:
    for target_value in target_values:
        _append_relation(
            relation_rows_by_type,
            relation_row_lookup,
            relation_type=relation_type,
            article_id=article_id,
            source_label=source_label,
            source_key=source_key,
            source_value=source_value,
            target_label=target_label,
            target_key="id",
            target_value=target_value,
            evidence_chunk_id=evidence_chunk_id,
            evidence_section=evidence_section,
            evidence_question_types=evidence_question_types,
        )


def _append_relation(
    relation_rows_by_type: dict[str, list[dict[str, Any]]],
    relation_row_lookup: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
    *,
    relation_type: str,
    article_id: str,
    source_label: str,
    source_key: str,
    source_value: str,
    target_label: str,
    target_key: str,
    target_value: str,
    evidence_chunk_id: str | None = None,
    evidence_section: str | None = None,
    evidence_question_types: list[str] | None = None,
) -> None:
    relation_key = (
        article_id,
        relation_type,
        source_label,
        source_value,
        target_label,
        target_value,
    )
    row = relation_row_lookup.get(relation_key)
    if row is None:
        row = {
            "article_id": article_id,
            "source_label": source_label,
            "source_key": source_key,
            "source_value": source_value,
            "target_label": target_label,
            "target_key": target_key,
            "target_value": target_value,
            "extraction_source": "article_projection",
            "evidence_chunk_ids": [],
            "evidence_sections": [],
            "evidence_question_types": [],
            "evidence_count": 0,
        }
        relation_row_lookup[relation_key] = row
        relation_rows_by_type[relation_type].append(row)
    _append_evidence_value(row["evidence_chunk_ids"], evidence_chunk_id)
    _append_evidence_value(row["evidence_sections"], evidence_section)
    for question_type in evidence_question_types or []:
        _append_evidence_value(row["evidence_question_types"], question_type)
    row["evidence_count"] = len(row["evidence_chunk_ids"])


def _normalize_for_match(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in value)


def _relation_evidence(chunk_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_chunk_id": str(chunk_context["chunk_id"]),
        "evidence_section": str(chunk_context["section"]),
        "evidence_question_types": list(chunk_context["question_types"]),
    }


def _chunk_supports_method_projection(chunk_context: dict[str, Any]) -> bool:
    if str(chunk_context["section"]) in METHOD_SECTION_NAMES:
        return True
    labels = chunk_context["entity_ids_by_label"]
    return any(
        labels.get(label)
        for label in ("Architecture", "TuningStrategy", "DataPreprocessing", "Pipeline")
    )


def _chunk_supports_evaluation_projection(chunk_context: dict[str, Any]) -> bool:
    if str(chunk_context["section"]) in EVALUATION_SECTION_NAMES:
        return True
    labels = chunk_context["entity_ids_by_label"]
    return any(
        labels.get(label)
        for label in ("Experiment", "ExperimentResult", "Dataset", "Benchmark", "Metric")
    )


def _append_evidence_value(values: list[str], value: str | None) -> None:
    if value is None:
        return
    cleaned = value.strip()
    if cleaned and cleaned not in values:
        values.append(cleaned)
