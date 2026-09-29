from __future__ import annotations

from math import sqrt
from typing import Any

from .answering import AnswerSource


def compute_answer_proxy_metrics(
    *,
    question: str,
    answer_text: str | None,
    sources: list[AnswerSource],
    embedder: Any | None = None,
    max_context_chars: int = 6000,
) -> dict[str, Any]:
    normalized_answer = (answer_text or "").strip()
    used_chunk_ids = [source.chunk_id for source in sources if source.chunk_id]
    referenced_chunk_id_count = sum(1 for chunk_id in used_chunk_ids if chunk_id in normalized_answer)
    context_text = _build_metric_context_text(sources, max_context_chars=max_context_chars)

    metrics: dict[str, Any] = {
        "answer_present": bool(normalized_answer),
        "answer_length_chars": len(normalized_answer),
        "source_count": len(sources),
        "has_used_chunks_line": "Использованные чанки:" in normalized_answer,
        "referenced_chunk_id_count": referenced_chunk_id_count,
        "referenced_chunk_id_rate": (
            referenced_chunk_id_count / len(used_chunk_ids) if used_chunk_ids else 0.0
        ),
        "question_answer_similarity": None,
        "answer_context_similarity": None,
        "question_context_similarity": None,
        "grounded_answer_score": None,
    }

    normalized_question = question.strip()
    if not embedder or not normalized_answer:
        return metrics

    embedding_inputs: list[str] = []
    question_index: int | None = None
    answer_index: int | None = None
    context_index: int | None = None
    if normalized_question:
        question_index = len(embedding_inputs)
        embedding_inputs.append(normalized_question)
    answer_index = len(embedding_inputs)
    embedding_inputs.append(normalized_answer)
    if context_text:
        context_index = len(embedding_inputs)
        embedding_inputs.append(context_text)

    vectors = embedder.embed_texts(embedding_inputs)
    question_vector = vectors[question_index] if question_index is not None else None
    answer_vector = vectors[answer_index]
    context_vector = vectors[context_index] if context_index is not None else None

    if question_vector is not None:
        metrics["question_answer_similarity"] = _cosine_similarity(question_vector, answer_vector)
    if context_vector is not None:
        metrics["answer_context_similarity"] = _cosine_similarity(answer_vector, context_vector)
    if question_vector is not None and context_vector is not None:
        metrics["question_context_similarity"] = _cosine_similarity(question_vector, context_vector)

    question_answer_similarity = metrics["question_answer_similarity"]
    answer_context_similarity = metrics["answer_context_similarity"]
    if question_answer_similarity is not None and answer_context_similarity is not None:
        metrics["grounded_answer_score"] = sqrt(
            max(float(question_answer_similarity), 0.0) * max(float(answer_context_similarity), 0.0)
        )
    return metrics


def aggregate_answer_proxy_metrics(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics_rows:
        return {
            "question_count": 0,
            "answer_presence_rate": 0.0,
            "used_chunks_line_rate": 0.0,
            "mean_referenced_chunk_id_rate": 0.0,
            "mean_question_answer_similarity": None,
            "mean_answer_context_similarity": None,
            "mean_question_context_similarity": None,
            "mean_grounded_answer_score": None,
        }

    return {
        "question_count": len(metrics_rows),
        "answer_presence_rate": _mean_bool(metrics_rows, "answer_present"),
        "used_chunks_line_rate": _mean_bool(metrics_rows, "has_used_chunks_line"),
        "mean_referenced_chunk_id_rate": _mean_number(metrics_rows, "referenced_chunk_id_rate"),
        "mean_question_answer_similarity": _mean_number(metrics_rows, "question_answer_similarity"),
        "mean_answer_context_similarity": _mean_number(metrics_rows, "answer_context_similarity"),
        "mean_question_context_similarity": _mean_number(metrics_rows, "question_context_similarity"),
        "mean_grounded_answer_score": _mean_number(metrics_rows, "grounded_answer_score"),
    }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension.")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    return dot_product / (left_norm * right_norm)


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 for row in rows if row.get(key)) / len(rows)


def _mean_number(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _build_metric_context_text(sources: list[AnswerSource], *, max_context_chars: int) -> str:
    parts: list[str] = []
    remaining = max_context_chars
    for source in sources:
        text = source.verbatim_text.strip() or source.text_excerpt.strip()
        if not text or remaining <= 0:
            continue
        if len(text) > remaining:
            parts.append(text[:remaining].rstrip())
            break
        parts.append(text)
        remaining -= len(text)
        if remaining > 2:
            remaining -= 2
    return "\n\n".join(parts)