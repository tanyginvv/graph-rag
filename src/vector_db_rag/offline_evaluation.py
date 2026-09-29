"""Small reproducible retrieval and evidence evaluation for the local corpus."""

from __future__ import annotations

import json
from pathlib import Path

from .chunks import ChunkFile
from .local_search import search_local
from .offline_graph import answer_offline


def evaluate_offline(corpus: ChunkFile, questions_path: str | Path) -> dict[str, object]:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation questions must be a nonempty JSON list")
    by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    rows: list[dict[str, object]] = []
    for item in questions:
        question = str(item["question"])
        expected = str(item["article_id"])
        results = search_local(corpus, question, limit=3)
        answer = answer_offline(corpus, question, max_sources=4)
        evidence_valid = all(
            source.excerpt in by_id[source.chunk_id].text
            for source in answer.sources
        )
        rows.append(
            {
                "question": question,
                "expected_article_id": expected,
                "top_3_article_ids": [result["article_id"] for result in results],
                "hit_at_1": bool(results and results[0]["article_id"] == expected),
                "hit_at_3": any(result["article_id"] == expected for result in results),
                "cited_chunk_ids": [source.chunk_id for source in answer.sources],
                "evidence_verbatim": evidence_valid,
            }
        )
    count = len(rows)
    return {
        "questions": count,
        "hit_at_1": sum(bool(row["hit_at_1"]) for row in rows) / count,
        "hit_at_3": sum(bool(row["hit_at_3"]) for row in rows) / count,
        "verbatim_evidence_rate": sum(bool(row["evidence_verbatim"]) for row in rows) / count,
        "details": rows,
    }
