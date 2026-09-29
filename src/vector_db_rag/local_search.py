"""Small reproducible BM25 baseline over the checked-in chunk corpus."""

from __future__ import annotations

import math
from collections import Counter

from .chunks import ChunkFile
from .sparse import tokenize
from .text_fields import build_sparse_text

STOPWORDS = frozenset({
    "a", "an", "and", "are", "by", "does", "for", "how", "in", "is", "of",
    "on", "the", "to", "use", "uses", "what", "which", "with",
    "в", "и", "из", "как", "на", "по", "с", "что",
})


def search_local(
    corpus: ChunkFile,
    query: str,
    *,
    limit: int = 5,
    article_id: str | None = None,
) -> list[dict[str, object]]:
    """Rank chunks with BM25 and return their source identifiers and evidence."""
    if limit < 1:
        raise ValueError("limit must be positive")
    query_terms = set(tokenize(query)) - STOPWORDS
    if not query_terms:
        raise ValueError("query must contain searchable words")

    documents = [
        (chunk, Counter(tokenize(build_sparse_text(chunk.model_dump()))))
        for chunk in corpus.chunks
    ]
    if not documents:
        return []

    document_count = len(documents)
    document_frequency = Counter(
        term for _, counts in documents for term in query_terms if term in counts
    )
    average_length = sum(sum(counts.values()) for _, counts in documents) / document_count
    results: list[dict[str, object]] = []

    for chunk, counts in documents:
        if article_id and chunk.article_id != article_id:
            continue
        document_length = sum(counts.values())
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log1p(
                (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            normalized_length = 1 - 0.75 + 0.75 * document_length / max(average_length, 1)
            score += inverse_frequency * frequency * 2.5 / (frequency + 1.5 * normalized_length)
            if term in set(tokenize(chunk.article_meta.title)):
                score += 5.0 * inverse_frequency
        if score:
            results.append(
                {
                    "score": round(score, 6),
                    "chunk_id": chunk.chunk_id,
                    "article_id": chunk.article_id,
                    "title": chunk.article_meta.title,
                    "section": " > ".join(chunk.section_hierarchy),
                    "summary": chunk.summary,
                    "text_excerpt": chunk.text[:700],
                }
            )

    results.sort(key=lambda item: (-float(item["score"]), str(item["chunk_id"])))
    return results[:limit]
