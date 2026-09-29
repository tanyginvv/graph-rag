"""Deterministic GraphRAG demonstration over the local research corpus.

The graph is bipartite: Chunk --MENTIONS--> Keyword. Search supplies seed
chunks; shared keywords add neighboring chunks; every statement is copied
from a cited source chunk. No external service or language model is needed.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from .chunks import Chunk, ChunkFile
from .local_search import search_local
from .sparse import tokenize


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    article_id: str
    title: str
    section: str
    retrieval: str
    via_keyword: str | None
    excerpt: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "title": self.title,
            "section": self.section,
            "retrieval": self.retrieval,
            "via_keyword": self.via_keyword,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class OfflineAnswer:
    question: str
    answer: str
    sources: list[Evidence]
    graph_edges_examined: int

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [source.as_dict() for source in self.sources],
            "graph_edges_examined": self.graph_edges_examined,
        }


def _keyword_graph(corpus: ChunkFile) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = defaultdict(list)
    for chunk in corpus.chunks:
        for keyword in {word.strip().casefold() for word in chunk.keywords if word.strip()}:
            graph[keyword].append(chunk.chunk_id)
    return graph


def _related_chunks(
    corpus: ChunkFile,
    seed_ids: set[str],
    *,
    max_keyword_degree: int = 20,
) -> tuple[list[tuple[str, str, float]], int]:
    """Return neighbors with an inverse-degree score and their connecting keyword."""
    by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    graph = _keyword_graph(corpus)
    scores: dict[str, float] = defaultdict(float)
    bridges: dict[str, tuple[str, float]] = {}
    examined = 0
    for seed_id in seed_ids:
        seed = by_id[seed_id]
        for keyword in {word.strip().casefold() for word in seed.keywords if word.strip()}:
            neighbors = graph[keyword]
            if not 1 < len(neighbors) <= max_keyword_degree:
                continue
            weight = math.log1p(len(corpus.chunks) / len(neighbors))
            for neighbor_id in neighbors:
                if neighbor_id in seed_ids:
                    continue
                examined += 1
                scores[neighbor_id] += weight
                if neighbor_id not in bridges or weight > bridges[neighbor_id][1]:
                    bridges[neighbor_id] = (keyword, weight)
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [(chunk_id, bridges[chunk_id][0], scores[chunk_id]) for chunk_id in ranked], examined


def _best_excerpt(chunk: Chunk, question: str, *, max_chars: int = 420) -> str:
    text = re.sub(r"(?m)^#{1,6}\s+.*$", "", chunk.text).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return chunk.text[:max_chars]
    query_terms = set(tokenize(question))
    best = max(
        enumerate(sentences),
        key=lambda pair: (len(set(tokenize(pair[1])) & query_terms), -pair[0]),
    )[1]
    return best[:max_chars]


def answer_offline(
    corpus: ChunkFile,
    question: str,
    *,
    max_sources: int = 4,
    expand_graph: bool = True,
    article_id: str | None = None,
) -> OfflineAnswer:
    if max_sources < 1:
        raise ValueError("max_sources must be positive")
    lexical = search_local(corpus, question, limit=min(max_sources, 2), article_id=article_id)
    seed_ids = {str(row["chunk_id"]) for row in lexical}
    by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    chosen: list[tuple[str, str, str | None]] = [
        (str(row["chunk_id"]), "BM25", None) for row in lexical
    ]
    examined = 0
    if expand_graph and seed_ids and len(chosen) < max_sources:
        neighbors, examined = _related_chunks(corpus, seed_ids)
        for chunk_id, keyword, _ in neighbors:
            if article_id and by_id[chunk_id].article_id != article_id:
                continue
            chosen.append((chunk_id, "keyword graph", keyword))
            if len(chosen) == max_sources:
                break

    sources = [
        Evidence(
            chunk_id=chunk_id,
            article_id=by_id[chunk_id].article_id,
            title=by_id[chunk_id].article_meta.title,
            section=" > ".join(by_id[chunk_id].section_hierarchy),
            retrieval=method,
            via_keyword=keyword,
            excerpt=_best_excerpt(by_id[chunk_id], question),
        )
        for chunk_id, method, keyword in chosen
    ]
    if not sources:
        answer = "В корпусе не найдены фрагменты, подтверждающие ответ."
    else:
        answer = "По найденным статьям:\n" + "\n".join(
            f"- {source.excerpt} [{source.chunk_id}]" for source in sources
        )
    return OfflineAnswer(question, answer, sources, examined)
