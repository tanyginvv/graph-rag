from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .retrieval import format_graph_summary


ANSWER_SYSTEM_PROMPT = """Ты работаешь как GraphRAG-ассистент по статьям о time-series LLM.

Отвечай на русском языке.
Отвечай только по переданному контексту.
Если контекста недостаточно, прямо скажи об этом и не выдумывай факты.
Не подменяй факты общими рассуждениями.
Опирайся на текст чанков в первую очередь, а графовый контекст используй только как структурное дополнение.
В конце ответа добавь строку "Использованные чанки:" и перечисли chunk_id через запятую."""


@dataclass(frozen=True)
class AnswerSource:
    chunk_id: str
    article_id: str
    title: str
    section_path: str
    score: float
    question_types: tuple[str, ...]
    text_excerpt: str
    verbatim_text: str
    graph_summary: str


@dataclass(frozen=True)
class AnswerContext:
    system_prompt: str
    user_prompt: str
    sources: list[AnswerSource]


@dataclass(frozen=True)
class AnswerRun:
    question: str
    profile: str
    answer_text: str | None
    graph_enabled: bool
    sources: list[AnswerSource]
    system_prompt: str
    user_prompt: str


def build_answer_context(
    question: str,
    results: list[dict[str, Any]],
    *,
    max_context_chunks: int = 5,
    max_chars_per_chunk: int = 1800,
) -> AnswerContext:
    sources = _build_sources(
        results,
        max_context_chunks=max_context_chunks,
        max_chars_per_chunk=max_chars_per_chunk,
    )
    context_blocks = [_format_source_block(index + 1, source) for index, source in enumerate(sources)]
    user_prompt = (
        f"Вопрос:\n{question.strip()}\n\n"
        f"Контекст:\n\n{'\n\n'.join(context_blocks)}"
    )
    return AnswerContext(
        system_prompt=ANSWER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        sources=sources,
    )


def format_sources_report(sources: list[AnswerSource]) -> str:
    lines = ["Источники:"]
    for source in sources:
        question_types = ", ".join(source.question_types) if source.question_types else "-"
        lines.append(
            f"- {source.chunk_id} | score={source.score:.4f} | {source.section_path} | question_types={question_types}"
        )
    return "\n".join(lines)


def build_answer_run(
    *,
    question: str,
    profile: str,
    graph_enabled: bool,
    answer_context: AnswerContext,
    answer_text: str | None,
) -> AnswerRun:
    normalized_answer_text = _normalize_answer_text(answer_text, answer_context.sources)
    return AnswerRun(
        question=question,
        profile=profile,
        answer_text=normalized_answer_text,
        graph_enabled=graph_enabled,
        sources=answer_context.sources,
        system_prompt=answer_context.system_prompt,
        user_prompt=answer_context.user_prompt,
    )


def answer_run_to_dict(answer_run: AnswerRun, *, include_prompts: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "question": answer_run.question,
        "profile": answer_run.profile,
        "graph_enabled": answer_run.graph_enabled,
        "used_chunk_ids": [source.chunk_id for source in answer_run.sources],
        "sources": [_source_to_dict(source) for source in answer_run.sources],
    }
    if answer_run.answer_text is not None:
        data["answer"] = answer_run.answer_text
    if include_prompts:
        data["system_prompt"] = answer_run.system_prompt
        data["user_prompt"] = answer_run.user_prompt
    return data


def answer_run_to_eval_item(answer_run: AnswerRun, *, question_id: str) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "answer": answer_run.answer_text,
        "chunks": [source.verbatim_text for source in answer_run.sources if source.verbatim_text],
    }


def build_fallback_answer(question: str, sources: list[AnswerSource], *, max_excerpt_chars: int = 280) -> str:
    if not sources:
        return (
            f"По вопросу \"{question.strip()}\" в текущем retrieval-прогоне не найдено "
            "достаточно релевантных фрагментов для содержательного ответа."
        )
    lead_source = sources[0]
    section_path = lead_source.section_path or "неизвестный раздел"
    chunk_ids = ", ".join(source.chunk_id for source in sources)
    return (
        f"По найденным фрагментам основной ответ на вопрос \"{question.strip()}\" "
        f"содержится в разделе \"{section_path}\". Подробное обоснование приложено в поле chunks; "
        "ответ нужно читать вместе с этими verbatim-фрагментами.\n\n"
        f"Использованные чанки: {chunk_ids}"
    )


def format_answer_run(answer_run: AnswerRun) -> str:
    lines: list[str] = []
    if answer_run.answer_text is not None:
        lines.append(answer_run.answer_text)
        lines.append("")
    else:
        lines.append("Вызов LLM не выполнялся.")
        lines.append("")
    lines.append(f"Профиль retrieval: {answer_run.profile}")
    lines.append(f"Graph context: {'on' if answer_run.graph_enabled else 'off'}")
    if answer_run.answer_text is None:
        lines.append("")
        lines.append("Context preview:")
        lines.append(answer_run.user_prompt)
        lines.append("")
    lines.append(format_sources_report(answer_run.sources))
    return "\n".join(lines)


def _build_sources(
    results: list[dict[str, Any]],
    *,
    max_context_chunks: int,
    max_chars_per_chunk: int,
) -> list[AnswerSource]:
    sources: list[AnswerSource] = []
    seen_chunk_ids: set[str] = set()
    for result in results:
        payload = result.get("payload") or {}
        chunk_id = str(payload.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        article_meta = payload.get("article_meta") or {}
        section_hierarchy = payload.get("section_hierarchy") or []
        verbatim_text = str(payload.get("text") or "").strip()
        sources.append(
            AnswerSource(
                chunk_id=chunk_id,
                article_id=str(payload.get("article_id") or "").strip(),
                title=str(article_meta.get("title") or "").strip(),
                section_path=" > ".join(str(item) for item in section_hierarchy if str(item).strip()),
                score=float(result.get("score") or 0.0),
                question_types=tuple(str(item) for item in payload.get("question_types") or []),
                text_excerpt=_trim_text(verbatim_text, max_chars=max_chars_per_chunk),
                verbatim_text=verbatim_text,
                graph_summary=format_graph_summary(result.get("graph_context")),
            )
        )
        if len(sources) >= max_context_chunks:
            break
    return sources


def _format_source_block(index: int, source: AnswerSource) -> str:
    lines = [
        f"[Источник {index}]",
        f"chunk_id: {source.chunk_id}",
        f"article_id: {source.article_id}",
        f"title: {source.title}",
        f"section: {source.section_path or '-'}",
        f"score: {source.score:.4f}",
    ]
    if source.question_types:
        lines.append(f"question_types: {', '.join(source.question_types)}")
    if source.graph_summary:
        lines.append(f"graph: {source.graph_summary}")
    lines.append("text:")
    lines.append(source.text_excerpt)
    return "\n".join(lines)


def _normalize_answer_text(answer_text: str | None, sources: list[AnswerSource]) -> str | None:
    if not answer_text:
        return None
    normalized = answer_text.strip()
    if not normalized:
        return None
    chunk_ids = [source.chunk_id for source in sources if source.chunk_id]
    if not chunk_ids:
        return normalized
    canonical_chunk_line = f"Использованные чанки: {', '.join(chunk_ids)}"
    marker = "Использованные чанки:"
    marker_index = normalized.find(marker)
    if marker_index < 0:
        return normalized
    normalized = normalized[:marker_index].rstrip()
    return f"{normalized}\n\n{canonical_chunk_line}"


def _trim_text(text: str, *, max_chars: int) -> str:
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."


def _source_to_dict(source: AnswerSource) -> dict[str, Any]:
    return {
        "chunk_id": source.chunk_id,
        "article_id": source.article_id,
        "title": source.title,
        "section_path": source.section_path,
        "score": source.score,
        "question_types": list(source.question_types),
        "text_excerpt": source.text_excerpt,
        "graph_summary": source.graph_summary,
    }
