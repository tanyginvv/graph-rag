from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from .answering import (
    AnswerRun,
    answer_run_to_dict,
    answer_run_to_eval_item,
    build_fallback_answer,
    build_answer_context,
    build_answer_run,
    format_answer_run,
)
from .answer_metrics import aggregate_answer_proxy_metrics, compute_answer_proxy_metrics
from .chunks import build_validation_report, load_chunk_file
from .chunk_adapter import adapt_basic_chunks, write_adapted_corpus
from .constants import DENSE_DIM, DENSE_MODEL
from .embeddings import EmbeddingError, OpenAIEmbeddingClient
from .eval_api import (
    DEFAULT_EVAL_API_BASE_URL,
    EvalApiError,
    build_submission_payload,
    detect_eval_question_type,
    load_eval_question_set,
    select_eval_questions,
    submit_eval_payload,
)
from .filters import FilterSpec, parse_keywords, parse_question_types
from .graph_payloads import build_graph_import_bundle
from .graph_schema import ensure_graph_schema
from .graph_store import (
    MissingNeo4jDriverError,
    check_neo4j,
    ensure_graph_enabled,
    expand_chunks,
    index_graph_bundle,
    make_driver as make_graph_driver,
)
from .indexing import index_dense_chunks, index_sparse_chunks
from .llm import LLMError, OpenAICompatibleChatClient
from .local_search import search_local
from .offline_graph import answer_offline
from .offline_evaluation import evaluate_offline
from .qdrant_store import (
    MissingQdrantClientError,
    check_qdrant,
    create_or_recreate_collection,
    ensure_payload_indexes,
    fetch_collection_chunks,
    make_client,
)
from .retrieval import attach_graph_context, format_result, search_by_profile
from .settings import load_config
from .sparse import load_sparse_encoder


@dataclass(frozen=True)
class RetrievalRun:
    profile: str
    results: list[dict[str, object]]


EVAL_SECTION_PRIORITIES: dict[str, tuple[str, ...]] = {
    "А": ("Abstract", "Introduction", "Conclusion", "Methodology", "Experiments", "Appendix"),
    "А+": ("Experiments", "Conclusion", "Methodology", "Abstract", "Introduction", "Appendix"),
    "В": ("Experiments", "Appendix", "Methodology", "Introduction", "Conclusion", "Abstract"),
    "З": ("Experiments", "Appendix", "Methodology", "Introduction", "Conclusion", "Abstract"),
    "Г": ("Methodology", "Introduction", "Conclusion", "Abstract", "Experiments", "Appendix"),
    "Б": ("Introduction", "Experiments", "Methodology", "Conclusion", "Abstract", "Appendix"),
    "Д": ("Methodology", "Introduction", "Conclusion", "Abstract", "Experiments", "Appendix"),
    "Е": ("Methodology", "Introduction", "Conclusion", "Abstract", "Experiments", "Appendix"),
    "И": ("Methodology", "Introduction", "Appendix", "Conclusion", "Abstract", "Experiments"),
    "Ж": ("Experiments", "Abstract", "Conclusion", "Appendix", "Methodology", "Introduction"),
    "К": ("Methodology", "Introduction", "Conclusion", "Abstract", "Experiments", "Appendix"),
    "Л": ("Methodology", "Introduction", "Appendix", "Conclusion", "Abstract", "Experiments"),
    "М": ("Methodology", "Conclusion", "Introduction", "Abstract", "Experiments", "Appendix"),
    "Спец": ("Experiments", "Appendix", "Conclusion", "Introduction", "Methodology", "Abstract"),
}

DEFAULT_EVAL_CHUNK_TYPE_PRIORITIES: dict[str, int] = {
    "prose": 0,
    "table_with_prose": 0,
    "table": 1,
    "formula_heavy": 2,
}

EVAL_CHUNK_TYPE_PRIORITIES: dict[str, dict[str, int]] = {
    "В": {"table_with_prose": 0, "prose": 1, "table": 2, "formula_heavy": 3},
    "З": {"table_with_prose": 0, "prose": 1, "table": 2, "formula_heavy": 3},
    "Ж": {"table_with_prose": 0, "prose": 1, "table": 2, "formula_heavy": 3},
}


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (
        ValidationError,
        ValueError,
        EmbeddingError,
        EvalApiError,
        LLMError,
        MissingQdrantClientError,
        MissingNeo4jDriverError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except ValueError:
            continue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vector-db-rag")
    parser.add_argument("--config", default="config/retrieval.yaml", help="Path to YAML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-chunks", help="Validate chunk JSON.")
    validate.add_argument("path", nargs="?", help="Path to dic_chunks.json.")
    validate.set_defaults(func=cmd_validate_chunks)

    local_query = subparsers.add_parser("local-query", help="Search the local corpus without services.")
    local_query.add_argument("text", help="Search query.")
    local_query.add_argument("--chunks", help="Optional path to a GraphRAG-compatible corpus.")
    local_query.add_argument("--limit", type=int, default=5)
    local_query.add_argument("--article-id")
    local_query.add_argument("--json", action="store_true")
    local_query.set_defaults(func=cmd_local_query)

    adapt = subparsers.add_parser("adapt-chunker", help="Convert basic chunker JSON to GraphRAG schema.")
    adapt.add_argument("--chunks", required=True, help="Basic dic_chunks.json from ts-llm-chunker.")
    adapt.add_argument("--articles", required=True, help="Normalized Markdown directory with frontmatter.")
    adapt.add_argument("--output", required=True, help="Output JSON path; existing files are not overwritten.")
    adapt.set_defaults(func=cmd_adapt_chunker)

    offline_answer = subparsers.add_parser(
        "offline-answer", help="Extract a cited answer using local BM25 and a keyword graph."
    )
    offline_answer.add_argument("text", help="Question.")
    offline_answer.add_argument("--chunks", help="GraphRAG-compatible corpus path.")
    offline_answer.add_argument("--limit", type=int, default=4)
    offline_answer.add_argument("--article-id")
    offline_answer.add_argument("--no-graph", action="store_true")
    offline_answer.add_argument("--json", action="store_true")
    offline_answer.set_defaults(func=cmd_offline_answer)

    offline_eval = subparsers.add_parser("offline-eval", help="Evaluate local retrieval and citations.")
    offline_eval.add_argument("--chunks", help="GraphRAG-compatible corpus path.")
    offline_eval.add_argument("--questions", default="evaluation/questions.json")
    offline_eval.set_defaults(func=cmd_offline_eval)

    check = subparsers.add_parser("check-qdrant", help="Check Qdrant connectivity.")
    check.set_defaults(func=cmd_check_qdrant)

    check_graph = subparsers.add_parser("check-graph", help="Check Neo4j connectivity and graph health.")
    check_graph.add_argument(
        "--with-qdrant",
        action="store_true",
        help="Also compare graph chunk ids with the current Qdrant collection.",
    )
    check_graph.set_defaults(func=cmd_check_graph)

    create = subparsers.add_parser("create-collection", help="Create Qdrant collection and indexes.")
    create.add_argument("--recreate", action="store_true", help="Delete and recreate collection.")
    create.add_argument("--yes", action="store_true", help="Confirm destructive recreate.")
    create.set_defaults(func=cmd_create_collection)

    graph_schema = subparsers.add_parser("create-graph-schema", help="Create Neo4j constraints for the graph.")
    graph_schema.set_defaults(func=cmd_create_graph_schema)

    index = subparsers.add_parser("index-chunks", help="Index chunks.")
    index.add_argument("--dense", action="store_true", help="Build and upload dense vectors.")
    index.add_argument("--sparse", action="store_true", help="Build and upload BM25 sparse vectors.")
    index.add_argument("--chunks", help="Path to dic_chunks.json.")
    index.set_defaults(func=cmd_index_chunks)

    index_graph = subparsers.add_parser("index-graph", help="Import chunks into Neo4j graph.")
    index_graph.add_argument("--chunks", help="Path to dic_chunks.json.")
    index_graph.add_argument(
        "--ensure-schema",
        action="store_true",
        help="Create graph constraints before import.",
    )
    index_graph.add_argument("--batch-size", type=int, help="Override graph batch size from config.")
    index_graph.set_defaults(func=cmd_index_graph)

    query = subparsers.add_parser("query", help="Run retrieval query.")
    query.add_argument("text", help="Query text.")
    query.add_argument(
        "--profile",
        default=None,
        choices=["dense_only", "sparse_only", "hybrid_default", "entity_exact", "table_numeric"],
    )
    query.add_argument("--limit", type=int)
    query.add_argument("--prefetch-limit", type=int)
    query.add_argument(
        "--expand-graph",
        action="store_true",
        help="Expand retrieval results with graph context from Neo4j.",
    )
    add_filter_args(query)
    query.set_defaults(func=cmd_query)

    answer = subparsers.add_parser("answer", help="Retrieve context and ask the LLM to answer the question.")
    answer.add_argument("text", help="Question text.")
    answer.add_argument(
        "--profile",
        default=None,
        choices=["dense_only", "sparse_only", "hybrid_default", "entity_exact", "table_numeric"],
    )
    answer.add_argument("--limit", type=int)
    answer.add_argument("--prefetch-limit", type=int)
    answer.add_argument("--max-context-chunks", type=int)
    answer.add_argument("--max-chars-per-chunk", type=int)
    answer.add_argument("--temperature", type=float)
    answer.add_argument("--max-tokens", type=int)
    answer.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of plain text.",
    )
    answer.add_argument(
        "--dry-run",
        action="store_true",
        help="Build retrieval and answer context without calling the LLM.",
    )
    answer.add_argument(
        "--include-prompts",
        action="store_true",
        help="Include system_prompt and user_prompt in JSON dry-run output.",
    )
    answer.add_argument(
        "--no-graph",
        action="store_true",
        help="Do not expand retrieval results with graph context.",
    )
    add_filter_args(answer)
    answer.set_defaults(func=cmd_answer)

    eval_run = subparsers.add_parser(
        "eval-run",
        help="Build and optionally submit Eval API payload with Russian answers and verbatim chunks.",
    )
    eval_run.add_argument(
        "--questions-path",
        default="eval-api/docs/questions_raw.json",
        help="Path to local Eval API questions_raw.json.",
    )
    eval_run.add_argument("--run-id", help="Optional Eval API run_id label.")
    eval_run.add_argument(
        "--base-url",
        default=os.getenv("EVAL_API_BASE_URL") or DEFAULT_EVAL_API_BASE_URL,
        help="Eval API base URL.",
    )
    eval_run.add_argument(
        "--api-key-env",
        default="EVAL_API_KEY",
        help="Environment variable that stores the Eval API key.",
    )
    eval_run.add_argument(
        "--profile",
        default=None,
        choices=["dense_only", "sparse_only", "hybrid_default", "entity_exact", "table_numeric"],
    )
    eval_run.add_argument("--limit", type=int)
    eval_run.add_argument("--prefetch-limit", type=int)
    eval_run.add_argument("--max-context-chunks", type=int)
    eval_run.add_argument("--max-chars-per-chunk", type=int)
    eval_run.add_argument("--temperature", type=float)
    eval_run.add_argument("--max-tokens", type=int)
    eval_run.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of plain text.",
    )
    eval_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the local payload without POSTing it to Eval API.",
    )
    eval_run.add_argument(
        "--with-llm",
        action="store_true",
        help="Generate answers via configured LLM instead of local Russian fallback answers.",
    )
    eval_run.add_argument(
        "--score-answers",
        action="store_true",
        help="Compute local proxy metrics for answers in addition to Eval API chunk metrics.",
    )
    eval_run.add_argument(
        "--include-prompts",
        action="store_true",
        help="Include system_prompt and user_prompt in dry-run JSON output.",
    )
    eval_run.add_argument(
        "--no-graph",
        action="store_true",
        help="Do not expand retrieval results with graph context.",
    )
    eval_run.add_argument(
        "--question-id",
        action="append",
        help="Restrict run to specific question_id values. Can be repeated or comma-separated.",
    )
    eval_run.add_argument(
        "--article-id",
        action="append",
        help="Restrict run to specific article_id values. Can be repeated or comma-separated.",
    )
    eval_run.add_argument("--max-questions", type=int, help="Limit the number of selected questions.")
    eval_run.set_defaults(func=cmd_eval_run)

    expand = subparsers.add_parser(
        "expand-chunks",
        help="Expand chunk ids through Neo4j entities and derived graph relations.",
    )
    expand.add_argument("chunk_ids", nargs="+", help="Chunk ids to expand.")
    expand.set_defaults(func=cmd_expand_chunks)

    return parser


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--article-id")
    parser.add_argument("--arxiv-id")
    parser.add_argument("--section")
    parser.add_argument("--chunk-type")
    parser.add_argument(
        "--question-type",
        action="append",
        help="Eval API question type filter. Can be repeated or comma-separated.",
    )
    parser.add_argument("--keyword", action="append", help="Keyword filter. Can be repeated or comma-separated.")
    parser.add_argument("--year-gte", type=int)
    parser.add_argument("--year-lte", type=int)


def cmd_validate_chunks(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    path = Path(args.path) if args.path else config.paths.chunks
    chunk_file = load_chunk_file(path)
    report = build_validation_report(chunk_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_local_query(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    corpus = load_chunk_file(args.chunks or config.paths.chunks)
    results = search_local(corpus, args.text, limit=args.limit, article_id=args.article_id)
    if args.json:
        print(json.dumps({"query": args.text, "results": results}, ensure_ascii=False, indent=2))
        return
    if not results:
        print("No matching chunks found.")
        return
    for position, result in enumerate(results, start=1):
        print(f"\n[{position}] {result['chunk_id']} | score={result['score']:.3f}")
        print(f"{result['title']} | {result['section']}")
        print(result["text_excerpt"].replace("\n", " "))


def cmd_adapt_chunker(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    corpus = adapt_basic_chunks(args.chunks, args.articles)
    write_adapted_corpus(corpus, output)
    print(json.dumps({"output": str(output), **build_validation_report(corpus)}, ensure_ascii=False, indent=2))


def cmd_offline_answer(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    corpus = load_chunk_file(args.chunks or config.paths.chunks)
    result = answer_offline(
        corpus, args.text, max_sources=args.limit,
        expand_graph=not args.no_graph, article_id=args.article_id,
    )
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer)
        print(f"\nGraph edges examined: {result.graph_edges_examined}")


def cmd_offline_eval(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    corpus = load_chunk_file(args.chunks or config.paths.chunks)
    report = evaluate_offline(corpus, args.questions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_check_qdrant(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    client = make_client(url=config.qdrant.url, api_key=config.qdrant.api_key)
    report = check_qdrant(client)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_check_graph(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_graph_enabled(config.graph.enabled)
    driver = make_graph_driver(
        uri=config.graph.uri,
        user=config.graph.user,
        password=config.graph.password,
    )
    try:
        qdrant_chunks = None
        if args.with_qdrant:
            client = make_client(url=config.qdrant.url, api_key=config.qdrant.api_key)
            qdrant_chunks = fetch_collection_chunks(client, collection=config.qdrant.collection)
        report = check_neo4j(driver, database=config.graph.database, qdrant_chunks=qdrant_chunks)
    finally:
        driver.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_create_collection(args: argparse.Namespace) -> None:
    if args.recreate and not args.yes:
        raise ValueError("--recreate requires --yes")
    config = load_config(args.config)
    client = make_client(url=config.qdrant.url, api_key=config.qdrant.api_key)
    create_or_recreate_collection(
        client,
        collection=config.qdrant.collection,
        recreate=args.recreate,
        dense_vector_name=config.dense.vector_name,
        dense_dim=config.dense.dim,
        sparse_vector_name=config.sparse.vector_name,
    )
    ensure_payload_indexes(client, collection=config.qdrant.collection)
    print(f"collection ready: {config.qdrant.collection}")


def cmd_index_chunks(args: argparse.Namespace) -> None:
    if not args.dense and not args.sparse:
        raise ValueError("Pass at least one indexing mode: --dense or --sparse.")
    config = load_config(args.config)
    chunks_path = Path(args.chunks) if args.chunks else config.paths.chunks
    chunk_file = load_chunk_file(chunks_path)
    client = make_client(url=config.qdrant.url, api_key=config.qdrant.api_key)
    manifest = None
    if args.dense:
        embedder = OpenAIEmbeddingClient(
            model=config.dense.model,
            expected_dim=config.dense.dim,
            cache_dir=config.paths.embedding_cache,
        )
        manifest = index_dense_chunks(
            client=client,
            collection=config.qdrant.collection,
            chunk_file=chunk_file,
            embedder=embedder,
            batch_size=config.dense.batch_size,
            manifest_path=str(config.paths.manifest),
        )
    if args.sparse:
        manifest = index_sparse_chunks(
            client=client,
            collection=config.qdrant.collection,
            chunk_file=chunk_file,
            sparse_cache_path=str(config.paths.sparse_cache),
            batch_size=config.sparse.batch_size,
            manifest_path=str(config.paths.manifest),
        )
    if manifest is None:
        raise ValueError("No indexing work was executed.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def cmd_create_graph_schema(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_graph_enabled(config.graph.enabled)
    driver = make_graph_driver(
        uri=config.graph.uri,
        user=config.graph.user,
        password=config.graph.password,
    )
    try:
        report = ensure_graph_schema(driver, database=config.graph.database)
    finally:
        driver.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_index_graph(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_graph_enabled(config.graph.enabled)
    chunks_path = Path(args.chunks) if args.chunks else config.paths.chunks
    chunk_file = load_chunk_file(chunks_path)
    bundle = build_graph_import_bundle(chunk_file)
    driver = make_graph_driver(
        uri=config.graph.uri,
        user=config.graph.user,
        password=config.graph.password,
    )
    try:
        if args.ensure_schema:
            ensure_graph_schema(driver, database=config.graph.database)
        report = index_graph_bundle(
            driver,
            database=config.graph.database,
            bundle=bundle,
            batch_size=args.batch_size or config.graph.batch_size,
        )
    finally:
        driver.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_query(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    client = make_client(url=config.qdrant.url, api_key=config.qdrant.api_key)
    retrieval_run = _run_retrieval(
        args=args,
        config=config,
        client=client,
        query_text=args.text,
        allow_sparse_fallback=False,
    )
    for index, result in enumerate(retrieval_run.results, start=1):
        print(f"\n[{index}]")
        print(format_result(result))


def cmd_answer(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    client = make_client(url=config.qdrant.url, api_key=config.qdrant.api_key)
    retrieval_run = _run_retrieval(
        args=args,
        config=config,
        client=client,
        query_text=args.text,
        allow_sparse_fallback=True,
        expand_graph=not args.no_graph,
    )
    answer_context = build_answer_context(
        args.text,
        retrieval_run.results,
        max_context_chunks=args.max_context_chunks or config.llm.max_context_chunks,
        max_chars_per_chunk=args.max_chars_per_chunk or config.llm.max_chars_per_chunk,
    )
    if not answer_context.sources:
        raise ValueError("No retrieval results were found for the answer prompt.")
    if args.dry_run:
        answer_run = build_answer_run(
            question=args.text,
            profile=retrieval_run.profile,
            graph_enabled=not args.no_graph,
            answer_context=answer_context,
            answer_text=None,
        )
        _print_answer_run(answer_run, json_output=args.json, include_prompts=args.include_prompts)
        return
    llm_client = OpenAICompatibleChatClient(
        provider=config.llm.provider,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        model=config.llm.model,
        thinking_enabled=config.llm.thinking_enabled,
        reasoning_effort=config.llm.reasoning_effort,
        timeout_seconds=config.llm.timeout_seconds,
    )
    answer = llm_client.generate(
        system_prompt=answer_context.system_prompt,
        user_prompt=answer_context.user_prompt,
        temperature=args.temperature if args.temperature is not None else config.llm.temperature,
        max_tokens=args.max_tokens or config.llm.max_tokens,
    )
    answer_run = build_answer_run(
        question=args.text,
        profile=retrieval_run.profile,
        graph_enabled=not args.no_graph,
        answer_context=answer_context,
        answer_text=answer,
    )
    _print_answer_run(answer_run, json_output=args.json, include_prompts=args.include_prompts)


def cmd_eval_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    question_set = load_eval_question_set(args.questions_path)
    selected_questions = select_eval_questions(
        question_set,
        question_ids=_parse_multi_values(args.question_id),
        article_ids=_parse_multi_values(args.article_id),
        limit=args.max_questions,
    )
    if not selected_questions:
        raise ValueError("No Eval API questions were selected.")

    client = make_client(url=config.qdrant.url, api_key=config.qdrant.api_key)
    llm_client = None
    if args.with_llm and not args.dry_run:
        llm_client = OpenAICompatibleChatClient(
            provider=config.llm.provider,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
            thinking_enabled=config.llm.thinking_enabled,
            reasoning_effort=config.llm.reasoning_effort,
            timeout_seconds=config.llm.timeout_seconds,
        )
    answer_metrics_embedder = None
    if args.score_answers and os.getenv("OPENAI_API_KEY"):
        answer_metrics_embedder = OpenAIEmbeddingClient(
            model=config.dense.model,
            expected_dim=config.dense.dim,
            cache_dir=config.paths.embedding_cache,
        )

    items: list[dict[str, object]] = []
    local_rows: list[dict[str, object]] = []
    for question in selected_questions:
        question_type = detect_eval_question_type(question.question)
        retrieval_args = SimpleNamespace(
            profile=args.profile,
            limit=args.limit,
            prefetch_limit=args.prefetch_limit,
            expand_graph=not args.no_graph,
            article_id=question.article_id,
            arxiv_id=None,
            section=None,
            chunk_type=None,
            question_type=[question_type],
            keyword=None,
            year_gte=None,
            year_lte=None,
        )
        retrieval_run, retrieval_mode = _run_eval_question_retrieval(
            retrieval_args=retrieval_args,
            config=config,
            client=client,
            expand_graph=not args.no_graph,
            query_text=question.question,
            question_type=question_type,
        )
        answer_context = build_answer_context(
            question.question,
            retrieval_run.results,
            max_context_chunks=args.max_context_chunks or config.llm.max_context_chunks,
            max_chars_per_chunk=args.max_chars_per_chunk or config.llm.max_chars_per_chunk,
        )
        answer_text: str | None = None
        if llm_client is not None and answer_context.sources:
            answer_text = llm_client.generate(
                system_prompt=answer_context.system_prompt,
                user_prompt=answer_context.user_prompt,
                temperature=args.temperature if args.temperature is not None else config.llm.temperature,
                max_tokens=args.max_tokens or config.llm.max_tokens,
            )
        if not answer_text:
            answer_text = build_fallback_answer(question.question, answer_context.sources)
        answer_run = build_answer_run(
            question=question.question,
            profile=retrieval_run.profile,
            graph_enabled=not args.no_graph,
            answer_context=answer_context,
            answer_text=answer_text,
        )
        item = answer_run_to_eval_item(answer_run, question_id=question.id)
        items.append(item)

        local_row: dict[str, object] = {
            "question_id": question.id,
            "article_id": question.article_id,
            "article_title": question.article_title,
            "question_type": question_type,
            "profile": retrieval_run.profile,
            "retrieval_mode": retrieval_mode,
            "used_chunk_ids": [source.chunk_id for source in answer_run.sources],
            "answer": answer_run.answer_text,
            "chunks": item["chunks"],
        }
        if args.score_answers:
            local_row["answer_metrics"] = compute_answer_proxy_metrics(
                question=question.question,
                answer_text=answer_run.answer_text,
                sources=answer_run.sources,
                embedder=answer_metrics_embedder,
            )
        if args.include_prompts:
            local_row["system_prompt"] = answer_run.system_prompt
            local_row["user_prompt"] = answer_run.user_prompt
        local_rows.append(local_row)

    payload = build_submission_payload(run_id=args.run_id, items=items)
    if args.dry_run:
        _print_eval_run_payload(
            question_set_version=question_set.version,
            payload=payload,
            local_rows=local_rows,
            json_output=args.json,
        )
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Eval API key is not set. Define environment variable {args.api_key_env} before running eval-run."
        )
    response = submit_eval_payload(
        base_url=args.base_url,
        api_key=api_key,
        payload=payload,
    )
    _print_eval_submit_result(
        question_set_version=question_set.version,
        payload=payload,
        local_rows=local_rows,
        response=response,
        json_output=args.json,
    )


def cmd_expand_chunks(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_graph_enabled(config.graph.enabled)
    driver = make_graph_driver(
        uri=config.graph.uri,
        user=config.graph.user,
        password=config.graph.password,
    )
    try:
        report = expand_chunks(driver, database=config.graph.database, chunk_ids=args.chunk_ids)
    finally:
        driver.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _run_retrieval(
    *,
    args: argparse.Namespace,
    config: object,
    client: object,
    query_text: str,
    allow_sparse_fallback: bool,
    expand_graph: bool | None = None,
) -> RetrievalRun:
    profile = _resolve_profile(
        requested_profile=args.profile or config.retrieval.default_profile,
        allow_sparse_fallback=allow_sparse_fallback,
    )
    needs_dense = profile in {"dense_only", "hybrid_default", "entity_exact", "table_numeric"}
    needs_sparse = profile in {"sparse_only", "hybrid_default", "entity_exact", "table_numeric"}
    embedder = (
        OpenAIEmbeddingClient(
            model=config.dense.model,
            expected_dim=config.dense.dim,
            cache_dir=config.paths.embedding_cache,
        )
        if needs_dense
        else None
    )
    sparse_encoder = None
    if needs_sparse:
        if not config.paths.sparse_cache.exists():
            raise ValueError(
                f"Sparse encoder cache not found: {config.paths.sparse_cache}. "
                "Run: vector-db-rag index-chunks --sparse"
            )
        sparse_encoder = load_sparse_encoder(config.paths.sparse_cache)
    filter_spec = _build_filter_spec(args)
    results = search_by_profile(
        profile=profile,
        client=client,
        collection=config.qdrant.collection,
        query=query_text,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        limit=args.limit or config.retrieval.default_limit,
        prefetch_limit=args.prefetch_limit or config.retrieval.prefetch_limit,
        filter_spec=filter_spec,
    )
    use_graph = args.expand_graph if expand_graph is None else expand_graph
    if use_graph:
        results = _expand_results_with_graph(results=results, config=config)
    return RetrievalRun(profile=profile, results=results)


def _run_eval_question_retrieval(
    *,
    retrieval_args: argparse.Namespace,
    config: object,
    client: object,
    expand_graph: bool,
    query_text: str,
    question_type: str,
) -> tuple[RetrievalRun, str]:
    limit = retrieval_args.limit or config.retrieval.default_limit
    candidate_limit = max(limit, 20)
    article_only_args = SimpleNamespace(**vars(retrieval_args))
    article_only_args.question_type = None
    article_only_args.limit = candidate_limit
    selected_run = _run_retrieval(
        args=article_only_args,
        config=config,
        client=client,
        query_text=query_text,
        allow_sparse_fallback=True,
        expand_graph=False,
    )
    selected_results = _rerank_eval_results(
        results=selected_run.results,
        question_type=question_type,
        limit=limit,
    )
    retrieval_mode = "article_reranked"

    if expand_graph:
        return (
            RetrievalRun(
                profile=selected_run.profile,
                results=_expand_results_with_graph(results=selected_results, config=config),
            ),
            retrieval_mode,
        )
    return RetrievalRun(profile=selected_run.profile, results=selected_results), retrieval_mode


def _rerank_eval_results(
    *,
    results: list[dict[str, object]],
    question_type: str,
    limit: int,
) -> list[dict[str, object]]:
    section_priorities = EVAL_SECTION_PRIORITIES.get(question_type, EVAL_SECTION_PRIORITIES["А"])
    section_order = {section: index for index, section in enumerate(section_priorities)}
    chunk_type_order = EVAL_CHUNK_TYPE_PRIORITIES.get(question_type, DEFAULT_EVAL_CHUNK_TYPE_PRIORITIES)

    def sort_key(result: dict[str, object]) -> tuple[int, int, float]:
        payload = result.get("payload") or {}
        section = str(payload.get("section") or "")
        chunk_type = str(payload.get("chunk_type") or "")
        section_rank = section_order.get(section, len(section_order) + 1)
        chunk_rank = chunk_type_order.get(chunk_type, len(chunk_type_order) + 1)
        return (section_rank, chunk_rank, -float(result.get("score") or 0.0))

    return sorted(results, key=sort_key)[:limit]


def _resolve_profile(*, requested_profile: str, allow_sparse_fallback: bool) -> str:
    if not allow_sparse_fallback:
        return requested_profile
    dense_profiles = {"dense_only", "hybrid_default", "entity_exact", "table_numeric"}
    if requested_profile not in dense_profiles:
        return requested_profile
    if os.getenv("OPENAI_API_KEY"):
        return requested_profile
    return "sparse_only"


def _build_filter_spec(args: argparse.Namespace) -> FilterSpec:
    return FilterSpec(
        article_id=args.article_id,
        arxiv_id=args.arxiv_id,
        section=args.section,
        chunk_type=args.chunk_type,
        keywords=parse_keywords(args.keyword),
        question_types=parse_question_types(args.question_type),
        year_gte=args.year_gte,
        year_lte=args.year_lte,
    )


def _expand_results_with_graph(*, results: list[dict[str, object]], config: object) -> list[dict[str, object]]:
    ensure_graph_enabled(config.graph.enabled)
    chunk_ids = [
        str((result.get("payload") or {}).get("chunk_id"))
        for result in results
        if (result.get("payload") or {}).get("chunk_id")
    ]
    if not chunk_ids:
        return results
    driver = make_graph_driver(
        uri=config.graph.uri,
        user=config.graph.user,
        password=config.graph.password,
    )
    try:
        graph_rows = expand_chunks(driver, database=config.graph.database, chunk_ids=chunk_ids)
    finally:
        driver.close()
    return attach_graph_context(results, graph_rows)


def _parse_multi_values(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    parsed: list[str] = []
    for value in values:
        parsed.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(dict.fromkeys(parsed))


def _print_answer_run(answer_run: AnswerRun, *, json_output: bool, include_prompts: bool) -> None:
    if json_output:
        print(
            json.dumps(
                answer_run_to_dict(answer_run, include_prompts=include_prompts),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(format_answer_run(answer_run))


def _print_eval_run_payload(
    *,
    question_set_version: str,
    payload: dict[str, object],
    local_rows: list[dict[str, object]],
    json_output: bool,
) -> None:
    if json_output:
        print(
            json.dumps(
                {
                    "questions_version": question_set_version,
                    "question_count": len(local_rows),
                    "submission_payload": payload,
                    "answer_metrics": _build_answer_metrics_report(local_rows),
                    "items": local_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"Prepared Eval API payload for {len(local_rows)} questions.")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_eval_submit_result(
    *,
    question_set_version: str,
    payload: dict[str, object],
    local_rows: list[dict[str, object]],
    response: dict[str, object],
    json_output: bool,
) -> None:
    if json_output:
        print(
            json.dumps(
                {
                    "questions_version": question_set_version,
                    "question_count": len(local_rows),
                    "submission_payload": payload,
                    "answer_metrics": _build_answer_metrics_report(local_rows),
                    "items": local_rows,
                    "response": response,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(json.dumps(response, ensure_ascii=False, indent=2))


def _build_answer_metrics_report(local_rows: list[dict[str, object]]) -> dict[str, object] | None:
    scored_rows = [row for row in local_rows if isinstance(row.get("answer_metrics"), dict)]
    if not scored_rows:
        return None
    return {
        "aggregate": aggregate_answer_proxy_metrics(
            [row["answer_metrics"] for row in scored_rows if isinstance(row["answer_metrics"], dict)]
        ),
        "per_question": [
            {
                "question_id": row["question_id"],
                "article_id": row["article_id"],
                "metrics": row["answer_metrics"],
            }
            for row in scored_rows
        ],
    }


if __name__ == "__main__":
    main()
