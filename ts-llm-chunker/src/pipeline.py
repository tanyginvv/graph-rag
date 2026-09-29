"""Single reproducible build from raw Markdown to normalized Markdown and chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.chunker.chunker import chunk_article
from src.chunker.run_chunking import VERSION
from src.common.io_utils import config_hash, iter_md_files, load_config, read_text, write_json, write_text
from src.common.schemas import Chunk, ChunksDocument
from src.preprocess.md_cleaner import clean_md


def build_corpus(input_dir: Path, output_dir: Path, config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    chunk_config = config.get("chunker", {})
    preprocess_config = config.get("preprocess", {})
    if chunk_config.get("strategy", "recursive") != "recursive":
        raise ValueError("Only the recursive chunking strategy is supported")
    max_size = int(chunk_config.get("max_chunk_size", 600))
    if max_size < 1:
        raise ValueError("max_chunk_size must be positive")
    files = iter_md_files(input_dir)
    if not files:
        raise ValueError(f"No Markdown articles found in {input_dir}")

    normalized_dir = output_dir / "articles"
    per_article_dir = output_dir / "chunks" / "per_article"
    all_chunks: list[Chunk] = []
    article_ids: set[str] = set()
    for path in files:
        clean = clean_md(
            read_text(path),
            filename=path.name,
            remove_references=preprocess_config.get("remove_references", True),
            remove_impact_statement=preprocess_config.get("remove_impact_statement", True),
            remove_images=preprocess_config.get("remove_image_placeholders", True),
            remove_emails=preprocess_config.get("remove_emails", True),
        )
        metadata, chunks = chunk_article(
            clean,
            filename=path.name,
            max_chunk_size=max_size,
            min_chunk_size=int(chunk_config.get("min_chunk_size", 150)),
            overlap_tokens=int(chunk_config.get("overlap_tokens", 80)),
            encoding=chunk_config.get("tokenizer", "cl100k_base"),
        )
        if metadata.article_id in article_ids:
            raise ValueError(f"Duplicate article_id: {metadata.article_id}")
        article_ids.add(metadata.article_id)
        write_text(normalized_dir / path.name, clean)
        write_json(per_article_dir / f"{path.stem}.json", [chunk.model_dump() for chunk in chunks])
        all_chunks.extend(chunks)

    document = ChunksDocument(
        chunker_version=VERSION,
        config_hash=config_hash(config),
        chunks=all_chunks,
    )
    write_json(output_dir / "chunks" / "dic_chunks.json", document.model_dump())
    report: dict[str, object] = {
        "articles": len(article_ids),
        "chunks": len(all_chunks),
        "unknown_section_chunks": sum(chunk.section == "Unknown" for chunk in all_chunks),
        "oversized_atomic_or_formula_chunks": sum(chunk.token_count > max_size for chunk in all_chunks),
        "max_chunk_tokens": max(chunk.token_count for chunk in all_chunks),
        "chunker_version": VERSION,
        "config_hash": document.config_hash,
    }
    write_json(output_dir / "quality_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized articles and chunks in one run")
    parser.add_argument("--input", type=Path, default=Path("articles_raw"))
    parser.add_argument("--output", type=Path, default=Path("build"))
    parser.add_argument("--config", type=Path, default=Path("config/chunker_config.yaml"))
    args = parser.parse_args()
    import json

    print(json.dumps(build_corpus(args.input, args.output, args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
