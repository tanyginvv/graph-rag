"""CLI entry point для чанкинга: articles/ -> chunks/dic_chunks.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.chunker.chunker import chunk_article
from src.common.io_utils import (
    config_hash,
    iter_md_files,
    load_config,
    read_text,
    write_json,
)
from src.common.logging_config import get_logger, setup_logging
from src.common.schemas import Chunk, ChunksDocument

log = get_logger(__name__)

DEFAULT_CONFIG = "config/chunker_config.yaml"
VERSION = "0.2.0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk normalized MD articles")
    parser.add_argument("--input", default="articles/", help="Input directory with clean MD")
    parser.add_argument("--output", default="chunks/", help="Output directory for JSON chunks")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    full_cfg = load_config(args.config)
    cfg = full_cfg.get("chunker", {})
    strategy = cfg.get("strategy", "recursive")
    if strategy != "recursive":
        parser.error(f"Unsupported chunking strategy: {strategy!r}. Use 'recursive'.")
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_md_files(input_dir)
    if not files:
        log.warning("No .md files found in %s", input_dir)
        return

    all_chunks: list[Chunk] = []

    for md_path in files:
        text = read_text(md_path)
        metadata, chunks = chunk_article(
            text,
            filename=md_path.name,
            max_chunk_size=cfg.get("max_chunk_size", 600),
            min_chunk_size=cfg.get("min_chunk_size", 150),
            overlap_tokens=cfg.get("overlap_tokens", 80),
            encoding=cfg.get("tokenizer", "cl100k_base"),
        )
        all_chunks.extend(chunks)

        # Также пишем per-article JSON
        per_article_dir = output_dir / "per_article"
        per_article_dir.mkdir(parents=True, exist_ok=True)
        per_article_path = per_article_dir / f"{md_path.stem}.json"
        write_json(per_article_path, [c.model_dump() for c in chunks])
        log.info("Written: %s (%d chunks)", per_article_path.name, len(chunks))

    # Пишем общий dic_chunks.json
    doc = ChunksDocument(
        chunker_version=VERSION,
        config_hash=config_hash(full_cfg),
        chunks=all_chunks,
    )
    dic_path = output_dir / "dic_chunks.json"
    write_json(dic_path, doc.model_dump())
    log.info("Written: %s (%d total chunks)", dic_path, len(all_chunks))


if __name__ == "__main__":
    main()
