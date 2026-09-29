"""CLI entry point для препроцессинга: articles_raw/ -> articles/."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.io_utils import iter_md_files, load_config, read_text, write_text
from src.common.logging_config import get_logger, setup_logging
from src.preprocess.md_cleaner import clean_md

log = get_logger(__name__)

DEFAULT_CONFIG = "config/chunker_config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw MD articles")
    parser.add_argument("--input", default="articles_raw/", help="Input directory with raw MD")
    parser.add_argument("--output", default="articles/", help="Output directory for clean MD")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    cfg = load_config(args.config).get("preprocess", {})
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_md_files(input_dir)
    if not files:
        log.warning("No .md files found in %s", input_dir)
        return

    for md_path in files:
        raw = read_text(md_path)
        clean = clean_md(
            raw,
            filename=md_path.name,
            remove_references=cfg.get("remove_references", True),
            remove_impact_statement=cfg.get("remove_impact_statement", True),
            remove_images=cfg.get("remove_image_placeholders", True),
            remove_emails=cfg.get("remove_emails", True),
        )
        out_path = output_dir / md_path.name
        write_text(out_path, clean)
        log.info("Written: %s (%d -> %d chars)", out_path.name, len(raw), len(clean))

    log.info("Preprocessed %d file(s)", len(files))


if __name__ == "__main__":
    main()
