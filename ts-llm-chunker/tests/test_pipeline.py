"""Exercise both CLIs on the checked-in sample articles."""

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_raw_markdown_to_chunk_json(tmp_path):
    normalized = tmp_path / "normalized"
    chunks = tmp_path / "chunks"
    config = PROJECT_ROOT / "config" / "chunker_config.yaml"
    subprocess.run(
        [
            sys.executable, "-m", "src.preprocess.run_preprocess",
            "--input", str(PROJECT_ROOT / "articles_raw"),
            "--output", str(normalized),
            "--config", str(config),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "src.chunker.run_chunking",
            "--input", str(normalized),
            "--output", str(chunks),
            "--config", str(config),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    document = json.loads((chunks / "dic_chunks.json").read_text(encoding="utf-8"))
    assert document["chunker_version"] == "0.2.0"
    assert len(list(normalized.glob("*.md"))) == 2
    assert len(list((chunks / "per_article").glob("*.json"))) == 2
    assert len({chunk["chunk_id"] for chunk in document["chunks"]}) == len(document["chunks"])
    assert {chunk["article_id"] for chunk in document["chunks"]} == {
        "arxiv_2403_05798", "arxiv_2603_29756"
    }
