from pathlib import Path

from src.common.io_utils import read_json
from src.pipeline import build_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_one_command_build_produces_matching_artifacts(tmp_path):
    report = build_corpus(
        PROJECT_ROOT / "articles_raw", tmp_path,
        PROJECT_ROOT / "config" / "chunker_config.yaml",
    )
    output = read_json(tmp_path / "chunks" / "dic_chunks.json")
    assert report["articles"] == 2
    assert report["chunks"] == len(output["chunks"])
    assert report["config_hash"] == output["config_hash"]
    assert len(list((tmp_path / "articles").glob("*.md"))) == 2
    assert len(list((tmp_path / "chunks" / "per_article").glob("*.json"))) == 2
