import json
from pathlib import Path

from vector_db_rag.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_validate_chunks_from_project_root(monkeypatch, capsys):
    monkeypatch.chdir(PROJECT_ROOT)
    main(["validate-chunks"])
    report = json.loads(capsys.readouterr().out)
    assert report["chunks"] == 141
    assert report["articles"] == 12


def test_local_query_json(monkeypatch, capsys):
    monkeypatch.chdir(PROJECT_ROOT)
    main(["local-query", "Time-LLM reprogramming", "--limit", "2", "--json"])
    output = json.loads(capsys.readouterr().out)
    assert len(output["results"]) == 2
    assert output["results"][0]["article_id"] == "time-llm"
