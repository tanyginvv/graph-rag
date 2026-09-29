from pathlib import Path

from vector_db_rag.chunks import load_chunk_file
from vector_db_rag.offline_evaluation import evaluate_offline


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_report_has_valid_evidence():
    corpus = load_chunk_file(ROOT / "chunks 2" / "docs" / "dic_chunks.json")
    report = evaluate_offline(corpus, ROOT / "evaluation" / "questions.json")
    assert report["questions"] == 6
    assert report["verbatim_evidence_rate"] == 1.0
    assert 0 <= report["hit_at_1"] <= report["hit_at_3"] <= 1
