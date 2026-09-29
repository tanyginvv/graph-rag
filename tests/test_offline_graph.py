from pathlib import Path

from vector_db_rag.chunks import load_chunk_file
from vector_db_rag.offline_graph import answer_offline


CORPUS = Path(__file__).resolve().parents[1] / "chunks 2" / "docs" / "dic_chunks.json"


def test_offline_answer_cites_verbatim_evidence_and_graph_paths():
    corpus = load_chunk_file(CORPUS)
    answer = answer_offline(corpus, "How does Time-LLM use text prototypes?", max_sources=4)
    assert answer.sources[0].article_id == "time-llm"
    assert answer.graph_edges_examined > 0
    by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    for source in answer.sources:
        assert source.excerpt in by_id[source.chunk_id].text
        assert f"[{source.chunk_id}]" in answer.answer
    assert any(source.via_keyword for source in answer.sources)


def test_offline_answer_respects_article_filter_and_can_disable_graph():
    corpus = load_chunk_file(CORPUS)
    answer = answer_offline(
        corpus, "text prototypes", article_id="time-llm", expand_graph=False
    )
    assert all(source.article_id == "time-llm" for source in answer.sources)
    assert answer.graph_edges_examined == 0
