"""Тесты для recursive_splitter."""

from src.chunker.splitters.recursive_splitter import recursive_split
from src.chunker.structure_parser import MDBlock
from src.chunker.token_counter import count_tokens


def _para(text: str, start: int = 0) -> MDBlock:
    return MDBlock(type="paragraph", content=text, start_line=start, end_line=start)


def _table(text: str, start: int = 0) -> MDBlock:
    return MDBlock(type="table", content=text, start_line=start, end_line=start)


def _display_math(text: str, start: int = 0) -> MDBlock:
    return MDBlock(type="display_math", content=text, start_line=start, end_line=start)


def test_single_small_paragraph():
    blocks = [_para("This is a short paragraph.")]
    chunks = recursive_split(blocks, max_chunk_size=600, overlap_tokens=0)
    assert len(chunks) == 1
    assert "short paragraph" in chunks[0]


def test_table_is_atomic():
    blocks = [
        _para("Before table."),
        _table("| A | B |\n|---|---|\n| 1 | 2 |"),
        _para("After table."),
    ]
    chunks = recursive_split(blocks, max_chunk_size=600, overlap_tokens=0)
    # Таблица — отдельный чанк
    table_chunks = [c for c in chunks if "| A | B |" in c]
    assert len(table_chunks) == 1
    # Таблица не смешана с другим текстом
    assert "Before" not in table_chunks[0]
    assert "After" not in table_chunks[0]


def test_display_math_is_atomic():
    blocks = [
        _para("Before math."),
        _display_math("$$E = mc^2$$"),
        _para("After math."),
    ]
    chunks = recursive_split(blocks, max_chunk_size=600, overlap_tokens=0)
    math_chunks = [c for c in chunks if "E = mc^2" in c]
    assert len(math_chunks) == 1


def test_large_paragraph_split():
    # Создаём большой параграф
    long_text = ". ".join([f"Sentence number {i}" for i in range(50)]) + "."
    blocks = [_para(long_text)]
    chunks = recursive_split(blocks, max_chunk_size=100, overlap_tokens=0)
    assert len(chunks) > 1
    for chunk in chunks:
        # Каждый чанк не должен сильно превышать лимит
        tokens = count_tokens(chunk)
        assert tokens <= 150  # допуск на границе предложения


def test_overlap_added():
    blocks = [
        _para("First paragraph about topic A. It has multiple sentences. This is the third."),
        _para("Second paragraph about topic B. It also has sentences."),
    ]
    chunks = recursive_split(blocks, max_chunk_size=30, overlap_tokens=20)
    if len(chunks) > 1:
        # Второй чанк должен содержать текст из конца первого
        assert len(chunks[1]) > len("Second paragraph about topic B.")


def test_empty_input():
    chunks = recursive_split([], max_chunk_size=600, overlap_tokens=0)
    assert chunks == []


def test_very_long_sentence_respects_limit():
    blocks = [_para("longword " * 400)]
    chunks = recursive_split(blocks, max_chunk_size=40, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 40 for chunk in chunks)


def test_overlap_does_not_join_atomic_table():
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    chunks = recursive_split(
        [_para("Before table. " * 15), _table(table), _para("After table. " * 15)],
        max_chunk_size=30,
        overlap_tokens=10,
    )
    assert table in chunks
    assert chunks[chunks.index(table)] == table


def test_large_table_splits_by_complete_rows_with_header():
    header = "| Dataset | MSE |\n| --- | --- |"
    rows = [f"| Dataset {i} | {i / 10:.1f} |" for i in range(80)]
    chunks = recursive_split([_table("\n".join([header, *rows]))], max_chunk_size=80)
    assert len(chunks) > 1
    assert all(chunk.startswith(header) for chunk in chunks)
    assert all(count_tokens(chunk) <= 80 for chunk in chunks)
    assert sum(chunk.count("| Dataset ") - 1 for chunk in chunks) == len(rows)
