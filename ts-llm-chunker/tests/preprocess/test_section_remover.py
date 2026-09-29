"""Тесты для section_remover."""

from src.preprocess.section_remover import remove_sections


SAMPLE = """\
## 5. Conclusion

This is the conclusion text.

## Impact Statement

This work has broader impact...

## References

Author1, 2024. Some paper.
Author2, 2023. Another paper.

## A. Experimental Details

### A.1. Implementation

Details here.
"""


def test_removes_references_and_impact():
    result = remove_sections(SAMPLE, remove_references=True, remove_impact_statement=True)
    assert "## References" not in result
    assert "Some paper" not in result
    assert "## Impact Statement" not in result
    assert "broader impact" not in result
    # Conclusion и Appendix остаются
    assert "## 5. Conclusion" in result
    assert "conclusion text" in result
    assert "## A. Experimental Details" in result
    assert "Details here." in result


def test_removes_only_references():
    result = remove_sections(SAMPLE, remove_references=True, remove_impact_statement=False)
    assert "## References" not in result
    assert "## Impact Statement" in result
    assert "broader impact" in result
    assert "## A. Experimental Details" in result


def test_removes_nothing():
    result = remove_sections(SAMPLE, remove_references=False, remove_impact_statement=False)
    assert "## References" in result
    assert "## Impact Statement" in result


def test_no_references_section():
    text = "## Introduction\\n\\nSome text.\\n\\n## Conclusion\\n\\nDone."
    result = remove_sections(text, remove_references=True, remove_impact_statement=True)
    assert result == text


def test_references_at_end_of_file():
    text = """\
## Conclusion

Done.

## References

Ref1.
Ref2."""
    result = remove_sections(text, remove_references=True, remove_impact_statement=True)
    assert "## References" not in result
    assert "Ref1." not in result
    assert "## Conclusion" in result
    assert "Done." in result
