"""Тесты для chunker (группировка по секциям, адаптивный порог уровня)."""

from src.chunker.chunker import (
    _detect_top_heading_level,
    _group_by_section,
    _has_sub_headings,
    _is_section_boundary,
)
from src.chunker.structure_parser import MDBlock


def _heading(text: str, level: int) -> MDBlock:
    prefix = "#" * level
    return MDBlock(
        type="heading",
        content=f"{prefix} {text}",
        heading_level=level,
        start_line=0,
        end_line=0,
    )


def _para(text: str) -> MDBlock:
    return MDBlock(type="paragraph", content=text, start_line=0, end_line=0)


class TestDetectTopHeadingLevel:
    def test_standard_arxiv(self):
        """## headings → top level = 2."""
        blocks = [
            _heading("Title", 1),
            _heading("1. Introduction", 2),
            _heading("1.1. Background", 3),
            _heading("2. Method", 2),
        ]
        assert _detect_top_heading_level(blocks) == 2

    def test_ieee_format(self):
        """#### headings → top level = 4."""
        blocks = [
            _heading("Title", 1),
            _heading("I. INTRODUCTION", 4),
            _heading("A. Traditional Approach", 4),
            _heading("II. RELATED WORK", 4),
        ]
        assert _detect_top_heading_level(blocks) == 4

    def test_mixed_supplementary(self):
        """#### in main body, ## in supplementary → top level = 4 (first heading wins)."""
        blocks = [
            _heading("Title", 1),
            _heading("I. INTRODUCTION", 4),
            _para("main text"),
            _heading("Supplementary", 2),
            _heading("A.1 Details", 2),
        ]
        assert _detect_top_heading_level(blocks) == 4

    def test_no_headings(self):
        blocks = [_para("text")]
        assert _detect_top_heading_level(blocks) == 2

    def test_only_title(self):
        blocks = [_heading("Title", 1)]
        assert _detect_top_heading_level(blocks) == 2


class TestHasSubHeadings:
    def test_arxiv_has_subheadings(self):
        """arXiv: ## + ### → has sub-headings."""
        blocks = [
            _heading("1. Intro", 2),
            _heading("1.1. Background", 3),
        ]
        assert _has_sub_headings(blocks, top_level=2) is True

    def test_ieee_flat(self):
        """IEEE: all #### → no sub-headings → flat."""
        blocks = [
            _heading("I. INTRODUCTION", 4),
            _heading("A. Background", 4),
            _heading("II. METHOD", 4),
        ]
        assert _has_sub_headings(blocks, top_level=4) is False

    def test_ieee_with_supplementary(self):
        """IEEE #### main + ## supplementary → ## is above top_level, not sub."""
        blocks = [
            _heading("I. INTRO", 4),
            _heading("Supplementary", 2),
        ]
        assert _has_sub_headings(blocks, top_level=4) is False


class TestIsSectionBoundary:
    def test_flat_main_body_label(self):
        """Flat hierarchy: heading with main body label → boundary."""
        block = _heading("I. INTRODUCTION", 4)
        assert _is_section_boundary(block, top_level=4, flat_hierarchy=True) is True

    def test_flat_appendix_label_not_boundary(self):
        """Flat hierarchy: subsection matching appendix pattern → NOT boundary."""
        block = _heading("B. In Modality Adaptation", 4)
        assert _is_section_boundary(block, top_level=4, flat_hierarchy=True) is False

    def test_flat_unknown_label(self):
        """Flat hierarchy: heading with unknown label → NOT boundary."""
        block = _heading("A. Traditional Approach", 4)
        assert _is_section_boundary(block, top_level=4, flat_hierarchy=True) is False

    def test_flat_higher_level_always_boundary(self):
        """Flat hierarchy: heading above top_level → always boundary."""
        block = _heading("Supplementary", 2)
        assert _is_section_boundary(block, top_level=4, flat_hierarchy=True) is True

    def test_multi_level_same_level_always_boundary(self):
        """Multi-level: heading at top_level → always boundary."""
        block = _heading("Acknowledgments", 2)
        assert _is_section_boundary(block, top_level=2, flat_hierarchy=False) is True

    def test_paragraph_never_boundary(self):
        block = _para("some text")
        assert _is_section_boundary(block, top_level=2, flat_hierarchy=False) is False


class TestGroupBySectionAdaptive:
    def test_ieee_sections_split(self):
        """#### known-label headings in IEEE create sections; subsections don't."""
        blocks = [
            _heading("Title", 1),
            _para("Author line"),
            _heading("I. INTRODUCTION", 4),
            _para("Intro text"),
            _heading("A. Background", 4),  # subsection, flat → Unknown → no split
            _para("Background text"),
            _heading("II. RELATED WORK", 4),
            _para("Related text"),
            _heading("III. METHODOLOGY", 4),
            _para("Method text"),
        ]
        sections = _group_by_section(blocks)
        labels = [s[0] for s in sections]
        assert labels == ["Unknown", "Introduction", "RelatedWork", "Methodology"]

    def test_ieee_supplementary_splits(self):
        """## headings in supplementary split even in #### document."""
        blocks = [
            _heading("Title", 1),
            _heading("I. INTRODUCTION", 4),
            _para("Intro"),
            _heading("Supplementary", 2),  # level 2 < top_level 4 → always boundary
            _para("Supplementary text"),
        ]
        sections = _group_by_section(blocks)
        labels = [s[0] for s in sections]
        assert len(sections) == 3  # Unknown (title), Introduction, Unknown (Supplementary)

    def test_standard_arxiv_still_works(self):
        """## headings with multi-level hierarchy still work."""
        blocks = [
            _heading("Title", 1),
            _heading("1. Introduction", 2),
            _para("Intro text"),
            _heading("1.1 Background", 3),
            _para("Background"),
            _heading("2. Related Work", 2),
            _para("Related text"),
            _heading("Acknowledgments", 2),  # Unknown label but still boundary (multi-level)
            _para("Thanks"),
        ]
        sections = _group_by_section(blocks)
        labels = [s[0] for s in sections]
        assert "Introduction" in labels
        assert "RelatedWork" in labels
        assert "Unknown" in labels  # Acknowledgments
