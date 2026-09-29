"""Тесты для section_mapper."""

from src.chunker.section_mapper import classify_heading


def test_abstract():
    assert classify_heading("## Abstract") == "Abstract"


def test_introduction():
    assert classify_heading("## 1. Introduction") == "Introduction"
    assert classify_heading("## 1 Introduction") == "Introduction"


def test_related_work():
    assert classify_heading("## 2. Related Work") == "RelatedWork"


def test_methodology():
    assert classify_heading("## 3. Methodology") == "Methodology"
    assert classify_heading("## 3 Method") == "Methodology"
    assert classify_heading("### 3.1. Proposed Framework") == "Methodology"


def test_experiments():
    assert classify_heading("## 4. Experiments") == "Experiments"


def test_conclusion():
    assert classify_heading("## 5. Conclusion") == "Conclusion"
    assert classify_heading("## 6. Conclusions") == "Conclusion"


def test_appendix_implementation():
    assert classify_heading("## A. Experimental Details") == "Appendix_Implementation"


def test_appendix_baselines():
    assert classify_heading("### A.2. Baseline Introduction") == "Appendix_Baselines"


def test_appendix_datasets():
    assert classify_heading("### A.3. Dataset Details") == "Appendix_Datasets"


def test_appendix_results():
    assert classify_heading("## B. Long-term Forecasting Results") == "Appendix_Results"
    assert classify_heading("## C. Full Short-term Forecasting Results") == "Appendix_Results"


def test_appendix_other():
    assert classify_heading("### A.4. Something Else") == "Appendix_Other"


def test_unknown():
    assert classify_heading("## Acknowledgments") == "Unknown"


def test_no_hash_prefix():
    assert classify_heading("Abstract") == "Abstract"
    assert classify_heading("1. Introduction") == "Introduction"


def test_roman_numerals():
    """IEEE-формат: I. INTRODUCTION, II. RELATED WORK, etc."""
    assert classify_heading("#### I. INTRODUCTION") == "Introduction"
    assert classify_heading("#### II. RELATED WORK") == "RelatedWork"
    assert classify_heading("#### III. METHODOLOGY") == "Methodology"
    assert classify_heading("#### IV. EXPERIMENT DETAILS") == "Experiments"
    assert classify_heading("#### V. EXPERIMENTAL RESULTS") == "Experiments"
    assert classify_heading("#### VIII. CONCLUSION AND FUTURE STUDY") == "Conclusion"


def test_roman_numerals_no_prefix():
    assert classify_heading("I. Introduction") == "Introduction"
    assert classify_heading("III. Methodology") == "Methodology"


def test_abstract_inline():
    """Abstract— без номера (IEEE-формат)."""
    assert classify_heading("## Abstract") == "Abstract"
    assert classify_heading("Abstract") == "Abstract"


def test_ablation_and_visual_analysis():
    """IEEE-стиль: VI. ABLATION STUDY, VII. VISUAL ANALYSIS."""
    assert classify_heading("#### VI. ABLATION STUDY") == "Experiments"
    assert classify_heading("#### VII. VISUAL ANALYSIS OF FORECASTING") == "Experiments"
