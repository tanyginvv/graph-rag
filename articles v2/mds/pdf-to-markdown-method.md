# PDF to Markdown Conversion Method

## Goal

Produce a Markdown conversion of a scientific PDF with maximum factual and structural accuracy, while avoiding OCR noise, layout artifacts, and speculative reconstruction.

## Core Principles

1. Use the PDF's native text layer as the primary source for prose, references, captions, and most formulas.
2. Use rendered page images as a mandatory verification layer, not as the primary text source.
3. Never treat figure images or page renders as a source for automatic prose generation.
4. Do not trust auto-generated Markdown blindly; use it only as a draft skeleton.
5. Every section, equation, table, figure caption, footnote, and appendix block must be manually verified against the rendered PDF page.
6. Figures are not embedded as images in the final Markdown; instead, they are kept as explicit in-flow references with their original captions.

## Final Conversion Strategy

### 1. Build the draft from the native PDF text layer

- Extract page text directly from the PDF.
- Preserve the article hierarchy: title, authors, abstract, index terms, numbered sections, subsections, acknowledgments, references, appendix/supplementary sections.
- Use any existing auto-converted `.md` only as a draft outline, not as an authoritative source.

### 2. Manually restore reading order

- Re-check every page against the rendered PDF because two-column layouts often interleave paragraphs incorrectly.
- Rejoin sentences broken by page columns, figures, tables, captions, or footnotes.
- Remove layout garbage such as:
  - `picture intentionally omitted`
  - OCR dumps of figure internals
  - broken ligatures / encoding artifacts
  - duplicated headers / page numbers
  - split words caused by line wrapping

### 2a. Use page-image verification explicitly

- Render each PDF page to an image and inspect it visually.
- Use page images to verify:
  - reading order in multi-column layouts
  - exact placement and captions of figures
  - table boundaries and whether a table is safe to transcribe
  - ambiguous symbols in formulas
  - appendix transitions, footnotes, and page breaks
- Do not convert those page images into prose automatically; they are a human verification layer over the native text extraction.

### 3. Restore formulas as math, not as image text

- Reconstruct equations from the PDF text layer into LaTeX math blocks.
- If the text layer is ambiguous, verify the formula visually on the rendered PDF page before writing the final LaTeX.
- Keep equation numbering exactly as in the source PDF.

Recommended format:

```md
$$
W_o + \Delta W = W_o + \beta_r YX
$$
```

### 4. Handle figures as references only

- Do not embed figure images in the final Markdown.
- Do not add raster copies of figures from the PDF into the final Markdown file.
- Keep each figure exactly at its original logical position in the text flow.
- Preserve the original caption wording as closely as possible.
- Mark figures as PDF references.

Recommended format:

```md
> Figure 2 (see PDF, p. 6). Comparison of model efficiency across different forecast horizons. ...
```

### 5. Handle tables conservatively

- If a table can be reconstructed cell-by-cell with full confidence, transcribe it manually into Markdown.
- If the table is too dense or the extracted structure is unreliable, do not guess. Keep:
  - the table number
  - the full caption
  - a note that the canonical layout remains in the PDF
- For high-stakes numeric tables, visual cross-checking against the PDF is mandatory even after manual transcription.

Recommended fallback format:

```md
> Table 7 (see PDF, p. 10). Comprehensive analysis of ...
> Canonical tabular layout is preserved in the source PDF.
```

Important clarification for this repository:

- A caption-only mention does **not** count as the table being fully preserved if the table contains recoverable scientific content.
- If the numeric or categorical content of a table is legible from the text layer plus page-image verification, transcribe it.
- For very wide tables, split them into several Markdown tables by dataset/block if needed, but preserve all cells.
- Use the fallback reference-only format only when the table truly cannot be reconstructed reliably, or when that fallback is explicitly acceptable for the current task.

### 5a. Table conversion nuances that must be handled explicitly

- Multi-row or multi-column headers must be preserved as hierarchy, not flattened into ambiguous single-line labels.
- Grouped method headers such as `Method -> MSE / MAE` should be represented with HTML tables when plain Markdown cannot preserve the original structure.
- Merged cells, row groups, and column groups must be reconstructed intentionally; do not silently duplicate or drop their meaning.
- Table typography can carry meaning and must be preserved when recoverable:
  - bold for best values
  - underline for second-best values
  - arrows such as `↓`, `↑`
  - superscripts, subscripts, and mathematical markers in headers or cells
- Model names, dataset names, and symbols inside tables must match the PDF exactly, including casing, hyphenation, and notation such as `S^2IP-LLM`, `ETTh1`, `w/o`, or `Avg.`.
- If one visual PDF table is logically composed of several metric blocks, dataset blocks, or horizon blocks, it is acceptable to split it into multiple Markdown tables only if no cells or labels are lost and the split is clearly documented.
- If a table spans pages, verify that continuation rows, repeated headers, notes, and summary rows are merged back into one coherent result.
- Table captions, notes, and legend text below the table are part of the scientific content and must not be dropped.
- If extracted text contains broken encodings or OCR-like artifacts inside table cells, correct them against the rendered PDF before finalizing the Markdown.
- If standard Markdown tables cannot preserve the semantics cleanly, prefer an inline HTML table over a lossy Markdown simplification.

### 6. Preserve scholarly apparatus exactly

- Keep citation numbers unchanged.
- Preserve footnotes, appendix labels, theorem/definition labels, and supplementary section numbering.
- Do not normalize terminology aggressively if it changes author intent.

## What Must Not Be Done

- Do not OCR figure contents into narrative text.
- Do not paraphrase captions when exact caption recovery is possible.
- Do not infer missing symbols in formulas without checking the rendered PDF.
- Do not leave conversion artifacts in the final file.
- Do not silently flatten or reorder tables, theorem blocks, or appendix content.

## Quality Gate Before Accepting a Conversion

The Markdown is acceptable only if all checks pass:

1. No encoding garbage or placeholder remnants remain.
2. No sentence is split incorrectly by column layout.
3. All formulas are present and readable as math.
4. Every figure is represented by an explicit in-flow reference and caption.
5. Every table is either accurately transcribed or explicitly preserved as a PDF reference.
6. Section numbering and reference numbering match the PDF.
7. Appendix / supplementary material is preserved, not dropped.

## Post-Conversion Verification Checklist

Run this checklist after the Markdown draft looks complete. Do not skip it.

### A. Structural inventory

1. Build an explicit inventory from the PDF:
   - title
   - authors
   - abstract
   - section and subsection headings
   - equations
   - tables
   - figures
   - footnotes
   - appendix / supplementary sections
   - references
2. Verify that every inventory item exists in the Markdown in some corresponding form.
3. Confirm that no PDF page has been skipped during verification, including appendix pages.

### B. Table completeness checks

1. List all table numbers present in the PDF.
2. List all table numbers present in the Markdown.
3. The two lists must match exactly.
4. For each table, verify one of the following is true:
   - the full table content is transcribed in Markdown
   - the table is intentionally preserved as a PDF reference-only block, and that fallback is justified and acceptable for the task
5. Do not treat a caption-only `Table N` mention as sufficient if the table data is recoverable.
6. For wide tables, verify that all rows and columns survived even if the table was split into smaller Markdown blocks.
7. For numeric tables, spot-check values row-by-row against the rendered PDF page, especially:
   - first row
   - last row
   - averages / summary rows
   - rows with missing-value markers such as `-`

### C. Figure checks

1. List all figure numbers present in the PDF.
2. Verify that every figure appears in the Markdown as an in-flow reference with caption text.
3. Verify that no PDF figures are embedded as Markdown images unless the task explicitly requests that.
4. Confirm that figure references are placed near their original logical position in the text.

### D. Formula and notation checks

1. Verify that every numbered equation in the PDF appears in the Markdown.
2. Check ambiguous symbols visually on the rendered page:
   - minus vs en dash
   - prime marks
   - subscripts / superscripts
   - Greek letters
   - calligraphic / bold symbols when semantically important
3. Confirm that equation numbering matches the PDF exactly.

### E. Appendix and supplementary checks

1. Verify that all appendix headings from the PDF are present in the Markdown.
2. Verify that appendix-only tables and figures are not omitted.
3. Confirm that supplementary metrics, dataset statistics, and ablation tables are included, not just cited.

### F. Artifact sweep

1. Search the Markdown for obvious conversion artifacts, including:
   - `picture intentionally omitted`
   - broken ligatures
   - duplicated headers / page numbers
   - malformed encoding
   - placeholder notes copied from draft tools
2. Search specifically for fallback phrases such as:
   - `Canonical tabular layout is preserved in the source PDF.`
3. For every fallback phrase found, manually confirm it is intentional and still acceptable.
4. Verify that no text block is duplicated because of two-column extraction errors.

### G. Final acceptance checks

1. Compare the Markdown against the PDF one last time page-by-page.
2. Confirm that all scientific content is present except items intentionally excluded by project rules, such as embedded figure images.
3. Confirm that anything omitted is omitted deliberately, documented, and acceptable for the task.
4. Only then mark the conversion as complete.

## Practical Workflow for Future Articles

1. Extract native PDF text.
2. Render pages to images for manual verification.
3. Build a draft Markdown skeleton.
4. Repair reading order page by page using the page renders as visual control.
5. Rebuild formulas into LaTeX.
6. Replace figures with captioned PDF references instead of embedding image files.
7. Transcribe tables only when structurally verifiable.
8. Run a final artifact sweep and compare the result against the PDF one last time.

## Decision for This Project

For this repository, the preferred high-accuracy method is:

- prose from the native PDF text layer
- page-image verification as the mandatory second layer
- formulas manually restored as LaTeX
- figures represented as references/captions only, without embedding PDF images
- tables transcribed only after manual verification, otherwise preserved as explicit PDF references
- final acceptance only after page-level visual cross-checking and the post-conversion verification checklist above
