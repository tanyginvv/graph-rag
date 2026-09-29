# LLM-Agents (DCATS): Извлечённые сущности (ревизия v1.2)

**article_id:** `llm-agents` | **Дата:** 2026-05-22
**Типы (v1.2, группа 3):** А (q00), А+ (q01), В (q02), З (q03), Д (q04), Е (q05), И (q06), Б (q07), Ж (q08), К (q09)

## Тип А
- **DCATS**: LLM-agent for data-centric TS AutoML; 4 компонента: TS dataset, metadata DB, LLM-agent, forecasting module
- **TaskParadigm: LLM-as-Agent (Tool Integration)**: LLM reasons over metadata → generates data enrichment proposals → iterative refinement
- **Prompt Design**: 5-section prompt (background, task, guidelines, neighbor sets, output format)

## Тип Ж
- **Task_Type**: Traffic volume forecasting (spatial-temporal)
- **Improvement_Percent**: 6% error reduction (average across all models + horizons)
- **Dataset**: Large-scale traffic volume (8600 locations, spatio-temporal)

## 3.2
- **Technique**: Prompt Engineering (Listing 1, 2), Iterative Proposal Refinement, Metadata-driven Data Selection

## 3.3
- **Baseline**: Models trained on all available time series vs. DCATS-selected subset
