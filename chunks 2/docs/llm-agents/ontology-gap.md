# LLM-Agents: Привязка к онтологии (ревизия v1.2)

**article_id:** `llm-agents` | **Дата:** 2026-05-22

**Ключевые пробелы:**
- **LLM-as-Agent (Tool Integration)** — новая парадигма: LLM как планировщик, не прямой обработчик TS
- **Data-Centric AutoML** — фокус на качестве данных, а не архитектуре модели
- **Metadata Reasoning** — LLM reasoning over TS metadata (не над самими TS)

**Покрытие:** ~80% (укороченный набор, 10 типов).
- **LLM-as-Agent** = "Tool Integration" (категория 5 из llm-ts-survey taxonomy)
- **Data-Centric paradigm**: LLM optimises data, not model. Method.strategy = "ToolIntegration" / "AgentBased"
- **Prompt-as-Controller**: промпт управляет AutoML процессом, а не предсказывает TS

## Chunking Plan

| ID | Раздел | Типы |
|----|--------|------|
| C01 | Abstract | А, Ж |
| C02 | 1. Introduction (DCATS data-centric idea) | А |
| C03 | 3. Methodology (4 components + prompt structure) | А |
| C04 | Experiments (6% improvement, traffic forecasting) | Ж |
