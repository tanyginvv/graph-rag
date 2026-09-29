# TS-RAG: Привязка к онтологии (ревизия v1.2)

**article_id:** `ts-rag` | **Дата:** 2026-05-22

## Ontology Gap

| Пробел | Предложение |
|--------|-------------|
| **RAG for TS** | Новая парадигма: retrieval-augmented generation адаптирована для time series. Method.strategy = "RAG" |
| **MoE Augmentation** | Mixture-of-Experts для динамического fusion. Техника, не покрываемая текущими типами |
| **Knowledge Base (TS-specific)** | Специфическая структура: triplets (x_i, e_i, y_i). Пересекается с общим RAG, но для TS |

## Chunking Plan

| ID | Раздел | Строки | Тип | Типы | Приоритет |
|----|--------|--------|-----|------|-----------|
| C01 | Abstract | 13 | prose | А, Б | Высокий |
| C02 | 1. Introduction (проблемы TSFM + RAG решение) | 23–27 | prose | А | Высокий |
| C03 | 2.1 TSFMs + 2.2 RAG for TS | 31–39 | prose | А | Высокий |
| C04 | 3.1 Retrieval Knowledge Base | 49–69 | prose | Б | Высокий |
| C05 | 3.2 RAG + MoE Augmentation | 71–80 | prose | А, Б | Высокий |
