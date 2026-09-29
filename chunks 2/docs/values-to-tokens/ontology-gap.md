# TokenCast: Привязка к онтологии (ревизия v1.2)

**article_id:** `values-to-tokens` | **Дата:** 2026-05-22

## Ontology Gap

| Пробел | Предложение |
|--------|-------------|
| **Symbolic Discretization** в Method.strategy | TokenCast — это "Quantization" (категория из llm-ts-survey) + "GenerativeFineTuning". Не укладывается в текущие Reprogramming/Inversion/Alignment. |
| **VQ Codebook** | Техника общая с DeWave, TOTEM, Chronos из обзора. Класс `VQCodebook` или `Technique` с type="VectorQuantization" |
| **Unified Vocabulary** | Расширение LLM vocabulary: V_orig + K temporal + S special tokens. Пересекается с VITRO (pseudo-words), S2IP-LLM (semantic anchors) |
| **Decoupled RevIN** | RevIN только на исторических данных (не на всём окне) — специфическая модификация стандартного RevIN |

## Chunking Plan

| ID | Раздел | Строки | Тип | Типы вопросов | Приоритет |
|----|--------|--------|-----|---------------|-----------|
| C01 | Abstract | 9 | prose | А, Г | Высокий |
| C02 | 1. Introduction (symbolic discretization идея) | 22–25 | prose | А | Высокий |
| C03 | 3.3.1 Time Series Tokenizer | 49–55 | prose | Г | Высокий |
| C04 | 3.4 LLM Backbone + Prompt Template | 76–78 | prose | А | Высокий |
| C05 | 3.5 Cross-Modality Alignment | 80–99 | prose + формулы | А, Г | Высокий |
| C06 | 3.6 Generative Fine-tuning | 101–105 | prose | А | Средний |
| C07 | 4.1.1 Datasets + Табл. 1 | 113–162 | table_with_prose | В | Низкий |

**Переиспользование:** C01 (Abstract) — А+Г; C05 (Alignment) — А+Г.
