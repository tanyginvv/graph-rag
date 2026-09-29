# TEST: Привязка к онтологии (ревизия v1.2)

**article_id:** `test` | **Дата:** 2026-05-22

## Ontology Gap

| Пробел | Предложение |
|--------|-------------|
| **TS-for-LLM** vs **LLM-for-TS** — мета-парадигмы | TEST вводит принципиальную дихотомию: model-centric vs data-centric. Method.strategy может принимать "DataCentric" или "ExternalEncoder" |
| **Text-Prototype-Aligned Contrast** | Специфическая техника: text embeddings как прототипы. Пересекается с VITRO (pseudo-words), S2IP-LLM (semantic anchors), Time-LLM (text prototypes) |
| **Feature-wise Contrast** | Уникальная техника TEST: columns of feature matrix as soft labels. `Technique` с type="FeatureContrast" |
| **Instance-wise Contrast** | Стандартная contrastive learning техника. `Technique` с type="ContrastiveLearning" |

## Chunking Plan

| ID | Раздел | Строки | Тип | Типы | Приоритет |
|----|--------|--------|-----|------|-----------|
| C01 | Abstract | 19 | prose | А, Г | Высокий |
| C02 | 1. Introduction (LLM-for-TS vs TS-for-LLM парадигмы) | 25–31 | prose | А | Высокий |
| C03 | Табл. 1 (категории TS+LLM работ) | 48–89 | table_with_prose | А | Высокий |
| C04 | 3.1 TS Token Augmentation and Encoding | 113–120 | prose | Г | Высокий |
| C05 | 3.2 Instance-wise + Feature-wise Contrast | 121–140 | prose + формулы | Г | Высокий |
| C06 | 3.3 Text-Prototype-Aligned Contrast | далее | prose | А, Г | Высокий |
| C07 | 3.4 Prompt Design | далее | prose | А | Средний |

**Переиспользование:** C01 (Abstract) — А+Г; C06 (Text-Prototype Alignment) — А+Г.
