# TEST: Извлечённые сущности (ревизия v1.2)

**article_id:** `test` | **Дата:** 2026-05-22
**Типы (v1.2):** А (q00), А+ (q01), В (q02), З (q03), Г (q04), Б (q05), Д (q06), Е (q07), И (q08), Ж (q09), К (q10), Спец (q11)

## 3.1. Сущности Eval API

### Тип А — Способы применения LLM

| Сущность | Значение | Раздел |
|----------|----------|--------|
| **TEST** | Text-prototype-aligned TS Embedding: TS tokenization → encoder (instance-wise + feature-wise + text-prototype-aligned contrast) → soft prompts → frozen LLM | Abstract, 3 |
| **LLM-for-TS (model-centric)** | Парадигма: design/train foundation model from scratch OR fine-tune pre-trained LLM for TS | 1 |
| **TS-for-LLM (data-centric)** | Парадигма: convert TS into LLM-friendly representation; freeze LLM; TEST реализует эту парадигму | 1 |
| GPT4TS, LLM4TS | LLM-for-TS через tuning | Табл. 1 |
| PromptCast, Health Learner, METS, Text2ECG | LLM-for-TS через tool-augmented | Табл. 1 |
| LM-of-TS (Ma et al., Earth transformer) | Foundation model с нуля | Табл. 1 |
| TaskParadigm: **External Encoder + Frozen LLM** | TS-for-LLM: внешний энкодер создаёт эмбеддинги, LLM заморожен, soft prompts | 3 |
| TaskParadigm: **Contrastive Alignment** | Text embeddings как прототипы для ограничения TS embedding space | 3.3 |

### Тип Г — Представление TS / Токенизация

| Сущность | Значение | Раздел |
|----------|----------|--------|
| **Sliding Window Tokenization** | TS → non-overlapping subsequences/tokens произвольной длины | 3.1 |
| **Instance-wise Contrast** | Anchor-positive-negative: positive = augmented (jitter-and-scale, permutation-and-jitter), negative = non-overlapping instances | 3.2 |
| **Feature-wise Contrast** | Columns of feature matrix as soft labels; contrast between same feature columns + different feature columns | 3.2 |
| **Text-Prototype-Aligned Contrast** | Text token embeddings from LLM as prototypes → constrain TS embedding space → align to LLM's cognitive space | 3.3 |
| **TS Token Augmentation** | Weak (jitter-and-scale) + Strong (permutation-and-jitter) аугментации | 3.1 |
| TS_Representation | Contrastive TS embedding aligned to LLM text embedding space | 3 |
| Tokenization_Method | Sliding window segmentation → non-overlapping subsequences | 3.1 |

## 3.2. Онтология без вопросов

| Класс | Сущность |
|-------|----------|
| BaseLLM | 8 frozen LLMs различных структур и размеров (не специфицированы) |
| TuningStrategy | Frozen LLM; обучается только encoder + soft prompts |
| Technique | Sliding Window, Jitter-and-Scale, Permutation-and-Jitter, Instance-wise CL, Feature-wise CL, Text-Prototype Alignment, Auto-encoding |

## 3.3. Расширяемые

| Класс | Сущность |
|-------|----------|
| PromptDesign | Soft prompts для активации LLM |
| TrainingRegime | Few-shot, zero-shot (через frozen LLM) |
| Baseline | GPT4TS, LLM4TS, PromptCast, LM-of-TS |
