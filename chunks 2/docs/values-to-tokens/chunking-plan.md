# TokenCast (values-to-tokens): План семантического чанкинга (ревизия v1.2)

**article_id:** `values-to-tokens` | **Вопросов:** 12 (q00–q11), типы А–Л | **Дата:** 2026-05-22

## 1. Актуальное соответствие

| question_id | Тип | Раздел-источник |
|-------------|-----|-----------------|
| `values-to-tokens_q00` | **А** | Abstract (9–10), 1.Introduction (23–25), 3.2 Framework (41–45) |
| `values-to-tokens_q01` | **А+** | 4.Experiments: results tables |
| `values-to-tokens_q02` | **В** | 4.Experiments: real-world datasets with contextual features |
| `values-to-tokens_q03` | **З** | MSE, MAE (standard metrics) |
| `values-to-tokens_q04` | **Г** | 3.3 Time Series Discretization: VQ-based tokenizer, codebook, causal encoder |
| `values-to-tokens_q05` | **Б** | Symbolic discretization, Unified vocabulary, Autoregressive alignment, Generative fine-tuning |
| `values-to-tokens_q06` | **Д** | 3.4 Pre-trained LLM backbone; 3.5 Extended vocabulary + shared embedding |
| `values-to-tokens_q07` | **Е** | 3.2: Tokenizer → LLM Alignment → Generative Fine-tuning → De-tokenizer |
| `values-to-tokens_q08` | **И** | 3.3.1 RIN normalization; 3.5 Embedding initialization; training configs |
| `values-to-tokens_q09` | **Ж** | 4: horizons, task types, results |
| `values-to-tokens_q10` | **К** | 3.5: frozen LLM + trainable embedding; 3.6: generative fine-tuning |
| `values-to-tokens_q11` | **Л** | 3.3.1 RIN normalization (historical-only statistics) |

## 2. Итоговая таблица чанков

| ID | Раздел | Типы вопросов | Приоритет |
|----|--------|---------------|-----------|
| C01 | Abstract | А | Высокий |
| C02 | 1. Introduction (TokenCast concept) | А | Высокий |
| C03 | 3.2 Framework Overview (3 stages) | А, Е | Высокий |
| C04 | 3.3.1 Time Series Tokenizer (VQ + RIN) | Г, Л | Высокий |
| C05 | 3.3.2 Training Objective (tokenizer loss) | Г, И | Средний |
| C06 | 3.4 Pre-trained LLM Backbone | А, Д | Высокий |
| C07 | 3.5 Cross-Modality Alignment | Г, Б, К | Высокий |
| C08 | 3.6 Generative Fine-tuning | Б, К | Высокий |
| C09 | 4. Experiments setup (datasets+metrics) | В, З | Высокий |
| C10 | 4. Results (tables + improvement text) | А+, Ж | Высокий |

**Всего:** 10 чанков. Переиспользуемые: C03 (А+Е), C04 (Г+Л), C07 (Г+Б+К), C08 (Б+К).

## 3. Принципы

1. **Дословность:** verbatim.
2. **TokenCast = VQ tokenizer + LLM vocabulary extension + generative fine-tuning.**
3. **Ключевая инновация:** symbolic discretization как мост между TS и текстом.
