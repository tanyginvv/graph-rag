# S2IP-LLM: Привязка к онтологии (ревизия v1.2)

**article_id:** `s2ip-llm` | **Дата:** 2026-05-22
**Добавлены:** А+, В, З, Г, Б, Е, И, Ж, К, Л.

**Ключевые пробелы:**
- **Semantic Anchors** — текстовые прототипы (как в Time-LLM), подтверждает TextPrototype
- **Decomposition Tokenization** (trend+seasonal+residual) — Technique
- **Partial Fine-tuning** (pos emb + layer norm only) — TuningStrategy
- **PseudoWordEmbedding** — подтверждён (vitro + S2IP-LLM)

**Покрытие:** ~90%.

## 1. Покрытие

| Сущность | Класс | Статус |
|----------|-------|--------|
| S²IP-LLM | `Method` (name="S2IP-LLM", category="LLM-based", strategy="PromptLearning", channel_strategy="Independent", scope="MultiTask") | ✓ |
| GPT-2 | `BaseLLM` | ✓ |
| Partial FT (pos emb + LN) | `TuningStrategy` (approach="PEFT", peft_method="PartialFT") | Частично: "PartialFT" нет в enum, но близко к "FPT" |
| Decomposition + Patching + Alignment | `Technique` | ✓ |
| 7+ датасетов | `Dataset` | ✓ |
| MSE, MAE, SMAPE, MASE, OWA | `Metric` | ✓ |

## 2. Пробелы

| Пробел | Предложение |
|--------|-------------|
| **PromptLearning** в Method.strategy | S²IP-LLM — это prompt learning (не Reprogramming, не Inversion, не Alignment). Нужно новое значение `"PromptLearning"` |
| **Semantic Anchors** | Специфическая техника: редукция E → E' → top-K alignment. Пересекается с VITRO (pseudo-words), Time-LLM (text prototypes) | Общий класс `PromptToken` или `SemanticAnchor` |
| **Additive Decomposition + Reconstruction** | Уникальная черта S²IP-LLM: trend+seasonal+residual → separate prediction → sum | `Technique` с type="Decomposition" (уже существует) |
| **PartialFT** (fine-tune only pos emb + LN) | Текущий `TuningStrategy.approach` не включает этот режим | Добавить `"PartialFT"` или использовать `trainable_components` |

## 3. Итог

- Покрытие: ~85%
- Ключевой пробел: `"PromptLearning"` в `Method.strategy` — новый тип взаимодействия LLM с TS
- Semantic Anchors — общая концепция с VITRO (pseudo-words), TEST (text prototypes) — кандидат на общий класс
- GPT-2 partial fine-tuning (pos emb + LN) — уточнение `TuningStrategy`
