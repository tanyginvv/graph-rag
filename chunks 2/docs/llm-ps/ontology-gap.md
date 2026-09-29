# LLM-PS: Привязка к онтологии (ревизия v1.2)

**article_id (Eval API):** `llm-ps` | **Дата:** 2026-05-22
**Добавлены:** А+, В, З, Г, Д, Е, И, Ж, К, Л к ранее покрытым А и Б.

**Ключевые пробелы:**
- LoRA (уже в онтологии как peft_method) — подтверждён
- MSCNN (Multi-Scale CNN) — Technique с type="MultiScaleConvolution"
- Wavelet Transform Decoupling — Technique с type="Decomposition"
- T2T (Time-to-Text) — Technique с type="SemanticExtraction"
- GPT-2 (6 layers) — BaseLLM

**Покрытие:** ~90% (12 типов).

## 1. Покрытие онтологии

| Сущность | Класс | Статус |
|----------|-------|--------|
| LLM-PS | `Method` (name="LLM-PS", category="LLM-based", strategy="Alignment", channel_strategy="Independent", scope="MultiTask", prompt_type="None") | ✓ |
| MSCNN | `Technique` (type="Tokenization" или новый type="PatternExtraction") | Частично |
| Wavelet Decoupling | `Technique` (type="Decomposition") | ✓ |
| T2T | `Technique` (type="Alignment") | Частично |
| LoRA | `TuningStrategy` (approach="PEFT", peft_method="LoRA") | ✓ |
| GPT-2 | `BaseLLM` | ✓ |
| 7 datasets | `Dataset` | ✓ |
| MSE, MAE, SMAPE, OWA | `Metric` | ✓ (MSAE отсутствует) |

## 2. Пробелы

| Пробел | Предложение |
|--------|-------------|
| **MSAE** (Mean Absolute Scaled Error) | Добавить в `Metric.name` |
| **MSCNN** — не просто Tokenization, а специфический multi-scale feature extractor | Допустимо как `Technique` с type="PatternExtraction" |
| **T2T (Time-to-Text)** — masked autoencoder для семантической разметки TS | Новый класс или `Technique` с type="SemanticExtraction" |
| **Wavelet Transform (DB4)** | `Technique` с type="Decomposition" (уже существует) |

## 3. Итог

- Покрытие: ~90% (наиболее полное среди проанализированных статей)
- Ключевой пробел: MSAE в Metric.name
- Менее критично: MSCNN и T2T — специфичные техники, но укладываются в `Technique`
- LoRA уже покрывается `TuningStrategy.peft_method`
