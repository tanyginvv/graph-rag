# One-for-All: Извлечённые сущности (ревизия v1.2)

**article_id:** `one-for-all` | **Дата:** 2026-05-22
**Типы (v1.2):** А (q00), А+ (q01), В (q02), З (q03), Г (q04), Б (q05), Д (q06), Е (q07), И (q08), Ж (q09), К (q10), Л (q11)

## Тип А

- **One-for-All**: rsLoRA PEFT framework; GPT-2 frozen backbone; rank-16 adapters → 0.55M trainable params
- **GPT4TS, Time-LLM**: baseline LLM methods
- **TaskParadigm: rsLoRA PEFT** — Gaussian rank-stabilized LoRA; provable gradient stability under non-stationarity

## Тип З

- **MSE, MAE**: long-term forecasting
- **SMAPE, MASE, OWA**: short-term (M3, M4)
- **Eff.*MSE**: parameter efficiency metric (5.50 vs TimesNet 30.00)

## 3.2

- **BaseLLM**: GPT-2
- **TuningStrategy**: PEFT via rsLoRA (rank=16, α=32); frozen attention, trainable pos emb + output
- **Technique**: RevIN, Patching, rsLoRA (Gaussian scaling β_r = α/√r)

## 3.3

- **ComputationalCost**: 0.55M params, 2.2 MiB memory
- **Baseline**: GPT4TS, Time-LLM, TimesNet, DLinear, PatchTST, Autoformer, FEDformer, ETSformer
