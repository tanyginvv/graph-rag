# One-for-All: Привязка к онтологии (ревизия v1.2)

**article_id:** `one-for-all` | **Дата:** 2026-05-22

**Ключевые пробелы:**
- **rsLoRA** — новый PEFT метод, требует "rsLoRA" в `TuningStrategy.peft_method`
- **Rank Stabilization** — математически обоснованная стабилизация градиентов при низких рангах
- **Extreme PEFT** — 0.55M параметров (0.2% от GPT-2)

**Покрытие:** ~90%.
- **Eff.*MSE**: новый тип метрики для parameter efficiency evaluation
- **Cross-task unification**: One-for-All — единственная модель, покрывающая forecasting + classification + anomaly detection → `Method.scope = "CrossTask"`

## Chunking Plan

| ID | Раздел | Типы |
|----|--------|------|
| C01 | Abstract | А, З |
| C02 | I. Novelty (rsLoRA + efficiency claims) | А |
| C03 | III. Methodology (rsLoRA mechanism) | А |
| C04 | IV. Experiments (metrics section) | З |
