# LLM-PS: План семантического чанкинга (ревизия v1.2)

**article_id (Eval API):** `llm-ps`
**Вопросов Eval API:** 12 (q00–q11), типы А, А+, В, З, Г, Б, Д, Е, И, Ж, К, Л
**Дата ревизии:** 2026-05-22

---

## 1. Актуальное соответствие «question_id → тип → раздел-источник»

| question_id | Тип | Семантика | Раздел-источник |
|-------------|-----|-----------|-----------------|
| `llm-ps_q00` | **А** | Все способы применения LLM | Abstract (15–16), 1.Introduction (25–31), 3.Approach (49–160), 5.Conclusion |
| `llm-ps_q01` | **А+** | Наилучшие способы | 4.1 (174): 15/18 best; 4.2 (228): best SMAPE/MASE/OWA; 4.3–4.4 few-shot/zero-shot |
| `llm-ps_q02` | **В** | Бенчмарки и наборы данных | 4.1 (172): ETT×4, Weather, Electricity, Traffic, ILI, ECG; 4.2: M4; App. A.2 |
| `llm-ps_q03` | **З** | Метрики оценки | 4.1: MSE, MAE; 4.2: SMAPE, MASE, OWA; App. A.3 |
| `llm-ps_q04` | **Г** | Представление TS | 3.1–3.2: MSCNN multi-scale features; 3.3: T2T semantic extraction |
| `llm-ps_q05` | **Б** | Способы улучшения | MSCNN (multi-scale), Wavelet decoupling, T2T semantics, LoRA fine-tuning |
| `llm-ps_q06` | **Д** | Нейросетевая архитектура | 3.4: GPT-2 (6 layers) backbone; MSCNN blocks; T2T encoder-decoder |
| `llm-ps_q07` | **Е** | Пайплайн и шаги | 3: MSCNN → Wavelet Decoupling → T2T → Feature Transfer → LoRA-tuned LLM |
| `llm-ps_q08` | **И** | Подготовка данных и обучение | 3.4: LoRA (r=8, α=32, dropout=0.1), Adam, LR=5e-4, λ=0.01 |
| `llm-ps_q09` | **Ж** | Тип задачи, горизонт, улучшение | 4.1: long-term H={96,192,336,720}; 4.2: short-term; 4.3–4.4: few/zero-shot |
| `llm-ps_q10` | **К** | Предобучение/дообучение | 3.4: LoRA fine-tuning of pre-trained GPT-2 (first 6 layers); parameter-efficient |
| `llm-ps_q11` | **Л** | Нормировка / преобразование | Standard normalization (RevIN reference); patching in T2T |

---

## 2. План чанков

### Тип А (`llm-ps_q00`)
| Чанк | Раздел | Строки | Размер |
|-------|--------|--------|--------|
| C01 | Abstract | 15–16 | ~120 |
| C02 | 1. Introduction: проблема + MSCNN + T2T | 25–31 | ~350 |
| C03 | 3. Approach: overview (Fig. 2 description) | 49–53 | ~200 |
| C04 | 3.4 Efficient Training (LoRA + GPT-2) | 141–160 | ~300 |

### Тип А+ (`llm-ps_q01`)
| Чанк | Раздел | Строки | Размер |
|-------|--------|--------|--------|
| C05 | 4.1 Results: 15/18 best, 6% over CALF | 174 | ~150 |
| C06 | 4.2 Short-term: best SMAPE/MASE/OWA | 228 | ~100 |

### Тип В (`llm-ps_q02`)
| Чанк | Раздел | Строки | Размер |
|-------|--------|--------|--------|
| C07 | 4.1 Setups: ETT×4, Weather, Electricity, Traffic, ILI, ECG | 172 | ~150 |
| C08 | 4.2: M4 dataset description | 226 | ~100 |

### Тип З (`llm-ps_q03`)
| C09 | 4.1: MSE, MAE; 4.2: SMAPE, MASE, OWA | 172, 226 | ~100 |

### Тип Г (`llm-ps_q04`)
| C10 | 3.1 MSCNN: multi-scale features with varying receptive fields | 55–77 | ~300 |
| C11 | 3.3 T2T: semantic extraction via masked patch reconstruction | 121–139 | ~300 |

### Тип Б (`llm-ps_q05`)
| C10 | 3.1 MSCNN (pattern capture) | 55–77 | ~300 |
| C12 | 3.2 Wavelet Decoupling + Assembling | 81–118 | ~350 |
| C11 | 3.3 T2T (semantic enrichment) | 121–139 | ~300 |
| C13 | 3.4 LoRA fine-tuning (efficiency) | 141–148 | ~150 |

### Тип Д (`llm-ps_q06`)
| C14 | 3.4: GPT-2 (6 layers) backbone | 168 | ~80 |
| C10 | 3.1 MSCNN architecture | 55–77 | ~300 |
| C15 | 3.3 T2T encoder-decoder structure | 121–125 | ~150 |

### Тип Е (`llm-ps_q07`)
| C03 | 3. Approach overview | 49–53 | ~200 |
| C16 | 3.1–3.4: full pipeline description | 55–160 | ~500 |

### Тип И (`llm-ps_q08`)
| C17 | 3.4: LoRA params (r=8, α=32, dropout=0.1), Adam, LR=5e-4, λ=0.01 | 141–168 | ~200 |
| C18 | 4. Implementation Details | 168 | ~100 |

### Тип Ж (`llm-ps_q09`)
| C19 | 4.1 Setups: long-term H={96,192,336,720} | 172 | ~100 |
| C05 | 4.1 Results: improvement % | 174 | ~150 |
| C20 | 4.2–4.4: short-term, few-shot, zero-shot setups | 226, (4.3, 4.4) | ~200 |

### Тип К (`llm-ps_q10`)
| C17 | 3.4: LoRA on GPT-2 (6 layers), parameter-efficient | 141–168 | ~200 |
| C21 | 2.Related Work: LLM fine-tuning approaches context | 39–41 | ~150 |

### Тип Л (`llm-ps_q11`)
| C22 | Standard normalization (RevIN) + patching in T2T | 125, App. | ~100 |

---

## 3. Итоговая таблица

| ID | Раздел | Типы вопросов | Приоритет |
|----|--------|---------------|-----------|
| C01 | Abstract | А | Высокий |
| C02 | 1. Introduction | А | Высокий |
| C03 | 3. Approach overview | А, Е | Высокий |
| C04 | 3.4 Efficient Training | А | Высокий |
| C05 | 4.1 Results (15/18 best, 6% over CALF) | А+, Ж | Высокий |
| C06 | 4.2 Short-term results | А+ | Средний |
| C07 | 4.1 Setups (datasets) | В | Высокий |
| C08 | 4.2 M4 dataset | В | Средний |
| C09 | 4.1–4.2 Metrics | З | Средний |
| C10 | 3.1 MSCNN | Г, Б, Д | Высокий |
| C11 | 3.3 T2T | Г, Б | Высокий |
| C12 | 3.2 Wavelet Decoupling | Б | Высокий |
| C13 | 3.4 LoRA (improvement) | Б | Средний |
| C14 | 3.4 GPT-2 backbone | Д | Высокий |
| C15 | 3.3 T2T encoder-decoder | Д | Средний |
| C16 | 3.1–3.4 Full pipeline | Е | Средний |
| C17 | 3.4 LoRA params + training | И, К | Высокий |
| C18 | 4. Implementation Details | И | Средний |
| C19 | 4.1 Long-term setups | Ж | Высокий |
| C20 | 4.2–4.4 Short/few/zero-shot | Ж | Средний |
| C21 | 2. Related Work (LLM fine-tuning) | К | Средний |
| C22 | Normalization + patching | Л | Средний |

**Всего:** 22 чанка. Переиспользуемые: C03 (А+Е), C05 (А++Ж), C10 (Г+Б+Д), C11 (Г+Б), C17 (И+К).

---

## 4. Принципы

1. **Дословность:** verbatim из markdown-версии.
2. **LLM-PS = MSCNN + T2T + LoRA-tuned GPT-2:** три ключевых компонента, каждый — отдельный чанк.
3. **Не чанкируются:** References, Appendix таблицы (кроме явно указанных).
