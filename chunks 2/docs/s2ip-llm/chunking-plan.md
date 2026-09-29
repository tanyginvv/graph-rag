# S2IP-LLM: План семантического чанкинга (ревизия v1.2)

**article_id (Eval API):** `s2ip-llm` | **Вопросов:** 12 (q00–q11), типы А–Л
**Дата ревизии:** 2026-05-22

---

## 1. Актуальное соответствие «question_id → тип → раздел-источник»

| question_id | Тип | Раздел-источник |
|-------------|-----|-----------------|
| `s2ip-llm_q00` | **А** | Abstract (12–13), 1.Introduction (24–30), 3.Methodology (42–110) |
| `s2ip-llm_q01` | **А+** | 4.1–4.3 Results; 4.4 Ablation |
| `s2ip-llm_q02` | **В** (расшир.) | 4.1 (120): Weather, Electricity, Traffic, ETT×4; M4; App. A.3 |
| `s2ip-llm_q03` | **З** | 4.1: MSE, MAE; 4.2: SMAPE, MASE, OWA |
| `s2ip-llm_q04` | **Г** (спец.²) | 3.2 Tokenization: decomposition + patching; 3.3 Semantic Anchors + Alignment |
| `s2ip-llm_q05` | **Б** | Decomposition tokenization, Semantic Anchors, Score-matching alignment, Prefix-prompt |
| `s2ip-llm_q06` | **Д** | 3.5: GPT-2 backbone, positional embedding + layer norm fine-tuned |
| `s2ip-llm_q07` | **Е** | 3: Tokenize → Align → Retrieve top-K → Prefix-prompt → GPT-2 → Project |
| `s2ip-llm_q08` | **И** | 3.2 RevIN; 3.5 frozen attention/FFN, fine-tune pos emb + layer norm |
| `s2ip-llm_q09` | **Ж** (без %) | 4.1–4.3: long-term/short-term/few-shot; horizons {96,192,336,720} |
| `s2ip-llm_q10` | **К** | 3.5: partial fine-tuning of GPT-2 (pos emb + layer norm only) |
| `s2ip-llm_q11` | **Л** | 3.2: RevIN normalization; decomposition |

---

## 2. Итоговая таблица чанков

| ID | Раздел | Типы вопросов | Приоритет |
|----|--------|---------------|-----------|
| C01 | Abstract | А | Высокий |
| C02 | 1. Introduction (S²IP-LLM concept) | А | Высокий |
| C03 | 3. Methodology overview | А, Е | Высокий |
| C04 | 3.2 Tokenization (decomposition+patching) | Г, Б, Л | Высокий |
| C05 | 3.3 Semantic Space Prompting | Г, Б | Высокий |
| C06 | 3.5 Backbone (GPT-2 + fine-tuning) | Д, К | Высокий |
| C07 | 3.4 Optimization Objective | Е, И | Средний |
| C08 | 4.1 Long-term Setups (datasets+metrics+horizons) | В, З, Ж | Высокий |
| C09 | 4.1 Results (Table 1 + text) | А+, Ж | Высокий |
| C10 | 4.2 Short-term (M4 setups+results) | В, Ж | Средний |
| C11 | 4.3 Few-shot | Ж | Средний |
| C12 | 4.4 Ablation | А+, Б | Средний |
| C13 | 3.2 RevIN normalization | Л, И | Средний |
| C14 | 3.5 Fine-tuning details | К, И | Средний |

**Всего:** 14 чанков. Переиспользуемые: C03 (А+Е), C04 (Г+Б+Л), C05 (Г+Б), C06 (Д+К), C08 (В+З+Ж), C09 (А++Ж).

---

## 3. Принципы

1. **Дословность:** verbatim из markdown-версии.
2. **q02=В расширенная формулировка, q04=Г спец. формулировка², q09=Ж без %⁵.**
3. **S²IP-LLM = decomposition tokenization + semantic anchors + GPT-2 partial fine-tuning.**
