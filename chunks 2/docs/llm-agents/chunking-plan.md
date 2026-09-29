# LLM-Agents (DCATS): План семантического чанкинга (ревизия v1.2)

**article_id:** `llm-agents` | **Вопросов:** 10 (q00–q09), группа 3 (укороченный набор) | **Дата:** 2026-05-22

## 1. Актуальное соответствие (группа 3)

| question_id | Тип | Семантика | Раздел-источник |
|-------------|-----|-----------|-----------------|
| `llm-agents_q00` | **А** | Все способы применения LLM | Abstract, 1.Introduction, 3.DCATS Framework |
| `llm-agents_q01` | **А+** | Наилучшие способы | 4.Results: 6% error reduction |
| `llm-agents_q02` | **В** (расшир.⁹) | Бенчмарки/датасеты | 4: traffic volume forecasting dataset |
| `llm-agents_q03` | **З** | Метрики | Forecasting error metrics (MSE/MAE) |
| `llm-agents_q04` | **Д** | Архитектура / backbone | 3: LLM-agent + forecasting module; 4 forecasting models |
| `llm-agents_q05` | **Е** | Пайплайн | 3: Metadata DB → LLM-agent reasoning → Data enrichment → Forecast |
| `llm-agents_q06` | **И** | Подготовка данных | 3: metadata processing, auxiliary TS selection |
| `llm-agents_q07` | **Б** | Способы улучшения | Data-centric enrichment, iterative refinement, metadata reasoning |
| `llm-agents_q08` | **Ж** | Тип задачи, горизонт | Traffic forecasting, multiple horizons |
| `llm-agents_q09` | **К** | Предобучение/дообучение | LLM-agent without fine-tuning (reasoning over metadata) |

**Примечание:** нет типов Г и Л в укороченном наборе.

## 2. Итоговая таблица чанков

| ID | Раздел | Типы вопросов | Приоритет |
|----|--------|---------------|-----------|
| C01 | Abstract | А | Высокий |
| C02 | 1. Introduction (DCATS concept) | А | Высокий |
| C03 | 3. DCATS Framework (4 components) | А, Д, Е | Высокий |
| C04 | 3. Metadata Reasoning + Data Enrichment | Б, И | Высокий |
| C05 | 3. Iterative Refinement | Б, Е | Средний |
| C06 | 4. Experiments setup (dataset+models+metrics) | В, З, Д | Высокий |
| C07 | 4. Results (6% error reduction, horizons) | А+, Ж | Высокий |
| C08 | 3. LLM-agent without fine-tuning | К | Средний |

**Всего:** 8 чанков. Переиспользуемые: C03 (А+Д+Е), C06 (В+З+Д).

## 3. Принципы

1. **DCATS = Data-Centric Agent:** LLM-agent reasoning over metadata, не fine-tuning моделей.
2. **Группа 3 (укороченный набор):** 10 вопросов, без Г и Л.
3. **q02=В расширенная формулировка⁹.**
