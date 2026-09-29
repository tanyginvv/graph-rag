# LLM-TS Survey: Привязка к онтологии и анализ пробелов (ревизия v1.2)

**article_id (Eval API):** `llm-ts-survey`
**Дата ревизии:** 2026-05-22
**Основание:** добавлены типы А+, З, Г, Б, Д, Е, И, Л к ранее покрытым А и В

---

## 1. Сопоставление извлечённых сущностей с классами онтологии

### Особенности обзорной статьи

LLM-TS Survey — **мета-статья**: не предлагает собственного метода, а систематизирует существующие. Это требует особого подхода к онтологии:

- **LLM_Method** здесь — не конкретный метод, а **категория методов** (Prompting, Quantization, Aligning, Vision Bridge, Tool)
- Статья выступает как **источник систематики** для графа знаний
- Каждая представительная работа (GPT4TS, Time-LLM, Lag-Llama и др.) — потенциальный узел в графе, связываемый с категорией
- **Тип А+** — не ablation-результат (как в method-статьях), а сравнительное руководство из раздела 4
- **Тип Д** — не единая архитектура, а перечень backbone-LLM
- **Тип Е** — общая схема LLM-пайплайна, а не конкретный пайплайн одного метода

### Покрытые классы (обновлено для v1.2)

| Сущность | Класс онтологии | Тип вопроса | Заполняется? |
|----------|----------------|-------------|-------------|
| 5 категорий таксономии | `Method` (категория методов) | А | Частично: `Method.category` не имеет значений типа "Prompting", "Quantization". Нужен `TaxonomyCategory`. |
| Рекомендации по выбору категории | `Method` (best) | А+ | Да — раздел 4 даёт ситуативные рекомендации |
| 16+ мультимодальных датасетов | `Dataset`, `Benchmark` | В | Да |
| MSE, MAE, RMSE, MAPE, Accuracy, F1, BLEU, ROUGE, METEOR, EM | `Evaluation_Metric` | З | Частично: NLP-метрики (BLEU, ROUGE, METEOR, EM) отсутствуют в `Metric.name` |
| Number-Agnostic/Specific, VQ-VAE, K-Means, Contrastive Alignment | `TS_Representation`, `Tokenization_Method` | Г | Да — богатый набор стратегий представления TS |
| Patching, Down-sampling, Domain Knowledge, End-to-end Training | `Improvement_Method`, `Optimization_Technique` | Б | Да |
| GPT-2, LLaMA-7B, BART, PaLM, GPT-4, CLIP, LLaVA | `BaseLLM`, `Neural_Architecture` | Д | Да |
| 5 стадий LLM-пайплайна + формальная модель | `Pipeline`, `Workflow_Step` | Е | Да — общая схема пайплайна |
| Zero-shot, Two-stage, End-to-end, Off-the-shelf | `TrainingRegime` | И | Да |
| Scaling, Fixed Precision, VQ-VAE, Binning, Patching, Decomposition | `DataPreprocessing` (normalization) | Л | Да — широкий спектр методов нормировки |
| 6 доменов (IoT, Finance, Healthcare, Audio, Music, Speech) | `Domain` | — (3.2) | Частично: "IoT", "Audio", "Music", "Speech" отсутствуют |
| Представительные работы (~25) | `Method` | — (3.2) | Да, если они уже есть в графе |

### Пробелы (обновлено)

| Сущность | Проблема | Предложение | Статус после ревизии |
|----------|----------|-------------|---------------------|
| **5 категорий таксономии** | Это не методы, а **мета-категории**. Не укладываются в `Method.strategy`. | Новый класс `TaxonomyCategory`. | **Подтверждён** — критичен для обзорных статей |
| **NLP-метрики** (BLEU, ROUGE, METEOR, EM) | Отсутствуют в `Metric.name`. | Добавить в `Metric.name`. | **Подтверждён** — теперь явно привязан к типу З (q03) |
| **Новые домены** (IoT, Audio, Music, Speech) | Отсутствуют в `Domain.name`. | Расширить `Domain.name`. | **Подтверждён** |
| **Связь «обзор → метод»** | Обзорная статья SURVEYS/CLASSIFIES, а не PROPOSES. | Связь `Article SURVEYS Method` или `Article CLASSIFIES TaxonomyCategory`. | **Подтверждён** |
| **Сравнительные рекомендации (тип А+)** | Для обзора нет ablation, но есть comparative guidance (раздел 4). | В онтологии `Method` нужен атрибут `recommendation_context` или связь `Method RECOMMENDED_FOR Scenario`. | **Новый пробел** — обнаружен при добавлении типа А+ |
| **LLM Pipeline Stages (тип Е)** | 5 стадий (input → tokenization → embedding → LLM → output) — общая схема, а не конкретный пайплайн. | `Pipeline` может иметь `pipeline_type: "generic" | "method_specific"`. | **Новый пробел** — для различения общих и конкретных пайплайнов |
| **Мультимодальность Dataset** | `Dataset.frequency` и `num_variates` плохо применимы к multimodal TS+text. | Атрибут `modalities: string[]` или `has_text_modality: bool`. | **Подтверждён** |

---

## 2. Кандидаты на расширение онтологии (обновлено)

| Кандидат | Обоснование | Приоритет | Предлагаемое решение |
|----------|-------------|-----------|---------------------|
| **TaxonomyCategory** | Уникальная сущность для обзорных статей: иерархическая категоризация методов. 5 категорий + подкатегории. | **Критический** | Новый класс: `TaxonomyCategory { name, description, parent_category, level }`. Связи: `Article → TaxonomyCategory` (SURVEYS), `Method → TaxonomyCategory` (BELONGS_TO). |
| **NLP-метрики в Metric** | BLEU, ROUGE, METEOR, EM — требуется для типа З. | Высокий | Расширить `Metric.name`. |
| **Новые домены** | IoT, Audio, Music, Speech. | Средний | Расширить `Domain.name`. |
| **Мультимодальность Dataset** | Атрибут `modalities` для multimodal TS+text датасетов. | Средний | Добавить `Dataset.modalities: string[]`. |
| **RecommendationContext / Scenario** | Для типа А+ в обзорах: ситуативные рекомендации (напр. «Prompting лучше при zero-shot»). | Средний | `Method.recommendation_context` или связь `Method RECOMMENDED_FOR Scenario`. |
| **Pipeline.generic vs specific** | Обзор даёт общую схему пайплайна (тип Е), method-статьи — конкретные пайплайны. | Низкий | `Pipeline.pipeline_type: enum["generic", "method_specific"]`. |

---

## 3. Проверка заполнения атрибутов существующих классов (без изменений)

### Method (для представительных работ, упоминаемых в обзоре)

Обзор **не детализирует** атрибуты конкретных методов — только относит их к категориям. Для заполнения атрибутов нужно обращаться к оригинальным статьям.

### Dataset (для мультимодальных датасетов)

`Dataset.frequency` и `Dataset.num_variates` плохо применимы к мультимодальным датасетам — подтверждён пробел `modalities`.

---

## 4. Итог (обновлено)

- **Покрытие онтологии:** ~75% (10 типов вопросов × разная степень покрытия; обзорная специфика снижает полноту для А+, Д, Е)
- **Ключевые пробелы (6):**
  1. **TaxonomyCategory** (критический) — для мета-категорий обзорных статей
  2. **NLP-метрики** в `Metric.name` (высокий) — теперь явно привязан к типу З
  3. **Новые домены** в `Domain.name` (средний)
  4. **Мультимодальность** в `Dataset` (средний)
  5. **RecommendationContext** (новый, средний) — для типа А+ в обзорах
  6. **Pipeline.pipeline_type** (новый, низкий) — generic vs method-specific
- **Снятые проблемы (0):** все пробелы из v1.0 подтверждены; новых ложных не выявлено.
- **Рекомендация:** `TaxonomyCategory` — первоочередное расширение онтологии. NLP-метрики и домены — накопить evidence из других статей. RecommendationContext и Pipeline.pipeline_type — рассмотреть после полной ревизии всех 12 статей.
