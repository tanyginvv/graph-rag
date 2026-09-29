from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

FIRST_STAGE_ENTITY_LABELS = (
    "Method",
    "Architecture",
    "Technique",
    "TuningStrategy",
    "DataPreprocessing",
    "Pipeline",
    "Dataset",
    "Benchmark",
    "Metric",
    "Domain",
    "Task",
    "Experiment",
    "ExperimentResult",
    "TaxonomyCategory",
)

EXACT_LABEL_OVERRIDES = {
    "accuracy saturation": "ExperimentResult",
    "adamw": "TuningStrategy",
    "automl": "Technique",
    "best-so-far": "ExperimentResult",
    "data preparation": "DataPreprocessing",
    "date-time features": "DataPreprocessing",
    "denormalization": "DataPreprocessing",
    "early stopping": "Experiment",
    "experiment results": "ExperimentResult",
    "few-shot": "Experiment",
    "forecasting module": "Pipeline",
    "from scratch": "TuningStrategy",
    "frozen backbone": "TuningStrategy",
    "frozen encoder": "TuningStrategy",
    "frozen llm": "TuningStrategy",
    "generalization": "ExperimentResult",
    "horizon": "Experiment",
    "hyperparameter search": "Experiment",
    "improvement": "ExperimentResult",
    "iqr": "DataPreprocessing",
    "lag features": "DataPreprocessing",
    "linear": "Method",
    "lookback window": "DataPreprocessing",
    "lora": "TuningStrategy",
    "median": "DataPreprocessing",
    "metadata": "Technique",
    "moe": "Technique",
    "outliers": "DataPreprocessing",
    "partial fine-tuning": "TuningStrategy",
    "peft": "TuningStrategy",
    "pipeline": "Pipeline",
    "pipeline stages": "Pipeline",
    "pretraining": "TuningStrategy",
    "probabilistic": "Task",
    "proposal generation": "Pipeline",
    "proposal refinement": "Pipeline",
    "qlora": "TuningStrategy",
    "residual": "Architecture",
    "rslora": "TuningStrategy",
    "sample trajectories": "Experiment",
    "scaling": "TaxonomyCategory",
    "seasonal": "Technique",
    "sota": "ExperimentResult",
    "stage 2": "Pipeline",
    "stratified sampling": "Experiment",
    "student's t": "Technique",
    "t2t": "Technique",
    "tool": "Technique",
    "trainable": "TuningStrategy",
    "trainable parameters": "TuningStrategy",
    "trend": "Technique",
    "ts-for-llm": "TaxonomyCategory",
    "value scaling": "DataPreprocessing",
    "zero-shot": "Experiment",
}

METRIC_KEYWORDS = frozenset(
    {
        "accuracy",
        "average rank",
        "average rank",
        "bleu",
        "crps",
        "eff.mse",
        "exact match",
        "f1",
        "mae",
        "mape",
        "mase",
        "meteor",
        "metric",
        "mse",
        "msmape",
        "owa",
        "precision",
        "recall",
        "rmse",
        "rouge",
        "scaled mae",
        "smape",
    }
)

BENCHMARK_KEYWORDS = frozenset(
    {
        "27 datasets",
        "benchmark",
        "m3",
        "m4",
        "monash",
        "uea",
    }
)

DATASET_KEYWORDS = frozenset(
    {
        "california",
        "ecg",
        "ett",
        "etth",
        "ettm",
        "exchange",
        "electricity",
        "google trends",
        "ili",
        "metr-la",
        "msl",
        "pems",
        "psm",
        "road network",
        "smap",
        "smd",
        "stock",
        "stock-na",
        "stock-ny",
        "swat",
        "traffic",
        "weather",
        "wikimedia",
    }
)

DOMAIN_KEYWORDS = frozenset(
    {
        "clinical",
        "economic",
        "financial",
        "health",
        "nature",
        "sensor",
        "vision",
        "web",
    }
)

TASK_KEYWORDS = frozenset(
    {
        "anomaly detection",
        "classification",
        "context-aware forecasting",
        "forecasting",
        "long-term",
        "probabilistic",
        "probabilistic forecasting",
        "regression",
        "short-term",
        "traffic forecasting",
    }
)

TAXONOMY_KEYWORDS = frozenset(
    {
        "ablation",
        "ablation study",
        "best methods",
        "challenges",
        "comparison",
        "data-centric",
        "data-centric ai",
        "desiderata",
        "future directions",
        "limitations",
        "model analysis",
        "model-centric",
        "practical guide",
        "qualitative analysis",
        "scaling analysis",
        "survey",
        "taxonomy",
        "theoretical understanding",
    }
)

ARCHITECTURE_KEYWORDS = frozenset(
    {
        "bart",
        "bert",
        "causal encoder",
        "chronos encoder",
        "clip",
        "decoder-only",
        "encoder",
        "encoder-decoder",
        "gpt-2",
        "gpt-4 turbo",
        "input embedding",
        "llm embedding space",
        "llama",
        "llama-7b",
        "mlp",
        "multi-head cross-attention",
        "output projection",
        "projection layer",
        "qwen2.5",
        "residual",
        "residual block",
        "rmsnorm",
        "rope",
        "stacked transformer",
        "transformer",
        "ts embedding",
        "tsfm backbone",
    }
)

TUNING_STRATEGY_KEYWORDS = frozenset(
    {
        "adamw",
        "fine-tuning",
        "finetuning",
        "from scratch",
        "frozen attention",
        "frozen backbone",
        "frozen encoder",
        "frozen ffn",
        "frozen llm",
        "frozen vs fine-tuned",
        "generative fine-tuning",
        "linear probing",
        "lora",
        "parameter efficiency",
        "parameter-efficient",
        "partial fine-tuning",
        "peft",
        "pretraining",
        "qlora",
        "rslora",
        "trainable",
        "trainable parameters",
    }
)

DATA_PREPROCESSING_KEYWORDS = frozenset(
    {
        "data preparation",
        "date-time features",
        "denormalization",
        "empirical cdf",
        "frequency-specific",
        "instance normalization",
        "iqr",
        "lag features",
        "lookback window",
        "median",
        "normalization",
        "outliers",
        "revin",
        "rin",
        "robust scaling",
        "robust standardization",
        "summary statistics",
        "value scaling",
        "z-score normalization",
    }
)

PIPELINE_KEYWORDS = frozenset(
    {
        "5 stages",
        "5 steps",
        "forecasting module",
        "pipeline",
        "pipeline stages",
        "proposal generation",
        "proposal refinement",
        "stage 2",
    }
)

TECHNIQUE_KEYWORDS = frozenset(
    {
        "adapter",
        "adaptive pretraining",
        "aligning",
        "alignment",
        "augmentation",
        "autoregressive decoding",
        "autoregressive generation",
        "autoregressive objective",
        "boundary markers",
        "causal attention",
        "concatenation",
        "contrastive",
        "contrastive learning",
        "contrastive loss",
        "contrastive strategies",
        "core lexicon reduction",
        "cross-modality alignment",
        "data preparation",
        "de-tokenizer",
        "decomposition",
        "decoupled tokenizer",
        "denormalization",
        "discrete tokenizer",
        "discrete wavelet transform",
        "distillation",
        "distribution head",
        "embedding",
        "empirical cdf",
        "feature transfer",
        "feature-wise contrast",
        "fine-tuning",
        "finetuning",
        "foundation model",
        "freq-mask",
        "freq-mix",
        "frozen backbone",
        "frozen encoder",
        "gaussian initialization",
        "gaussian scaling",
        "hard prompt",
        "inversion",
        "instance normalization",
        "instance-wise contrast",
        "instruction-driven",
        "iterative refinement",
        "jitter-and-scale",
        "k-means",
        "lag features",
        "lag indices",
        "lag tokens",
        "linear adapter",
        "linear projection",
        "lora",
        "low-rank adapter",
        "metadata reasoning",
        "multi-modal",
        "multimodal",
        "normalization",
        "overlapping patching",
        "parameter efficiency",
        "parameter-efficient",
        "partial fine-tuning",
        "patch embedding",
        "patch embeddings",
        "patch masking",
        "patch reprogramming",
        "patch tokens",
        "patching",
        "peft",
        "positional embedding",
        "positional encoding",
        "prefix tuning",
        "prefix-prompt",
        "prefix-prompts",
        "pretraining",
        "prompt learning",
        "prompt engineering",
        "prompt template",
        "prompt-as-prefix",
        "projection layer",
        "pseudo-word embeddings",
        "pseudo-words",
        "q lora",
        "qlora",
        "quantization",
        "quantization-aware",
        "rank decomposition",
        "rank stabilization",
        "reprogramming",
        "retrieval",
        "retrieval knowledge base",
        "retrieval-augmented",
        "retriever",
        "revin",
        "rin",
        "robust scaling",
        "robust standardization",
        "rslora",
        "semantic anchors",
        "semantic extraction",
        "sliding window",
        "soft prompt",
        "soft prompts",
        "standard normalization",
        "structured prompt",
        "summary statistics",
        "symbolic discretization",
        "text embedding",
        "text embeddings",
        "text prototype",
        "text prototypes",
        "text-prototype",
        "text-prototype alignment",
        "time series tokenizer",
        "token-based decoding",
        "tokenization",
        "trainable prompt",
        "two-stage",
        "value scaling",
        "vector quantization",
        "vision as bridge",
        "vocabulary inversion",
        "wavelet",
        "word embeddings",
        "z-score normalization",
    }
)

EXPERIMENT_KEYWORDS = frozenset(
    {
        "early stopping",
        "few-shot",
        "horizon",
        "hyperparameter search",
        "sample trajectories",
        "stratified sampling",
        "zero-shot",
    }
)

EXPERIMENT_RESULT_KEYWORDS = frozenset(
    {
        "accuracy saturation",
        "best-so-far",
        "experiment results",
        "generalization",
        "improvement",
        "sota",
    }
)

METHOD_KEYWORDS = frozenset(
    {
        "aide",
        "autoformer",
        "chronos",
        "crossformer",
        "darts",
        "dcats",
        "dlinear",
        "gpt4ts",
        "informer",
        "itransformer",
        "lag-llama",
        "largest",
        "llm",
        "llm+ts",
        "llm-agent",
        "llm-agents",
        "llm-based",
        "llm-for-ts",
        "llm-ps",
        "llm4ts",
        "llmtime",
        "moment",
        "moirai",
        "mscnn",
        "ofa",
        "one-for-all",
        "patchtst",
        "promptcast",
        "rag",
        "retrieval knowledge base",
        "s2ip-llm",
        "simmtm",
        "sparsetsf",
        "tempo",
        "test",
        "time-llm",
        "timedart",
        "timegpt-1",
        "timesfm",
        "timesnet",
        "tokencast",
        "ts-rag",
        "ts-specific",
        "tsfm",
        "ultrastf",
        "unitime",
        "vitro",
    }
)

TECHNIQUE_SUBSTRINGS = (
    "adapter",
    "align",
    "augment",
    "contrast",
    "decom",
    "embedding",
    "fine-tun",
    "inversion",
    "mask",
    "normaliz",
    "patch",
    "pretrain",
    "prompt",
    "projection",
    "quant",
    "reprogram",
    "retriev",
    "scale",
    "token",
    "vocabulary",
    "wavelet",
)

METHOD_SUBSTRINGS = (
    "agent",
    "foundation model",
    "rag",
    "time-llm",
    "timesfm",
)


@dataclass(frozen=True)
class EntityMention:
    label: str
    entity_id: str
    name: str
    source: str = "keywords"


def canonical_entity_id(label: str, name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if not slug:
        slug = "entity"
    return f"{label}:{slug}"


def infer_entity_label(keyword: str) -> str | None:
    normalized = _normalize(keyword)
    if not normalized:
        return None

    if normalized in EXACT_LABEL_OVERRIDES:
        return EXACT_LABEL_OVERRIDES[normalized]
    if normalized in METRIC_KEYWORDS:
        return "Metric"
    if normalized in BENCHMARK_KEYWORDS or "benchmark" in normalized:
        return "Benchmark"
    if normalized in DATASET_KEYWORDS:
        return "Dataset"
    if normalized in DOMAIN_KEYWORDS:
        return "Domain"
    if normalized in TASK_KEYWORDS or normalized.endswith(" forecasting"):
        return "Task"
    if normalized in EXPERIMENT_RESULT_KEYWORDS:
        return "ExperimentResult"
    if normalized in EXPERIMENT_KEYWORDS:
        return "Experiment"
    if normalized in PIPELINE_KEYWORDS:
        return "Pipeline"
    if normalized in DATA_PREPROCESSING_KEYWORDS:
        return "DataPreprocessing"
    if normalized in TUNING_STRATEGY_KEYWORDS:
        return "TuningStrategy"
    if normalized in TAXONOMY_KEYWORDS:
        return "TaxonomyCategory"
    if normalized in ARCHITECTURE_KEYWORDS or _looks_like_architecture(normalized):
        return "Architecture"
    if normalized in TECHNIQUE_KEYWORDS or _contains_any(normalized, TECHNIQUE_SUBSTRINGS):
        return "Technique"
    if normalized in METHOD_KEYWORDS or _contains_any(normalized, METHOD_SUBSTRINGS):
        return "Method"
    return None


def build_entity_mention(keyword: str) -> EntityMention | None:
    label = infer_entity_label(keyword)
    if label is None:
        return None
    name = keyword.strip()
    return EntityMention(
        label=label,
        entity_id=canonical_entity_id(label, name),
        name=name,
    )


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _looks_like_architecture(value: str) -> bool:
    if value in {"gpt-2", "bert", "bart", "llama", "llama-7b"}:
        return True
    return any(
        token in value
        for token in ("encoder", "decoder", "transformer", "backbone", "attention")
    )
