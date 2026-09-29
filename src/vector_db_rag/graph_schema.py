from __future__ import annotations

from typing import Any

GRAPH_CONSTRAINT_STATEMENTS = (
    "CREATE CONSTRAINT article_id_unique IF NOT EXISTS "
    "FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT method_id_unique IF NOT EXISTS "
    "FOR (n:Method) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT architecture_id_unique IF NOT EXISTS "
    "FOR (n:Architecture) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT technique_id_unique IF NOT EXISTS "
    "FOR (n:Technique) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT tuning_strategy_id_unique IF NOT EXISTS "
    "FOR (n:TuningStrategy) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT data_preprocessing_id_unique IF NOT EXISTS "
    "FOR (n:DataPreprocessing) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT pipeline_id_unique IF NOT EXISTS "
    "FOR (n:Pipeline) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT dataset_id_unique IF NOT EXISTS "
    "FOR (n:Dataset) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT benchmark_id_unique IF NOT EXISTS "
    "FOR (n:Benchmark) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT metric_id_unique IF NOT EXISTS "
    "FOR (n:Metric) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT domain_id_unique IF NOT EXISTS "
    "FOR (n:Domain) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT task_id_unique IF NOT EXISTS "
    "FOR (n:Task) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT experiment_id_unique IF NOT EXISTS "
    "FOR (n:Experiment) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT experiment_result_id_unique IF NOT EXISTS "
    "FOR (n:ExperimentResult) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT taxonomy_category_id_unique IF NOT EXISTS "
    "FOR (n:TaxonomyCategory) REQUIRE n.id IS UNIQUE",
)


def ensure_graph_schema(driver: Any, *, database: str) -> dict[str, Any]:
    with driver.session(database=database) as session:
        for statement in GRAPH_CONSTRAINT_STATEMENTS:
            session.run(statement).consume()
    return {
        "database": database,
        "constraints_applied": len(GRAPH_CONSTRAINT_STATEMENTS),
    }
