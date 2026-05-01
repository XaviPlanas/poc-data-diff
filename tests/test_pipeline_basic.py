# tests/test_pipeline_basic.py

from tfg.canonical_engine.pipeline import CanonicalPipeline
from tfg.canonical_engine.dialects.mysql import MySQLDialect


def test_pipeline_generates_plan():
    pipeline = CanonicalPipeline(
        table="titanic",
        dialect=MySQLDialect(),
        config={
            "columns": {
                "PassengerId": {"type": "integer"},
                "Name": {"type": "text"}
            }
        }
    )

    plan = pipeline.build_plan()

    assert plan is not None
    assert len(plan.columns) == 2


def test_pipeline_generates_sql():
    pipeline = CanonicalPipeline(
        table="titanic",
        dialect=MySQLDialect(),
        config={
            "columns": {
                "PassengerId": {"type": "integer"}
            }
        }
    )

    plan = pipeline.build_plan()
    sql = pipeline.generate_sql(plan)

    assert "SELECT" in sql
    assert "FROM titanic" in sql