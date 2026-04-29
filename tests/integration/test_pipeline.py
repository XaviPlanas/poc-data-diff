# tests/integration/test_pipeline.py

from tfg.pipeline.full_analysis import run_full_analysis


def fake_canonical(a, b):
    return a, b, {"rules": []}


def fake_diff(a, b):
    return [{"id": 1}]


def fake_classifier(diffs):
    from tfg.datadiff_classifier.models import DiffClassification, DiffCategory
    return [
        DiffClassification(
            key=1,
            categoria=DiffCategory.EQUIVALENT,
            confianza=0.9,
            columnas_afectadas=["name"],
            explicacion="mock",
            normalizacion_sugerida=None,
            row_a={"name": "A"},
            row_b={"name": "a"}
        )
    ]


def test_full_pipeline_runs():
    result = run_full_analysis(
        canonical_fn=fake_canonical,
        diff_fn=fake_diff,
        classify_fn=fake_classifier,
        source_a=[{"name": "A"}],
        source_b=[{"name": "a"}]
    )

    assert "metrics" in result
    assert "patterns" in result
    assert result["metrics"]["total"] == 1