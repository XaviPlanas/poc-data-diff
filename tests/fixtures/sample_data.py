import pytest
from tfg.datadiff_classifier.models import DiffClassification, DiffCategory

@pytest.fixture
def sample_classifications():
    return [
        DiffClassification(
            key=1,
            categoria=DiffCategory.EQUIVALENT,
            confianza=0.95,
            columnas_afectadas=["name"],
            explicacion="case difference",
            normalizacion_sugerida="lowercase",
            row_a={"name": "Juan"},
            row_b={"name": "juan"}
        ),
        DiffClassification(
            key=2,
            categoria=DiffCategory.DIFFERENT,
            confianza=0.9,
            columnas_afectadas=["age"],
            explicacion="real diff",
            normalizacion_sugerida=None,
            row_a={"age": 30},
            row_b={"age": 40}
        ),
        DiffClassification(
            key=3,
            categoria=DiffCategory.UNCERTAIN,
            confianza=0.4,
            columnas_afectadas=["city"],
            explicacion="uncertain",
            normalizacion_sugerida=None,
            row_a={"city": "València"},
            row_b={"city": "Valencia"}
        ),
    ]