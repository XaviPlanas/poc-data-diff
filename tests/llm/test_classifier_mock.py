# tests/llm/test_classifier_mock.py

def test_classifier_output_structure():
    from tfg.datadiff_classifier.models import DiffClassification

    # Simulación de salida del LLM
    result = DiffClassification(
        key=1,
        categoria="equivalent",
        confianza=0.8,
        columnas_afectadas=["name"],
        explicacion="mock",
        normalizacion_sugerida="lowercase",
        row_a={"name": "A"},
        row_b={"name": "a"}
    )

    assert result.key == 1
    assert result.confianza >= 0