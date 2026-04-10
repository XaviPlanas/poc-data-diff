from collections import defaultdict
from models import ClassificationResult, DiffCategory

def generate_report(results: list[ClassificationResult]) -> dict:
    """
    Agrega los resultados en un informe accionable que alimentará
    directamente la Solución 2.
    """
    by_category = defaultdict(list)
    for r in results:
        by_category[r.categoria].append(r)

    # Extraer sugerencias de normalización únicas por columna
    normalization_hints = defaultdict(set)
    for r in by_category[DiffCategory.FALSO_POSITIVO_TIPO] + \
             by_category[DiffCategory.FALSO_POSITIVO_NORM]:
        if r.normalizacion_sugerida:
            for col in r.columnas_afectadas:
                normalization_hints[col].add(r.normalizacion_sugerida)

    return {
        "resumen": {
            cat.value: len(rows) 
            for cat, rows in by_category.items()
        },
        "tasa_falsos_positivos": (
            len(by_category[DiffCategory.FALSO_POSITIVO_TIPO]) +
            len(by_category[DiffCategory.FALSO_POSITIVO_NORM])
        ) / len(results) if results else 0,
        "normalization_hints": {
            col: list(hints) 
            for col, hints in normalization_hints.items()
        },
        "requieren_revision": [
            {"key": r.key, "explicacion": r.explicacion}
            for r in by_category[DiffCategory.AMBIGUA]
        ]
    }