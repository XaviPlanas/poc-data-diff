"""
uso_report.py

Cómo integrar DiffReport con el classifier.py real.
Muestra la sustitución de report_statistics() y report_details()
y la generación del informe completo en JSON y Markdown.

Ejecutar:
    PYTHONPATH=src ANTHROPIC_API_KEY=sk-... python3 uso_report.py
"""

import json
from tfg.datadiff_classifier.classifier import DiffClassifier
from tfg.datadiff_classifier.report     import (
    DiffReport, ReportNarrator, ReportExporter
)
from tfg.datadiff_classifier.prompts    import SCHEMA_CONTEXT_TITANIC
from tfg.logging_config                 import setup_logging

setup_logging(level="INFO")

# ─────────────────────────────────────────────────────────────────
# 1. Obtener clasificaciones (flujo normal del pipeline)
# ─────────────────────────────────────────────────────────────────

clf = DiffClassifier(
    schema_context = SCHEMA_CONTEXT_TITANIC,
    llm_provider   = "anthropic",
    model          = "claude-haiku-4-5",
    few_shot       = True,
)

# En el pipeline real esto viene de:
#   diffrows = clf.parse_to_diffrows(metadata, diffs)
#   classifications = clf.classify_row_by_row(diffrows)
#
# Para este ejemplo usamos clasificaciones preconstruidas:
from tfg.datadiff_classifier.models import (
    DiffClassification, DiffCategory, DiffAction
)

classifications = [
    DiffClassification(
        key=1, accion=DiffAction.UPDATE,
        categoria=DiffCategory.CANONIZABLE, confianza=0.97,
        columnas_afectadas=["name"],
        explicacion="Diferencia de casing y espacios.",
        normalizacion_sugerida="LOWER(TRIM(name))",
        row_a={"PassengerId": 1, "name": "  JACK  "},
        row_b={"PassengerId": 1, "name": "jack"},
    ),
    DiffClassification(
        key=14, accion=DiffAction.UPDATE,
        categoria=DiffCategory.EQUIVALENT, confianza=0.91,
        columnas_afectadas=["embarked"],
        explicacion="\"C\" y \"Cherbourg\" son el mismo puerto.",
        normalizacion_sugerida=None,
        row_a={"PassengerId": 14, "embarked": "C"},
        row_b={"PassengerId": 14, "embarked": "Cherbourg"},
    ),
    DiffClassification(
        key=33, accion=DiffAction.UPDATE,
        categoria=DiffCategory.DIFFERENT_STRUCTURAL, confianza=0.99,
        columnas_afectadas=["survived"],
        explicacion="Cambio de supervivencia: 0 → 1.",
        normalizacion_sugerida=None,
        row_a={"PassengerId": 33, "survived": 0},
        row_b={"PassengerId": 33, "survived": 1},
    ),
    DiffClassification(
        key=55, accion=DiffAction.UPDATE,
        categoria=DiffCategory.DIFFERENT_CONTEXTUAL, confianza=0.84,
        columnas_afectadas=["fare", "currency"],
        explicacion="Distinta moneda, podría ser equivalente con tipo de cambio.",
        normalizacion_sugerida=None,
        row_a={"PassengerId": 55, "fare": 26.0,  "currency": "GBP"},
        row_b={"PassengerId": 55, "fare": 32.5,  "currency": "USD"},
    ),
    DiffClassification(
        key=78, accion=DiffAction.UPDATE,
        categoria=DiffCategory.UNCERTAIN, confianza=0.62,
        columnas_afectadas=["cabin"],
        explicacion="No hay suficiente contexto para clasificar la diferencia.",
        normalizacion_sugerida=None,
        row_a={"PassengerId": 78, "cabin": "C85"},
        row_b={"PassengerId": 78, "cabin": "c85/c86"},
    ),
    DiffClassification(
        key=200, accion=DiffAction.INSERT,
        categoria=DiffCategory.DIFFERENT_STRUCTURAL, confianza=1.0,
        columnas_afectadas=["__ROW__"],
        explicacion="INSERT del registro con clave 200.",
        normalizacion_sugerida=None,
        row_a=None,
        row_b={"PassengerId": 200, "name": "nuevo"},
    ),
    DiffClassification(
        key=7, accion=DiffAction.UPDATE,
        categoria=DiffCategory.CANONIZABLE, confianza=0.95,
        columnas_afectadas=["name", "ticket"],
        explicacion="Diferencia de casing en name.",
        normalizacion_sugerida="LOWER(TRIM(name))",
        row_a={"PassengerId": 7, "name": "  MCCARTHY, MR. TIMOTHY J", "ticket": "17463"},
        row_b={"PassengerId": 7, "name": "mccarthy, mr. timothy j",   "ticket": "17463"},
    ),
]

# ─────────────────────────────────────────────────────────────────
# 2. Construir DiffReport (una sola vez)
# ─────────────────────────────────────────────────────────────────

report = DiffReport(classifications)

# ─────────────────────────────────────────────────────────────────
# 3. Métodos de consola (reemplazan los del classifier)
# ─────────────────────────────────────────────────────────────────

# Antes: clf.report_statistics(classifications)
report.print_summary()

# Antes: clf.report_details(classifications)
report.print_details(false_positives=True, review=True)

# Nuevo: análisis por columna
report.print_column_analysis()

# Nuevo: reglas sugeridas
report.print_canonizable_rules()

# ─────────────────────────────────────────────────────────────────
# 4. Acceso programático a los datos
# ─────────────────────────────────────────────────────────────────

summ  = report.summary()
print(f"\nFalse positive rate  : {summ['tasas']['false_positive_rate']:.1%}")
print(f"Structural diff rate : {summ['tasas']['structural_diff_rate']:.1%}")
print(f"Review rate          : {summ['tasas']['review_rate']:.1%}")

# Reglas para retroalimentar el motor de canonización
rules = report.canonizable_rules()
print("\nReglas para añadir al motor:")
for rule, info in rules.items():
    print(f"  {rule}  ({info['apariciones']}×)  cols={info['columnas']}")

# Cola de revisión humana
queue = report.review_queue(max_items=5)
print(f"\nCola de revisión ({len(queue)} items):")
for item in queue:
    print(f"  key={item['key']}  {item['categoria']}  conf={item['confianza']:.2f}")

# Métricas del pipeline (si se tiene la info de las etapas)
pipeline = DiffReport.reduction_pipeline(
    n_raw           = 142,    # diffs sin canonización
    n_pre           = 23,     # tras PRE SQL
    n_post          = len(classifications),   # tras POST Python
    classifications = classifications,
)
print("\nPipeline:")
print(json.dumps(pipeline, indent=2))

# ─────────────────────────────────────────────────────────────────
# 5. Narrativa LLM (una sola llamada)
# ─────────────────────────────────────────────────────────────────

import os
if os.environ.get("ANTHROPIC_API_KEY"):
    narrator  = ReportNarrator()
    narrative = narrator.full_narrative(report, pipeline_stats=pipeline)

    print("\n── DIAGNÓSTICO ──────────────────────────────────────────")
    print(narrative["diagnose"])
    print("\n── RECOMENDACIONES ──────────────────────────────────────")
    print(narrative["recommend"])
    print("\n── RESUMEN EJECUTIVO ────────────────────────────────────")
    print(narrative["executive_summary"])
else:
    narrative = None
    print("\n(Narrativa LLM omitida: ANTHROPIC_API_KEY no definida)")

# ─────────────────────────────────────────────────────────────────
# 6. Exportar
# ─────────────────────────────────────────────────────────────────

ReportExporter.to_json(report, narrative, path="output_report.json")
ReportExporter.to_markdown(report, narrative, path="output_report.md")
print("\nInformes guardados: output_report.json  output_report.md")
