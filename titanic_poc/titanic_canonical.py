###################################
# Escenario: 
#   Pipeline completo de canonización y comparación.
#   Ejecuta los cuatro pasos en secuencia:
#       paso 1. Cargar configuración canónica
#       paso 2. Construir y aplicar el plan de canonización en cada motor
#       paso 3. Comparar con data-diff sobre las vistas canónicas
#       paso 4. Clasificar las diferencias con el clasificador IA
#       paso 5. Generar informe detallado de resultados
###################################

import json

from data_diff import connect_to_table, diff_tables

from sqlalchemy import inspect, Engine

from tfg.datadiff_classifier.classifier import DiffClassifier, DiffRow
from tfg.datadiff_classifier.report     import generate_report

from tfg.canonical_engine.pipeline      import CanonicalPipeline
from tfg.canonical_engine.config.loader import CanonicalConfigLoader

from .titanic_utils import Config

###################################
# Globales y configuración
###################################

conf = Config()

DEBUG = True
SEPARATOR = "─" * 60

MYSQL_URI = conf.getConnectionString(Config.MYSQL)
PG_URI    = conf.getConnectionString(Config.POSTGRES)

MYSQL_URI_DDIFF = conf.getConnectionString(Config.MYSQL, datadiff = True)
PG_URI_DDIFF = conf.getConnectionString(Config.POSTGRES, datadiff = True)

CFG_FILE  = "tfg/titanic_poc/titanic_canonical.yaml" # ejecutando desde tfg/src

###################################
# Métodos con los pasos del pipeline
###################################

def __header_print(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(f"{SEPARATOR}\n")

def paso1_load_config():
    __header_print("PASO 1 — Cargar configuración canónica")

    loaded = CanonicalConfigLoader.from_file(CFG_FILE)
    print(loaded.report())
    return loaded

def paso2_build_and_apply_plans(loaded):

    __header_print("PASO 2 — Construir y aplicar planes de canonización")

    # ── MySQL ──────────────────────────────────────────────────────
    print("\n  MySQL:")
    pipeline_mysql = CanonicalPipeline(
        connection_uri = MYSQL_URI,
        table_name     = "titanic",
    )
    plan_mysql = pipeline_mysql.build_plan()
    print(plan_mysql.report())
    pipeline_mysql.apply_plan(plan_mysql)

    # ── PostgreSQL ────────────────────────────────────────────────
    print("\n  PostgreSQL:")
    pipeline_pg = CanonicalPipeline(
        connection_uri = PG_URI,
        table_name     = "titanic",
    )
    plan_pg = pipeline_pg.build_plan()
    print(plan_pg.report())
    pipeline_pg.apply_plan(plan_pg)

    return plan_mysql, plan_pg

def paso3_compare_with_datadiff():
    __header_print("PASO 3 — Comparación con data-diff sobre vistas canónicas")

    # Sin canonización (baseline: muestra el problema)
    print("\n  [Baseline] Sin canonización:")
    table1_raw = connect_to_table(
        MYSQL_URI_DDIFF, "titanic", "PassengerId"
    )
    table2_raw = connect_to_table(
        PG_URI_DDIFF, "titanic", "PassengerId"
    )
    
    #TODO: Validación cruzada de cols ANTES de empezar con los esquemas
    #      def_validate_schema_match(table1.cols, table2.cols):
    #      ¿comparar nombres o sólo igualdad de elementos?
    # Mientras asumimos que table1.cols == table2.cols y tomamos table1.cols
   
    #table1_cols = conf.get_all_column_names( conf.get_mysql_engine(), 'titanic' )
    table1_cols = table1_raw.get_schema().keys() # Queremos usar TableSegment en lugar de inspección SQLAlchemy

    diffs_raw = list(diff_tables(table1 = table1_raw,
                                 table2 = table2_raw,
                                 extra_columns= table1_cols                    
                                ))
    print(f"  Diferencias detectadas (sin canonizar): {len(diffs_raw)}")

    # Con canonización (vistas canónicas)
    print("\n  [Canonizado] Con vistas canónicas:")
    table_mysql_can = connect_to_table(
        MYSQL_URI_DDIFF, "titanic_canonical", "PassengerId"
    )
    table_pg_can = connect_to_table(
        PG_URI_DDIFF, "titanic_canonical", "PassengerId"
    )
    diffs_can = list(diff_tables(table_mysql_can, table_pg_can))
    print(f"  Diferencias detectadas (canonizadas):   {len(diffs_can)}")

    reduction = len(diffs_raw) - len(diffs_can)
    print(f"\n  Reducción de falsos positivos: {reduction} filas")
    print(f"  ({reduction/len(diffs_raw)*100:.1f}% de reducción)"
          if diffs_raw else "")

    return diffs_raw, diffs_can


def paso4_classify_differences(diffs_can):
    __header_print("PASO 4 — Clasificación IA de diferencias restantes")

    if not diffs_can:
        print("\n  No hay diferencias que clasificar.")
        print("  La canonización ha eliminado todos los falsos positivos.")
        return []


    schema_context = """
    Tabla: titanic (dataset Kaggle, 891 pasajeros del Titanic)
    Motor A: MySQL 8.0
    Motor B: PostgreSQL 15
    Columnas canonizadas: fare (ROUND 3 dec), name (NFC+lower+trim),
                          sex (lower+trim), survived (0/1), cabin (lower+trim)
    """

    classifier = DiffClassifier(schema_context=schema_context)

    diff_rows = [
        DiffRow(
            key      = diff[1],
            row_a    = diff[2],
            row_b    = diff[3],
            source_a = "mysql://pocdb/titanic_canonical",
            source_b = "postgresql://tfgdb/titanic_canonical",
        )
        for diff in diffs_can
    ]

    print(f"\n  Clasificando {len(diff_rows)} diferencias...")
    results = classifier.classify_batch(diff_rows)
    report  = generate_report(results)

    print(f"\n  Resumen de clasificación:")
    for category, count in report["resumen"].items():
        print(f"    {category:<35} {count} filas")

    print(f"\n  Tasa de falsos positivos residual: "
          f"{report['tasa_falsos_positivos']:.1%}")

    if report["requieren_revision"]:
        print(f"\n  Requieren revisión manual:")
        for item in report["requieren_revision"]:
            print(f"    PassengerId {item['key']}: {item['explicacion']}")

    if report["normalization_hints"]:
        print(f"\n  Sugerencias de normalización adicional:")
        for col, hints in report["normalization_hints"].items():
            for hint in hints:
                print(f"    {col}: {hint}")

    # Guardar informe completo
    with open("output_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Informe guardado en: output_report.json")

    return results


def main():
    print("=" * 60)
    print("  MOTOR DE CANONIZACIÓN — CASO DE USO TITANIC")
    print("=" * 60)

    loaded                  = paso1_load_config()   # Paso 1 - Cargar Configuración 
    plan_mysql, plan_pg     = paso2_build_and_apply_plans(loaded) # Paso 2 - Canonizar en cada motor
    diffs_raw, diffs_can    = paso3_compare_with_datadiff() # Paso 3 - data-diff contra datasets canonizados

  
    #TODO: results = paso4_classify_differences(diffs_can) # Paso 4 - clasificador IA


    print(f"\n{'=' * 60}")
    print("  EJECUCIÓN COMPLETADA")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()