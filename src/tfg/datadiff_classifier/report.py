import re
import unicodedata
from collections import Counter, defaultdict
from statistics import mean
from collections import defaultdict
from collections import defaultdict
from tfg.datadiff_classifier.models import DiffClassification, DiffCategory

def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip().lower()

def _is_number(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def generate_report(results: list[DiffClassification]) -> dict:
    by_category = defaultdict(list)
    for r in results:
        by_category[r.categoria].append(r)
    
    total = len(results)
    equiv = len(by_category[DiffCategory.SEMANTICALLY_EQUIVALENT])
    diff  = len(by_category[DiffCategory.SEMANTICALLY_DIFFERENT])
    # Extraer sugerencias de normalización únicas por columna
    normalization_hints = defaultdict(set)
    for r in by_category[DiffCategory.SEMANTICALLY_EQUIVALENT] + \
             by_category[DiffCategory.SEMANTICALLY_DIFFERENT]:
        if r.normalizacion_sugerida:
            for col in r.columnas_afectadas:
                normalization_hints[col].add(r.normalizacion_sugerida)    
    return {
        "total_diffs": total,
        "false_positive_rate": equiv / total if total else 0,
        "by_category": {cat.value: len(rows) for cat, rows in by_category.items()},
        "normalization_hints": {normalization_hints},   
        "uncertain_cases": [r.key for r in by_category[DiffCategory.UNCERTAIN]]
    }


def report_metrics(classifications: list) -> dict:
    if not classifications:
        return {}

    total = len(classifications)

    # ---------------------------
    # 1. Conteo por categoría
    # ---------------------------
    category_counter = Counter(c.categoria for c in classifications)

    category_percentages = {
        str(cat): round(count / total * 100, 2)
        for cat, count in category_counter.items()
    }

    # ---------------------------
    # 2. Confianza
    # ---------------------------
    confidences = [c.confianza for c in classifications if c.confianza is not None]

    confidence_metrics = {
        "media": round(mean(confidences), 4) if confidences else None,
        "min": min(confidences) if confidences else None,
        "max": max(confidences) if confidences else None,
    }

    # Bucketización simple (clave para análisis)
    confidence_buckets = {
        "alta": 0,     # >= 0.8
        "media": 0,    # 0.5 - 0.79
        "baja": 0      # < 0.5
    }

    for c in classifications:
        if c.confianza is None:
            continue
        if c.confianza >= 0.8:
            confidence_buckets["alta"] += 1
        elif c.confianza >= 0.5:
            confidence_buckets["media"] += 1
        else:
            confidence_buckets["baja"] += 1

    # ---------------------------
    # 3. Columnas conflictivas
    # ---------------------------
    column_counter = Counter()
    for c in classifications:
        if c.columnas_afectadas:
            column_counter.update(c.columnas_afectadas)

    top_columns = column_counter.most_common(10)

    # ---------------------------
    # 4. Detección simple de incertidumbre
    # ---------------------------
    uncertain = [
        c for c in classifications
        if c.confianza is not None and c.confianza < 0.5
    ]

    # Clasificación dominante
    dominant_category = category_counter.most_common(1)[0][0]


    # ---------------------------
    # 5. Resultado final
    # ---------------------------
    return {
        "total_registros": total,

        "categorias": {
            "conteo": dict(category_counter),
            "porcentaje": category_percentages,
        },

        "confianza": {
            "metricas": confidence_metrics,
            "distribucion": confidence_buckets,
        },

        "columnas_conflictivas": {
            "top": top_columns
        },

        "incertidumbre": {
            "total": len(uncertain),
            "porcentaje": round(len(uncertain) / total * 100, 2)
        },

        "resumen": {
            "categoria_dominante": str(dominant_category),
            "columna_mas_conflictiva": top_columns[0][0] if top_columns else None,
            "nivel_confianza_global": (
                "alto" if confidence_metrics["media"] and confidence_metrics["media"] >= 0.75
                else "medio" if confidence_metrics["media"] and confidence_metrics["media"] >= 0.5
                else "bajo"
            )
        }
    }

def report_patterns(classifications: list) -> dict:
    """
    Módulo de detección de patrones
    Analiza los resultados para identificar patrones comunes en las diferencias clasificadas.
    """
    patterns = defaultdict(list)

    for c in classifications:
        row_a = c.row_a or {}
        row_b = c.row_b or {}

        for col in c.columnas_afectadas or []:
            val_a = row_a.get(col)
            val_b = row_b.get(col)

            detected = []

            # -----------------------
            # 1. NULL vs valor
            # -----------------------
            if (val_a is None and val_b is not None) or (val_b is None and val_a is not None):
                detected.append("null_vs_value")

            # -----------------------
            # 2. Tipo diferente
            # -----------------------
            if val_a is not None and val_b is not None:
                if type(val_a) != type(val_b):
                    detected.append("type_mismatch")

            # Solo seguir si ambos son strings
            if isinstance(val_a, str) and isinstance(val_b, str):

                # -----------------------
                # 3. Espacios
                # -----------------------
                if val_a.strip() == val_b.strip() and val_a != val_b:
                    detected.append("whitespace_issue")

                # -----------------------
                # 4. Mayúsculas/minúsculas
                # -----------------------
                if val_a.lower() == val_b.lower() and val_a != val_b:
                    detected.append("case_difference")

                # -----------------------
                # 5. Normalización unicode
                # -----------------------
                if _normalize_text(val_a) == _normalize_text(val_b) and val_a != val_b:
                    detected.append("encoding_issue")

                # -----------------------
                # 6. Diferencia de longitud significativa
                # -----------------------
                if abs(len(val_a) - len(val_b)) > 3:
                    detected.append("length_difference")

                # -----------------------
                # 7. Posible typo (muy simple)
                # -----------------------
                if (
                    _normalize_text(val_a) != _normalize_text(val_b)
                    and abs(len(val_a) - len(val_b)) <= 2
                ):
                    detected.append("possible_typo")

            # -----------------------
            # 8. Números con distinto formato
            # -----------------------
            if _is_number(val_a) and _is_number(val_b):
                if float(val_a) == float(val_b) and val_a != val_b:
                    detected.append("numeric_format")

            # -----------------------
            # Guardado
            # -----------------------
            for p in detected:
                patterns[p].append({
                    "key": c.key,
                    "column": col,
                    "value_a": val_a,
                    "value_b": val_b,
                    "categoria": str(c.categoria),
                    "confianza": c.confianza
                })

    # -----------------------
    # Resumen final
    # -----------------------
    summary = {
        pattern: {
            "count": len(items),
            "examples": items[:5]  # limitar ejemplos
        }
        for pattern, items in patterns.items()
    }

    return {
        "patterns": summary,
        "total_patterns_detected": len(summary)
    }

# def generate_report(results: list[DiffClassification]) -> dict:
#     """
#     Agrega los resultados en un informe accionable que alimentará
#     directamente la Solución 2.
#     """
#     by_category = defaultdict(list)
#     for r in results:
#         by_category[r.categoria].append(r)

#     # Extraer sugerencias de normalización únicas por columna
#     normalization_hints = defaultdict(set)
#     for r in by_category[DiffCategory.FALSO_POSITIVO_TIPO] + \
#              by_category[DiffCategory.FALSO_POSITIVO_NORM]:
#         if r.normalizacion_sugerida:
#             for col in r.columnas_afectadas:
#                 normalization_hints[col].add(r.normalizacion_sugerida)

#     return {
#         "resumen": {
#             cat.value: len(rows) 
#             for cat, rows in by_category.items()
#         },
#         "tasa_falsos_positivos": (
#             len(by_category[DiffCategory.FALSO_POSITIVO_TIPO]) +
#             len(by_category[DiffCategory.FALSO_POSITIVO_NORM])
#         ) / len(results) if results else 0,
#         "normalization_hints": {
#             col: list(hints) 
#             for col, hints in normalization_hints.items()
#         },
#         "requieren_revision": [
#             {"key": r.key, "explicacion": r.explicacion}
#             for r in by_category[DiffCategory.AMBIGUA]
#         ]
#     }