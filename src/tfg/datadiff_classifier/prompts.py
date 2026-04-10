CLASSIFY_PROMPT = """
Eres un experto en bases de datos heterogéneas. Estás analizando 
diferencias detectadas por data-diff entre {source_a} y {source_b}.

Contexto del esquema:
{schema_context}

Fila en {source_a}:
{row_a}

Fila en {source_b}:
{row_b}

Clasifica la diferencia en UNA de estas categorías:

- REAL: los datos son semánticamente distintos. La diferencia 
  es significativa e indica una divergencia real entre los sistemas.

- FALSO_POSITIVO_TIPO: mismo valor, representación distinta por 
  diferencias en tipos de datos entre motores (coma flotante, 
  precisión numérica, booleanos, timestamps).

- FALSO_POSITIVO_NORMALIZACION: mismo valor semántico, representación 
  distinta por normalización asimétrica (encoding, collation, 
  mayúsculas, espacios, charset).

- AMBIGUA: no se puede determinar la categoría con los datos 
  disponibles.

Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{{
    "categoria": "REAL|FALSO_POSITIVO_TIPO|FALSO_POSITIVO_NORMALIZACION|AMBIGUA",
    "confianza": <float entre 0.0 y 1.0>,
    "columnas_afectadas": ["col1", "col2"],
    "explicacion": "<explicación concisa en español>",
    "normalizacion_sugerida": "<expresión SQL o null>"
}}
"""