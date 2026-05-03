import textwrap
from typing import List, Dict, Any

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

CLASSIFY_PROMPT_FROM_DICT_OPTIMIZED_1 = """
Assume the role of an expert in data integration and schema alignment, with deep understanding 
of data inconsistencies across heterogeneous systems. Your task is to help me evaluate a difference
 between two records (row_a and row_b), provided as dictionary structures keyed by primary key. 
 Using the schema definitions (including encoded categorical values like embarked, survival, etc.), 
 determine if the discrepancy represents a real divergence or a false positive caused by type 
 differences or normalization inconsistencies. Perform a structured comparison column by column, 
 identify affected fields, and infer semantic equivalence where possible. Output must be strictly 
 valid JSON with fields: categoria, confianza (0–1), columnas_afectadas, explicacion (concise Spanish),
   and normalizacion_sugerida (SQL or null). One output per primary key. Ensure no extra text outside the JSON.
 ###  This is the data schema:
   | Columna     | Tipo      | Descripción                                      | Valores posibles / Notas                  |
|-------------|-----------|--------------------------------------------------|-------------------------------------------|
| survival    | int       | Sobrevivió o no                                  | 0 = No, 1 = Sí (target)                  |
| pclass      | int       | Clase del ticket                                 | 1 = 1ª, 2 = 2ª, 3 = 3ª                    |
| sex         | string    | Sexo                                             | male / female                             |
| age         | float     | Edad en años                                     | Puede tener decimales y valores nulos     |
| sibsp       | int       | Hermanos/esposos a bordo                         | Numérico                                  |
| parch       | int       | Padres/hijos a bordo                             | Numérico                                  |
| ticket      | string    | Número de ticket                                 | Alfanumérico                              |
| fare        | float     | Tarifa pagada                                    | Numérico                                  |
| cabin       | string    | Número de camarote                               | Muchos valores nulos (~77%)               |
| embarked    | string    | Puerto de embarque                               | C=Cherbourg, Q=Queenstown, S=Southampton (2 nulos) |

 ###  These are the data rows to compare by primary keys
   for {source_a} the data is {row_a}
   for  {source_b} the data is {row_b} 
  
"""

CLASSIFY_PROMPT_FROM_DICT_OPTIMIZED_2 = """
You are an expert in data reconciliation across heterogeneous systems.

<<DATA_STRUCTURE>>

* **Key:**  Each line in the dataset corresponds to a single passenger on the Titanic. For each passenger, there are keys associated with different attributes (e.g., PassengerId). 
* **Passenger Details:** Each line holds details about a specific passenger including their:
    - **Survival Status:** Whether the passenger survived or not (`Survived`)
    - **Pclass:**  Their class on the ship (1st, 2nd, or 3rd)
    - **Name:**  The passenger's full name.
    - **Sex:** The passenger's gender 
    - **Age:**  The passenger's age at time of travel.   
    - **SibSp:** Number of siblings/spouses traveling with them. 
    - **Parch:** Number of parents/children traveling with them.
    - **Ticket:** Their ticket number or the unique identifier for each individual.
    - **Fare:** The cost of their ticket.
    - **Cabin:** The cabin number they were assigned on the ship (if applicable). 
<<DATA_STRUCTURE>>

I will provide two records DATASET_A from {source_a} and DATASET_B from {source_b}) with the same primary key and a shared schema.
The primary key is the first key of the dictionary of dictionaries.

You are a data quality expert.

Your task is to determine whether two values are semantically equivalent.

Definition:
- SEMANTICALLY_EQUIVALENT: same meaning, even if representation differs
- SEMANTICALLY_DIFFERENT: meaning changes
- UNCERTAIN: not enough information

Rules:
- Ignore formatting differences
- Consider numeric tolerance for floats
- Be conservative: if unsure → UNCERTAIN
- Do NOT assume business context

Output: 


Output format:
Return ONLY valid JSON:
{{
"id" : "passenger_id",
"category": "...",
"confidence_of_category": 0.0,
"afected_columns": ["..."],
"explanation": "...",
"normalization_suggested": "SQL o null"
}}

Constraints:
- No extra text
- Create multiple JSON: One JSON per each primary key from the two datasets 
- Be conservative before labeling as "diferente"

IMPORTANT:
- No text outside JSON
- No markdown
- No questions
- No metadata
- No schema description
- You MUST process ALL keys
- If dataset is large, continue until end without stopping

<< DATA >>
<< DATASET_A {source_a} >>
 {row_a} 
<</DATASET_A>>

<<RDATASET_B {source_b}>>
{row_b}
<</DATASET_B>>
<< /DATA >>
"""

CLASSIFY_PROMPT_FROM_DICT_OPTIMIZED_GEMMA2_2b = """
You are an expert in data reconciliation. 

### SCHEMA (Titanic)
- Primary key: PassengerId (first key in each dict)
- Columns: survival, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked

I will give you two datasets with identical schema.

Task:
1. Compare row-by-row using PassengerId as key.
2. For each PassengerId decide ONLY:
   - "diferente" → real difference
   - "equivalente" → same value or only formatting/type
   - "inconsistencia_normalizacion" → normalization artifact

Return EXACTLY this and NOTHING ELSE:
[
  {{"categoria": "diferente|equivalente|inconsistencia_normalizacion", "confianza": 0.XX, "columnas_afectadas": ["col1","col2"], "explicacion": "breve en español", "normalizacion_sugerida": "SQL o null"}},
  ...
]

No markdown, no texto extra, no explicación, no ```json. Solo el array JSON.

<<DATASET_A {source_a}>>
{row_a}
<</DATASET_A>>

<<DATASET_B {source_b}>>
{row_b}
<</DATASET_B>>
"""

CLASSIFY_PROMPT_FROM_DICT_OPTIMIZED_QWEN2_5 = """

Eres un experto en reconciliación de datos.

### SCHEMA (Titanic)
- Primary key: PassengerId (es la primera clave en cada diccionario)
- Columnas: survival, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked

Te voy a proporcionar dos datasets con el mismo schema.

### TAREA (sigue estos pasos **estrictamente** en orden):
1. Compara fila por fila usando **PassengerId** como clave primaria.
2. Para cada PassengerId decide **SOLO** una de estas tres categorías:
   - "diferente" → hay una diferencia real en el valor
   - "equivalente" → los valores son iguales o solo difieren ligeramente en formato/tipo, pero mantienen exactamente el mismo significado
   - "inconsistencia_normalizacion" → es un artefacto de normalización

### REQUISITOS DE SALIDA (OBLIGATORIOS):
Devuelve **ÚNICAMENTE** un array JSON válido y nada más.
No uses markdown, no agregues texto explicativo, no uses ```json, no pongas nada antes ni después del array.

El formato EXACTO de tu salida es para cada registro o par de registros de acuerdo a su clave única:
  {{
    "categoria": "diferente|equivalente|inconsistencia_normalizacion",
    "confianza": 0.XX,
    "columnas_afectadas": ["col1", "col2"],
    "explicacion": "breve explicación en español",
    "normalizacion_sugerida": "SQL o null"
  }}
NO incluyas NINGUNA INFORMACIÓN adicional. Responde únicamente con JSON válido.

Estos son los datos de los 2 datasets a comparar fila a fila de acuerdo con la primary key:
<<DATASET_A {source_a}>>
{row_a}
<</DATASET_A>>

<<DATASET_B {source_b}>>
{row_b}
<</DATASET_B>>
"""

CLASSIFY_PROMPT_FROM_DICT_OPTIMIZED_QWEN2_5b = """
You are Qwen, created by Alibaba Cloud. You are an expert in data reconciliation.


Te daré dos datasets que comparten el mismo SCHEMA: son las diferencias que ha encontrado data-diff entre dos bases de datos ({source_a} y {source_b}).
Clasificarás el impacto semántico de las diferencias detectadas.

TAREA:
 Paso 1 : Relaciona los dos datasets fila a fila a partir de su clave primaria (PassengerId)
 Paso 2 : Clasifica en una de las 4 categorías a CADA UNA de las filas obtenidas en el paso anterior
 Paso 3 : Crea un JSON para cada una de las claves primarias con la categoría en la que lo hayas clasificado
 
Las CATEGORIAS de clasificación posibles son:
- REAL: diferencia semánticamente significativa entre los datos
- FALSO_POSITIVO_TIPO: mismo significado, distinta representación numérica
- FALSO_POSITIVO_NORMALIZACION: mismo significado, distinto encoding/collation
- AMBIGUA: no se puede determinar con certeza

REGLAS OBLIGATORIAS - SIGUE ESTO AL PIE DE LA LETRA:
- Cada PassengerId debe producir EXACTAMENTE un único objeto JSON.
- Está prohibido generar múltiples hipótesis o múltiples entradas por PassengerId.
- Debes resolver la ambigüedad internamente y elegir UNA sola categoría final.
- Devuelve ÚNICAMENTE un array JSON válido. 
- Si existen múltiples interpretaciones posibles, debes:
1. Evaluarlas internamente
2. Elegir la categoría con mayor probabilidad
3. No mostrar alternativas
- NO escribas ninguna explicación, texto, ni observaciones, ni introducción ni conclusión.
- NO uses markdown, NO uses ```json, NO uses "Based on...", NO uses listas ni tablas.
- La respuesta debe empezar con '[' y terminar con ']' sin ningún carácter adicional antes o después.
- es MUY IMPORTANTE que devuelvas un json para CADA clave primaria. TODOS los registros han de tener su clasificación.

Formato exacto requerido:
[
  {{
    "PassengerId" : clave primaria
    "categoria": "REAL|FALSO_POSITIVO_TIPO|FALSO_POSITIVO_NORMALIZACION|AMBIGUA",
    "confianza": 0.XX,
    "descripción": "motivo de la clasificación"
  }}
]
por ejemplo:
[
  {{
    "PassengerId" : "13"
    "categoría" : "REAL"
    "confianza" : 0.5,
    "descripción" : "El pasajero tiene valores distintos en la columna Ticket en {source_a} 'A/5. 2151' y en {source_b} es 'A/5. 2151_mod' , aunque la diferencia parece seguir el patrón _mod"
  }},
  {{
    "PassengerId" : "816"
    "categoría" : "REAL"
    "confianza" : 0.5,
    "descripción" : "El pasajero tiene valores distintos en la columna Name en {source_a} 'Fry, Mr. Richard' y en {source_b} es 'Fry, Mr. Richard_mod', aunque la diferencia parece seguir el patrón _mod"
  }},
...
]

Responde SOLO con el JSON anterior. Nada más. Has de clasificar EXACTAMENTE un JSON por cada clave primaria.

<<DATASET_A {source_a}>>
{row_a}
<</DATASET_A>>

<<DATASET_B {source_b}>>
{row_b}
<</DATASET_B>>
"""

SYSTEM_PROMPT = """
Eres un experto en bases de datos heterogéneas especializado en
detectar falsos positivos en comparaciones entre motores distintos.

CONTEXTO TÉCNICO CRÍTICO — memoriza esto antes de clasificar:

1. FALSOS POSITIVOS POR TIPO NUMÉRICO:
   MySQL almacena FLOAT con precisión ~7 dígitos significativos.
   PostgreSQL almacena DOUBLE PRECISION con ~15 dígitos.
   El mismo valor CSV (ej: 10.4625) puede quedar como 10.462 en MySQL
   y 10.463 en PostgreSQL. Diferencias menores a 0.01 en columnas
   numéricas son casi siempre falsos positivos por representación.

2. FALSOS POSITIVOS POR ENCODING/COLLATION:
   MySQL puede usar utf8mb4_general_ci (case-insensitive) mientras
   PostgreSQL usa collations case-sensitive por defecto.
   Diferencias solo en mayúsculas/minúsculas son falsos positivos.
   Caracteres especiales (ö, é, ñ) pueden tener representaciones
   binarias distintas entre motores: también son falsos positivos.

3. FALSOS POSITIVOS POR NULOS:
   MySQL puede tratar '' (cadena vacía) como NULL en ciertos contextos.
   PostgreSQL mantiene distinción estricta. Si un lado tiene NULL
   y el otro tiene '' (cadena vacía), es probable falso positivo.

4. DIFERENCIAS REALES:
   Cambios en valores como Survived (0→1 o 1→0), cambios grandes
   en Fare (>1%), cambios en nombres, filas que solo existen en
   un lado. Estas son diferencias reales que deben reportarse.

REGLAS DE CLASIFICACIÓN:
- REAL: diferencia semánticamente significativa entre los datos
- FALSO_POSITIVO_TIPO: mismo significado, distinta representación numérica
- FALSO_POSITIVO_NORMALIZACION: mismo significado, distinto encoding/collation
- AMBIGUA: no se puede determinar con certeza

FORMATO DE RESPUESTA — responde ÚNICAMENTE con JSON válido.
No incluyas texto antes ni después del JSON.
No uses bloques de código markdown.
"""

CLASSIFY_PROMPT = """
Analiza esta diferencia detectada por data-diff entre {source_a} y {source_b}.

Fila en {source_a}:
{row_a}

Fila en {source_b}:
{row_b}

Responde con este JSON exacto:
{{
    "Passengerid" : "clave primaria del registro (PassengerId)" 
    "categoria": "REAL" o "FALSO_POSITIVO_TIPO" o "FALSO_POSITIVO_NORMALIZACION" o "AMBIGUA",
    "confianza": número entre 0.0 y 1.0,
    "columnas_afectadas": ["col1", "col2"],
    "explicacion": "explicación concisa en español de máximo 2 frases",
}}
"""
SCHEMA_CONTEXT_TITANIC = """
| Columna     | Tipo      | Descripción                                      | Valores posibles / Notas                  |
|-------------|-----------|--------------------------------------------------|-------------------------------------------|
| survival    | int       | Sobrevivió o no                                  | 0 = No, 1 = Sí (target)                  |
| pclass      | int       | Clase del ticket                                 | 1 = 1ª, 2 = 2ª, 3 = 3ª                    |
| sex         | string    | Sexo                                             | male / female                             |
| age         | float     | Edad en años                                     | Puede tener decimales y valores nulos     |
| sibsp       | int       | Hermanos/esposos a bordo                         | Numérico                                  |
| parch       | int       | Padres/hijos a bordo                             | Numérico                                  |
| ticket      | string    | Número de ticket                                 | Alfanumérico                              |
| fare        | float     | Tarifa pagada                                    | Numérico                                  |
| cabin       | string    | Número de camarote                               | Muchos valores nulos (~77%)               |
| embarked    | string    | Puerto de embarque                               | C=Cherbourg, Q=Queenstown, S=Southampton (2 nulos) |
"""

SCHEMA_CONTEXT_JSON = """
{
  "schema_context": {
    "dataset": "passenger in titanic",
    "task": "classificacion de todos los registros únicos en 1 de 4 categories",
    "target": "Primary Key: PassengerId" ,
    "description": "Resultado de comparación de data-diff entre dos datasets similares"
    "columns": {
      "PassengerId": {
        "type": "integer",
        "role": "target",
        "description": "Identifica al pasajero",
        "values": "a unique value for each passenger"
      },

      "survival": {
        "type": "integer",
        "description": "Indica si el pasajero sobrevivió o no",
        "values": {
          "0": "No sobrevivió",
          "1": "Sobrevivió"
        }
      },

      "pclass": {
        "type": "integer",
        "description": "Clase del billete del pasajero",
        "values": {
          "1": "Primera clase",
          "2": "Segunda clase",
          "3": "Tercera clase"
        }
      },

      "sex": {
        "type": "string",
        "description": "Sexo del pasajero",
        "values": ["male", "female"]
      },

      "age": {
        "type": "float",
        "description": "Edad del pasajero en años",
        "notes": "Puede contener valores decimales y valores nulos"
      },

      "sibsp": {
        "type": "integer",
        "description": "Número de hermanos o cónyuges a bordo"
      },

      "parch": {
        "type": "integer",
        "description": "Número de padres o hijos a bordo"
      },

      "ticket": {
        "type": "string",
        "description": "Número del ticket",
        "notes": "Valor alfanumérico"
      },

      "fare": {
        "type": "float",
        "description": "Tarifa pagada por el pasajero"
      },

      "cabin": {
        "type": "string",
        "description": "Número de cabina",
        "notes": "Alta proporción de valores nulos (~77%)"
      },

      "embarked": {
        "type": "string",
        "description": "Puerto de embarque",
        "values": {
          "C": "Cherbourg",
          "Q": "Queenstown",
          "S": "Southampton"
        },
        "notes": "Algunos valores nulos (~2)"
      }
    }
  }
}
"""

JSON_CONSTRAINED_OUTPUT = """
{{
    "type": "json",
    "json_schema": {{
      "name": "titanic_classification",
      "strict": true,
      "schema": {{
        "type": "array",
        "items": {{
          "type": "object",
          "additionalProperties": false,
          "properties": {{
            "PassengerId": {{ "type": "integer" {{,
            "categoria": {{
              "type": "string",
              "enum": [
                "REAL",
                "FALSO_POSITIVO_TIPO",
                "FALSO_POSITIVO_NORMALIZACION",
                "AMBIGUA"
              ]
            }}
          {{,
          "required": ["PassengerId", "categoria"]
        }}
      }}
    }}
  }}
"""

PROMPT_ROW_BY_ROW = """
    You are an expert in data reconciliation.

    TASK:
    Classify the semantic difference between two rows from two source ({source_a} and {source_b}) with the same PassengerId ({pk}).

    IMPORTANT RULES:
    - Cada PassengerId debe producir EXACTAMENTE un único objeto JSON.
    - Está prohibido generar múltiples hipótesis o múltiples entradas por PassengerId.
    - Debes resolver la ambigüedad internamente y elegir UNA sola categoría final.
    - Do NOT return arrays
    - Do NOT include explanations outside JSON
    - PassengerId MUST be {pk}

    CATEGORIES:
    REAL | FALSO_POSITIVO_TIPO | FALSO_POSITIVO_NORMALIZACION | AMBIGUA

    INPUT:
      {source_a} : {row_a}
      {source_b} : {row_b}
    OUTPUT FORMAT:
    {{
      "PassengerId": {passenger_id},
      "categoria": "...",
      "confianza": 0.0-1.0,
      "columnas_afectadas": [],
      "explicacion": "...",
      "normalizacion_sugerida": null
    }}
"""
CONSTRAINED_OUTPUT_ROW_BY_ROW_2 = """
{
    "type": "object",
    "required": [
        "key",
        "categoria",
        "confianza",
        "columnas_afectadas",
        "explicacion",
        "normalizacion_sugerida"
    ],
    "properties": {
        "key": {
            "type": "string"
        },
        "categoria": {
            "type": "string",
            "enum": [
                "REAL",
                "FALSO_POSITIVO_TIPO",
                "FALSO_POSITIVO_NORMALIZACION",
                "AMBIGUA"
            ]
        },
        "confianza": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "columnas_afectadas": {
            "type": "array",
            "items": {"type": "string"}
        },
        "explicacion": {
            "type": "string"
        },
        "normalizacion_sugerida": {
            "type": ["string", "null"]
        }
    },
    "additionalProperties": False
}
"""
PROMPT_ROW_BY_ROW_2 = """
You are a data reconciliation system. You compore two datasets and decides if the are semantic differences between them.

Return EXACTLY one JSON object per each comparation.

key or PassengerId is {pk} and is the primery key of the records to compare.

Both pretend to be the same data from two diferent datasets : {source_a} and {source_b}
 
INPUT:
{source_a}: {row_a}
{source_b}: {row_b}

RULES:
- If one row is null and the other is not → categoria = REAL
- If values differ only by numeric precision → FALSO_POSITIVO_TIPO
- If values differ only by encoding/string formatting → FALSO_POSITIVO_NORMALIZACION
- Otherwise → REAL
- If uncertain → AMBIGUA
- If input rows do not contain data, DO NOT infer or invent values from training data
- Use ONLY the provided input
- NEVER invent data
- NEVER change PassengerId value.

OUTPUT:
{{
  "PassengerId": "{pk}",
  "categoria": "REAL|FALSO_POSITIVO_TIPO|FALSO_POSITIVO_NORMALIZACION|AMBIGUA",
  "confianza": 0.0,
  "columnas_afectadas": [],
  "explicacion": "",
  "normalizacion_sugerida": null
}}
""".strip()

PROMPT_PHI3 =  """∫
Eres un experto analista de calidad de datos y comparaciones semánticas entre bases de datos. Trabajas para un equipo de ingeniería que replica datasets entre MySQL y PostgreSQL. Un sistema de comparación literal ha detectado diferencias y tú debes clasificar si esas diferencias son semánticamente significativas o son falsos positivos.

Reglas estrictas:
- Nunca inventes datos ni asumas valores que no estén en row_a o row_b.
- Solo usa la información exacta proporcionada en la entrada.
- Si row_a o row_b es None, significa que la fila completa falta en esa fuente.
- Razona paso a paso internamente (Chain-of-Thought) antes de responder.
- Identifica claramente las columnas donde hay diferencias literales.

Entrada que recibirás siempre en este formato exacto:
Key: <key>
Row: DiffRow(key='<key>', row_a=<dict o None>, row_b=<dict o None>, source_a='mysql', source_b='postgresql')

Tu tarea:
1. Compara row_a y row_b campo por campo.
2. Clasifica la diferencia en una sola categoría:
   - "REAL" → diferencia semántica real (valores distintos con significado diferente, o fila completa ausente en una fuente).
   - "FALSO_POSITIVO_TIPO" → solo diferencia de tipo de dato (ej: '1' vs 1, 'true' vs True, etc.).
   - "FALSO_POSITIVO_NORMALIZACION" → diferencia de formato/normalización (espacios extra, mayúsculas/minúsculas, precisión decimal equivalente, None vs null, etc.).
   - "AMBIGUA" → no puedes decidir con certeza.
3. Asigna una confianza (0.0 a 1.0).
4. Lista las columnas afectadas. Si es ausencia completa de fila (row_a o row_b es None), usa ["__ROW_EXISTENCE__"].
5. Escribe una explicación clara y concisa (máximo 2-3 frases).
6. Si la categoría es "FALSO_POSITIVO_NORMALIZACION", sugiere una normalización concreta; si no, pon null.

Responde EXCLUSIVAMENTE con un JSON válido y nada más (ni texto antes ni después). El JSON debe seguir exactamente este esquema:

{{
    "key": "string",
    "categoria": "REAL" | "FALSO_POSITIVO_TIPO" | "FALSO_POSITIVO_NORMALIZACION" | "AMBIGUA",
    "confianza": number entre 0.0 y 1.0,
    "columnas_afectadas": array de strings,
    "explicacion": "string",
    "normalizacion_sugerida": "string" o null
}}

Ejemplo de cómo razonar internamente (no lo incluyas en la salida):
- Key 924: row_a tiene datos completos, row_b=None → ausencia total de fila → REAL, confianza 1.0, columnas=["__ROW_EXISTENCE__"].

Ahora procesa la siguiente comparación y responde solo con el JSON.
Key (PassengerId): {pk}
Rows : {row}
"""

CLASSIFY_PROMPT_SEMANTIC_ROW_DIFF = """

You are a senior data quality engineer specialized in semantic reconciliation between heterogeneous sources.

### Task

Given two records that represent the **same entity** (same primary key) from two different sources, determine if they are semantically equivalent.

### Input

- Source A ({source_a}): {row_a}
- Source B ({source_b}): {row_b}
- Permissiveness: 0.1 # float between 0.0 and 1.0

### Definitions

- **SEMANTICALLY_EQUIVALENT**: The differences do not change the real-world meaning or ontology of the entity.
- **SEMANTICALLY_DIFFERENT**: The differences alter the meaning or ontology of the entity in a substantive way.
- **UNCERTAIN**: Not enough information or the case is ambiguous.

### Permissiveness Factor (0.0 → 1.0)

- **0.0 (STRICT)**: Any detectable difference is considered semantically relevant → tend toward DIFFERENT.
- **0.5 (MODERATE)**: Ignore small formatting and minor numeric differences.
- **1.0 (LENIENT)**: Accept differences as long as they do not substantially change the ontology/meaning of the entity.

Higher permissiveness = more tolerant to variations (formatting, small numeric deviations, equivalent representations).
Lower permissiveness = more strict, even small differences matter.
 
### Evaluation Rules
1. Evaluate the record **as a whole** (entity-level semantics), not just field by field.
2. Ignore pure formatting differences (case, whitespace, punctuation, encoding).
3. Apply reasonable tolerance for numeric values.
4. Focus exclusively on whether the **real-world meaning** changes.
5. Do NOT infer external business rules or domain knowledge beyond the data.
6. When in doubt → output UNCERTAIN (be conservative with "DIFFERENT").

### Output Format (STRICT)
Return **ONLY** valid JSON. One object per entity:

```json
{{
"id": "<primary_key_value>",
"category": "SEMANTICALLY_EQUIVALENT | SEMANTICALLY_DIFFERENT | UNCERTAIN",
"confidence_of_category": 0.0..1.0,
"affected_columns": ["column1", "column2"],
"explanation": "Brief and precise reasoning focusing on the semantic impact",
"normalization_suggested": "SQL expression or null"
}}
"""
FEW_SHOT_EXAMPLES = [
    # ── CANONIZABLE ──────────────────────────────────────────────
    {
        "role": "user",
        "content": """\
Clasifica la diferencia entre estos dos registros.

Fuente A (mysql):
{"PassengerId": 1, "name": "  Braund, Mr. Owen Harris  ", "fare": 7.25, "survived": 1}

Fuente B (postgresql):
{"PassengerId": 1, "name": "braund, mr. owen harris", "fare": 7.25, "survived": 1}

Responde con este JSON exacto y nada más:
{
  "key": "1",
  "categoria": "CANONIZABLE|EQUIVALENT_SEMANTIC|DIFFERENT_STRUCTURAL|DIFFERENT_CONTEXTUAL|UNCERTAIN",
  "confianza": 0.0,
  "columnas_afectadas": [],
  "explicacion": "",
  "normalizacion_sugerida": null
}"""
    },
    {
        "role": "assistant",
        "content": """\
{
  "key": "1",
  "categoria": "CANONIZABLE",
  "confianza": 0.97,
  "columnas_afectadas": ["name"],
  "explicacion": "La columna name difiere solo en espacios iniciales/finales y capitalización. El valor semántico es idéntico.",
  "normalizacion_sugerida": "LOWER(TRIM(name))"
}"""
    },

    # ── EQUIVALENT_SEMANTIC ───────────────────────────────────────
    {
        "role": "user",
        "content": """\
Clasifica la diferencia entre estos dos registros.

Fuente A (mysql):
{"PassengerId": 14, "embarked": "C", "cabin": null, "survived": 0}

Fuente B (postgresql):
{"PassengerId": 14, "embarked": "Cherbourg", "cabin": null, "survived": 0}

Responde con este JSON exacto y nada más:
{
  "key": "14",
  "categoria": "CANONIZABLE|EQUIVALENT_SEMANTIC|DIFFERENT_STRUCTURAL|DIFFERENT_CONTEXTUAL|UNCERTAIN",
  "confianza": 0.0,
  "columnas_afectadas": [],
  "explicacion": "",
  "normalizacion_sugerida": null
}"""
    },
    {
        "role": "assistant",
        "content": """\
{
  "key": "14",
  "categoria": "EQUIVALENT_SEMANTIC",
  "confianza": 0.92,
  "columnas_afectadas": ["embarked"],
  "explicacion": "\"C\" y \"Cherbourg\" son representaciones del mismo puerto de embarque. La equivalencia requiere conocimiento del dominio (código IATA/abreviatura del esquema Titanic), no es resoluble con una regla de transformación simple.",
  "normalizacion_sugerida": null
}"""
    },

    # ── DIFFERENT_CONTEXTUAL — caso moneda ────────────────────────
    {
        "role": "user",
        "content": """\
Clasifica la diferencia entre estos dos registros.

Fuente A (mysql):
{"PassengerId": 55, "fare": 26.0, "currency": "GBP", "survived": 1}

Fuente B (postgresql):
{"PassengerId": 55, "fare": 32.5, "currency": "USD", "survived": 1}

Responde con este JSON exacto y nada más:
{
  "key": "55",
  "categoria": "CANONIZABLE|EQUIVALENT_SEMANTIC|DIFFERENT_STRUCTURAL|DIFFERENT_CONTEXTUAL|UNCERTAIN",
  "confianza": 0.0,
  "columnas_afectadas": [],
  "explicacion": "",
  "normalizacion_sugerida": null
}"""
    },
    {
        "role": "assistant",
        "content": """\
{
  "key": "55",
  "categoria": "DIFFERENT_CONTEXTUAL",
  "confianza": 0.85,
  "columnas_afectadas": ["fare", "currency"],
  "explicacion": "Los valores difieren en moneda y cantidad. Podrían ser equivalentes si se aplica el tipo de cambio GBP/USD histórico de la fecha de la transacción, pero esa equivalencia depende de datos externos no disponibles aquí.",
  "normalizacion_sugerida": null
}"""
    },

    # ── DIFFERENT_CONTEXTUAL — caso teléfono (fronterra con CANONIZABLE) ──
    {
        "role": "user",
        "content": """\
Clasifica la diferencia entre estos dos registros.

Fuente A (mysql):
{"PassengerId": 78, "ticket": "PC 17599", "phone": "+44 20 7946 0958", "survived": 0}

Fuente B (postgresql):
{"PassengerId": 78, "ticket": "PC 17599", "phone": "02079460958", "survived": 0}

Responde con este JSON exacto y nada más:
{
  "key": "78",
  "categoria": "CANONIZABLE|EQUIVALENT_SEMANTIC|DIFFERENT_STRUCTURAL|DIFFERENT_CONTEXTUAL|UNCERTAIN",
  "confianza": 0.0,
  "columnas_afectadas": [],
  "explicacion": "",
  "normalizacion_sugerida": null
}"""
    },
    {
        "role": "assistant",
        "content": """\
{
  "key": "78",
  "categoria": "DIFFERENT_CONTEXTUAL",
  "confianza": 0.78,
  "columnas_afectadas": ["phone"],
  "explicacion": "Ambos valores podrían representar el mismo número telefónico londinense (+44 20 = 020 en formato nacional), pero la equivalencia depende de conocer que el prefijo internacional es +44 y que la fuente B omite el cero de marcación. No es determinista sin contexto de país.",
  "normalizacion_sugerida": null
}"""
    },
]

SYSTEM_PROMPT_V2 = """\
Eres un sistema especializado en reconciliación semántica de datos entre \
fuentes heterogéneas. Recibes pares de registros que representan la misma \
entidad en dos sistemas distintos y clasificas el impacto semántico de \
sus diferencias.

## ESQUEMA DE LA TABLA
{schema_context}

## CATEGORÍAS DE CLASIFICACIÓN

Clasifica cada par en exactamente una de estas cinco categorías:

CANONIZABLE
  Los valores son semánticamente idénticos. La diferencia se puede eliminar \
con una regla determinista: TRIM, LOWER, ROUND, normalización de fecha, \
booleano, encoding Unicode. Proporciona siempre normalizacion_sugerida \
como expresión SQL o Python.
  Señales: diferencia de espacios, casing, precisión numérica trivial, \
formato de fecha distinto para el mismo instante, NFC vs NFD.

EQUIVALENT
  Los valores son semánticamente equivalentes pero la equivalencia requiere \
conocimiento de dominio que no es expresable como regla determinista.
  Señales: abreviaturas conocidas ("BCN"/"Barcelona"), sinónimos, \
"N/A" vs cadena vacía, distintas unidades de medida del mismo SI.

DIFFERENT_STRUCTURAL
  Los valores son semánticamente distintos de forma objetiva y verificable. \
Sin ambigüedad de interpretación posible.
  Señales: estados opuestos ("approved"/"rejected"), números claramente \
distintos, fechas en días diferentes, texto con distinto significado real.

DIFFERENT_CONTEXTUAL
  Los valores son distintos pero la relevancia depende de reglas de negocio \
externas. La diferencia es real pero podría ser aceptable según el contexto.
  Señales: distinta moneda, prefijo telefónico nacional vs internacional, \
títulos de puesto equivalentes en distintos organigramas.

UNCERTAIN
  Información insuficiente para clasificar con confianza ≥ {uncertainty_threshold}. \
Úsala solo si ninguna de las anteriores se puede aplicar con certeza razonable.

## PROCESO DE DECISIÓN

1. ¿Son semánticamente equivalentes?
   Sí + regla exacta disponible  → CANONIZABLE
   Sí + requiere conocimiento     → EQUIVALENT_SEMANTIC

2. Son distintos. ¿La diferencia es objetiva?
   Sí, sin ambigüedad            → DIFFERENT_STRUCTURAL
   Depende del contexto externo  → DIFFERENT_CONTEXTUAL

3. No puedes decidir             → UNCERTAIN

## REGLAS DE OUTPUT

- Responde ÚNICAMENTE con el objeto JSON del prompt de usuario.
- Sin texto antes ni después. Sin ```. Sin markdown.
- normalizacion_sugerida: expresión SQL/Python si CANONIZABLE, null en otro caso.
- confianza: float [0.0, 1.0]. Reserva > 0.95 para casos sin ninguna ambigüedad.
- columnas_afectadas: solo las columnas con diferencias observables. \
Si falta la fila completa en un lado, usa ["__ROW__"].\
"""

USER_PROMPT_V2 = """\
Clasifica la diferencia entre estos dos registros.

Fuente A ({source_a}):
{row_a}

Fuente B ({source_b}):
{row_b}

Responde con este JSON exacto y nada más:
{{
  "key": "{pk}",
  "categoria": "CANONIZABLE|EQUIVALENT_SEMANTIC|DIFFERENT_STRUCTURAL|DIFFERENT_CONTEXTUAL|UNCERTAIN",
  "confianza": 0.0,
  "columnas_afectadas": [],
  "explicacion": "",
  "normalizacion_sql_sugerida": null
}}\
"""

FEW_SHOT_EXAMPLES: List[Dict] = [

    # ── CANONIZABLE: espacios + casing ───────────────────────────
    {
        "role": "user",
        "content": textwrap.dedent("""\
            Clasifica la diferencia entre estos dos registros.

            Fuente A (mysql):
            {"PassengerId": 1, "name": "  Braund, Mr. Owen Harris  ", "fare": 7.25}

            Fuente B (postgresql):
            {"PassengerId": 1, "name": "braund, mr. owen harris", "fare": 7.25}

            Responde con este JSON exacto y nada más:
            {
              "key": "1",
              "categoria": "CANONIZABLE|EQUIVALENT_SEMANTIC|DIFFERENT_STRUCTURAL|DIFFERENT_CONTEXTUAL|UNCERTAIN",
              "confianza": 0.0,
              "columnas_afectadas": [],
              "explicacion": "",
              "normalizacion_sugerida": null
            }"""),
    },
    {
        "role": "assistant",
        "content": """\
{
  "key": "1",
  "categoria": "CANONIZABLE",
  "confianza": 0.97,
  "columnas_afectadas": ["name"],
  "explicacion": "La columna name difiere solo en espacios iniciales/finales y capitalización. El valor semántico es idéntico.",
  "normalizacion_sugerida": "LOWER(TRIM(name))"
}""",
    },

    # ── EQUIVALENT_SEMANTIC: abreviatura de dominio ───────────────
    {
        "role": "user",
        "content": textwrap.dedent("""\
            Clasifica la diferencia entre estos dos registros.

            Fuente A (mysql):
            {"PassengerId": 14, "embarked": "C", "survived": 0}

            Fuente B (postgresql):
            {"PassengerId": 14, "embarked": "Cherbourg", "survived": 0}

            Responde con este JSON exacto y nada más:
            {
              "key": "14",
              "categoria": "CANONIZABLE|EQUIVALENT_SEMANTIC|DIFFERENT_STRUCTURAL|DIFFERENT_CONTEXTUAL|UNCERTAIN",
              "confianza": 0.0,
              "columnas_afectadas": [],
              "explicacion": "",
              "normalizacion_sugerida": null
            }"""),
    },
    {
        "role": "assistant",
        "content": """\
{
  "key": "14",
  "categoria": "EQUIVALENT_SEMANTIC",
  "confianza": 0.92,
  "columnas_afectadas": ["embarked"],
  "explicacion": "\"C\" y \"Cherbourg\" designan el mismo puerto. La equivalencia requiere conocimiento del esquema Titanic, no es una regla de transformación determinista.",
  "normalizacion_sugerida": null
}""",
    },

    # ── DIFFERENT_STRUCTURAL: cambio de estado real ───────────────
    {
        "role": "user",
        "content": textwrap.dedent("""\
            Clasifica la diferencia entre estos dos registros.

            Fuente A (mysql):
            {"PassengerId": 33, "survived": 0, "fare": 8.05}

            Fuente B (postgresql):
            {"PassengerId": 33, "survived": 1, "fare": 8.05}

            Responde con este JSON exacto y nada más:
            {
              "key": "33",
              "categoria": "CANONIZABLE|EQUIVALENT_SEMANTIC|DIFFERENT_STRUCTURAL|DIFFERENT_CONTEXTUAL|UNCERTAIN",
              "confianza": 0.0,
              "columnas_afectadas": [],
              "explicacion": "",
              "normalizacion_sugerida": null
            }"""),
    },
    {
        "role": "assistant",
        "content": """\
{
  "key": "33",
  "categoria": "DIFFERENT_STRUCTURAL",
  "confianza": 0.99,
  "columnas_afectadas": ["survived"],
  "explicacion": "El valor de survived cambia de 0 (no sobrevivió) a 1 (sobrevivió). Es una diferencia semántica objetiva e inequívoca.",
  "normalizacion_sugerida": null
}""",
    },

    # ── DIFFERENT_CONTEXTUAL: moneda distinta ─────────────────────
    {
        "role": "user",
        "content": textwrap.dedent("""\
            Clasifica la diferencia entre estos dos registros.

            Fuente A (mysql):
            {"PassengerId": 55, "fare": 26.0, "currency": "GBP"}

            Fuente B (postgresql):
            {"PassengerId": 55, "fare": 32.5, "currency": "USD"}

            Responde con este JSON exacto y nada más:
            {
              "key": "55",
              "categoria": "CANONIZABLE|EQUIVALENT_SEMANTIC|DIFFERENT_STRUCTURAL|DIFFERENT_CONTEXTUAL|UNCERTAIN",
              "confianza": 0.0,
              "columnas_afectadas": [],
              "explicacion": "",
              "normalizacion_sugerida": null
            }"""),
    },
    {
        "role": "assistant",
        "content": """\
{
  "key": "55",
  "categoria": "DIFFERENT_CONTEXTUAL",
  "confianza": 0.85,
  "columnas_afectadas": ["fare", "currency"],
  "explicacion": "Los valores difieren en moneda y cantidad. Podrían ser equivalentes aplicando el tipo de cambio histórico GBP/USD, pero esa equivalencia depende de datos externos no disponibles en este contexto.",
  "normalizacion_sugerida": null
}""",
    },
]
