from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DiffCategory(Enum):
    # REAL                    = "REAL"
    # FALSO_POSITIVO_TIPO     = "FALSO_POSITIVO_TIPO"
    # FALSO_POSITIVO_NORM     = "FALSO_POSITIVO_NORMALIZACION"
    # AMBIGUA                 = "AMBIGUA"
    SEMANTICALLY_EQUIVALENT = "SEMANTICALLY_EQUIVALENT"
    SEMANTICALLY_DIFFERENT  = "SEMANTICALLY_DIFFERENT"
    UNCERTAIN               = "UNCERTAIN"
    ERROR                   = "ERROR"

class DiffAction(Enum):
    INSERT = "INSERT"
    DELETE = "DELETE"
    UPDATE = "UPDATE"
@dataclass
class DiffRow:
    """Par de filas divergentes devueltas por data-diff."""
    key:        any
    row_a:      dict
    row_b:      dict
    source_a:   str   # ej. "mysql://...titanic"
    source_b:   str   # ej. "postgresql://...titanic"
@dataclass
class DiffClassification:
    key:                    any
    accion:                 DiffAction
    categoria:              DiffCategory
    confianza:              float
    columnas_afectadas:     list[str]
    explicacion:            str
    normalizacion_sugerida: Optional[str]
    row_a:                  dict
    row_b:                  dict
@dataclass
class DiffEvent:
    """Evento de diferencia, que contiene una columna divergente.
       Una comparación puede generar múltiples eventos si hay varias columnas divergentes.
    """
    key:        any
    columna:    str
    valor_a:    any
    valor_b:    any
    accion:     DiffAction

@dataclass
class SegmentStructure:
    """Estructura de un segmento de datos, que puede ser una tabla, un bloque de filas, o una fila individual."""
    columnas:  list[str]
    pk:        str   # "table", "row_block", "row"