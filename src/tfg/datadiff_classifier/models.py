from dataclasses import dataclass
from enum import Enum
from typing import Optional

class IAModel(Enum):
    ANTHROPIC   = "claude-opus-4-5"
    OLLAMA      = "OLLAMA"
    QWEN        = "QWEN"

class DiffCategory(Enum):
    REAL                    = "REAL"
    FALSO_POSITIVO_TIPO     = "FALSO_POSITIVO_TIPO"
    FALSO_POSITIVO_NORM     = "FALSO_POSITIVO_NORMALIZACION"
    AMBIGUA                 = "AMBIGUA"

@dataclass
class DiffRow:
    """Par de filas divergentes devueltas por data-diff."""
    key:        any
    row_a:      dict
    row_b:      dict
    source_a:   str   # ej. "mysql://...titanic"
    source_b:   str   # ej. "postgresql://...titanic"

@dataclass
class ClassificationResult:
    key:                    any
    categoria:              DiffCategory
    confianza:              float
    columnas_afectadas:     list[str]
    explicacion:            str
    normalizacion_sugerida: Optional[str]
    row_a:                  dict
    row_b:                  dict