from dataclasses import asdict, dataclass
from enum import Enum
import json
import hashlib  
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
class DiffEvent:
    """Evento de diferencia, que contiene una columna divergente.
       Una comparación puede generar múltiples eventos si hay varias columnas divergentes.
    """
    key:        any
    columna:    str
    valor_a:    any
    valor_b:    any
@dataclass
class DiffRow:
    """Par de filas divergentes devueltas por data-diff."""
    key:        any
    row_a:      dict
    row_b:      dict
    source_a:   str   # ej. "mysql://...titanic"
    source_b:   str   # ej. "postgresql://...titanic"
    accion:     DiffAction = None
    eventos:    list[DiffEvent] = None

@dataclass
class DiffClassification:
    key: any
    accion: DiffAction
    categoria: DiffCategory
    confianza: float
    columnas_afectadas: list[str]
    explicacion: str
    normalizacion_sugerida: str | None
    row_a: dict
    row_b: dict

    def to_dict(self) -> dict:
        data = asdict(self)
        # convertir enums
        data["accion"] = self.accion.value if self.accion else None
        data["categoria"] = self.categoria.value if self.categoria else None
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

@dataclass
class SegmentStructure:
    """Estructura de un segmento de datos, que puede ser una tabla, un bloque de filas, o una fila individual."""
    columnas:  list[str]
    pk:        str   # "table", "row_block", "row"

    def _normalized(self) -> str:
        """Representación determinista del schema"""
        return json.dumps({
            "columnas": sorted(self.columnas),  # orden estable
            "pk": self.pk
        }, sort_keys=True)

    def schema_version(self) -> str:
        """Fingerprint del schema"""
        normalized = self._normalized()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()