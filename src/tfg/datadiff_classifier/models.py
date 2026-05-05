from dataclasses import asdict, dataclass
from enum import Enum
import json
import hashlib

class DiffCategory(Enum):
    CANONIZABLE          = "canonizable"
    EQUIVALENT           = "equivalent_semantic"
    DIFFERENT_CONTEXTUAL = "different_contextual"
    DIFFERENT_STRUCTURAL = "different_structural"
    DIFFERENT_SEMANTICAL = "different_semantical"
    UNCERTAIN            = "uncertain"
    ERROR                = "error"
    
# Grupos funcionales: frozenset de valores string (no de miembros del enum)
# para evitar el problema de asignación en enum post-definición
_FALSE_POSITIVE_VALUES = frozenset({
    "CANONIZABLE",
    "EQUIVALENT_SEMANTIC",
})
_NEEDS_REVIEW_VALUES = frozenset({
    "DIFFERENT_CONTEXTUAL",
    "UNCERTAIN",
    "ERROR",
})
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
    
    def is_real_difference(self) -> bool:
        """True solo para DIFFERENT_STRUCTURAL."""
        return self.categoria == DiffCategory.DIFFERENT_STRUCTURAL
    
    def is_false_positive(self) -> bool:
        """True si la diferencia es resuelta por canonización o equivalencia."""
        return self.categoria.value in _FALSE_POSITIVE_VALUES

    def needs_review(self) -> bool:
        """True si requiere revisión humana."""
        return self.categoria.value in _NEEDS_REVIEW_VALUES
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