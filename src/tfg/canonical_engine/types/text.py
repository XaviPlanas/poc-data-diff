from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List
from .base import CanonicalType
from ..dialect.base import UnsupportedTransformation

class TextTransformation:
    TRIM            = "trim"
    LOWERCASE       = "lowercase"
    NFC             = "unicode_nfc"
    NFKC            = "unicode_nfkc"
    ASCII_FOLD      = "ascii_fold"
    REMOVE_PUNCT    = "remove_punct"
    COLLAPSE_SPACES = "collapse_spaces"

@dataclass
class TextCanonical(CanonicalType):
    """
    Tipo texto con pipeline de transformaciones configurable.
    Las transformaciones se aplican en el orden en que se declaran.
    Cada transformación se delega al dialecto para garantizar
    que la expresión SQL resultante es válida en el motor destino.
    """
    transformations: List[str] = field(default_factory=lambda: [
        TextTransformation.TRIM,
        TextTransformation.LOWERCASE,
        TextTransformation.NFC,
    ])
    encoding:        str = "utf8mb4"
    max_length:      int = 255

    information_loss: str = (
        "Se pierde la distinción de mayúsculas/minúsculas. "
        "La normalización NFC unifica representaciones Unicode "
        "equivalentes pero preserva caracteres especiales."
    )

    def to_sql(self, dialect) -> str:
        """
        Construye la expresión SQL aplicando las transformaciones
        en orden, cada una envolviendo a la anterior.
        El resultado es una expresión SQL anidada del tipo:
        LOWER(TRIM(NORMALIZE(col, NFC)))
        """
        # Punto de partida: referencia a la columna con encoding
        expr = dialect.ensure_encoding(self.column_name, self.encoding)
        for t in self.transformations:
            expr = self._apply_sql(t, expr, dialect)
        return self.with_null_handling(expr, dialect)

    # ── to_sql_partial: split PRE/POST por dialectos ──────────────

    def to_sql_partial(
        self,
        dialect,
        peer_dialect,
    ) -> Tuple[str, List[str]]:
        """
        Compila las transformaciones que AMBOS dialectos soportan en SQL.

        Recorre el pipeline en orden; en cuanto una transformación falla
        en cualquiera de los dos dialectos, esa y todas las siguientes
        se marcan como pendientes para POST.

        Devuelve:
            (sql_expr,        # expresión SQL para *dialect* (PRE)
             pending_transforms)  # lista de transforms para POST Python
        """
        expr    = dialect.ensure_encoding(self.column_name, self.encoding)
        applied = []

        for t in self.transformations:
            # Comprobar soporte en AMBOS dialectos antes de aplicar
            if not self._is_supported(t, dialect):
                pending = self.transformations[len(applied):]
                return self.with_null_handling(expr, dialect), pending

            if not self._is_supported(t, peer_dialect):
                pending = self.transformations[len(applied):]
                return self.with_null_handling(expr, dialect), pending

            # Ambos lo soportan → incluir en SQL
            expr = self._apply_sql(t, expr, dialect)
            applied.append(t)

        return self.with_null_handling(expr, dialect), []

    # ── to_python: pipeline Python completo ──────────────────────

    def to_python(self) -> Callable:
        """
        Equivalente Python de to_sql() para el pipeline completo.
        PostCanonicalizer lo usa para las transformaciones POST.

        Nota: aplica TODAS las transformaciones, no solo las pendientes.
        PostCanonicalizer lo invoca sobre valores que ya pasaron por
        el PRE SQL, por lo que en la práctica solo las últimas
        (NFC, NFKC…) producirán cambio observable.
        """
        transforms = list(self.transformations)
        nullable   = self.nullable

        def transform(value) -> str:
            if value is None:
                return "" if nullable else None
            v = str(value)
            for t in transforms:
                v = _apply_python(t, v)
            return v

        transform.__name__ = f"TextCanonical({'+'.join(transforms)})"
        return transform

    def to_python_for(self, pending: List[str]) -> Callable:
        """
        Devuelve un callable que aplica únicamente las transformaciones
        de *pending* (las que no pudieron hacerse en SQL).
        Lo usa CanonicalPipeline para registrar el callable exacto
        que necesita PostCanonicalizer.
        """
        nullable = self.nullable

        def transform(value) -> str:
            if value is None:
                return "" if nullable else None
            v = str(value)
            for t in pending:
                v = _apply_python(t, v)
            return v

        label = "+".join(pending) if pending else "identity"
        transform.__name__ = f"TextPost({label})"
        return transform

    def validate(self, value) -> bool:
        return isinstance(value, str)

    # ── Helpers privados ─────────────────────────────────────────

    def _apply_sql(self, transform: str, expr: str, dialect) -> str:
        """Aplica una transformación al dialecto; propaga la excepción."""
        if transform == TextTransformation.TRIM:
            return dialect.trim(expr)
        if transform == TextTransformation.LOWERCASE:
            return dialect.lowercase(expr)
        if transform == TextTransformation.NFC:
            return dialect.normalize_unicode(expr, "NFC")
        if transform == TextTransformation.NFKC:
            return dialect.normalize_unicode(expr, "NFKC")
        if transform == TextTransformation.ASCII_FOLD:
            return dialect.ascii_fold(expr)
        if transform == TextTransformation.COLLAPSE_SPACES:
            return dialect.collapse_spaces(expr)
        if transform == TextTransformation.REMOVE_PUNCT:
            return dialect.remove_punct(expr)
        raise ValueError(f"Transformación desconocida: {transform!r}")

    def _is_supported(self, transform: str, dialect) -> bool:
        """True si el dialecto soporta la transformación sin lanzar."""
        try:
            self._apply_sql(transform, "x", dialect)
            return True
        except UnsupportedTransformation:
            return False

# ── Implementaciones Python de cada transformación ────────────────

def _apply_python(transform: str, value: str) -> str:
    if transform == TextTransformation.TRIM:
        return value.strip()
    if transform == TextTransformation.LOWERCASE:
        return value.lower()
    if transform == TextTransformation.NFC:
        return unicodedata.normalize("NFC", value)
    if transform == TextTransformation.NFKC:
        return unicodedata.normalize("NFKC", value)
    if transform == TextTransformation.ASCII_FOLD:
        nfd = unicodedata.normalize("NFD", value)
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    if transform == TextTransformation.COLLAPSE_SPACES:
        return re.sub(r"\s+", " ", value).strip()
    if transform == TextTransformation.REMOVE_PUNCT:
        return re.sub(r"[^\w\s]", "", value)
    raise ValueError(f"Transformación Python desconocida: {transform!r}")