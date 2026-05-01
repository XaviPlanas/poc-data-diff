from dataclasses import dataclass
from typing      import Callable
from .base       import CanonicalType


@dataclass
class NumericCanonical(CanonicalType):
    """
    Tipo numérico con precisión fija. Resuelve los falsos positivos
    por diferencias de representación entre FLOAT, DOUBLE PRECISION,
    REAL y NUMERIC entre motores distintos.
    """
    precision: int = 3
    scale:     int = 10

    information_loss: str = (
        "Se pierden los dígitos a partir del decimal {precision}. "
        "Diferencias menores a 10^-{precision} se tratan como iguales."
    )

    def to_sql(self, dialect) -> str:
        col  = self.column_name
        expr = dialect.round_numeric(col, self.precision, self.scale)
        return self.with_null_handling(expr, dialect)

    def to_python(self) -> Callable:
        precision = self.precision
        nullable  = self.nullable

        def transform(value):
            if value is None:
                return 0 if nullable else None
            try:
                return round(float(value), precision)
            except (TypeError, ValueError):
                return value

        transform.__name__ = f"NumericCanonical(precision={precision})"
        return transform

    def validate(self, value) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False


@dataclass
class IntegerCanonical(CanonicalType):
    """
    Tipo entero. Normaliza representaciones de booleanos
    como TINYINT(1) en MySQL.
    """
    information_loss: str = "Ninguna para enteros bien formados."

    def to_sql(self, dialect) -> str:
        col  = self.column_name
        expr = dialect.cast_integer(col)
        return self.with_null_handling(expr, dialect)

    def to_python(self) -> Callable:
        nullable = self.nullable

        def transform(value):
            if value is None:
                return 0 if nullable else None
            try:
                return int(value)
            except (TypeError, ValueError):
                return value

        transform.__name__ = "IntegerCanonical"
        return transform

    def validate(self, value) -> bool:
        try:
            int(value)
            return True
        except (TypeError, ValueError):
            return False