from dataclasses import dataclass
from typing      import Callable
from .base       import CanonicalType


@dataclass
class BooleanCanonical(CanonicalType):
    """
    Tipo booleano canónico. Resuelve la divergencia entre motores
    que implementan booleanos de forma nativa (PostgreSQL: BOOLEAN)
    y motores que los representan como enteros (MySQL: TINYINT(1)).

    El problema concreto es el siguiente: cuando data-diff compara
    una columna BOOLEAN de PostgreSQL con su equivalente TINYINT(1)
    de MySQL, puede encontrar representaciones binarias distintas
    para el mismo valor lógico dependiendo de cómo el conector
    ORM haya inferido y transmitido el tipo. La canonización
    normaliza ambas representaciones a un entero 0/1 antes de
    que el valor entre en el hash, eliminando esa fuente de
    falsos positivos.

    Adicionalmente, en la práctica aparecen representaciones textuales
    de booleanos ('true', 'yes', 'on', '1') provenientes de imports
    desde CSV, Excel o APIs REST. La canonización cubre también
    estos casos mediante una normalización SQL que los colapsa
    a 0 o 1 de forma determinista.
    """

    information_loss: str = (
        "Se pierde la representación original. "
        "Todos los valores se normalizan a 0 o 1."
    )

    _TRUTHY  = frozenset({"true", "yes", "on", "1", "t"})
    _FALSY   = frozenset({"false", "no", "off", "0", "f"})

    def to_sql(self, dialect) -> str:
        expr = dialect.normalize_boolean(self.column_name)
        return self.with_null_handling(expr, dialect)

    def to_python(self) -> Callable:
        truthy   = self._TRUTHY
        falsy    = self._FALSY
        nullable = self.nullable

        def transform(value):
            if value is None:
                return None
            if isinstance(value, bool):
                return 1 if value else 0
            if isinstance(value, int):
                return 1 if value else 0
            if isinstance(value, str):
                s = value.lower()
                if s in truthy:
                    return 1
                if s in falsy:
                    return 0
            return 1 if value else 0

        transform.__name__ = "BooleanCanonical"
        return transform

    def validate(self, value) -> bool:
        """
        Acepta cualquier representación habitual de un booleano:
        tipo bool nativo, entero 0/1, o cadenas textuales.
        """
        if isinstance(value, bool):
            return True
        if isinstance(value, int) and value in (0, 1):
            return True
        if isinstance(value, str) and value.lower() in (
            self._TRUTHY | self._FALSY
        ):
            return True
        return False