from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

@dataclass
class CanonicalType(ABC):
    """
    Tipo canónico abstracto. Cada subtipo representa una categoría
    semántica de dato (numérico, texto, temporal...) y sabe
    compilarse a SQL para cada dialecto.
    
    Regla de routing PRE / POST (decidida en CanonicalPipeline):
      PRE : to_sql() tiene éxito en TODOS los dialectos involucrados.
            La expresión entra en la query de data-diff directamente.
      POST: to_sql() lanza UnsupportedTransformation en al menos uno.
            La columna entra sin transformar en la query y
            PostCanonicalizer aplica to_python() a ambos lados del
            DiffRow antes del clasificador LLM.
    """
    column_name:      str
    nullable:         bool = True
    information_loss: str  = ""   # Documentación obligatoria

    @abstractmethod
    def to_sql(self, dialect: "BaseDialect") -> str:
        """
        Compila la expresión SQL canónica para el dialecto dado.
        Devuelve una expresión SQL que puede usarse directamente
        en un SELECT o en una vista.
        """
        ...

    @abstractmethod
    def to_python(self) -> Callable:
        """
        Equivalente Python de to_sql().
        Devuelve un callable (valor_raw) -> valor_canonizado.
        Gestiona None respetando self.nullable.
        """
        ...
        
    @abstractmethod
    def validate(self, value) -> bool:
        """
        Valida que un valor de muestra es compatible con este tipo.
        Usado durante la introspección para confirmar el tipo inferido.
        """
        ...

    def with_null_handling(self, expr: str, dialect: "BaseDialect") -> str:
        """
        Envuelve la expresión con manejo de nulos si la columna
        es nullable. Delega la sintaxis al dialecto.
        """
        if self.nullable:
            return dialect.coalesce(expr, dialect.null_replacement(self))
        return expr