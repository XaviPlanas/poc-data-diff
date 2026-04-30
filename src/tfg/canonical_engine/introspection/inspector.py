
from sqlalchemy import create_engine, inspect, text
from .type_mapper import TypeMapper

import re
import unicodedata

import logging
logger = logging.getLogger(__name__)
class SchemaInspector:
    """
    Inspecciona el esquema de una tabla remota y devuelve
    los tipos canónicos inferidos para cada columna.
    No descarga datos: usa solo los metadatos del motor.
    """

    def __init__(self, connection_uri: str):
        logger.info(f"Connection URL: {connection_uri}")
        self.engine  = create_engine(connection_uri)
        self.mapper  = TypeMapper()
        self.dialect = self._detect_dialect()

    def _detect_dialect(self):
        from ..dialect.registry import DialectRegistry
        dialect_name = self.engine.dialect.name  # 'mysql', 'postgresql', etc.
        return DialectRegistry.get(dialect_name)
    
    @staticmethod
    def _normalize_column_name(name: str) -> str:
        """Creamos una versión del nombre de columa"""
        logger.trace(f"Columna con valor {name} al entrar")
        # 1. quitar acentos
        name = unicodedata.normalize('NFKD', name)
        name = name.encode('ascii', 'ignore').decode('ascii')
        logger.trace(f"Columna con valor {name} tras paso 1/5 (sin acentos)")
        # 2. camelCase → snake_case
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        logger.trace(f"Columna con valor {name} tras paso 2/5 (camelCase)")
        # 3. minúsculas
        name = name.lower()
        logger.trace(f"Columna con valor {name} tras paso 3/5 (minúsculas)")
        # 4. reemplazar caracteres no válidos por _
        name = re.sub(r'[^a-z0-9]+', '_', name)
        logger.trace(f"Columna con valor {name} tras paso 4/5 (_)")
        # 5. quitar underscores extremos
        name = name.strip('_')
        logger.trace(f"Columna con valor {name} tras paso final 5/5 (strip de _)")
        #controlamos que no se generen nombres vacíos:
        if not name:
            raise ValueError("La normalización produjo un nombre de columna vacío")
        return name
    
    def inspect_table(self, table_name: str) -> dict:
        """
        Devuelve un diccionario {column_name: CanonicalType}
        con los tipos canónicos inferidos para cada columna.
        """
        inspector = inspect(self.engine)
        columns   = inspector.get_columns(table_name)

        canonical_types = {}
        for col in columns:
            if self._detect_dialect() == "mysql" : #mySQL no cumple con ANSI SQL
                col_name    = f"{col["name"]}" 
            else : 
                col_name    = f"\"{col["name"]}\"" #ANSI SQL usa \"
            view_name   = self._normalize_column_name(f"{col["name"]}")
            sql_type    = col["type"]
            nullable    = col["nullable"]
            canonical   = self.mapper.map(
                col_name  = col_name,
                sql_type  = sql_type,
                nullable  = nullable,
                dialect   = self.dialect,
            )
            canonical_types[col_name] = canonical

        return canonical_types

    def get_sample(self, table_name: str, n: int = 5) -> list:
        """
        Descarga una muestra mínima de datos para validar
        los tipos inferidos. Es la única descarga de datos
        que realiza el motor de canonización.
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT * FROM {table_name} LIMIT {n}")
            )
            return [dict(row) for row in result]