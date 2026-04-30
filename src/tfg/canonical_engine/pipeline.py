# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, text
from .introspection.inspector  import SchemaInspector
from .dialect.base             import UnsupportedTransformation
from .engine                   import PythonFallback
from dataclasses               import dataclass
from typing                    import Dict, List, Optional
import unicodedata
import re

import logging
logger = logging.getLogger(__name__)
@dataclass
class CanonicalColumn:
    name:             str
    sql_expression:   str           # Expresión SQL remota (si es posible)
    python_fallback:  callable      # Función Python (si SQL no es posible)
    requires_download: bool         # True si se necesita descargar el dato
    information_loss: str
    view_col_name:        Optional[str] = None

    def __post_init__(self):        # Si no se ha definido un view_name , se asume name
        if self.view_col_name is None:
            self.view_col_name = self.name

@dataclass
class CanonicalPlan:
    """
    Plan de canonización para una tabla.
    Documenta qué ocurre con cada columna y dónde se canoniza.
    """
    table_name:       str
    dialect_name:     str
    columns:          Dict[str, CanonicalColumn]
    view_sql:         str           # SQL de la vista canónica completa
    download_columns: List[str]     # Columnas que requieren fallback Python

    def report(self) -> str:
        lines = [
            f"Plan de canonización: {self.table_name} ({self.dialect_name})",
            f"{'─' * 60}",
        ]
        for col_name, col in self.columns.items():
            location = "Python (descarga)" if col.requires_download \
                       else "SQL (remoto)"
            lines.append(f"  {col_name:<20} → {location}")
            if col.information_loss:
                lines.append(f"  {'':20}   ⚠ {col.information_loss}")
        lines.append(f"{'─' * 60}")
        lines.append(
            f"Columnas remotas:  "
            f"{len(self.columns) - len(self.download_columns)}"
        )
        lines.append(
            f"Columnas descarga: {len(self.download_columns)}"
        )
        logger.debug(lines)
        
        return "\n".join(lines)

class CanonicalPipeline:

    def __init__(self, connection_uri: str, table_name: str):
        self.inspector  = SchemaInspector(connection_uri)
        self.dialect    = self.inspector.dialect
        self.table      = table_name
        self.engine     = create_engine(connection_uri)

    def build_plan(self) -> CanonicalPlan:
        """
        Construye el plan de canonización inspeccionando el esquema
        remoto y compilando las expresiones SQL para cada columna.
        No ejecuta ninguna transformación todavía.
        """
        canonical_types   = self.inspector.inspect_table(self.table)
        columns           = {}
        download_columns  = []
        logger.debug(f"CanonicalPipeline.build_plan: canonical_types = {canonical_types}")
        for col_name, canonical_type in canonical_types.items():
            try:
                sql_expr = canonical_type.to_sql(self.dialect)
                # if self.dialect.name == "mysql" : #mySQL no cumple con ANSI SQL
                #      column_name    = col_name.replace("\"","") 
                # else : 
                #      column_name    = col_name #ANSI SQL usa \"    
                
                logger.trace(f"Dialect : {self.dialect.name} Columna {col_name} , se normaliza a {col_name}.Se puede canonizar en SQL con expresión: {sql_expr}")
                
                columns[col_name] = CanonicalColumn(
                    name              = col_name,
                    sql_expression    = f"{sql_expr} AS {self.inspector._normalize_column_name(col_name)}",
                    python_fallback   = None,
                    requires_download = False,
                    information_loss  = canonical_type.information_loss,
                )

            except UnsupportedTransformation as e:
                # El motor no soporta esta transformación de forma nativa.
                # Se marca para fallback Python y se documenta.
                fallback_fn = self._resolve_fallback(e.transformation)
                
                if self.dialect.name == "mysql" : #mySQL no cumple con ANSI SQL
                     column_name    = col_name.replace("\"","`") 
                else : 
                     column_name    = col_name #ANSI SQL usa \"  
                     
                columns[col_name] = CanonicalColumn(
                    name              = column_name,
                    sql_expression    = column_name,  # Sin transformar
                    python_fallback   = fallback_fn,
                    requires_download = True,
                    information_loss  = canonical_type.information_loss,
                )
                download_columns.append(col_name)

        view_sql = self._build_view_sql(columns)

        return CanonicalPlan(
            table_name        = self.table,
            dialect_name      = self.dialect.name,
            columns           = columns,
            view_sql          = view_sql,
            download_columns  = download_columns,
        )

    def _resolve_fallback(self, transformation: str) -> callable:
        mapping = {
            "NORMALIZE(NFC)":  lambda v: PythonFallback.normalize_unicode(v, "NFC"),
            "NORMALIZE(NFKC)": lambda v: PythonFallback.normalize_unicode(v, "NFKC"),
            "ascii_fold":      PythonFallback.ascii_fold,
            "collapse_spaces": PythonFallback.collapse_spaces,
        }
        return mapping.get(transformation, lambda v: v)

    def _build_view_sql(self, columns: dict) -> str:
        expressions = [col.sql_expression for col in columns.values()] 
        view_sql = f"""SELECT
            {',\n    '.join(expressions)}
            FROM {self.table}"""
        logger.debug("_build_view_sql: {view_sql}")
        return(view_sql)

    def apply_plan(self, plan: CanonicalPlan) -> None:
        """
        Materializa el plan creando una VISTA SQL en el motor remoto
        para las columnas que pueden canonizarse en SQL,
        y aplicando los fallbacks Python al resultado para
        las columnas que requieren descarga.
        """
        view_name = f"{self.table}_canonical" 

        with self.engine.connect() as conn:
           
            drop_sql = f"DROP VIEW IF EXISTS {view_name}"
            logger.debug(drop_sql)
            conn.execute(
                text(drop_sql)
            )

            create_sql = f"CREATE VIEW {view_name} AS {plan.view_sql}"
            logger.debug(create_sql)
            conn.execute(
                text(create_sql)
            )
            conn.commit()

        logger.info(f"Vista canónica creada: {view_name}")
        if plan.download_columns:
            logger.warning(f"Las siguientes columnas requieren fallback Python (se descargarán los valores): {plan.download_columns}")
            