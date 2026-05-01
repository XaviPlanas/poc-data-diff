# -*- coding: utf-8 -*-
from __future__ import annotations
from sqlalchemy import create_engine, text
from .introspection.inspector  import SchemaInspector
from .dialect.base             import UnsupportedTransformation
# from .engine                   import PythonFallback
from dataclasses               import dataclass
from typing                    import Dict, List, Optional
import unicodedata
import re

import logging
from typing import Optional

from .dialect.base          import UnsupportedTransformation, BaseDialect
from .dialect.registry      import DialectRegistry
from .introspection.inspector import SchemaInspector
from .plan                  import CanonicalColumn, CanonicalPlan, Layer
from .types.text            import TextCanonical

logger = logging.getLogger(__name__)
# @dataclass
# class CanonicalColumn:
#     name:             str
#     sql_expression:   str           # Expresión SQL remota (si es posible)
#     python_fallback:  callable      # Función Python (si SQL no es posible)
#     requires_download: bool         # True si se necesita descargar el dato
#     information_loss: str
#     view_col_name:        Optional[str] = None

#     def __post_init__(self):        # Si no se ha definido un view_name , se asume name
#         if self.view_col_name is None:
#             self.view_col_name = self.name

# @dataclass
# class CanonicalPlan:
#     """
#     Plan de canonización para una tabla.
#     Documenta qué ocurre con cada columna y dónde se canoniza.
#     """
#     table_name:       str
#     dialect_name:     str
#     columns:          Dict[str, CanonicalColumn]
#     view_sql:         str           # SQL de la vista canónica completa
#     download_columns: List[str]     # Columnas que requieren fallback Python

#     def report(self) -> str:
#         lines = [
#             f"Plan de canonización: {self.table_name} ({self.dialect_name})",
#             f"{'─' * 60}",
#         ]
#         for col_name, col in self.columns.items():
#             location = "Python (descarga)" if col.requires_download \
#                        else "SQL (remoto)"
#             lines.append(f"  {col_name:<20} → {location}")
#             if col.information_loss:
#                 lines.append(f"  {'':20}   ⚠ {col.information_loss}")
#         lines.append(f"{'─' * 60}")
#         lines.append(
#             f"Columnas remotas:  "
#             f"{len(self.columns) - len(self.download_columns)}"
#         )
#         lines.append(
#             f"Columnas descarga: {len(self.download_columns)}"
#         )
#         logger.debug(lines)
        
#         return "\n".join(lines)

class CanonicalPipeline:

    def __init__(self, connection_uri: str, table_name: str):
        self.inspector  = SchemaInspector(connection_uri)
        self.dialect    = self.inspector.dialect
        self.table      = table_name
        self.engine     = create_engine(connection_uri)
        self.uri        = connection_uri
        logger.debug(
            "CanonicalPipeline init: table=%s dialect=%s",
            table_name, self.dialect.name,
        )

    # ── API pública ───────────────────────────────────────────────

    def build_plan(
        self,
        peer_dialect: Optional[BaseDialect] = None,
    ) -> CanonicalPlan:
        """
        Construye el plan de canonización inspeccionando el esquema
        remoto y compilando las expresiones SQL para cada columna.
        No ejecuta ninguna transformación todavía.
        """
        canonical_types = self.inspector.inspect_table(self.table)
        columns: dict[str, CanonicalColumn] = {}

        for col_name, ctype in canonical_types.items():
            col = self._classify_column(col_name, ctype, peer_dialect)
            columns[col_name] = col

            logger.debug(
                "column=%s layer=%s pending=%s",
                col_name, col.layer, col.pending_transforms,
            )

        plan = CanonicalPlan(
            table_name   = self.table,
            dialect_name = self.dialect.name,
            columns      = columns,
        )

        pre   = sum(1 for c in columns.values() if c.layer == Layer.PRE)
        split = sum(1 for c in columns.values() if c.layer == Layer.SPLIT)
        post  = sum(1 for c in columns.values() if c.layer == Layer.POST)

        logger.info(
            "Plan construido: table=%s dialect=%s PRE=%d SPLIT=%d POST=%d",
            self.table, self.dialect.name, pre, split, post,
            extra={"pre": pre, "split": split, "post": post},
        )
        if split or post:
            post_names = [
                n for n, c in columns.items()
                if c.layer in (Layer.SPLIT, Layer.POST)
            ]
            logger.warning(
                "Columnas con transformaciones POST (Python post-diff): %s",
                post_names,
            )

        return plan

    def resolve_peer_dialect(self, peer_uri: str) -> BaseDialect:
        """
        Devuelve el dialecto del motor par a partir de su URI de conexión.
        Helper para construir el peer_dialect sin instanciar un inspector.
        """
        from sqlalchemy import create_engine
        engine = create_engine(peer_uri)
        peer   = DialectRegistry.get(engine.dialect.name)
        logger.debug("Peer dialect resuelto: %s", peer.name)
        return peer

    # ── Clasificación de columnas ─────────────────────────────────

    def _classify_column(
        self,
        col_name:     str,
        ctype,
        peer_dialect: Optional[BaseDialect],
    ) -> CanonicalColumn:
        """
        Determina la capa (PRE / SPLIT / POST) de una columna y
        construye el CanonicalColumn correspondiente.

        Para TextCanonical usamos to_sql_partial() que permite routing
        granular: las primeras N transforms van a SQL y las restantes a POST.

        Para otros tipos (Numeric, Boolean, Timestamp…) el routing es
        binario: todo PRE si el dialecto (y el par) lo soportan, todo POST
        si alguno falla.
        """
        if isinstance(ctype, TextCanonical) and peer_dialect is not None:
            return self._classify_text_split(col_name, ctype, peer_dialect)

        return self._classify_binary(col_name, ctype, peer_dialect)

    def _classify_text_split(
        self,
        col_name:     str,
        ctype:        TextCanonical,
        peer_dialect: BaseDialect,
    ) -> CanonicalColumn:
        """
        Routing granular para TextCanonical con peer_dialect.

        Usa to_sql_partial() para obtener:
          - sql_expr        : expresión con los transforms soportados por ambos
          - pending         : lista de transforms restantes para Python

        Si pending está vacío → capa PRE (todo en SQL).
        Si pending no vacío   → capa SPLIT (parte SQL + parte Python).
        """
        sql_expr, pending = ctype.to_sql_partial(self.dialect, peer_dialect)

        if not pending:
            # Todos los transforms soportados en ambos dialectos
            return CanonicalColumn(
                name               = col_name,
                layer              = Layer.PRE,
                sql_expression     = f"{sql_expr} AS {col_name}",
                post_fn            = None,
                pending_transforms = [],
                information_loss   = ctype.information_loss,
            )
        else:
            # Parte SQL segura + Python para los restantes
            post_fn = ctype.to_python_for(pending)
            return CanonicalColumn(
                name               = col_name,
                layer              = Layer.SPLIT,
                sql_expression     = f"{sql_expr} AS {col_name}",
                post_fn            = post_fn,
                pending_transforms = pending,
                information_loss   = ctype.information_loss,
            )

    def _classify_binary(
        self,
        col_name:     str,
        ctype,
        peer_dialect: Optional[BaseDialect],
    ) -> CanonicalColumn:
        """
        Routing binario: intenta to_sql() en el dialecto local y,
        si hay peer, también en él.  Si alguno falla → POST completo.
        """
        # Probar dialecto local
        try:
            sql_expr   = ctype.to_sql(self.dialect)
            local_ok   = True
        except UnsupportedTransformation:
            sql_expr   = col_name    # passthrough
            local_ok   = False

        # Probar dialecto par (si aplica)
        peer_ok = True
        if peer_dialect is not None and local_ok:
            try:
                ctype.to_sql(peer_dialect)
            except UnsupportedTransformation:
                peer_ok  = False
                sql_expr = col_name  # revertir a passthrough

        both_ok = local_ok and peer_ok

        if both_ok:
            return CanonicalColumn(
                name               = col_name,
                layer              = Layer.PRE,
                sql_expression     = f"{sql_expr} AS {col_name}",
                post_fn            = None,
                pending_transforms = [],
                information_loss   = ctype.information_loss,
            )
        else:
            post_fn = ctype.to_python()
            return CanonicalColumn(
                name               = col_name,
                layer              = Layer.POST,
                sql_expression     = col_name,   # passthrough en la query
                post_fn            = post_fn,
                pending_transforms = [],          # binario: todo en Python
                information_loss   = ctype.information_loss,
            )
