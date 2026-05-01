"""
canonical_engine/segment.py

CanonicalSegment: construye el TableSegment de data-diff incorporando
las expresiones SQL del plan PRE directamente, sin vistas persistentes.

Estrategia de implementación (data-diff 0.11.2):
────────────────────────────────────────────────
data-diff 0.11.2 no expone un parámetro `column_expressions` en
TableSegment.  Para inyectar transformaciones SQL sin crear vistas
permanentes usamos una vista de sesión:

  1. Conectamos al motor con SQLAlchemy (conexión propia, no la de data-diff).
  2. Creamos una vista usando las expresiones PRE del plan.
  3. Construimos un TableSegment de data-diff apuntando a esa vista.
  4. Al salir del contexto, eliminamos la vista.

La arquitectura está diseñada para migrar a expresiones inline cuando
data-diff las soporte: solo cambia el cuerpo de _build_segment();
el resto del pipeline (plan, PostCanonicalizer, titanic_canonical) no
necesita ningún cambio.

Punto de migración documentado:
    pre_expressions()  →  pasar directamente a TableSegment(column_expressions=...)

Uso:
    with CanonicalSegment(plan, MYSQL_URI_DDIFF, "titanic", "PassengerId") as seg:
        diffs = list(diff_tables(seg.table_segment, other_seg.table_segment))
"""

from __future__ import annotations

import logging
from typing import List, Tuple
from data_diff import connect_to_table
from sqlalchemy import create_engine, text

from .plan import CanonicalPlan

logger = logging.getLogger(__name__)


class CanonicalSegment:
    """
    Context manager que expone un TableSegment de data-diff con las
    transformaciones PRE del CanonicalPlan incorporadas como vista
    de sesión efímera.

    Parámetros:
        plan        : CanonicalPlan generado por CanonicalPipeline.
        db_uri      : URI de data-diff (sin driver, ej. "mysql://u:p@h/db").
        table       : nombre de la tabla original.
        key_column  : columna PK para data-diff.
        extra_cols  : columnas adicionales a comparar (por defecto: todas).
    """

    def __init__(
        self,
        plan:       CanonicalPlan,
        db_uri:     str,
        table:      str,
        key_column: str,
        extra_cols: List[str] | None = None,
    ):
        self._plan       = plan
        self._db_uri     = db_uri
        self._table      = table
        self._key_col    = key_column
        self._extra_cols = self._canonize_column(extra_cols) if extra_cols else None
        self._view_name  = f"{table}_can_{plan.dialect_name}"
        self._sa_engine  = None
        #self.table_segment = None   # data-diff TableSegment, disponible en __enter:
        self._sa_engine = create_engine(
            # SQLAlchemy URI: necesita el driver completo
            self._uri_for_sqlalchemy(),
        )
        self._create_view()
        self.table_segment = self._build_segment()

    # ── Context manager ───────────────────────────────────────────

    # def __enter__(self) -> "CanonicalSegment":
    #     self._sa_engine = create_engine(
    #         # SQLAlchemy URI: necesita el driver completo
    #         self._uri_for_sqlalchemy(),
    #     )
    #     self._create_view()
    #     self.table_segment = self._build_segment()
    #     return self

    # def __exit__(self, *_):
    #     logger.debug("Saliendo de CanonicalSegment, limpiando recursos...")
    #     self._drop_view()
    #     if self._sa_engine:
    #         self._sa_engine.dispose()

    # ── API auxiliar ──────────────────────────────────────────────

    def pre_expressions(self) -> dict[str, str]:
        """
        Devuelve {col: sql_expr} de las columnas PRE y SPLIT.
        Punto de migración: cuando data-diff soporte column_expressions,
        pasar este dict directamente a TableSegment sin crear vistas.
        """
        return self._plan.pre_expressions()

    def post_callables(self) -> dict:
        """Devuelve {col: callable} para PostCanonicalizer."""
        return self._plan.post_callables()

    # ── Construcción interna de la vista ─────────────────────────

    def _canonize_column(self, column_name) :
        if isinstance(column_name, list):
            return [self._canonize_column(c) for c in column_name]
        
        return column_name.strip('"').strip('`').strip('\'')
        # if self._sa_engine.name == "mysql":
        #     cn =  column_name.strip('"')
        #     # if not (cn.startswith("`") and cn.endswith("`")):
        #     #    return f"`{cn}`"
        #     return cn
        # return column_name
    
    
    def _create_view(self) -> None:
        """
        Crea la vista canónica en el motor usando las expresiones PRE.
        Las columnas POST se incluyen sin transformar (passthrough).
        """
        select_parts = self._build_select_parts()
        view_sql = (
            f"CREATE OR REPLACE VIEW {self._view_name} AS\n"
            f"SELECT\n"
            f"    {',\n    '.join(select_parts)}\n"
            f"FROM {self._table}"
        )

        logger.info(
            "Creando vista canónica efímera: %s  dialect=%s",
            self._view_name, self._plan.dialect_name,
        )
        logger.debug("Vista SQL:\n%s", view_sql)

        with self._sa_engine.connect() as conn:
            conn.execute(text(f"DROP VIEW IF EXISTS {self._view_name}"))
            conn.execute(text(view_sql))
            conn.commit()

    def _drop_view(self) -> None:
        """Elimina la vista al salir del contexto."""
        if self._sa_engine is None:
            return
        try:
            with self._sa_engine.connect() as conn:
                conn.execute(text(f"DROP VIEW IF EXISTS {self._view_name}"))
                conn.commit()
            logger.info("Vista canónica eliminada: %s", self._view_name)
        except Exception as exc:
            logger.warning("No se pudo eliminar la vista %s: %s",
                           self._view_name, exc)

    def _build_select_parts(self) -> List[str]:
        """
        Construye la lista de expresiones SELECT:
          - Columnas PRE/SPLIT: expresión SQL canónica (ya incluye "AS col")
          - Columnas POST:      nombre de columna sin transformar
          - Columna PK:        siempre sin transformar
        """
        parts = []
        seen  = set()
        
        #pk = f"\"{self._canonize_column(self._key_col)}\""
        pk = self._canonize_column(self._key_col)
        
        if self._sa_engine.name == "postgresql":
            parts.append(f'"{pk}"')
        else:
            parts.append(pk)

        seen.add(pk)

        for col_name, col in self._plan.columns.items():
            bare = self._canonize_column(col_name)
            if bare in seen:
                continue
            seen.add(bare)
            parts.append(col.sql_expression)   # ya tiene "AS col_name"

        return parts

    def _build_segment(self):
        """
        Construye el TableSegment de data-diff apuntando a la vista.
        Punto de migración: sustituir por TableSegment con
        column_expressions=self.pre_expressions() cuando esté disponible.
        """

        all_cols = self._extra_cols or self._canonize_column(list(self._plan.columns.keys()))

        segment = connect_to_table(
            self._db_uri,
            self._view_name,
            self._canonize_column(self._key_col),
            extra_columns=self._canonize_column(all_cols),
        )

        logger.debug(
            "TableSegment construido: vista=%s  cols=%d",
            self._view_name, len(all_cols),
        )
        return segment

    # ── Helpers de URI ────────────────────────────────────────────

    def _uri_for_sqlalchemy(self) -> str:
        """
        Convierte la URI de data-diff (sin driver) a URI SQLAlchemy
        añadiendo el driver por defecto para cada dialecto.

        data-diff usa: "mysql://u:p@h/db"
        SQLAlchemy usa: "mysql+mysqlconnector://u:p@h/db"
        """
        if self._db_uri.startswith("mysql://"):
            return self._db_uri.replace(
                "mysql://", "mysql+mysqlconnector://", 1
            )
        if self._db_uri.startswith("postgresql://"):
            return self._db_uri.replace(
                "postgresql://", "postgresql+psycopg2://", 1
            )
        # Para otros dialectos asumir que la URI ya incluye el driver
        return self._db_uri
