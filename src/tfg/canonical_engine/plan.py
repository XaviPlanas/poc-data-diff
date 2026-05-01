"""
canonical_engine/plan.py

Dataclasses del plan de canonización con separación explícita PRE/POST.

  CanonicalColumn  → metadatos de una columna: expresión SQL, callable Python,
                     capa asignada (pre | split | post) y transforms pendientes.

  CanonicalPlan    → plan completo para una tabla en un dialecto concreto.
                     Expone:
                       pre_expressions()   → dict col→sql  para data-diff
                       post_callables()    → dict col→fn   para PostCanonicalizer
                       report()            → resumen legible para consola/memoria
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing      import Callable, Dict, List


# ── Capas posibles ────────────────────────────────────────────────

class Layer:
    PRE   = "pre"    # toda la canonización en SQL  (data-diff query)
    SPLIT = "split"  # parte en SQL, parte en Python posterior
    POST  = "post"   # toda la canonización en Python posterior


@dataclass
class CanonicalColumn:
    """
    Metadatos de canonización de una columna individual.

    Atributos:
        name             : nombre de la columna en la tabla.
        layer            : "pre" | "split" | "post"
        sql_expression   : expresión SQL para la query de data-diff.
                           Para capas "split" y "pre" contiene
                           las transformaciones SQL seguras.
                           Para "post" es simplemente el nombre de la
                           columna (passthrough, sin transformar).
        post_fn          : callable Python para PostCanonicalizer.
                           None si layer == "pre" (todo en SQL).
        pending_transforms: lista de transforms que van a POST.
                           Vacía si layer == "pre".
        information_loss : documentación de pérdida semántica.
    """
    name:               str
    layer:              str
    sql_expression:     str
    post_fn:            Callable | None
    pending_transforms: List[str]
    information_loss:   str


@dataclass
class CanonicalPlan:
    """
    Plan de canonización para una tabla en un dialecto concreto.

    Generado por CanonicalPipeline.build_plan().
    Se pasa a CanonicalSegment (para construir el TableSegment)
    y a PostCanonicalizer (para normalizar los DiffRow).
    """
    table_name:   str
    dialect_name: str
    columns:      Dict[str, CanonicalColumn]

    # ── Accesores por capa ────────────────────────────────────────

    def pre_expressions(self) -> Dict[str, str]:
        """
        Devuelve {col_name: sql_expression} para las columnas con
        transformación SQL (capas "pre" y "split").

        Este dict es la entrada de CanonicalSegment: se usa para
        construir la SELECT que data-diff ejecutará, evitando la
        creación de vistas persistentes.

        Cuando data-diff exponga un parámetro column_expressions en
        TableSegment, bastará con pasar este dict directamente.
        """
        result = {}

        for name, col in self.columns.items():
            if col.layer in (Layer.PRE, Layer.SPLIT):
                result[name] = col.sql_expression

        return result

    def post_callables(self) -> Dict[str, Callable]:
        """
        Devuelve {col_name: callable} para las columnas con
        transformación Python post-diff (capas "split" y "post").

        Este dict es la entrada de PostCanonicalizer.
        """
        return {
            name: col.post_fn
            for name, col in self.columns.items()
            if col.layer in (Layer.SPLIT, Layer.POST) and col.post_fn
        }

    def passthrough_columns(self) -> List[str]:
        """
        Columnas que entran sin transformar en la query de data-diff
        (capa "post").  PostCanonicalizer las normaliza completamente.
        """
        return [
            name for name, col in self.columns.items()
            if col.layer == Layer.POST
        ]

    # ── Informe ───────────────────────────────────────────────────

    def report(self) -> str:
        pre   = [c for c in self.columns.values() if c.layer == Layer.PRE]
        split = [c for c in self.columns.values() if c.layer == Layer.SPLIT]
        post  = [c for c in self.columns.values() if c.layer == Layer.POST]

        lines = [
            f"CanonicalPlan  tabla={self.table_name}  "
            f"dialecto={self.dialect_name}",
            f"{'─' * 64}",
            f"  PRE   ({len(pre):2d} cols) — canonización completa en SQL:",
        ]
        for c in pre:
            lines.append(f"    {c.name:<26} {c.sql_expression}")

        lines.append(
            f"  SPLIT ({len(split):2d} cols) — SQL parcial + Python post-diff:"
        )
        for c in split:
            pending = ", ".join(c.pending_transforms)
            lines.append(
                f"    {c.name:<26} SQL={c.sql_expression}"
            )
            lines.append(
                f"    {'':26} POST_fn={c.post_fn.__name__}  "
                f"pendiente=[{pending}]"
            )

        lines.append(
            f"  POST  ({len(post):2d} cols) — passthrough SQL, Python completo:"
        )
        for c in post:
            fn_name = c.post_fn.__name__ if c.post_fn else "—"
            lines.append(f"    {c.name:<26} post_fn={fn_name}")

        if any(c.information_loss for c in self.columns.values()):
            lines.append(f"{'─' * 64}")
            lines.append("  Pérdidas de información:")
            for c in self.columns.values():
                if c.information_loss:
                    lines.append(f"    {c.name}: {c.information_loss}")

        lines.append(f"{'─' * 64}")
        lines.append(
            f"  Total PRE={len(pre)}  SPLIT={len(split)}  POST={len(post)}"
        )
        return "\n".join(lines)
