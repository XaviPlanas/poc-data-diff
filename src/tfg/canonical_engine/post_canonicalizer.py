"""
canonical_engine/post_canonicalizer.py

PostCanonicalizer: normaliza los DiffRow devueltos por data-diff
aplicando las transformaciones Python (capa POST) ANTES de que
lleguen al clasificador LLM.

Resuelve el problema de asimetría de dialecto:
    MySQL no puede hacer NORMALIZE(NFC) en SQL.
    PostgreSQL sí puede.
    → Ambas queries dejan la columna sin la transform NFC.
    → data-diff puede reportar diferencias que no son reales.
    → PostCanonicalizer aplica NFC en Python a AMBOS lados del DiffRow.
    → La diferencia desaparece antes del LLM.

Flujo completo (ver titanic_canonical.py):
    ┌─ data-diff ──────────────────────────────────────────────┐
    │  Compara vistas con transforms PRE (TRIM, LOWER, etc.)   │
    │  Devuelve DiffRows con cols POST sin normalizar           │
    └───────────────────────────────────────────────────────────┘
              │ DiffRow list
              ▼
    ┌─ PostCanonicalizer ──────────────────────────────────────┐
    │  Aplica to_python_for(pending) a row_a y row_b           │
    │  Detecta falsos positivos: diff_a == diff_b tras POST    │
    │  Los elimina de la lista antes del clasificador          │
    └───────────────────────────────────────────────────────────┘
              │ DiffRow list (filtrado)
              ▼
    ┌─ DiffClassifier (LLM) ───────────────────────────────────┐
    │  Solo recibe diferencias que sobreviven la canonización  │
    └───────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing      import Dict, Callable, List, Tuple

from .plan import CanonicalPlan

logger = logging.getLogger(__name__)


class PostCanonicalizer:
    """
    Aplica las transformaciones Python del plan POST a cada DiffRow.

    Parámetros:
        plan_a : plan del motor A (ej. MySQL)
        plan_b : plan del motor B (ej. PostgreSQL)

    Ambos planes deben tener los mismos nombres de columnas en
    post_callables() para que la normalización sea simétrica.
    En la práctica, cuando se usa peer_dialect en build_plan(),
    los callables son funcionalmente equivalentes aunque las
    expresiones SQL intermedias sean distintas.
    """

    def __init__(self, plan_a: CanonicalPlan, plan_b: CanonicalPlan):
        self._fns_a = plan_a.post_callables()
        self._fns_b = plan_b.post_callables()

        all_post = set(self._fns_a) | set(self._fns_b)
        if all_post:
            logger.info(
                "PostCanonicalizer configurado: columnas POST=%s",
                sorted(all_post),
            )
        else:
            logger.info(
                "PostCanonicalizer: sin columnas POST "
                "(todo resuelto en SQL)."
            )

    # ── API pública ───────────────────────────────────────────────

    def apply(self, diff_row) -> "DiffRow":
        """
        Aplica las transformaciones POST a un DiffRow.
        Devuelve un nuevo DiffRow con los valores normalizados.
        row_a y row_b se normalizan con sus callables respectivos.
        """
        new_a = self._normalize_side(diff_row.row_a, self._fns_a)
        new_b = self._normalize_side(diff_row.row_b, self._fns_b)
        return replace(diff_row, row_a=new_a, row_b=new_b)

    def apply_batch(
        self,
        diff_rows: List,
    ) -> Tuple[List, List]:
        """
        Normaliza una lista de DiffRow y separa los resultados en:
          - resolved : filas cuyas diferencias desaparecen tras POST
                       (eran falsos positivos de las transforms PRE
                        que data-diff no pudo comparar en SQL).
          - remaining: filas que siguen siendo distintas tras POST
                       (diferencias reales → pasan al clasificador).

        Devuelve (remaining, resolved) para que el llamador pueda
        registrar métricas de reducción de falsos positivos.
        """
        remaining = []
        resolved  = []

        for row in diff_rows:
            normalized = self.apply(row)

            if self._rows_equal(normalized):
                resolved.append(normalized)
            else:
                remaining.append(normalized)

        logger.info(
            "PostCanonicalizer.apply_batch: "
            "entrada=%d  eliminados=%d  restantes=%d",
            len(diff_rows), len(resolved), len(remaining),
            extra={
                "input":     len(diff_rows),
                "resolved":  len(resolved),
                "remaining": len(remaining),
            },
        )
        return remaining, resolved

    def report(self) -> str:
        """Resumen de las columnas que normaliza este PostCanonicalizer."""
        all_post = set(self._fns_a) | set(self._fns_b)
        if not all_post:
            return "PostCanonicalizer: sin columnas POST."

        lines = ["PostCanonicalizer — columnas normalizadas en Python:"]
        for col in sorted(all_post):
            fn_a = self._fns_a.get(col)
            fn_b = self._fns_b.get(col)
            lines.append(
                f"  {col:<26} "
                f"fn_a={fn_a.__name__ if fn_a else '—':<30} "
                f"fn_b={fn_b.__name__ if fn_b else '—'}"
            )
        return "\n".join(lines)

    # ── Helpers privados ─────────────────────────────────────────

    @staticmethod
    def _normalize_side(
        row_data: dict | None,
        fns:      Dict[str, Callable],
    ) -> dict | None:
        """
        Aplica los callables POST a un lado del DiffRow.
        Si row_data es None (INSERT/DELETE), lo devuelve tal cual.
        """
        if row_data is None:
            return None
        result = dict(row_data)
        for col, fn in fns.items():
            if col in result:
                raw           = result[col]
                result[col]   = fn(raw)
        return result

    @staticmethod
    def _rows_equal(normalized_row) -> bool:
        """
        True si row_a y row_b son iguales tras la normalización POST,
        es decir, la diferencia era un falso positivo de las transforms
        pendientes que data-diff no comparó en SQL.
        """
        if normalized_row.row_a is None or normalized_row.row_b is None:
            # INSERT o DELETE real — no es un falso positivo
            return False
        return normalized_row.row_a == normalized_row.row_b