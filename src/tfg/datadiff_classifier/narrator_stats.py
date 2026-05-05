"""
datadiff_classifier/exporter.py

ReportExporter: serializa DiffReport + narrativa a JSON y Markdown.
"""

from __future__ import annotations

import json
from datetime import datetime
from .report   import DiffReport


class ReportExporter:

    @staticmethod
    def to_json(
        report:    DiffReport,
        narrative: dict = None,
        path:      str  = "output_report.json",
    ) -> str:
        """
        Exporta el reporte completo a JSON.
        narrative: {"diagnose": str, "recommend": str, "executive": str}
        """
        data = {
            "generated_at": datetime.now().isoformat(),
            "statistics":   report.to_dict(),
        }
        if narrative:
            data["narrative"] = narrative

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return path

    @staticmethod
    def to_markdown(
        report:    DiffReport,
        narrative: dict = None,
        path:      str  = "output_report.md",
    ) -> str:
        """Exporta el reporte a Markdown para la memoria del TFG."""
        summ      = report.summary()
        by_col    = report.by_column()
        rules     = report.canonizable_rules()
        conf_dist = report.confidence_distribution()

        lines = [
            "# Informe de Comparación de Datos",
            f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        ]

        # ── Resumen ejecutivo (si hay narrativa) ──────────────────
        if narrative and narrative.get("executive_summary"):
            lines += [
                "## Resumen ejecutivo\n",
                narrative["executive_summary"], "\n",
            ]

        # ── Métricas globales ─────────────────────────────────────
        lines += [
            "## Métricas globales\n",
            f"| Métrica | Valor |",
            f"|---------|-------|",
            f"| Total diferencias analizadas | **{summ['total']}** |",
            f"| Tasa de falsos positivos | **{summ['tasas']['false_positive_rate']:.1%}** |",
            f"| Diferencias reales (estructurales) | **{summ['tasas']['structural_diff_rate']:.1%}** |",
            f"| Casos para revisión humana | **{summ['tasas']['review_rate']:.1%}** |",
            f"| Confianza media del clasificador | **{summ['confianza']['media']:.3f}** |",
            "",
        ]

        # ── Por categoría ─────────────────────────────────────────
        lines += ["## Distribución por categoría\n",
                  "| Categoría | N | % |",
                  "|-----------|---|---|"]
        for cat, n in summ["por_categoria"].items():
            pct = n / summ["total"] * 100 if summ["total"] else 0
            lines.append(f"| {cat} | {n} | {pct:.1f}% |")
        lines.append("")

        # ── Por columna ───────────────────────────────────────────
        lines += ["## Análisis por columna\n",
                  "| Columna | Diffs | FP rate | ¿Canonizar? |",
                  "|---------|-------|---------|-------------|"]
        for col, data in by_col.items():
            prio = "✓ Prioritario" if data["prioridad_canonizacion"] else "—"
            lines.append(
                f"| {col} | {data['total_diffs']} "
                f"| {data['false_positive_rate']:.1%} | {prio} |"
            )
        lines.append("")

        # ── Reglas de canonización ────────────────────────────────
        if rules:
            lines += ["## Reglas de canonización recomendadas\n"]
            for rule, info in list(rules.items())[:10]:
                cols = ", ".join(info["columnas"])
                lines.append(
                    f"- **`{rule}`** — {info['apariciones']} casos "
                    f"— columnas: {cols}"
                )
            lines.append("")

        # ── Distribución de confianza ─────────────────────────────
        lines += [
            "## Distribución de confianza del clasificador\n",
            f"- Casos con confianza alta (≥ 0.85): "
            f"**{conf_dist['pct_alta_confianza']:.1%}**",
            f"- Casos en zona gris [0.70, 0.85): "
            f"**{conf_dist['pct_zona_gris']:.1%}**\n",
        ]

        # ── Diagnóstico y recomendaciones (narrativa LLM) ─────────
        if narrative:
            if narrative.get("diagnose"):
                lines += ["## Diagnóstico\n", narrative["diagnose"], ""]
            if narrative.get("recommend"):
                lines += ["## Recomendaciones\n", narrative["recommend"], ""]

        content = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path