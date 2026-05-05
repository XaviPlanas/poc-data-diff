"""
datadiff_classifier/report.py

DiffReport: estadísticas descriptivas sobre List[DiffClassification].
Adaptado al classifier.py y models.py reales del proyecto.

Sin LLM. Sin efectos secundarios. Completamente determinista.
Cada método devuelve un dict serializable a JSON.

Uso integrado con classifier.py:

    classifications = classifier.classify_row_by_row(diffrows)

    report = DiffReport(classifications)
    report.print_summary()                     # reemplaza report_statistics()
    report.print_details()                     # reemplaza report_details()

    # Para la memoria del TFG
# ──────────────────────────────────────────────────────
    narrator = ReportNarrator()
    narrative = narrator.full_narrative(report, pipeline_stats)
    ReportExporter.to_markdown(report, narrative, "output_report.md")
"""

from __future__ import annotations

import json
import statistics
from collections  import Counter, defaultdict
from dataclasses  import dataclass
from datetime     import datetime
from typing       import Dict, List, Optional

import logging
logger = logging.getLogger(__name__)

from tfg.datadiff_classifier.models import ( DiffClassification, DiffCategory, DiffAction, 
                    _FALSE_POSITIVE_VALUES, _NEEDS_REVIEW_VALUES, _CAT_ICON
                    )

# Constantes de presentación
_W     = 60    # ancho de línea para print
_LABEL = 46    # ancho etiqueta en tablas de consola

# ─────────────────────────────────────────────────────────────────
# DiffReport
# ─────────────────────────────────────────────────────────────────

@dataclass
class DiffReport:
    """
    Contenedor de estadísticas descriptivas para una lista de
    DiffClassification producidas por DiffClassifier.

    Construir una sola vez, consultar las veces necesarias:
        report = DiffReport(classifications)
        summ   = report.summary()
        cols   = report.by_column()
    """

    classifications: List[DiffClassification]

    # ── Métricas globales ─────────────────────────────────────────

    def summary(self) -> dict:
        """
        Resumen global: conteos, tasas y distribución de confianza.

        Métricas clave para la memoria del TFG:
          false_positive_rate  : (CANONIZABLE + EQUIVALENT_SEMANTIC) / total
          structural_diff_rate : DIFFERENT_STRUCTURAL / total
          review_rate          : (DIFFERENT_CONTEXTUAL + UNCERTAIN + ERROR) / total

        false_positive_rate es la métrica principal: mide cuántas de las
        diferencias que data-diff detectó eran en realidad el mismo dato
        representado de forma distinta.
        """
        total = len(self.classifications)
        if total == 0:
            return {"total": 0, "advertencia": "lista de clasificaciones vacía"}

        counts = Counter(c.categoria for c in self.classifications)
        conf   = [c.confianza for c in self.classifications]

        false_positives = sum(
            1 for c in self.classifications if c.is_false_positive()
        )
        needs_review = sum(
            1 for c in self.classifications if c.needs_review()
        )
        real_diffs = sum(
            1 for c in self.classifications if c.is_real_difference()
        )

        return {
            "total": total,
            "por_categoria": {
                cat.value: counts.get(cat, 0)
                for cat in DiffCategory
                if not cat.name.startswith("SEMANTICALLY")   # ocultar aliases
                and cat.name != "DIFFERENT_SEMANTICAL"
            },
            "por_accion": {
                action.value: sum(
                    1 for c in self.classifications if c.accion == action
                )
                for action in DiffAction
            },
            "tasas": {
                "false_positive_rate":  _pct(false_positives, total),
                "structural_diff_rate": _pct(real_diffs, total),
                "review_rate":          _pct(needs_review, total),
                "uncertain_rate":       _pct(counts.get(DiffCategory.UNCERTAIN, 0), total),
                "error_rate":           _pct(counts.get(DiffCategory.ERROR, 0), total),
            },
            "confianza": {
                "media":   _r(statistics.mean(conf)),
                "mediana": _r(statistics.median(conf)),
                "minima":  _r(min(conf)),
                "maxima":  _r(max(conf)),
                "stdev":   _r(statistics.stdev(conf)) if len(conf) > 1 else 0.0,
            },
        }

    # ── Análisis por columna ──────────────────────────────────────

    def by_column(self) -> dict:
        """
        Distribución de categorías por columna afectada.

        Responde: ¿qué columnas generan más falsos positivos?
        Las que tienen prioridad_canonizacion=True son candidatas
        directas a añadir una regla al motor de canonización.
        """
        col_counts:     Dict[str, Counter] = defaultdict(Counter)
        col_confidence: Dict[str, list]    = defaultdict(list)

        for c in self.classifications:
            for col in (c.columnas_afectadas or []):
                if col == "__ROW__":
                    continue
                col_counts[col][c.categoria] += 1
                col_confidence[col].append(c.confianza)

        out = {}
        for col, counts in col_counts.items():
            total_col = sum(counts.values())
            fp_count  = sum(
                counts.get(cat, 0)
                for cat in col_counts[col]
                if cat.value in _FALSE_POSITIVE_VALUES
            )
            out[col] = {
                "total_diffs":            total_col,
                "por_categoria":          {c.value: n for c, n in counts.items()},
                "false_positive_rate":    _pct(fp_count, total_col),
                "confianza_media":        _r(statistics.mean(col_confidence[col])),
                "prioridad_canonizacion": fp_count / total_col > 0.5,
            }

        return dict(
            sorted(out.items(), key=lambda x: x[1]["total_diffs"], reverse=True)
        )

    # ── Distribución de confianza ─────────────────────────────────

    def confidence_distribution(self, bins: int = 5) -> dict:
        """
        Histograma de confianza en `bins` intervalos.

        Distribución sana: mayoría de casos ≥ 0.85.
        Muchos casos en [0.70, 0.85) → few-shot insuficientes
        o schema_context demasiado genérico.
        """
        conf = [c.confianza for c in self.classifications]
        if not conf:
            return {}

        step = 1.0 / bins
        histogram = {}
        for i in range(bins):
            lo = round(i * step, 2)
            hi = round((i + 1) * step, 2)
            histogram[f"[{lo:.2f},{hi:.2f})"] = sum(
                1 for v in conf if lo <= v < hi
            )
        # Incluir 1.0 en el último intervalo
        last = list(histogram.keys())[-1]
        histogram[last] += sum(1 for v in conf if v == 1.0)

        return {
            "histograma":          histogram,
            "pct_alta_confianza":  _pct(sum(1 for v in conf if v >= 0.85), len(conf)),
            "pct_zona_gris":       _pct(sum(1 for v in conf if 0.70 <= v < 0.85), len(conf)),
            "pct_baja_confianza":  _pct(sum(1 for v in conf if v < 0.70), len(conf)),
        }

    # ── Reglas de canonización sugeridas ─────────────────────────

    def canonizable_rules(self) -> dict:
        """
        Agrega las normalizacion_sugerida de los casos CANONIZABLE.

        Cada entrada es una regla SQL/Python que debería existir en
        el motor de canonización pero no existe todavía. Las reglas
        con más apariciones son las más prioritarias.

        Retroalimentación directa al motor:
            rules = report.canonizable_rules()
            # Las top-3 son las reglas a añadir primero
        """
        rule_counts:   Counter         = Counter()
        rule_columns:  Dict[str, set]  = defaultdict(set)
        rule_examples: Dict[str, list] = defaultdict(list)

        for c in self.classifications:
            if (c.categoria == DiffCategory.CANONIZABLE
                    and c.normalizacion_sugerida):
                rule = c.normalizacion_sugerida.strip()
                rule_counts[rule] += 1
                for col in (c.columnas_afectadas or []):
                    rule_columns[rule].add(col)
                if len(rule_examples[rule]) < 3:
                    rule_examples[rule].append({
                        "key":   str(c.key),
                        "row_a": c.row_a,
                        "row_b": c.row_b,
                    })

        return {
            rule: {
                "apariciones": n,
                "columnas":    sorted(rule_columns[rule]),
                "ejemplos":    rule_examples[rule],
            }
            for rule, n in rule_counts.most_common()
        }

    # ── Cola de revisión humana ───────────────────────────────────

    def review_queue(self, max_items: int = 20) -> list:
        """
        Lista priorizada de casos para revisión humana.
        Orden: UNCERTAIN primero, luego DIFFERENT_CONTEXTUAL y ERROR,
        dentro de cada grupo por confianza ascendente.
        """
        priority = {
            DiffCategory.UNCERTAIN:            0,
            DiffCategory.ERROR:                1,
            DiffCategory.DIFFERENT_CONTEXTUAL: 2,
        }
        candidates = [
            c for c in self.classifications if c.needs_review()
        ]
        candidates.sort(key=lambda c: (
            priority.get(c.categoria, 9), c.confianza
        ))

        return [
            {
                "key":          str(c.key),
                "categoria":    c.categoria.value,
                "confianza":    c.confianza,
                "columnas":     c.columnas_afectadas,
                "explicacion":  c.explicacion,
                "row_a":        c.row_a,
                "row_b":        c.row_b,
            }
            for c in candidates[:max_items]
        ]

    # ── Métricas del pipeline PRE / POST / LLM ───────────────────

    @staticmethod
    def reduction_pipeline(
        n_raw:           int,
        n_pre:           int,
        n_post:          int,
        classifications: List[DiffClassification],
    ) -> dict:
        """
        Métricas de reducción de falsos positivos en las tres etapas.

        Uso desde titanic_canonical.py:
            pipeline = DiffReport.reduction_pipeline(
                n_raw           = len(diffs_raw),
                n_pre           = len(diffs_pre),
                n_post          = len(diff_rows),
                classifications = results,
            )
        """
        if n_raw == 0:
            return {"error": "n_raw = 0, sin datos baseline"}

        fp_llm  = sum(1 for c in classifications if c.is_false_positive())
        reales  = max(0, n_post - fp_llm)

        red_pre  = n_raw  - n_pre
        red_post = n_pre  - n_post
        red_llm  = fp_llm
        red_tot  = n_raw  - reales

        return {
            "etapas": {
                "sin_canonizacion": n_raw,
                "tras_PRE_sql":     n_pre,
                "tras_POST_python": n_post,
                "diferencias_reales": reales,
            },
            "reduccion": {
                "PRE_sql": {
                    "absoluta":   red_pre,
                    "porcentaje": _pct(red_pre, n_raw),
                },
                "POST_python": {
                    "absoluta":   red_post,
                    "porcentaje": _pct(red_post, n_raw),
                },
                "LLM_false_pos": {
                    "absoluta":   red_llm,
                    "porcentaje": _pct(red_llm, n_raw),
                },
                "total": {
                    "absoluta":   red_tot,
                    "porcentaje": _pct(red_tot, n_raw),
                },
            },
        }

    # ── Serialización ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "summary":               self.summary(),
            "by_column":             self.by_column(),
            "confidence_distribution": self.confidence_distribution(),
            "canonizable_rules":     self.canonizable_rules(),
            "review_queue":          self.review_queue(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False,
                          indent=indent, default=str)

    # ── Métodos de consola (reemplazan report_statistics / report_details) ──

    def print_summary(self) -> None:
        """
        Reemplaza classifier.report_statistics().
        Misma presentación en consola, datos desde DiffReport.
        """
        summ   = self.summary()
        total  = summ["total"]
        tasas  = summ["tasas"]
        conf   = summ["confianza"]

        print(f"\n{'=' * _W}")
        print("REPORTE DE CLASIFICACIÓN".center(_W))
        print(f"{'=' * _W}")
        print(f"{'Total procesado':<{_LABEL}} : {total:>4d}")

        print("\n--- Clasificación por categoría ---")
        for cat_str, count in summ["por_categoria"].items():
            try:
                cat  = DiffCategory(cat_str)
                icon = _CAT_ICON.get(cat, "  ")
            except ValueError:
                icon = "  "
            pct = count / total * 100 if total else 0
            print(f"  {icon} {cat_str:<{_LABEL - 4}} : {count:>4d}  ({pct:5.1f}%)")

        print("\n--- Clasificación por acción ---")
        for action_str, count in summ["por_accion"].items():
            print(f"  {action_str:<{_LABEL}} : {count:>4d}")

        print("\n--- Tasas de calidad ---")
        print(f"  {'Falsos positivos (canonizable + equivalente)':<{_LABEL}} : "
              f"{tasas['false_positive_rate']:.2%}")
        print(f"  {'Diferencias reales (structural)':<{_LABEL}} : "
              f"{tasas['structural_diff_rate']:.2%}")
        print(f"  {'Requieren revisión humana':<{_LABEL}} : "
              f"{tasas['review_rate']:.2%}")

        print("\n--- Confianza del clasificador ---")
        print(f"  {'Media':<{_LABEL}} : {conf['media']:.3f}")
        print(f"  {'Mediana':<{_LABEL}} : {conf['mediana']:.3f}")
        print(f"  {'Desviación estándar':<{_LABEL}} : {conf['stdev']:.3f}")
        print(f"  {'Mínima / Máxima':<{_LABEL}} : "
              f"{conf['minima']:.3f} / {conf['maxima']:.3f}")
        print(f"{'=' * _W}")

    def print_details(
        self,
        false_positives: bool = True,
        review:          bool = True,
    ) -> None:
        """
        Reemplaza classifier.report_details().
        Parámetros para mostrar solo el subconjunto deseado.
        """
        def _print_one(c: DiffClassification) -> None:
            icon = _CAT_ICON.get(c.categoria, "  ")
            print(f"\n  {'─' * (_W - 4)}")
            print(f"  Key      : {c.key}")
            print(f"  Acción   : {c.accion.name if c.accion else '—'}")
            print(f"  {icon} Categoría: {c.categoria.value}")
            print(f"  Confianza: {c.confianza:.2f}")
            print(f"  Columnas : {c.columnas_afectadas}")
            print(f"  Explicación: {c.explicacion}")
            if c.normalizacion_sugerida:
                print(f"  Norm. sug.: {c.normalizacion_sugerida}")
            if c.row_a:
                print(f"  Row A : {json.dumps(c.row_a, ensure_ascii=False)}")
            if c.row_b:
                print(f"  Row B : {json.dumps(c.row_b, ensure_ascii=False)}")

        print(f"\n{'=' * _W}")
        print("REPORTE DE DETALLES".center(_W))
        print(f"{'=' * _W}")

        if false_positives:
            fp = [c for c in self.classifications if c.is_false_positive()]
            print(f"\n🟢 Falsos positivos ({len(fp)}):")
            for c in fp:
                _print_one(c)

        if review:
            rv = [c for c in self.classifications if c.needs_review()]
            print(f"\n🟠 Requieren revisión ({len(rv)}):")
            for c in rv:
                _print_one(c)

        real = [c for c in self.classifications if c.is_real_difference()]
        print(f"\n🔴 Diferencias reales ({len(real)}):")
        for c in real:
            _print_one(c)

        print(f"\n{'=' * _W}")

    def print_column_analysis(self) -> None:
        """Análisis por columna en consola."""
        by_col = self.by_column()
        print(f"\n{'=' * _W}")
        print("ANÁLISIS POR COLUMNA".center(_W))
        print(f"{'=' * _W}")
        print(f"  {'Columna':<20} {'Diffs':>6} {'FP%':>7} {'Conf.':>6}  Prioridad")
        print(f"  {'─' * 55}")
        for col, data in by_col.items():
            prio = "✓ Canonizar" if data["prioridad_canonizacion"] else ""
            print(f"  {col:<20} {data['total_diffs']:>6} "
                  f"{data['false_positive_rate']:>7.1%} "
                  f"{data['confianza_media']:>6.3f}  {prio}")
        print(f"{'=' * _W}")

    def print_canonizable_rules(self) -> None:
        """Reglas de canonización sugeridas en consola."""
        rules = self.canonizable_rules()
        if not rules:
            print("Sin reglas de canonización sugeridas.")
            return
        print(f"\n{'=' * _W}")
        print("REGLAS DE CANONIZACIÓN SUGERIDAS".center(_W))
        print(f"{'=' * _W}")
        for rule, info in rules.items():
            cols = ", ".join(info["columnas"])
            print(f"  [{info['apariciones']:>3}x] {rule}")
            print(f"         columnas: {cols}")
        print(f"{'=' * _W}")


# ─────────────────────────────────────────────────────────────────
# ReportNarrator
# ─────────────────────────────────────────────────────────────────

class ReportNarrator:
    """
    Genera narrativa e interpretación sobre un DiffReport
    mediante UNA SOLA llamada a claude-haiku-4-5.

    El LLM recibe únicamente el resumen estadístico (no los datos
    individuales), lo que garantiza respuestas coherentes con los
    números y evita alucinaciones sobre datos no vistos.
    """

    _SYSTEM = """\
Eres un experto en integración y calidad de datos. Recibes estadísticas \
agregadas de una comparación entre dos fuentes SQL (MySQL y PostgreSQL) \
y produces análisis concisos y recomendaciones accionables.

Reglas:
- Usa solo los datos del resumen proporcionado. No inventes cifras.
- Sé directo: primero diagnóstico, luego acción concreta.
- Responde en español.\
"""

    def __init__(
        self,
        model:      str = "claude-haiku-4-5",
        max_tokens: int = 700,
        api_key:    str = None,
    ):
        import os
        import anthropic
        self.model      = model
        self.max_tokens = max_tokens
        self.client     = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def full_narrative(
        self,
        report:         DiffReport,
        pipeline_stats: dict = None,
    ) -> dict:
        """
        Genera los tres bloques de narrativa en una sola llamada LLM.
        Devuelve {"diagnose": str, "recommend": str, "executive_summary": str}.
        """
        context = report.to_dict()
        if pipeline_stats:
            context["pipeline"] = pipeline_stats

        prompt = f"""\
Analiza estas estadísticas de comparación de datos y produce los tres \
bloques siguientes separados por las etiquetas exactas indicadas.

{json.dumps(context, ensure_ascii=False, indent=2, default=str)}

## DIAGNÓSTICO
(2-3 frases: qué tipo de problema domina, columnas más críticas, \
señales del clasificador)

## RECOMENDACIONES
(lista priorizada: ALTA/MEDIA/BAJA — Columna — Regla SQL/Python — impacto estimado)

## RESUMEN EJECUTIVO
(máximo 150 palabras, apto para la memoria de un TFG: qué se detectó, \
cómo el pipeline redujo los falsos positivos en cada etapa, conclusión \
sobre la efectividad del sistema)
"""
        raw = self._call(prompt)
        return self._split_narrative(raw)

    def _call(self, prompt: str) -> str:
        response = self.client.messages.create(
            model      = self.model,
            max_tokens = self.max_tokens,
            system     = self._SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    @staticmethod
    def _split_narrative(raw: str) -> dict:
        """Separa la respuesta en los tres bloques por etiquetas."""
        import re
        blocks = {"diagnose": "", "recommend": "", "executive_summary": ""}
        parts  = re.split(r"##\s*(DIAGNÓSTICO|RECOMENDACIONES|RESUMEN EJECUTIVO)",
                          raw, flags=re.IGNORECASE)
        label_map = {
            "diagnóstico":      "diagnose",
            "recomendaciones":  "recommend",
            "resumen ejecutivo":"executive_summary",
        }
        for i in range(1, len(parts) - 1, 2):
            key  = label_map.get(parts[i].lower().strip())
            text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if key:
                blocks[key] = text
        return blocks


# ─────────────────────────────────────────────────────────────────
# ReportExporter
# ─────────────────────────────────────────────────────────────────

class ReportExporter:
    """Exporta DiffReport + narrativa a JSON y Markdown."""

    @staticmethod
    def to_json(
        report:    DiffReport,
        narrative: dict = None,
        path:      str  = "output_report.json",
    ) -> str:
        data = {
            "generated_at": datetime.now().isoformat(),
            "statistics":   report.to_dict(),
        }
        if narrative:
            data["narrative"] = narrative
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Reporte JSON guardado en %s", path)
        return path

    @staticmethod
    def to_markdown(
        report:    DiffReport,
        narrative: dict = None,
        path:      str  = "output_report.md",
    ) -> str:
        summ     = report.summary()
        by_col   = report.by_column()
        rules    = report.canonizable_rules()
        conf_d   = report.confidence_distribution()
        total    = summ["total"]

        lines = [
            "# Informe de Comparación de Datos",
            f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        ]

        # Resumen ejecutivo
        if narrative and narrative.get("executive_summary"):
            lines += [
                "## Resumen ejecutivo\n",
                narrative["executive_summary"], "\n",
            ]

        # Métricas globales
        t = summ["tasas"]
        c = summ["confianza"]
        lines += [
            "## Métricas globales\n",
            "| Métrica | Valor |",
            "|---------|-------|",
            f"| Total diferencias analizadas | **{total}** |",
            f"| Tasa de falsos positivos | **{t['false_positive_rate']:.1%}** |",
            f"| Diferencias reales (structural) | **{t['structural_diff_rate']:.1%}** |",
            f"| Casos para revisión humana | **{t['review_rate']:.1%}** |",
            f"| Confianza media | **{c['media']:.3f}** ± {c['stdev']:.3f} |",
            "",
        ]

        # Por categoría
        lines += [
            "## Distribución por categoría\n",
            "| Categoría | N | % |",
            "|-----------|---|---|",
        ]
        for cat_str, n in summ["por_categoria"].items():
            pct = n / total * 100 if total else 0
            try:
                icon = _CAT_ICON.get(DiffCategory(cat_str), "")
            except ValueError:
                icon = ""
            lines.append(f"| {icon} {cat_str} | {n} | {pct:.1f}% |")
        lines.append("")

        # Por columna
        lines += [
            "## Análisis por columna\n",
            "| Columna | Diffs | FP % | Conf. media | Prioridad |",
            "|---------|-------|------|-------------|-----------|",
        ]
        for col, data in by_col.items():
            prio = "✓ Canonizar" if data["prioridad_canonizacion"] else "—"
            lines.append(
                f"| {col} | {data['total_diffs']} "
                f"| {data['false_positive_rate']:.1%} "
                f"| {data['confianza_media']:.3f} | {prio} |"
            )
        lines.append("")

        # Reglas
        if rules:
            lines += ["## Reglas de canonización recomendadas\n"]
            for rule, info in list(rules.items())[:10]:
                lines.append(
                    f"- **`{rule}`** — {info['apariciones']}× "
                    f"— columnas: {', '.join(info['columnas'])}"
                )
            lines.append("")

        # Confianza
        lines += [
            "## Distribución de confianza\n",
            f"- Alta (≥ 0.85): **{conf_d.get('pct_alta_confianza', 0):.1%}**",
            f"- Zona gris [0.70, 0.85): **{conf_d.get('pct_zona_gris', 0):.1%}**",
            f"- Baja (< 0.70): **{conf_d.get('pct_baja_confianza', 0):.1%}**\n",
        ]

        # Diagnóstico y recomendaciones
        if narrative:
            if narrative.get("diagnose"):
                lines += ["## Diagnóstico\n", narrative["diagnose"], ""]
            if narrative.get("recommend"):
                lines += ["## Recomendaciones\n", narrative["recommend"], ""]

        content = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Reporte Markdown guardado en %s", path)
        return path


# ─────────────────────────────────────────────────────────────────
# Helpers privados
# ─────────────────────────────────────────────────────────────────

def _pct(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0

def _r(v: float) -> float:
    return round(v, 3)
