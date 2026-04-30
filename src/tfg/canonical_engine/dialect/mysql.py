from .base import BaseDialect, UnsupportedTransformation

class MySQLDialect(BaseDialect):

    name = "mysql"
    
    @staticmethod
    def _clean(col: str) -> str:
        # Quita comillas externas si vienen ya citadas
        #if col.startswith(("'", '"', "`")) and col.endswith(("'", '"', "`")):
        if col.startswith('"') and col.endswith('"'):
            col = col[1:-1]
        return f"`{col}`"

    def round_numeric(self, col, precision, scale):
        # MySQL: DECIMAL en lugar de NUMERIC para comportamiento estándar
        col = self._clean(col)
        return f"ROUND(CAST({col} AS DECIMAL({scale},{precision})), {precision})"

    def cast_integer(self, col):
        col = self._clean(col)
        return f"CAST({col} AS SIGNED)"

    def ensure_encoding(self, col, encoding):
        # Conversión explícita de encoding, crítica en MySQL
        # donde el encoding puede variar por columna
        col = self._clean(col)
        return f"CONVERT({col} USING {encoding})"

    def trim(self, expr):
        return f"TRIM({expr})"

    def lowercase(self, expr):
        return f"LOWER({expr})"

    def normalize_unicode(self, expr, form):
        # MySQL no tiene NORMALIZE() nativo.
        # Se lanza UnsupportedTransformation para que el motor
        # aplique el fallback Python.
        raise UnsupportedTransformation("mysql", f"NORMALIZE({form})")

    def ascii_fold(self, expr):
        # MySQL tampoco tiene ascii_fold nativo.
        raise UnsupportedTransformation("mysql", "ascii_fold")

    def collapse_spaces(self, expr):
        # MySQL soporta REGEXP_REPLACE desde 8.0
        return f"REGEXP_REPLACE({expr}, '\\\\s+', ' ')"

    def remove_punct(self, expr):
        return f"REGEXP_REPLACE({expr}, '[^[:alnum:][:space:]]', '')"

    def to_utc(self, expr):
        return f"CONVERT_TZ({expr}, @@session.time_zone, '+00:00')"

    def truncate_timestamp(self, expr, precision):
        # MySQL no tiene DATE_TRUNC. Se simula con DATE_FORMAT.
        mysql_format = {
            "second": "%Y-%m-%d %H:%i:%s",
            "minute": "%Y-%m-%d %H:%i:00",
            "hour":   "%Y-%m-%d %H:00:00",
            "day":    "%Y-%m-%d 00:00:00",
        }
        if precision == "microsecond":
            return expr  # MySQL soporta microsegundos nativamente
        fmt = mysql_format.get(precision, "%Y-%m-%d %H:%i:%s")
        return f"STR_TO_DATE(DATE_FORMAT({expr}, '{fmt}'), '{fmt}')"

    def coalesce(self, expr, default):
        return f"COALESCE({expr}, {default})"

    def null_replacement(self, canonical_type):
        from ..types.numeric  import NumericCanonical
        from ..types.text     import TextCanonical
        from ..types.temporal import TimestampCanonical
        if isinstance(canonical_type, NumericCanonical):  return "0"
        if isinstance(canonical_type, TextCanonical):     return "''"
        if isinstance(canonical_type, TimestampCanonical):
            return "'1970-01-01 00:00:00'"
        return "NULL"
    
    def normalize_boolean(self, col: str) -> str:
    # MySQL representa booleanos como TINYINT(1).
    # IF convierte cualquier valor truthy a 1, falsy a 0,
    # que es el comportamiento esperado para TINYINT(1).
    # El CASE exterior cubre representaciones textuales
    # provenientes de imports CSV o de columnas VARCHAR
    # usadas como booleanos por convención.
        col = self._clean(col)
        return (
            f"CASE "
            f"  WHEN LOWER({col}) IN ('true','yes','on','1','t') THEN 1 "
            f"  WHEN LOWER({col}) IN ('false','no','off','0','f') THEN 0 "
            f"  ELSE IF({col}, 1, 0) "
            f"END"
        )