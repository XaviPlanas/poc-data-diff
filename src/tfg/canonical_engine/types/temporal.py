from dataclasses import dataclass
from datetime    import datetime, timezone
from typing      import Callable
from .base       import CanonicalType

class TemporalPrecision:
    MICROSECOND = "microsecond"
    SECOND      = "second"
    MINUTE      = "minute"
    HOUR        = "hour"
    DAY         = "day"


_TRUNC = {
    TemporalPrecision.MICROSECOND: lambda dt: dt,
    TemporalPrecision.SECOND:      lambda dt: dt.replace(microsecond=0),
    TemporalPrecision.MINUTE:      lambda dt: dt.replace(second=0, microsecond=0),
    TemporalPrecision.HOUR:        lambda dt: dt.replace(minute=0, second=0, microsecond=0),
    TemporalPrecision.DAY:         lambda dt: dt.replace(
                                       hour=0, minute=0, second=0, microsecond=0),
}


@dataclass
class TimestampCanonical(CanonicalType):
    """
    Tipo temporal normalizado a UTC con precisión configurable.
    Resuelve diferencias por zona horaria y por precisión
    en subsegundos entre motores.
    """
    precision:  str  = TemporalPrecision.SECOND
    force_utc:  bool = True

    information_loss: str = (
        "Se pierde la información de zona horaria original "
        "al normalizar a UTC. Se trunca la precisión temporal "
        "al nivel especificado."
    )

    def to_sql(self, dialect) -> str:
        col  = self.column_name
        expr = col
        if self.force_utc:
            expr = dialect.to_utc(expr)
        expr = dialect.truncate_timestamp(expr, self.precision)
        return self.with_null_handling(expr, dialect)

    def to_python(self) -> Callable:
        force_utc = self.force_utc
        precision = self.precision
        nullable  = self.nullable
        trunc_fn  = _TRUNC.get(precision, lambda dt: dt)
        sentinel  = "1970-01-01T00:00:00+00:00" if force_utc else "1970-01-01"

        def transform(value):
            if value is None:
                return sentinel if nullable else None
            try:
                if isinstance(value, str):
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                elif isinstance(value, datetime):
                    dt = value
                else:
                    return value

                if force_utc:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)

                return trunc_fn(dt).isoformat()

            except (ValueError, AttributeError):
                return value

        transform.__name__ = f"TimestampCanonical(utc={force_utc},prec={precision})"
        return transform

    def validate(self, value) -> bool:
   
        return isinstance(value, (datetime, str))