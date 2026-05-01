# tests/test_dialect_mysql.py

from tfg.canonical_engine.dialect.mysql import MySQLDialect


def test_round_numeric_mysql():
    d = MySQLDialect()
    sql = d.round_numeric(price", precision=10, scale=2)

    assert "DECIMAL(10,2)" in sql
    assert "ROUND" in sql


def test_cast_integer_mysql():
    d = MySQLDialect()
    sql = d.cast_integer("col")

    assert sql == "CAST(`col` AS SIGNED)"


def test_normalize_boolean_mysql():
    d = MySQLDialect()
    sql = d.normalize_boolean("flag")

    assert "CASE" in sql
    assert "LOWER" in sql