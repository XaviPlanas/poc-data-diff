# tests/test_schema_validation.py

import pytest

from tfg.canonical_engine.validator import validate_schema


def test_schema_no_empty_names():
    columns = ["id", "name", "age"]
    validate_schema(columns)  # no debería fallar


def test_schema_detects_empty_name():
    columns = ["id", "", "age"]

    with pytest.raises(ValueError):
        validate_schema(columns)


def test_schema_detects_duplicates():
    columns = ["id", "name", "name"]

    with pytest.raises(ValueError):
        validate_schema(columns)