# tests/test_normalization.py

import pytest

from tfg.canonical_engine.utils import normalize_sql_name


@pytest.mark.parametrize("input_name,expected", [
    ("PassengerId", "passenger_id"),
    ("Cabin Number", "cabin_number"),
    ("Edad (años)", "edad_anos"),
])
def test_normalize_sql_name_basic(input_name, expected):
    assert normalize_sql_name(input_name) == expected


def test_normalize_sql_name_not_empty():
    with pytest.raises(ValueError):
        normalize_sql_name("!!!")