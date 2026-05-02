# tests/conftest.py

import pytest

@pytest.fixture
def sample_config():
    return {
        "columns": {
            "PassengerId": {"type": "integer"},
            "Name": {"type": "text"},
            "Age": {"type": "numeric"}
        }
    }