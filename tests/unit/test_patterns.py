# tests/unit/test_patterns.py

from tfg.analytics.patterns import detect_patterns


def test_detect_patterns_encoding(sample_classifications):
    patterns = detect_patterns(sample_classifications)

    # Esperamos detectar problema de encoding en València vs Valencia
    assert "encoding_issue" in patterns


def test_detect_patterns_case(sample_classifications):
    patterns = detect_patterns(sample_classifications)

    assert "case_difference" in patterns