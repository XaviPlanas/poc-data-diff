from tfg.analytics.metrics import compute_metrics


def test_compute_metrics_basic(sample_classifications):
    metrics = compute_metrics(sample_classifications)

    assert metrics["total"] == 3

    # categorías
    assert metrics["categorias"]["conteo"]["equivalent"] == 1
    assert metrics["categorias"]["conteo"]["different"] == 1
    assert metrics["categorias"]["conteo"]["uncertain"] == 1

    # confianza
    assert metrics["confianza"]["metricas"]["media"] is not None

    # incertidumbre
    assert metrics["incertidumbre"]["total"] == 1