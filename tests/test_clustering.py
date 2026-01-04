import unittest

from pathlib import Path

try:
    import numpy as np
    import pandas as pd

    from src.clustering.evaluate import objective_score
    from src.clustering.features import build_typology_features
    from src.clustering.models import prepare_feature_matrix, fit_kmeans
    from src.clustering.tuning import make_sqlite_storage_uri

    _HAS_DEPS = True
except ModuleNotFoundError:
    _HAS_DEPS = False


class TestClusteringPipelinePieces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _HAS_DEPS:
            raise unittest.SkipTest("Optional test deps missing (numpy/pandas/sklearn).")

    def test_make_sqlite_storage_uri(self):
        uri = make_sqlite_storage_uri(Path("output/optuna/optuna.db"))
        self.assertTrue(uri.startswith("sqlite:////"))

    def test_typology_features_shape(self):
        city_long = pd.DataFrame(
            {
                "geo_code": ["DE001C", "DE001C", "FR001C", "FR001C"],
                "year": [2020, 2021, 2020, 2021],
                "indicator_code": ["X", "X", "X", "X"],
                "value": [10.0, 12.0, 20.0, 21.0],
            }
        )
        # No country enrichment in this unit test
        wide = build_typology_features(
            city_long=city_long,
            country_enriched_long=None,
            year_min=2020,
            year_max=2021,
        )
        self.assertIn("geo_code", wide.columns)
        # Expect at least mean/last/slope/volatility columns for indicator X
        cols = [c for c in wide.columns if c != "geo_code"]
        self.assertTrue(any(c.startswith("X__") for c in cols))
        self.assertEqual(len(wide), 2)

    def test_prepare_and_kmeans(self):
        df = pd.DataFrame(
            {
                "geo_code": ["A", "B", "C", "D"],
                "f1": [0.0, 0.1, 10.0, 10.2],
                "f2": [0.0, 0.2, 9.9, 10.1],
            }
        )
        pm = prepare_feature_matrix(df, scaler="standard")
        res = fit_kmeans(pm.X_scaled, k=2, seed=42, n_init=10)
        self.assertEqual(len(res.labels), 4)

    def test_objective_score_constraints(self):
        from src.clustering.evaluate import ClusterMetrics

        m = ClusterMetrics(
            silhouette=0.5,
            davies_bouldin=1.0,
            calinski_harabasz=10.0,
            stability_ari=0.5,
            n_clusters=1,
            min_cluster_size=4,
            max_cluster_frac=1.0,
        )
        self.assertLess(objective_score(m), -1e6)


if __name__ == "__main__":
    unittest.main()


