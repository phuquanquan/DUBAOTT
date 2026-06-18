from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from xsmb_pipeline.database import create_schema, get_connection, insert_features_daily, insert_results
from xsmb_pipeline.feature_generators import compute_all_features
from xsmb_pipeline.models.xgboost_ranker import (
    build_xgboost_feature_artifact,
    evaluate_xgboost_ranking_backtest,
    export_xgboost_model_payload,
    save_xgboost_feature_artifact,
    train_xgboost_ranking_model,
    train_xgboost_with_selected_features,
)
from xsmb_pipeline.tests.test_database import make_sample_results


class XGBoostRankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = make_sample_results()
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "xsmb.duckdb"
        connection = get_connection(str(self.db_path))
        create_schema(connection)
        insert_results(connection, self.results)
        for result in self.results:
            rows = compute_all_features(
                loto_history=[(item.date, item.special[-2:]) for item in self.results if item.date <= result.date],
                digit_history=[
                    (item.date, "special", 0, position_index, digit)
                    for item in self.results
                    if item.date <= result.date
                    for position_index, digit in enumerate(item.special)
                ],
                as_of_date=result.date,
            )
            insert_features_daily(connection, rows)
        connection.close()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_train_xgboost_ranking_model_builds_predictions(self) -> None:
        model = train_xgboost_ranking_model(self.results, db_path=str(self.db_path), target_name="loto2", top_k=3, min_train_size=2)
        self.assertEqual(model.target, "loto2")
        self.assertEqual(len(model.predict()), 3)
        self.assertTrue(model.feature_importance)
        self.assertTrue(model.selected_features)
        self.assertTrue(model.shap_importance)
        self.assertTrue(model.mutual_info)
        self.assertTrue(model.permutation_importance)
        self.assertTrue(model.rfe_ranking)
        payload = export_xgboost_model_payload(model)
        artifact = build_xgboost_feature_artifact(model)
        self.assertIn("shap_importance", payload)
        self.assertIn("mutual_info", payload)
        self.assertIn("permutation_importance", payload)
        self.assertIn("rfe_ranking", payload)
        self.assertIn("feature_artifacts", artifact)
        self.assertIn("shap", artifact["feature_artifacts"])
        self.assertIn("mutual_info", artifact["feature_artifacts"])
        self.assertIn("permutation_importance", artifact["feature_artifacts"])
        self.assertIn("rfe", artifact["feature_artifacts"])
        self.assertGreaterEqual(len(payload["shap_importance"]), len(model.feature_names))
        self.assertGreaterEqual(len(payload["mutual_info"]), len(model.feature_names))
        self.assertGreaterEqual(len(payload["permutation_importance"]), len(model.feature_names))
        self.assertGreaterEqual(len(payload["rfe_ranking"]), len(model.feature_names))

    def test_train_xgboost_with_selected_features_returns_selected_payload(self) -> None:
        model = train_xgboost_with_selected_features(self.results, db_path=str(self.db_path), target_name="loto2", top_k=3, min_train_size=2, top_k_features=5)
        payload = export_xgboost_model_payload(model)
        artifact = build_xgboost_feature_artifact(model)
        self.assertEqual(model.target, "loto2")
        self.assertLessEqual(len(model.feature_names), 5)
        self.assertEqual(payload["model"], "xgboost-ranking")
        self.assertIn("selected_features", payload)
        self.assertIn("feature_importance", payload)
        self.assertIn("shap_importance", payload)
        self.assertIn("mutual_info", payload)
        self.assertIn("permutation_importance", payload)
        self.assertIn("rfe_ranking", payload)
        self.assertIn("feature_artifacts", artifact)

    def test_evaluate_xgboost_ranking_backtest_returns_metrics(self) -> None:
        metrics = evaluate_xgboost_ranking_backtest(self.results, db_path=str(self.db_path), target_name="loto2", top_k=3, min_train_size=2)
        self.assertEqual(metrics.target, "loto2")
        self.assertEqual(metrics.model, "xgboost-ranking")
        self.assertTrue(metrics.feature_importance)
        self.assertTrue(metrics.selected_features)
        self.assertTrue(metrics.shap_importance)
        self.assertTrue(metrics.mutual_info)
        self.assertTrue(metrics.permutation_importance)
        self.assertTrue(metrics.rfe_ranking)
        self.assertEqual(len(metrics.shap_importance), len(metrics.feature_importance))
        self.assertEqual(len(metrics.mutual_info), len(metrics.feature_importance))
        self.assertEqual(len(metrics.permutation_importance), len(metrics.feature_importance))
        self.assertEqual(len(metrics.rfe_ranking), len(metrics.feature_importance))
        temp_output = self.db_path.parent / "artifact.json"
        artifact = save_xgboost_feature_artifact(train_xgboost_ranking_model(self.results, db_path=str(self.db_path), target_name="loto2", top_k=3, min_train_size=2), output_path=str(temp_output))
        self.assertTrue(temp_output.exists())
        self.assertIn("feature_artifacts", artifact)


if __name__ == "__main__":
    unittest.main()
