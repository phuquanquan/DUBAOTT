import unittest

from xsmb_pipeline.evaluate import (
    benchmark_loto2_data_windows,
    blend_model_rankings,
    compare_loto2_input_models,
    compare_loto2_models,
    compare_meta_models,
    compare_provider_models,
    compare_specialized_models,
    evaluate_named_ranking_backtest,
    evaluate_tuned_weighted_backtest,
    layer1_ensemble_predictions,
    loto2_data_windows,
    loto2_input_configurations,
    meta_stack_predictions,
    rolling_loto2_walkforward_benchmark,
)
from xsmb_pipeline.database import create_schema, get_connection, insert_features_daily, insert_results
from xsmb_pipeline.feature_generators import compute_all_features
from xsmb_pipeline.models.sklearn_ranker import (
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    SKLEARN_AVAILABLE,
    available_model_names,
    benchmarkable_model_names,
    fit_named_sklearn_ranking_model,
)
from xsmb_pipeline.models.xgboost_ranker import (
    XGBOOST_AVAILABLE,
    export_xgboost_model_payload,
    train_xgboost_ranking_model,
)
from xsmb_pipeline.models.neural_ranker import (
    build_gru_ranker,
    build_lstm_ranker,
    build_transformer_ranker,
    export_neural_ranker_payload,
)
from xsmb_pipeline.evaluate import build_full_dashboard_payload, provider_model_predictions
from xsmb_pipeline.models import TORCH_AVAILABLE
from xsmb_pipeline.schema import LotteryResult


SKLEARN_ONLY = unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")


def build_result(day: int, special: str, first: str, second: list[str], third: list[str], fourth: list[str], fifth: list[str], sixth: list[str], seventh: list[str]) -> LotteryResult:
    return LotteryResult(
        date=f"{day:02d}/01/2026",
        region="xsmb",
        special=special,
        first=[first],
        second=second,
        third=third,
        fourth=fourth,
        fifth=fifth,
        sixth=sixth,
        seventh=seventh,
    )


def sample_results() -> list[LotteryResult]:
    return [
        build_result(1, "12345", "54321", ["11111", "22222"], ["33333", "44444", "55555", "66666", "77777", "88888"], ["1234", "2345", "3456", "4567"], ["1000", "2001", "3002", "4003", "5004", "6005"], ["101", "202", "303"], ["12", "23", "34", "45"]),
        build_result(2, "67890", "98765", ["13579", "24680"], ["10203", "20304", "30405", "40506", "50607", "60708"], ["5678", "6789", "7890", "8901"], ["7006", "8007", "9008", "1110", "2221", "3332"], ["404", "505", "606"], ["56", "67", "78", "89"]),
        build_result(3, "11223", "32109", ["99887", "77665"], ["12312", "23423", "34534", "45645", "56756", "67867"], ["1357", "2468", "3579", "4680"], ["4444", "5555", "6666", "7777", "8888", "9999"], ["707", "808", "909"], ["11", "22", "33", "44"]),
        build_result(4, "44556", "65432", ["12121", "34343"], ["21212", "32323", "43434", "54545", "65656", "76767"], ["1122", "2233", "3344", "4455"], ["1230", "2341", "3452", "4563", "5674", "6785"], ["111", "222", "333"], ["55", "66", "77", "88"]),
        build_result(5, "77889", "43210", ["56565", "78787"], ["80808", "91919", "12012", "23023", "34034", "45045"], ["5566", "6677", "7788", "8899"], ["4321", "5432", "6543", "7654", "8765", "9876"], ["444", "555", "666"], ["90", "01", "12", "23"]),
        build_result(6, "99001", "21098", ["90909", "80808"], ["10101", "21212", "32323", "43434", "54545", "65656"], ["9081", "8172", "7263", "6354"], ["2468", "1357", "0246", "9135", "8024", "7913"], ["777", "888", "999"], ["34", "45", "56", "67"]),
        build_result(7, "10293", "90817", ["31313", "42424"], ["21234", "32345", "43456", "54567", "65678", "76789"], ["0192", "1283", "2374", "3465"], ["1590", "2681", "3772", "4863", "5954", "6045"], ["121", "232", "343"], ["14", "25", "36", "47"]),
        build_result(8, "56473", "81726", ["51515", "62626"], ["11123", "22234", "33345", "44456", "55567", "66678"], ["9087", "8176", "7265", "6354"], ["7140", "8251", "9362", "0473", "1584", "2695"], ["454", "565", "676"], ["58", "69", "70", "81"]),
    ]


def extended_sample_results() -> list[LotteryResult]:
    rows: list[LotteryResult] = []
    templates = sample_results()
    for year in (2024, 2025, 2026):
        for index, result in enumerate(templates, start=1):
            rows.append(
                LotteryResult(
                    date=f"{index:02d}/01/{year}",
                    region=result.region,
                    special=result.special,
                    first=result.first,
                    second=result.second,
                    third=result.third,
                    fourth=result.fourth,
                    fifth=result.fifth,
                    sixth=result.sixth,
                    seventh=result.seventh,
                )
            )
    return rows


class ModelBenchmarkTests(unittest.TestCase):
    def test_available_model_names_exposes_phase5_backends(self):
        names = set(available_model_names())
        self.assertIn("mlp", names)
        self.assertIn("xgboost", names)
        if LIGHTGBM_AVAILABLE:
            self.assertIn("lightgbm", names)
        if CATBOOST_AVAILABLE:
            self.assertIn("catboost", names)

    def test_benchmarkable_model_names_exposes_phase5_backends(self):
        names = set(benchmarkable_model_names("loto2"))
        self.assertIn("xgboost", names)
        self.assertIn("ridge", names)
        self.assertIn("elasticnet", names)
        if LIGHTGBM_AVAILABLE:
            self.assertIn("lightgbm", names)
        if CATBOOST_AVAILABLE:
            self.assertIn("catboost", names)
        self.assertIn("random_forest", names)
        self.assertIn("extra_trees", names)

    @SKLEARN_ONLY
    def test_fit_named_sklearn_ranking_model_supports_ridge(self):
        model = fit_named_sklearn_ranking_model(sample_results(), target_name="loto2", top_k=3, min_train_size=3, model_name="ridge")
        self.assertEqual(model.model_name, "ridge")
        self.assertEqual(len(model.predict()), 3)

    @SKLEARN_ONLY
    def test_fit_named_sklearn_ranking_model_supports_elasticnet(self):
        model = fit_named_sklearn_ranking_model(sample_results(), target_name="loto2", top_k=3, min_train_size=3, model_name="elasticnet")
        self.assertEqual(model.model_name, "elasticnet")
        self.assertEqual(len(model.predict()), 3)

    @SKLEARN_ONLY
    def test_compare_loto2_models_includes_phase6_meta_models(self):
        payload = compare_loto2_models(sample_results(), split_ratio=0.75, top_k=3, min_train_size=3)
        names = {item["model"] for item in payload["evaluations"]}
        self.assertIn("sklearn-ridge-ranking", names)
        self.assertIn("sklearn-elasticnet-ranking", names)

    @SKLEARN_ONLY
    def test_fit_named_sklearn_ranking_model_supports_lightgbm(self):
        if not LIGHTGBM_AVAILABLE:
            self.skipTest("lightgbm is not installed")
        model = fit_named_sklearn_ranking_model(sample_results(), target_name="loto2", top_k=3, min_train_size=3, model_name="lightgbm")
        self.assertEqual(model.model_name, "lightgbm")
        self.assertEqual(len(model.predict()), 3)

    @SKLEARN_ONLY
    def test_fit_named_sklearn_ranking_model_supports_catboost(self):
        if not CATBOOST_AVAILABLE:
            self.skipTest("catboost is not installed")
        model = fit_named_sklearn_ranking_model(sample_results(), target_name="loto2", top_k=3, min_train_size=3, model_name="catboost")
        self.assertEqual(model.model_name, "catboost")
        self.assertEqual(len(model.predict()), 3)

    @SKLEARN_ONLY
    def test_fit_named_sklearn_ranking_model_supports_mlp(self):
        model = fit_named_sklearn_ranking_model(sample_results(), target_name="loto2", top_k=3, min_train_size=3, model_name="mlp")
        self.assertEqual(model.model_name, "mlp")
        self.assertEqual(len(model.predict()), 3)

    @SKLEARN_ONLY
    def test_evaluate_named_ranking_backtest_supports_extra_trees(self):
        metrics = evaluate_named_ranking_backtest(sample_results(), split_ratio=0.75, target_name="loto2", top_k=3, min_train_size=3, model_name="extra_trees")
        self.assertEqual(metrics.target, "loto2")
        self.assertEqual(metrics.top_k, 3)

    def test_evaluate_tuned_weighted_backtest_returns_tuned_model_name(self):
        metrics = evaluate_tuned_weighted_backtest(sample_results(), split_ratio=0.75, target_name="loto2", top_k=3, min_train_size=3)
        self.assertEqual(metrics.model, "tuned-weighted-ranking")

    def test_loto2_input_configurations_include_no_signal_baseline(self):
        configs = loto2_input_configurations(sample_results(), top_k=3, min_train_size=3)
        names = {item["name"] for item in configs}
        self.assertIn("no_signals", names)
        self.assertIn("default_signals", names)

    def test_loto2_data_windows_include_named_ranges(self):
        windows = loto2_data_windows(extended_sample_results())
        names = {item["name"] for item in windows}
        self.assertIn("recent", names)
        self.assertIn("y2025_2026", names)
        self.assertIn("y2024_2026", names)
        self.assertIn("full", names)

    def test_build_full_dashboard_payload_exposes_top20_explanations(self):
        from xsmb_pipeline.evaluate import build_full_dashboard_payload
        payload = build_full_dashboard_payload(extended_sample_results(), split_ratio=0.75, top_k=5, min_train_size=3)
        self.assertIn("top20", payload)
        self.assertIn("top20_details", payload)
        self.assertEqual(payload["top20"]["target"], "loto2")
        self.assertIsInstance(payload["top20"]["rows"], list)
        self.assertLessEqual(len(payload["top20"]["rows"]), 20)
        if payload["top20_details"]:
            first_detail = payload["top20_details"][0]
            self.assertIn("signal_score", first_detail)
            self.assertIn("top_reasons", first_detail)
            self.assertTrue(first_detail["top_reasons"])

    @SKLEARN_ONLY
    def test_compare_loto2_input_models_returns_ranked_configurations(self):
        payload = compare_loto2_input_models(sample_results(), split_ratio=0.75, top_k=3, min_train_size=3)
        self.assertEqual(payload["target"], "loto2")
        self.assertTrue(payload["evaluations"])
        self.assertIn("best_configuration", payload)
        self.assertIn("input_config", payload["evaluations"][0])

    @SKLEARN_ONLY
    def test_benchmark_loto2_data_windows_returns_window_rows(self):
        payload = benchmark_loto2_data_windows(extended_sample_results(), split_ratio=0.75, top_k=3, min_train_size=3)
        self.assertEqual(payload["target"], "loto2")
        self.assertTrue(payload["windows"])
        self.assertIn("window", payload["windows"][0])
        self.assertIn("best_model", payload["windows"][0])

    def test_rolling_loto2_walkforward_benchmark_returns_train_windows(self):
        payload = rolling_loto2_walkforward_benchmark(extended_sample_results(), top_k=3, min_train_size=3, train_window_sizes=(4, 6))
        self.assertEqual(payload["target"], "loto2")
        self.assertTrue(payload["windows"])
        self.assertIn("train_window", payload["windows"][0])

    def test_walkforward_yearly_backtest_returns_summary_payload(self):
        from xsmb_pipeline.evaluate import walkforward_yearly_backtest, yearly_backtest_report
        payload = walkforward_yearly_backtest(extended_sample_results(), target_name="loto2", top_k=3, min_train_size=3)
        self.assertEqual(payload["target"], "loto2")
        self.assertIn("summary", payload)
        self.assertIn("hit_rate", payload["summary"])
        self.assertIn("precision@5", payload["summary"])
        self.assertIn("precision@10", payload["summary"])
        self.assertIn("roi", payload["summary"])
        report = yearly_backtest_report(extended_sample_results(), target_name="loto2", top_k=3, min_train_size=3)
        self.assertEqual(report["target"], "loto2")
        self.assertIn("years", report)

    def test_blend_model_rankings_prefers_high_rank_consensus(self):
        blended = blend_model_rankings([[("01", 1.0), ("02", 0.9)], [("02", 1.0), ("01", 0.8)]])
        self.assertEqual(blended[0][0], "01")

    def test_layer1_ensemble_predictions_returns_blended_rankings(self):
        payload = layer1_ensemble_predictions(sample_results(), top_k=3, min_train_size=3)
        self.assertEqual(payload["model"], "layer1-ensemble")
        self.assertTrue(payload["components"])
        self.assertTrue(payload["component_rankings"])
        self.assertEqual(len(payload["top_predictions"]), 3)
        # Layer 1 chỉ gồm tree-based models, không có logistic/ridge/elasticnet/mlp
        self.assertIn("xgboost", payload["components"])
        self.assertIn("random_forest", payload["components"])
        self.assertIn("extra_trees", payload["components"])
        self.assertNotIn("logistic", payload["components"])
        self.assertNotIn("ridge", payload["components"])
        self.assertNotIn("elasticnet", payload["components"])
        self.assertNotIn("mlp", payload["components"])
        if LIGHTGBM_AVAILABLE:
            self.assertIn("lightgbm", payload["components"])
        if CATBOOST_AVAILABLE:
            self.assertIn("catboost", payload["components"])

    def test_compare_loto2_models_includes_weighted_models(self):
        if SKLEARN_AVAILABLE:
            payload = compare_loto2_models(sample_results(), split_ratio=0.75, top_k=3, min_train_size=3)
            names = {item["model"] for item in payload["evaluations"]}
            self.assertIn("weighted-ranking", names)
            self.assertIn("tuned-weighted-ranking", names)
            self.assertIn("sklearn-mlp-ranking", names)
            if XGBOOST_AVAILABLE:
                self.assertIn("sklearn-xgboost-ranking", names)
            self.assertEqual(payload["target"], "loto2")
            self.assertIn("best_model", payload)
            self.assertIn("input_benchmark", payload)
            self.assertIn("best_configuration", payload["input_benchmark"])
        else:
            with self.assertRaises(ModuleNotFoundError):
                compare_loto2_models(sample_results(), split_ratio=0.75, top_k=3, min_train_size=3)

    @SKLEARN_ONLY
    def test_compare_meta_models_returns_layer2_models(self):
        payload = compare_meta_models(sample_results(), split_ratio=0.75, top_k=3, min_train_size=3)
        names = {item["model"] for item in payload["evaluations"]}
        self.assertIn("sklearn-logistic-ranking", names)
        self.assertIn("sklearn-ridge-ranking", names)
        self.assertIn("sklearn-elasticnet-ranking", names)
        self.assertIn("ensemble", payload)
        self.assertEqual(payload["target"], "loto2")

    @SKLEARN_ONLY
    def test_compare_specialized_models_returns_presets(self):
        payload = compare_specialized_models(sample_results(), split_ratio=0.75, top_k=3, min_train_size=3)
        self.assertEqual(payload["target"], "loto2")
        self.assertTrue(payload["evaluations"])
        self.assertIn("presets", payload)
        self.assertTrue(all(item["target"] == "loto2" for item in payload["evaluations"]))

    def test_compare_provider_models_is_layered_on_loto2_presets(self):
        payload = compare_provider_models(sample_results(), split_ratio=0.75, top_k=3, min_train_size=3)
        self.assertEqual(payload["targets"], ["dau", "duoi", "cham", "tong", "so00_99"])
        self.assertTrue(payload["evaluations"])
        self.assertTrue(all(item["model"] in {"sklearn-logistic-ranking", "sklearn-ridge-ranking", "sklearn-elasticnet-ranking"} for item in payload["evaluations"]))

    def test_targets_remain_locked_to_loto2(self):
        from xsmb_pipeline.targets import actual_targets, target_width
        with self.assertRaises(ValueError):
            actual_targets(sample_results()[0], "dau")
        with self.assertRaises(ValueError):
            target_width("dau")

    @unittest.skipUnless(XGBOOST_AVAILABLE, "xgboost is not installed")
    def test_xgboost_baseline_scores_are_non_zero(self):
        """baseline_scores phải có giá trị thực tế, không phải 0.0 cứng."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as workdir:
            from xsmb_pipeline.database import create_schema, get_connection, insert_features_daily, insert_results
            from xsmb_pipeline.feature_generators import compute_all_features
            db_path = str(Path(workdir) / "xsmb.duckdb")
            connection = get_connection(db_path)
            create_schema(connection)
            results = sample_results()
            insert_results(connection, results)
            for result in results:
                rows = compute_all_features(
                    loto_history=[(item.date, item.special[-2:]) for item in results if item.date <= result.date],
                    digit_history=[
                        (item.date, "special", 0, position_index, digit)
                        for item in results
                        if item.date <= result.date
                        for position_index, digit in enumerate(item.special)
                    ],
                    as_of_date=result.date,
                )
                insert_features_daily(connection, rows)
            connection.close()
            model = train_xgboost_ranking_model(results, db_path=db_path, target_name="loto2", top_k=3, min_train_size=3)
            baseline = model.predict_baseline()
            self.assertEqual(len(baseline), 3)
            # Baseline không nên toàn 0.0
            scores = [score for _, score in baseline]
            self.assertTrue(any(score > 0.0 for score in scores), "baseline_scores toàn 0.0")

    @unittest.skipUnless(XGBOOST_AVAILABLE, "xgboost is not installed")
    def test_xgboost_shap_fallback_returns_zeros_when_shap_missing(self):
        """Khi shap=None, _compute_shap_importance phải trả list zeros."""
        from xsmb_pipeline.models.xgboost_ranker import _compute_shap_importance, _feature_names
        import unittest.mock as mock
        feature_names = _feature_names()
        # Giả lập shap = None bằng cách patch module level
        import xsmb_pipeline.models.xgboost_ranker as xgboost_module
        original_shap = xgboost_module.shap
        try:
            xgboost_module.shap = None
            result = _compute_shap_importance(None, [], feature_names)
            self.assertEqual(len(result), len(feature_names))
            self.assertTrue(all(score == 0.0 for _, score in result))
        finally:
            xgboost_module.shap = original_shap

    @unittest.skipUnless(XGBOOST_AVAILABLE, "xgboost is not installed")
    def test_xgboost_rfe_fallback_returns_default_ranking_when_sklearn_missing(self):
        """Khi SKLEARN_SELECTION_AVAILABLE=False, _compute_rfe_ranking trả default ranking."""
        from xsmb_pipeline.models.xgboost_ranker import _compute_rfe_ranking, _feature_names
        import xsmb_pipeline.models.xgboost_ranker as xgboost_module
        original = xgboost_module.SKLEARN_SELECTION_AVAILABLE
        try:
            xgboost_module.SKLEARN_SELECTION_AVAILABLE = False
            feature_names = _feature_names()
            result = _compute_rfe_ranking([], [], feature_names)
            self.assertEqual(len(result), len(feature_names))
            ranks = [rank for _, rank in result]
            self.assertEqual(sorted(ranks), list(range(1, len(feature_names) + 1)))
        finally:
            xgboost_module.SKLEARN_SELECTION_AVAILABLE = original

    @SKLEARN_ONLY
    def test_meta_stack_predictions_returns_stack_payload(self):
        payload = meta_stack_predictions(sample_results(), top_k=3, min_train_size=3)
        self.assertEqual(payload["model"], "meta-stack")
        self.assertTrue(payload["components"])
        self.assertTrue(payload["component_rankings"])
        self.assertEqual(len(payload["top_predictions"]), 3)

    def test_available_model_names_exposes_dl_path_even_without_runtime(self):
        self.assertIn("mlp", available_model_names())
        self.assertIn("xgboost", available_model_names())

    def test_dashboard_top20_payload_uses_signal_explanations(self):
        payload = build_full_dashboard_payload(extended_sample_results(), split_ratio=0.75, top_k=5, min_train_size=3)
        self.assertIn("top20_details", payload)
        self.assertTrue(payload["top20_details"])
        self.assertIn("top_reasons", payload["top20_details"][0])
        self.assertIn("signal_score", payload["top20_details"][0])

    @unittest.skipUnless(XGBOOST_AVAILABLE, "xgboost is not installed")
    def test_xgboost_payload_exposes_importance_and_selected_features(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as workdir:
            db_path = str(Path(workdir) / "xsmb.duckdb")
            connection = get_connection(db_path)
            create_schema(connection)
            results = sample_results()
            insert_results(connection, results)
            for result in results:
                rows = compute_all_features(
                    loto_history=[(item.date, item.special[-2:]) for item in results if item.date <= result.date],
                    digit_history=[
                        (item.date, "special", 0, position_index, digit)
                        for item in results
                        if item.date <= result.date
                        for position_index, digit in enumerate(item.special)
                    ],
                    as_of_date=result.date,
                )
                insert_features_daily(connection, rows)
            connection.close()
            model = train_xgboost_ranking_model(results, db_path=db_path, target_name="loto2", top_k=3, min_train_size=3)
            payload = export_xgboost_model_payload(model)
            self.assertEqual(payload["model"], "xgboost-ranking")
            self.assertIn("feature_importance", payload)
            self.assertIn("selected_features", payload)
            self.assertTrue(payload["top_predictions"])


if __name__ == "__main__":
    unittest.main()
