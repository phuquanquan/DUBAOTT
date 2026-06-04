import unittest

from xsmb_pipeline.evaluate import benchmark_loto2_data_windows, blend_model_rankings, compare_loto2_input_models, compare_loto2_models, evaluate_named_ranking_backtest, evaluate_tuned_weighted_backtest, loto2_data_windows, loto2_input_configurations, rolling_loto2_walkforward_benchmark
from xsmb_pipeline.models.sklearn_ranker import SKLEARN_AVAILABLE, available_model_names, fit_named_sklearn_ranking_model


SKLEARN_ONLY = unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
from xsmb_pipeline.schema import LotteryResult


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
    def test_available_model_names_exposes_dl_path(self):
        self.assertIn("mlp", available_model_names())

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

    def test_blend_model_rankings_prefers_high_rank_consensus(self):
        blended = blend_model_rankings([[("01", 1.0), ("02", 0.9)], [("02", 1.0), ("01", 0.8)]])
        self.assertEqual(blended[0][0], "01")

    def test_compare_loto2_models_includes_weighted_models(self):
        if SKLEARN_AVAILABLE:
            payload = compare_loto2_models(sample_results(), split_ratio=0.75, top_k=3, min_train_size=3)
            names = {item["model"] for item in payload["evaluations"]}
            self.assertIn("weighted-ranking", names)
            self.assertIn("tuned-weighted-ranking", names)
            self.assertIn("sklearn-mlp-ranking", names)
            self.assertEqual(payload["target"], "loto2")
            self.assertIn("best_model", payload)
            self.assertIn("input_benchmark", payload)
            self.assertIn("best_configuration", payload["input_benchmark"])
        else:
            with self.assertRaises(ModuleNotFoundError):
                compare_loto2_models(sample_results(), split_ratio=0.75, top_k=3, min_train_size=3)

    def test_available_model_names_exposes_dl_path_even_without_runtime(self):
        self.assertIn("mlp", available_model_names())


if __name__ == "__main__":
    unittest.main()
