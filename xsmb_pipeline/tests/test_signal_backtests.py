import unittest

from xsmb_pipeline.evaluate import all_signals_backtest, ensemble_backtest, signal_backtest, signal_group_backtest, walkforward_signal_backtest
from xsmb_pipeline.schema import LotteryResult
from xsmb_pipeline.signals import (
    SIGNAL_DEFINITIONS, signal_group_names,
    days_since_last_score, prize_position_penalty_score, thirty_day_frequency_penalty_score,
)


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
    ]


# --- Tests for Junlangzi Scoring Algorithms ---

class JunlangziScoringTests(unittest.TestCase):
    def test_days_since_last_score_returns_valid_score(self):
        results = sample_results()
        sig = days_since_last_score(results, "99", "loto2")
        self.assertEqual(sig.name, "days_since_last")
        self.assertGreaterEqual(sig.score, 0.0)
        self.assertLessEqual(sig.score, 1.0)
        self.assertIn("days", sig.details)

    def test_days_since_last_score_zero_for_recent_appearance(self):
        results = sample_results()
        sig = days_since_last_score(results, "45", "loto2")
        self.assertEqual(sig.details["days"], 0)

    def test_days_since_last_score_returns_nonzero_days(self):
        results = sample_results()
        sig = days_since_last_score(results, "99", "loto2")
        self.assertIn("days", sig.details)

    def test_days_since_last_score_returns_zero_for_non_loto2(self):
        results = sample_results()
        sig = days_since_last_score(results, "5", "dau")
        self.assertEqual(sig.score, 0.0)

    def test_prize_position_penalty_score_returns_valid_signal(self):
        results = sample_results()
        sig = prize_position_penalty_score(results, "45", "loto2")
        self.assertEqual(sig.name, "prize_position_penalty")
        self.assertGreaterEqual(sig.score, 0.0)
        self.assertLessEqual(sig.score, 1.0)
        self.assertIn("matched_prizes", sig.details)

    def test_prize_position_penalty_score_match_count(self):
        results = sample_results()
        sig = prize_position_penalty_score(results, "45", "loto2")
        self.assertGreater(sig.details["matched_prizes"], 0)

    def test_prize_position_penalty_score_no_match(self):
        results = sample_results()
        sig = prize_position_penalty_score(results, "66", "loto2")
        self.assertEqual(sig.details["matched_prizes"], 0)

    def test_prize_position_penalty_returns_zero_for_non_loto2(self):
        results = sample_results()
        sig = prize_position_penalty_score(results, "5", "dau")
        self.assertEqual(sig.score, 0.0)

    def test_thirty_day_frequency_penalty_score_returns_valid_signal(self):
        results = sample_results()
        sig = thirty_day_frequency_penalty_score(results, "45", "loto2")
        self.assertEqual(sig.name, "thirty_day_freq_penalty")
        self.assertGreaterEqual(sig.score, 0.0)
        self.assertLessEqual(sig.score, 1.0)

    def test_thirty_day_frequency_penalty_score_appeared(self):
        results = sample_results()
        sig = thirty_day_frequency_penalty_score(results, "45", "loto2")
        self.assertIn("appeared", sig.details)

    def test_thirty_day_frequency_penalty_returns_zero_for_non_loto2(self):
        results = sample_results()
        sig = thirty_day_frequency_penalty_score(results, "5", "dau")
        self.assertEqual(sig.score, 0.0)

    def test_new_signals_appear_in_signal_definitions(self):
        names = {d.name for d in SIGNAL_DEFINITIONS}
        self.assertIn("days_since_last", names)
        self.assertIn("prize_position_penalty", names)
        self.assertIn("thirty_day_freq_penalty", names)

    def test_new_signals_in_ensemble_names(self):
        from xsmb_pipeline.signals import TARGET_ENSEMBLE_SIGNAL_NAMES
        ens = TARGET_ENSEMBLE_SIGNAL_NAMES["loto2"]
        self.assertIn("days_since_last", ens)
        self.assertIn("prize_position_penalty", ens)
        self.assertIn("thirty_day_freq_penalty", ens)

    def test_new_signals_in_model_names(self):
        from xsmb_pipeline.signals import TARGET_MODEL_SIGNAL_NAMES
        model = TARGET_MODEL_SIGNAL_NAMES["loto2"]
        self.assertIn("days_since_last", model)
        self.assertIn("prize_position_penalty", model)
        self.assertIn("thirty_day_freq_penalty", model)


class SignalBacktestTests(unittest.TestCase):
    def test_walkforward_signal_backtest_uses_past_only(self):
        rows = walkforward_signal_backtest(sample_results(), target_name="loto2", signal_name="touch", top_k=3, min_train_size=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["date"], "04/01/2026")
        self.assertEqual(rows[0]["signal"], "touch")
        self.assertEqual(len(rows[0]["predicted"]), 3)
        self.assertNotIn("04/01/2026", rows[0]["predicted"])

    def test_signal_backtest_returns_evaluation_and_recent_rows(self):
        payload = signal_backtest(sample_results(), target_name="loto2", signal_name="bridge", top_k=3, min_train_size=3, recent_rows=2)
        self.assertEqual(payload["mode"], "single")
        self.assertEqual(payload["signal"], "bridge")
        self.assertEqual(payload["evaluation"]["test_size"], 3)
        self.assertEqual(len(payload["rows"]), 2)

    def test_all_signals_backtest_covers_catalog(self):
        payload = all_signals_backtest(sample_results(), target_name="loto2", top_k=3, min_train_size=3, recent_rows=2)
        signal_names = {item["signal"] for item in payload["evaluations"]}
        self.assertEqual(signal_names, {definition.name for definition in SIGNAL_DEFINITIONS})
        self.assertEqual(payload["mode"], "all")
        self.assertIn("verdict_counts", payload)
        self.assertIn("bridge_focus", payload)
        self.assertIn("group_summary", payload)
        self.assertIn("filter_summary", payload)
        self.assertEqual(sum(payload["verdict_counts"].values()), len(payload["evaluations"]))
        self.assertTrue(all("research_verdict" in item for item in payload["evaluations"]))
        self.assertEqual({item["group"] for item in payload["group_summary"]["groups"]}, set(signal_group_names()))

    def test_signal_group_backtest_returns_group_summary(self):
        payload = signal_group_backtest(sample_results(), target_name="loto2", top_k=3, min_train_size=3, recent_rows=2)
        self.assertEqual(payload["mode"], "groups")
        self.assertIn("group_summary", payload)
        self.assertIn("groups", payload["group_summary"])
        self.assertEqual({item["group"] for item in payload["group_summary"]["groups"]}, set(signal_group_names()))

    def test_ensemble_backtest_returns_ensemble_evaluation(self):
        payload = ensemble_backtest(sample_results(), target_name="loto2", top_k=3, min_train_size=3, recent_rows=2)
        self.assertEqual(payload["mode"], "ensemble")
        self.assertEqual(payload["signal"], "ensemble")
        self.assertEqual(payload["evaluation"]["signal"], "ensemble")
        self.assertIn(payload["evaluation"]["research_verdict"], {"keep", "watch", "drop"})

    def test_walkforward_yearly_backtest_exposes_precision_and_roi(self):
        from xsmb_pipeline.evaluate import walkforward_yearly_backtest
        payload = walkforward_yearly_backtest(sample_results(), target_name="loto2", top_k=3, min_train_size=3)
        self.assertEqual(payload["target"], "loto2")
        self.assertIn("summary", payload)
        self.assertIn("precision@5", payload["summary"])
        self.assertIn("precision@10", payload["summary"])
        self.assertIn("roi", payload["summary"])

    def test_filter_summary_tracks_kept_groups_and_signals(self):
        payload = all_signals_backtest(sample_results(), target_name="loto2", top_k=3, min_train_size=3, recent_rows=2)
        summary = payload["filter_summary"]
        self.assertIn("kept_groups", summary)
        self.assertIn("dropped_groups", summary)
        self.assertIn("kept_signals_by_group", summary)
        self.assertTrue(set(summary["kept_groups"]).issubset(set(signal_group_names())))
        self.assertTrue(set(summary["dropped_groups"]).issubset(set(signal_group_names())))


if __name__ == "__main__":
    unittest.main()
