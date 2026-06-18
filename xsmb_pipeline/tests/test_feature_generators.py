from __future__ import annotations

"""Test cho ``xsmb_pipeline/feature_generators.py`` (Phase 3 - P2.1..P2.19).

Mỗi nhóm feature có 1 test class. Sample data nhỏ (3-7 ngày) để verify
nhanh shape/giá trị mong đợi mà không cần dataset thật.
"""

import unittest
from typing import List, Tuple

from xsmb_pipeline.feature_generators import (
    apriori_features,
    compute_all_features,
    cross_day_features,
    cross_position_features,
    cycle_features,
    dau_duoi_cham_tong_features,
    discovered_features,
    feature_summary,
    fp_growth_features,
    frequency_features,
    gan_features,
    gp_features,
    graph_features,
    hmm_features,
    markov_features,
    pattern_features,
    position_features,
    roi_features,
    sequential_pattern_features,
    symbolic_regression_features,
)


def make_sample_loto_history() -> List[Tuple[str, str]]:
    """3 ngày, mỗi ngày có vài lô. Sort tăng theo ngày."""
    return [
        ("01/01/2010", "12"),
        ("01/01/2010", "27"),
        ("01/01/2010", "54"),
        ("02/01/2010", "12"),
        ("02/01/2010", "33"),
        ("02/01/2010", "27"),
        ("03/01/2010", "27"),
        ("03/01/2010", "78"),
        ("03/01/2010", "33"),
    ]


def make_sample_digit_history() -> List[Tuple[str, str, int, int, str]]:
    """Digit history cho 3 ngày: special + G1 đủ 5 vị trí."""
    rows: List[Tuple[str, str, int, int, str]] = []
    days = {
        "01/01/2010": ("37754", "84983"),
        "02/01/2010": ("22732", "34570"),
        "03/01/2010": ("44591", "41716"),
    }
    for date, (special, g1) in days.items():
        for pos, digit in enumerate(special):
            rows.append((date, "special", 0, pos, digit))
        for pos, digit in enumerate(g1):
            rows.append((date, "G1", 0, pos, digit))
    return rows


# ===========================================================================
# P2.1
# ===========================================================================
class FrequencyTests(unittest.TestCase):
    def test_freq_shape(self) -> None:
        rows = frequency_features(
            make_sample_loto_history(), "03/01/2010", windows=(3, 7)
        )
        # 2 windows × 100 lô = 200 features.
        self.assertEqual(len(rows), 200)
        # All have prefix freq_, all on same as_of date.
        for date, name, _value in rows:
            self.assertEqual(date, "03/01/2010")
            self.assertTrue(name.startswith("freq_"))

    def test_freq_count_correct(self) -> None:
        rows = frequency_features(
            make_sample_loto_history(), "03/01/2010", windows=(3,)
        )
        as_dict = {name: value for _date, name, value in rows}
        # Lô 27 ra cả 3 ngày trong window 3d.
        self.assertEqual(as_dict["freq_27_3d"], 3.0)
        # Lô 33 ra 2 lần (02 và 03).
        self.assertEqual(as_dict["freq_33_3d"], 2.0)
        # Lô 99 không bao giờ ra.
        self.assertEqual(as_dict["freq_99_3d"], 0.0)


# ===========================================================================
# P2.2
# ===========================================================================
class GanTests(unittest.TestCase):
    def test_gan_shape(self) -> None:
        rows = gan_features(make_sample_loto_history(), "03/01/2010")
        # 100 lô × 4 metric = 400.
        self.assertEqual(len(rows), 400)

    def test_gan_current_for_27(self) -> None:
        rows = gan_features(make_sample_loto_history(), "03/01/2010")
        as_dict = {name: value for _d, name, value in rows}
        # Lô 27 ra ngày 03 (last) -> gan_current = 0.
        self.assertEqual(as_dict["gan_27_current"], 0.0)
        # Lô 12 ra 01 + 02, last là day-2 -> current = 1.
        self.assertEqual(as_dict["gan_12_current"], 1.0)
        # Lô 99 chưa ra -> -1.
        self.assertEqual(as_dict["gan_99_current"], -1.0)


# ===========================================================================
# P2.3 + P2.4
# ===========================================================================
class PositionAndCrossPositionTests(unittest.TestCase):
    def test_position_shape(self) -> None:
        rows = position_features(
            make_sample_digit_history(), "03/01/2010", window_days=3
        )
        # special 1×5 + G1 1×5 + G2 2×5 + G3 6×5 + G4 4×4 + G5 6×4 + G6 3×3 + G7 4×2
        # = 5+5+10+30+16+24+9+8 = 107 (prize,idx,pos) × 10 digit = 1070
        self.assertEqual(len(rows), 1070)

    def test_cross_position_shape(self) -> None:
        rows = cross_position_features(
            make_sample_digit_history(), "03/01/2010", window_days=3
        )
        # 4 cặp × 10 digit = 40 feature.
        self.assertEqual(len(rows), 40)


# ===========================================================================
# P2.5
# ===========================================================================
class CrossDayTests(unittest.TestCase):
    def test_cross_day_co_occur_lag1(self) -> None:
        rows = cross_day_features(
            make_sample_loto_history(), "03/01/2010", lags=(1,)
        )
        as_dict = {name: value for _d, name, value in rows}
        # T-1 (02) = {12, 27, 33}; T (03) = {27, 33, 78} -> shared = 2.
        self.assertEqual(as_dict["co_occur_lag1"], 2.0)


# ===========================================================================
# P2.6
# ===========================================================================
class PatternTests(unittest.TestCase):
    def test_pattern_returns_4_features(self) -> None:
        rows = pattern_features(make_sample_loto_history(), "03/01/2010")
        names = [name for _d, name, _v in rows]
        self.assertEqual(
            names,
            ["pascal_count", "reverse_count", "mirror_count", "modulo_sum"],
        )


# ===========================================================================
# P2.7
# ===========================================================================
class RoiTests(unittest.TestCase):
    def test_roi_shape(self) -> None:
        rows = roi_features(make_sample_loto_history(), "03/01/2010")
        # 100 lô × 5 (roi_1d, roi_2d, roi_3d, roi_db, roi_g1) = 500.
        self.assertEqual(len(rows), 500)

    def test_roi_1d_for_27(self) -> None:
        rows = roi_features(make_sample_loto_history(), "03/01/2010")
        as_dict = {name: value for _d, name, value in rows}
        # 27 ra ngày 03 (last). roi_1d_27 = 1 nếu lô 27 ra T-1 (02) -> đúng.
        self.assertEqual(as_dict["roi_1d_27"], 1.0)


# ===========================================================================
# P2.8
# ===========================================================================
class DauDuoiChamTongTests(unittest.TestCase):
    def test_shape(self) -> None:
        rows = dau_duoi_cham_tong_features(
            make_sample_loto_history(), "03/01/2010", window_days=3
        )
        # 10 × 3 (dau, duoi, cham) + tong_de + tong_lo = 32.
        self.assertEqual(len(rows), 32)


# ===========================================================================
# P2.9 + P2.10 + P2.11
# ===========================================================================
class DiscoveredAndGpAndSrTests(unittest.TestCase):
    def test_discovered_shape(self) -> None:
        rows = discovered_features(
            make_sample_digit_history(), "03/01/2010", window_days=3
        )
        self.assertEqual(len(rows), 5)

    def test_gp_shape(self) -> None:
        rows = gp_features(
            make_sample_digit_history(), "03/01/2010", window_days=3
        )
        # 5 candidate × 3 op × 2 stat = 30.
        self.assertEqual(len(rows), 30)

    def test_symbolic_regression_returns_3_params(self) -> None:
        rows = symbolic_regression_features(
            make_sample_digit_history(), "03/01/2010", window_days=3
        )
        names = [name for _d, name, _v in rows]
        self.assertEqual(
            names, ["sr_db_v4_coef_a", "sr_db_v4_coef_b", "sr_db_v4_error"]
        )


# ===========================================================================
# P2.12 + P2.13 + P2.14
# ===========================================================================
class MarkovHmmCycleTests(unittest.TestCase):
    def test_markov_shape(self) -> None:
        rows = markov_features(
            make_sample_loto_history(), "03/01/2010", orders=(1,)
        )
        self.assertEqual(len(rows), 100)

    def test_markov_probabilities_sum_to_at_most_1(self) -> None:
        """Transition matrix thật: tổng xác suất markov1 ≤ 1.0 (nhiều lô ra 1 ngày)."""
        rows = markov_features(
            make_sample_loto_history(), "03/01/2010", orders=(1,)
        )
        total = sum(value for _d, name, value in rows if name.startswith("markov1_"))
        # Tổng xác suất markov bậc 1 phải ≤ 1.0 + epsilon (vì nhiều lô ra/ngày)
        self.assertLessEqual(total, 1.01)

    def test_markov_values_are_non_negative(self) -> None:
        rows = markov_features(
            make_sample_loto_history(), "03/01/2010", orders=(1, 2)
        )
        for _d, name, value in rows:
            self.assertGreaterEqual(value, 0.0, msg=f"{name} có giá trị âm: {value}")

    def test_hmm_returns_state_features(self) -> None:
        rows = hmm_features(
            make_sample_loto_history(), "03/01/2010", window_days=3
        )
        names = {name for _d, name, _v in rows}
        self.assertIn("hmm_state_current", names)
        self.assertIn("hmm_transition_to_0", names)
        self.assertIn("hmm_transition_to_2", names)

    def test_cycle_shape(self) -> None:
        rows = cycle_features(
            make_sample_loto_history(), "03/01/2010", periods=(7, 14, 30)
        )
        self.assertEqual(len(rows), 3)


# ===========================================================================
# P2.15 + P2.16 + P2.17 + P2.18
# ===========================================================================
class AprioriFpGrowthSeqGraphTests(unittest.TestCase):
    def test_apriori_returns_top_k_plus_1(self) -> None:
        rows = apriori_features(
            make_sample_loto_history(),
            "03/01/2010",
            window_days=3,
            min_support=1,
            top_k=3,
        )
        # 1 count + 3 top pair (hoặc pad) = 4.
        self.assertEqual(len(rows), 4)

    def test_fp_growth_pad(self) -> None:
        rows = fp_growth_features(
            make_sample_loto_history(),
            "03/01/2010",
            window_days=3,
            top_k=2,
        )
        # 1 + top_k = 3.
        self.assertEqual(len(rows), 3)

    def test_seq_pattern_top_k(self) -> None:
        rows = sequential_pattern_features(
            make_sample_loto_history(),
            "03/01/2010",
            window_days=3,
            top_k=4,
        )
        self.assertEqual(len(rows), 4)

    def test_graph_shape(self) -> None:
        rows = graph_features(
            make_sample_loto_history(),
            "03/01/2010",
            window_days=3,
            pagerank_iterations=3,
        )
        # 100 × 4 = 400.
        self.assertEqual(len(rows), 400)


# ===========================================================================
# P2.19 - Orchestrator
# ===========================================================================
class ComputeAllFeaturesTests(unittest.TestCase):
    def test_compute_all_features_runs(self) -> None:
        rows = compute_all_features(
            loto_history=make_sample_loto_history(),
            digit_history=make_sample_digit_history(),
            as_of_date="03/01/2010",
        )
        # Tổng phải > 1000 (frequency 800 + gan 400 + roi 300 + ...).
        self.assertGreater(len(rows), 2000)
        # Mọi feature_value đều là float.
        for _date, _name, value in rows:
            self.assertIsInstance(value, float)

    def test_compute_all_features_writes_into_features_daily(self) -> None:
        from xsmb_pipeline.database import (
            create_schema,
            fetch_features_for_date,
            get_connection,
            insert_features_daily,
        )

        rows = compute_all_features(
            loto_history=make_sample_loto_history(),
            digit_history=make_sample_digit_history(),
            as_of_date="03/01/2010",
        )
        conn = get_connection(":memory:")
        try:
            create_schema(conn)
            inserted = insert_features_daily(conn, rows)
            self.assertEqual(inserted, len(rows))
            features = fetch_features_for_date(conn, "03/01/2010")
            self.assertEqual(len(features), len(rows))
            self.assertIn("freq_27_3d", features)
            self.assertIn("gan_27_current", features)
        finally:
            conn.close()

    def test_feature_summary_groups_by_prefix(self) -> None:
        rows = compute_all_features(
            loto_history=make_sample_loto_history(),
            digit_history=make_sample_digit_history(),
            as_of_date="03/01/2010",
        )
        summary = feature_summary(rows)
        self.assertIn("freq", summary)
        self.assertIn("gan", summary)
        self.assertGreaterEqual(summary["freq"], 800)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
