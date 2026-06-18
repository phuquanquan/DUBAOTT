from __future__ import annotations

"""Test cho ``xsmb_pipeline/database.py``.

Dùng DuckDB in-memory (``:memory:``) để test nhanh, không đụng file thật.ông p
Phạm vi test:
    P1.1 - schema + bảng ``draws``:
        - ``create_schema`` tạo đủ 6 bảng theo thiết kế.
        - ``create_draws_table`` tạo bảng ``draws`` với 4 cột yêu cầu
          (``draw_date``, ``region``, ``special``, ``first_prize``).
        - ``insert_draws`` idempotent (không trùng PK).
        - ``fetch_draw`` đọc lại đúng kỳ quay theo ngày.

    P1.2 - bảng ``prizes`` (chi tiết từng giải / từng dãy):
        - ``create_prizes_table`` tạo 4 cột (``draw_date``, ``prize_name``,
          ``prize_index``, ``prize_value``) - PK (date, name, index).
        - ``insert_prizes`` ghi đủ ``special`` + G1..G7, đúng số dãy
          mỗi giải XSMB (DB=1, G1=1, G2=2, G3=6, G4=4, G5=6, G6=3, G7=4).
        - ``insert_prizes`` idempotent trên cùng (date, name, index).
        - ``insert_prizes`` rỗng -> trả 0.
"""

import unittest
from typing import List

import duckdb

from xsmb_pipeline.database import (
    count_rows,
    create_draws_table,
    create_prizes_table,
    create_schema,
    fetch_digit_history,
    fetch_digits_for_date,
    fetch_draw,
    fetch_features_for_date,
    fetch_loto_for_date,
    fetch_loto_history,
    fetch_pattern_scores,
    fetch_prizes_for_date,
    get_connection,
    insert_digit_positions,
    insert_draws,
    insert_features_daily,
    insert_loto_results,
    insert_pattern_scores,
    insert_prizes,
    insert_results,
    integrity_check,
    migrate_csv,
)
from xsmb_pipeline.dataset import write_csv
from xsmb_pipeline.schema import LotteryResult


def make_sample_results() -> List[LotteryResult]:
    """Sinh 3 kỳ quay mẫu (đúng định dạng XSMB) cho test."""
    return [
        LotteryResult(
            date="01/01/2010",
            region="XSMB",
            special="37754",
            first=["84983"],
            second=["12345", "67890"],
            third=["11111", "22222", "33333", "44444", "55555", "66666"],
            fourth=["1234", "5678", "9012", "3456"],
            fifth=["12345", "23456", "34567", "45678", "56789", "67890"],
            sixth=["123", "456", "789"],
            seventh=["12", "34", "56", "78"],
        ),
        LotteryResult(
            date="02/01/2010",
            region="XSMB",
            special="22732",
            first=["34570"],
            second=["10001", "20002"],
            third=["30003", "40004", "50005", "60006", "70007", "80008"],
            fourth=["1111", "2222", "3333", "4444"],
            fifth=["11122", "22233", "33344", "44455", "55566", "66677"],
            sixth=["111", "222", "333"],
            seventh=["11", "22", "33", "44"],
        ),
        LotteryResult(
            date="03/01/2010",
            region="XSMB",
            special="44591",
            first=["41716"],
            second=["00001", "00002"],
            third=["00003", "00004", "00005", "00006", "00007", "00008"],
            fourth=["0001", "0002", "0003", "0004"],
            fifth=["00011", "00022", "00033", "00044", "00055", "00066"],
            sixth=["001", "002", "003"],
            seventh=["01", "02", "03", "04"],
        ),
    ]


class CreateSchemaTests(unittest.TestCase):
    """Kiểm tra schema được tạo đầy đủ và bảng ``draws`` đúng đặc tả."""

    def setUp(self) -> None:
        self.conn: duckdb.DuckDBPyConnection = get_connection(":memory:")

    def tearDown(self) -> None:
        self.conn.close()

    def test_create_schema_creates_six_tables(self) -> None:
        create_schema(self.conn)
        rows = self.conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchall()
        table_names = [row[0] for row in rows]
        expected = [
            "digit_positions",
            "draws",
            "features_daily",
            "loto_results",
            "pattern_scores",
            "prizes",
        ]
        self.assertEqual(table_names, expected)

    def test_draws_table_columns_match_design(self) -> None:
        create_draws_table(self.conn)
        rows = self.conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'draws'
            ORDER BY ordinal_position
            """
        ).fetchall()
        columns = {name: dtype for name, dtype in rows}
        self.assertEqual(
            list(columns.keys()),
            ["draw_date", "region", "special", "first_prize"],
        )
        for dtype in columns.values():
            self.assertEqual(dtype, "VARCHAR")

    def test_create_schema_is_idempotent(self) -> None:
        create_schema(self.conn)
        # Gọi lại lần 2 không được raise (dùng IF NOT EXISTS).
        create_schema(self.conn)
        count = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main'
            """
        ).fetchone()
        self.assertIsNotNone(count)
        assert count is not None  # for type checker
        self.assertEqual(int(count[0]), 6)


class InsertDrawsTests(unittest.TestCase):
    """Kiểm tra ``insert_draws`` + ``fetch_draw`` + ``count_rows``."""

    def setUp(self) -> None:
        self.conn: duckdb.DuckDBPyConnection = get_connection(":memory:")
        create_schema(self.conn)
        self.results = make_sample_results()

    def tearDown(self) -> None:
        self.conn.close()

    def test_insert_draws_returns_row_count(self) -> None:
        inserted = insert_draws(self.conn, self.results)
        self.assertEqual(inserted, 3)

    def test_insert_draws_persists_rows(self) -> None:
        insert_draws(self.conn, self.results)
        counts = count_rows(self.conn, ("draws",))
        self.assertEqual(counts["draws"], 3)

    def test_insert_draws_uses_first_prize_first_value(self) -> None:
        insert_draws(self.conn, self.results)
        record = fetch_draw(self.conn, "01/01/2010")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["draw_date"], "01/01/2010")
        self.assertEqual(record["region"], "XSMB")
        self.assertEqual(record["special"], "37754")
        self.assertEqual(record["first_prize"], "84983")

    def test_insert_draws_idempotent_on_same_date(self) -> None:
        insert_draws(self.conn, self.results)
        # Ghi lại lần 2 cùng draw_date - không được nhân đôi.
        insert_draws(self.conn, self.results)
        counts = count_rows(self.conn, ("draws",))
        self.assertEqual(counts["draws"], 3)

    def test_insert_draws_empty_returns_zero(self) -> None:
        inserted = insert_draws(self.conn, [])
        self.assertEqual(inserted, 0)
        counts = count_rows(self.conn, ("draws",))
        self.assertEqual(counts["draws"], 0)

    def test_fetch_draw_missing_returns_none(self) -> None:
        insert_draws(self.conn, self.results)
        self.assertIsNone(fetch_draw(self.conn, "31/12/1999"))

    def test_insert_draws_handles_empty_first_prize(self) -> None:
        empty_first = LotteryResult(
            date="04/01/2010",
            region="XSMB",
            special="11111",
            first=[],
            second=[],
            third=[],
            fourth=[],
            fifth=[],
            sixth=[],
            seventh=[],
        )
        insert_draws(self.conn, [empty_first])
        record = fetch_draw(self.conn, "04/01/2010")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["first_prize"], "")


class CreatePrizesTableTests(unittest.TestCase):
    """P1.2 - kiểm tra schema bảng ``prizes`` đúng thiết kế."""

    def setUp(self) -> None:
        self.conn: duckdb.DuckDBPyConnection = get_connection(":memory:")

    def tearDown(self) -> None:
        self.conn.close()

    def test_prizes_table_columns_match_design(self) -> None:
        create_prizes_table(self.conn)
        rows = self.conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'prizes'
            ORDER BY ordinal_position
            """
        ).fetchall()
        columns = {name: dtype for name, dtype in rows}
        self.assertEqual(
            list(columns.keys()),
            ["draw_date", "prize_name", "prize_index", "prize_value"],
        )
        self.assertEqual(columns["draw_date"], "VARCHAR")
        self.assertEqual(columns["prize_name"], "VARCHAR")
        self.assertEqual(columns["prize_index"], "INTEGER")
        self.assertEqual(columns["prize_value"], "VARCHAR")


class InsertPrizesTests(unittest.TestCase):
    """P1.2 - kiểm tra ``insert_prizes`` đầy đủ giải đặc biệt + G1..G7."""

    # Số dãy mỗi giải XSMB chuẩn (theo sample data make_sample_results).
    PRIZE_COUNT_PER_DRAW = {
        "special": 1,
        "G1": 1,
        "G2": 2,
        "G3": 6,
        "G4": 4,
        "G5": 6,
        "G6": 3,
        "G7": 4,
    }
    TOTAL_PER_DRAW = sum(PRIZE_COUNT_PER_DRAW.values())  # = 27

    def setUp(self) -> None:
        self.conn: duckdb.DuckDBPyConnection = get_connection(":memory:")
        create_schema(self.conn)
        self.results = make_sample_results()

    def tearDown(self) -> None:
        self.conn.close()

    def test_insert_prizes_returns_total_rows(self) -> None:
        inserted = insert_prizes(self.conn, self.results)
        self.assertEqual(inserted, self.TOTAL_PER_DRAW * len(self.results))

    def test_insert_prizes_persists_correct_count_per_prize(self) -> None:
        insert_prizes(self.conn, self.results)
        rows = self.conn.execute(
            """
            SELECT prize_name, COUNT(*)
            FROM prizes
            GROUP BY prize_name
            ORDER BY prize_name
            """
        ).fetchall()
        actual = {name: int(count) for name, count in rows}
        expected = {
            name: count * len(self.results)
            for name, count in self.PRIZE_COUNT_PER_DRAW.items()
        }
        self.assertEqual(actual, expected)

    def test_insert_prizes_special_value_correct(self) -> None:
        insert_prizes(self.conn, self.results)
        row = self.conn.execute(
            """
            SELECT prize_value
            FROM prizes
            WHERE draw_date = '01/01/2010' AND prize_name = 'special'
            """
        ).fetchone()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row[0], "37754")

    def test_insert_prizes_g3_indexes_zero_based(self) -> None:
        insert_prizes(self.conn, self.results)
        rows = self.conn.execute(
            """
            SELECT prize_index, prize_value
            FROM prizes
            WHERE draw_date = '01/01/2010' AND prize_name = 'G3'
            ORDER BY prize_index
            """
        ).fetchall()
        indexes = [int(row[0]) for row in rows]
        values = [row[1] for row in rows]
        self.assertEqual(indexes, [0, 1, 2, 3, 4, 5])
        # Giữ nguyên thứ tự gốc trong LotteryResult.third.
        self.assertEqual(
            values,
            ["11111", "22222", "33333", "44444", "55555", "66666"],
        )

    def test_insert_prizes_idempotent_on_same_pk(self) -> None:
        insert_prizes(self.conn, self.results)
        before = count_rows(self.conn, ("prizes",))["prizes"]
        # Ghi lại - PK (draw_date, prize_name, prize_index) -> không trùng.
        insert_prizes(self.conn, self.results)
        after = count_rows(self.conn, ("prizes",))["prizes"]
        self.assertEqual(before, after)

    def test_insert_prizes_empty_returns_zero(self) -> None:
        inserted = insert_prizes(self.conn, [])
        self.assertEqual(inserted, 0)
        counts = count_rows(self.conn, ("prizes",))
        self.assertEqual(counts["prizes"], 0)

    def test_insert_prizes_query_by_prize_name(self) -> None:
        """Verify use case: query toàn bộ G7 cho 1 ngày để sinh feature."""
        insert_prizes(self.conn, self.results)
        rows = self.conn.execute(
            """
            SELECT prize_value
            FROM prizes
            WHERE draw_date = '02/01/2010' AND prize_name = 'G7'
            ORDER BY prize_index
            """
        ).fetchall()
        values = [row[0] for row in rows]
        self.assertEqual(values, ["11", "22", "33", "44"])


# ===========================================================================
# P1.3 - loto_results
# ===========================================================================
class LotoResultsTests(unittest.TestCase):
    """P1.3 - bảng ``loto_results`` + insert + fetch_loto_for_date/history."""

    def setUp(self) -> None:
        self.conn: duckdb.DuckDBPyConnection = get_connection(":memory:")
        create_schema(self.conn)
        self.results = make_sample_results()

    def tearDown(self) -> None:
        self.conn.close()

    def test_loto_results_table_columns(self) -> None:
        rows = self.conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'loto_results'
            ORDER BY ordinal_position
            """
        ).fetchall()
        columns = {name: dtype for name, dtype in rows}
        self.assertEqual(
            list(columns.keys()), ["draw_date", "loto_number"]
        )
        for dtype in columns.values():
            self.assertEqual(dtype, "VARCHAR")

    def test_insert_loto_results_dedupe_in_day(self) -> None:
        insert_loto_results(self.conn, self.results)
        # Ngày 02/01/2010 có nhiều giá trị "X..11" "X..22" "X..33" "X..44"
        # nhưng loto_results chỉ giữ unique theo PK (date, loto_number).
        rows = self.conn.execute(
            """
            SELECT loto_number
            FROM loto_results
            WHERE draw_date = '02/01/2010'
            ORDER BY loto_number
            """
        ).fetchall()
        values = [row[0] for row in rows]
        self.assertEqual(values, sorted(set(values)))
        # Tất cả phải dài đúng 2 ký tự là chữ số.
        for value in values:
            self.assertEqual(len(value), 2)
            self.assertTrue(value.isdigit())

    def test_insert_loto_results_idempotent(self) -> None:
        insert_loto_results(self.conn, self.results)
        before = count_rows(self.conn, ("loto_results",))["loto_results"]
        insert_loto_results(self.conn, self.results)
        after = count_rows(self.conn, ("loto_results",))["loto_results"]
        self.assertEqual(before, after)

    def test_insert_loto_results_empty_returns_zero(self) -> None:
        self.assertEqual(insert_loto_results(self.conn, []), 0)

    def test_fetch_loto_for_date(self) -> None:
        insert_loto_results(self.conn, self.results)
        loto = fetch_loto_for_date(self.conn, "01/01/2010")
        self.assertEqual(loto, sorted(set(loto)))
        self.assertGreater(len(loto), 0)
        # Spot-check: 2 chữ số cuối của ``special=37754`` -> "54".
        self.assertIn("54", loto)
        # 2 chữ số cuối của G7 ngày 01/01/2010 = 12,34,56,78
        self.assertIn("12", loto)
        self.assertIn("78", loto)

    def test_fetch_loto_for_date_unknown_returns_empty(self) -> None:
        insert_loto_results(self.conn, self.results)
        self.assertEqual(fetch_loto_for_date(self.conn, "31/12/1999"), [])

    def test_fetch_loto_history_orders_by_date(self) -> None:
        insert_loto_results(self.conn, self.results)
        history = fetch_loto_history(self.conn)
        # Phải có ít nhất 3 ngày, sort theo thời gian (DD/MM/YYYY).
        dates_in_order = [date for date, _ in history]
        unique_in_order: List[str] = []
        for date in dates_in_order:
            if not unique_in_order or unique_in_order[-1] != date:
                unique_in_order.append(date)
        self.assertEqual(
            unique_in_order, ["01/01/2010", "02/01/2010", "03/01/2010"]
        )

    def test_fetch_loto_history_with_date_range(self) -> None:
        insert_loto_results(self.conn, self.results)
        history = fetch_loto_history(
            self.conn, start_date="02/01/2010", end_date="02/01/2010"
        )
        for date, _loto in history:
            self.assertEqual(date, "02/01/2010")
        self.assertGreater(len(history), 0)


# ===========================================================================
# P1.4 - digit_positions
# ===========================================================================
class DigitPositionsTests(unittest.TestCase):
    """P1.4 - bảng ``digit_positions`` + insert + fetch_digits_for_date/history."""

    # XSMB chuẩn: tổng số chữ số/ngày = 5 (DB) + 5 (G1) + 10 (G2)
    # + 30 (G3) + 16 (G4) + 30 (G5) + 9 (G6) + 8 (G7) = 113.
    DIGITS_PER_DAY = 5 + 5 + 10 + 30 + 16 + 30 + 9 + 8  # = 113

    def setUp(self) -> None:
        self.conn: duckdb.DuckDBPyConnection = get_connection(":memory:")
        create_schema(self.conn)
        self.results = make_sample_results()

    def tearDown(self) -> None:
        self.conn.close()

    def test_digit_positions_table_columns(self) -> None:
        rows = self.conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'digit_positions'
            ORDER BY ordinal_position
            """
        ).fetchall()
        columns = {name: dtype for name, dtype in rows}
        self.assertEqual(
            list(columns.keys()),
            [
                "draw_date",
                "prize_name",
                "prize_index",
                "position_index",
                "digit",
            ],
        )
        self.assertEqual(columns["prize_index"], "INTEGER")
        self.assertEqual(columns["position_index"], "INTEGER")
        self.assertEqual(columns["digit"], "VARCHAR")

    def test_insert_digit_positions_total_count(self) -> None:
        inserted = insert_digit_positions(self.conn, self.results)
        self.assertEqual(inserted, self.DIGITS_PER_DAY * len(self.results))

    def test_digits_are_single_char(self) -> None:
        insert_digit_positions(self.conn, self.results)
        rows = self.conn.execute(
            "SELECT digit FROM digit_positions"
        ).fetchall()
        for (digit,) in rows:
            self.assertEqual(len(digit), 1)
            self.assertTrue(digit.isdigit())

    def test_position_index_range_per_prize(self) -> None:
        insert_digit_positions(self.conn, self.results)
        rows = self.conn.execute(
            """
            SELECT prize_name, MIN(position_index), MAX(position_index)
            FROM digit_positions
            GROUP BY prize_name
            ORDER BY prize_name
            """
        ).fetchall()
        ranges = {name: (int(low), int(high)) for name, low, high in rows}
        # 5 chữ số -> position 0..4; 4 chữ số -> 0..3; 3 -> 0..2; 2 -> 0..1.
        self.assertEqual(ranges["special"], (0, 4))
        self.assertEqual(ranges["G1"], (0, 4))
        self.assertEqual(ranges["G2"], (0, 4))
        self.assertEqual(ranges["G3"], (0, 4))
        self.assertEqual(ranges["G4"], (0, 3))
        self.assertEqual(ranges["G5"], (0, 4))
        self.assertEqual(ranges["G6"], (0, 2))
        self.assertEqual(ranges["G7"], (0, 1))

    def test_special_digits_match_source(self) -> None:
        insert_digit_positions(self.conn, self.results)
        rows = self.conn.execute(
            """
            SELECT position_index, digit
            FROM digit_positions
            WHERE draw_date = '01/01/2010' AND prize_name = 'special'
            ORDER BY position_index
            """
        ).fetchall()
        digits = "".join(digit for _pos, digit in rows)
        self.assertEqual(digits, "37754")

    def test_fetch_digits_for_date(self) -> None:
        insert_digit_positions(self.conn, self.results)
        digits = fetch_digits_for_date(self.conn, "01/01/2010")
        self.assertEqual(len(digits), self.DIGITS_PER_DAY)
        # special[V0..V4] phải tồn tại đúng 5 dòng.
        special = [
            entry for entry in digits if entry["prize_name"] == "special"
        ]
        self.assertEqual(len(special), 5)
        special_digits = "".join(str(entry["digit"]) for entry in special)
        self.assertEqual(special_digits, "37754")

    def test_fetch_digit_history_for_special_last_position(self) -> None:
        insert_digit_positions(self.conn, self.results)
        history = fetch_digit_history(
            self.conn,
            prize_name="special",
            prize_index=0,
            position_index=4,
        )
        # special đuôi: 37754->4, 22732->2, 44591->1
        self.assertEqual(
            history,
            [
                ("01/01/2010", "4"),
                ("02/01/2010", "2"),
                ("03/01/2010", "1"),
            ],
        )

    def test_insert_digit_positions_idempotent(self) -> None:
        insert_digit_positions(self.conn, self.results)
        before = count_rows(self.conn, ("digit_positions",))["digit_positions"]
        insert_digit_positions(self.conn, self.results)
        after = count_rows(self.conn, ("digit_positions",))["digit_positions"]
        self.assertEqual(before, after)


# ===========================================================================
# P1.5 - features_daily
# ===========================================================================
class FeaturesDailyTests(unittest.TestCase):
    """P1.5 - bảng ``features_daily`` + UPSERT + fetch_features_for_date."""

    def setUp(self) -> None:
        self.conn: duckdb.DuckDBPyConnection = get_connection(":memory:")
        create_schema(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def test_features_daily_table_columns(self) -> None:
        rows = self.conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'features_daily'
            ORDER BY ordinal_position
            """
        ).fetchall()
        columns = {name: dtype for name, dtype in rows}
        self.assertEqual(
            list(columns.keys()),
            ["draw_date", "feature_name", "feature_value"],
        )
        self.assertEqual(columns["draw_date"], "VARCHAR")
        self.assertEqual(columns["feature_name"], "VARCHAR")
        self.assertEqual(columns["feature_value"], "DOUBLE")

    def test_insert_features_daily_basic(self) -> None:
        rows = [
            ("01/01/2010", "freq_27_30d", 5.0),
            ("01/01/2010", "gan_27_current", 3.0),
            ("02/01/2010", "freq_27_30d", 6.0),
        ]
        inserted = insert_features_daily(self.conn, rows)
        self.assertEqual(inserted, 3)
        self.assertEqual(
            count_rows(self.conn, ("features_daily",))["features_daily"], 3
        )

    def test_insert_features_daily_upsert_overwrites_value(self) -> None:
        insert_features_daily(
            self.conn, [("01/01/2010", "freq_27_30d", 5.0)]
        )
        # Ghi đè bằng giá trị mới - PK trùng -> UPDATE.
        insert_features_daily(
            self.conn, [("01/01/2010", "freq_27_30d", 9.5)]
        )
        features = fetch_features_for_date(self.conn, "01/01/2010")
        self.assertEqual(features["freq_27_30d"], 9.5)
        self.assertEqual(
            count_rows(self.conn, ("features_daily",))["features_daily"], 1
        )

    def test_fetch_features_for_date_groups_by_feature_name(self) -> None:
        insert_features_daily(
            self.conn,
            [
                ("01/01/2010", "freq_27_30d", 5.0),
                ("01/01/2010", "gan_27_current", 3.0),
                ("02/01/2010", "freq_27_30d", 6.0),
            ],
        )
        features = fetch_features_for_date(self.conn, "01/01/2010")
        self.assertEqual(
            features, {"freq_27_30d": 5.0, "gan_27_current": 3.0}
        )

    def test_fetch_features_for_unknown_date_returns_empty(self) -> None:
        self.assertEqual(fetch_features_for_date(self.conn, "31/12/1999"), {})

    def test_insert_features_daily_empty_returns_zero(self) -> None:
        self.assertEqual(insert_features_daily(self.conn, []), 0)


# ===========================================================================
# P1.6 - pattern_scores
# ===========================================================================
class PatternScoresTests(unittest.TestCase):
    """P1.6 - bảng ``pattern_scores`` + UPSERT + fetch_pattern_scores."""

    def setUp(self) -> None:
        self.conn: duckdb.DuckDBPyConnection = get_connection(":memory:")
        create_schema(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def test_pattern_scores_table_columns(self) -> None:
        rows = self.conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'pattern_scores'
            ORDER BY ordinal_position
            """
        ).fetchall()
        columns = {name: dtype for name, dtype in rows}
        self.assertEqual(
            list(columns.keys()),
            ["pattern_id", "hit_rate", "roi", "precision", "stability"],
        )
        self.assertEqual(columns["pattern_id"], "VARCHAR")
        for metric in ("hit_rate", "roi", "precision", "stability"):
            self.assertEqual(columns[metric], "DOUBLE")

    def test_insert_pattern_scores_basic(self) -> None:
        rows = [
            ("pascal_v1", 0.42, 1.15, 0.30, 0.80),
            ("markov_b1", 0.55, 1.25, 0.40, 0.75),
        ]
        inserted = insert_pattern_scores(self.conn, rows)
        self.assertEqual(inserted, 2)

    def test_insert_pattern_scores_upsert_overwrites_metrics(self) -> None:
        insert_pattern_scores(
            self.conn, [("pascal_v1", 0.42, 1.15, 0.30, 0.80)]
        )
        insert_pattern_scores(
            self.conn, [("pascal_v1", 0.50, 1.40, 0.35, 0.90)]
        )
        scores = fetch_pattern_scores(self.conn, pattern_id="pascal_v1")
        self.assertEqual(len(scores), 1)
        score = scores[0]
        self.assertEqual(score["hit_rate"], 0.50)
        self.assertEqual(score["roi"], 1.40)
        self.assertEqual(score["precision"], 0.35)
        self.assertEqual(score["stability"], 0.90)

    def test_fetch_pattern_scores_sorted_by_hit_rate_desc(self) -> None:
        insert_pattern_scores(
            self.conn,
            [
                ("low", 0.10, 1.0, 0.1, 0.5),
                ("high", 0.90, 2.0, 0.5, 0.9),
                ("mid", 0.50, 1.5, 0.3, 0.7),
            ],
        )
        scores = fetch_pattern_scores(self.conn)
        ids = [score["pattern_id"] for score in scores]
        self.assertEqual(ids, ["high", "mid", "low"])

    def test_fetch_pattern_scores_specific_id_not_found(self) -> None:
        self.assertEqual(
            fetch_pattern_scores(self.conn, pattern_id="missing"), []
        )

    def test_insert_pattern_scores_empty_returns_zero(self) -> None:
        self.assertEqual(insert_pattern_scores(self.conn, []), 0)


# ===========================================================================
# P1.7 - migrate_csv + integrity_check end-to-end
# ===========================================================================
class MigrateAndIntegrityTests(unittest.TestCase):
    """P1.7 - ``migrate_csv`` chạy end-to-end + ``integrity_check``."""

    def test_migrate_csv_then_integrity_check_ok(self) -> None:
        import tempfile
        from pathlib import Path

        results = make_sample_results()
        with tempfile.TemporaryDirectory() as workdir:
            csv_path = Path(workdir) / "sample.csv"
            db_path = Path(workdir) / "sample.duckdb"
            write_csv(csv_path, results)

            summary = migrate_csv(
                csv_path=str(csv_path), db_path=str(db_path)
            )
            self.assertEqual(summary["csv_rows"], 3)
            self.assertEqual(summary["inserted"], 3)
            counts = summary["table_counts"]
            assert isinstance(counts, dict)
            self.assertEqual(counts["draws"], 3)
            self.assertEqual(counts["prizes"], 27 * 3)
            # 4 bảng nguồn phải có 3 distinct days.
            conn = get_connection(str(db_path), read_only=True)
            try:
                report = integrity_check(conn)
            finally:
                conn.close()
            self.assertTrue(report["ok"], msg=str(report.get("errors")))
            self.assertEqual(report["errors"], [])
            counts_report = report["counts"]
            assert isinstance(counts_report, dict)
            for table in (
                "draws",
                "prizes",
                "loto_results",
                "digit_positions",
            ):
                self.assertEqual(counts_report[table]["days"], 3)

    def test_integrity_check_detects_missing_prize_rows(self) -> None:
        conn = get_connection(":memory:")
        try:
            create_schema(conn)
            results = make_sample_results()
            insert_results(conn, results)
            # Cố tình xóa 1 dòng prize -> integrity phải fail.
            conn.execute(
                """
                DELETE FROM prizes
                WHERE draw_date = '01/01/2010'
                  AND prize_name = 'G7'
                  AND prize_index = 0
                """
            )
            report = integrity_check(conn)
            self.assertFalse(report["ok"])
            errors = report["errors"]
            assert isinstance(errors, list)
            joined = "\n".join(errors)
            self.assertIn("prizes", joined)
        finally:
            conn.close()

    def test_integrity_check_detects_loto_mismatch(self) -> None:
        conn = get_connection(":memory:")
        try:
            create_schema(conn)
            results = make_sample_results()
            insert_results(conn, results)
            # Xoa 1 loto cu the -> khong khop voi 2 chu so cuoi cua prizes.
            conn.execute(
                """
                DELETE FROM loto_results
                WHERE draw_date = '01/01/2010' AND loto_number = '54'
                """
            )
            report = integrity_check(conn)
            self.assertFalse(report["ok"])
            errors = report["errors"]
            assert isinstance(errors, list)
            self.assertTrue(
                any("loto_results" in str(err) for err in errors),
                msg=str(errors),
            )
        finally:
            conn.close()


# ===========================================================================
# Cross-table query: kiểm tra bằng prizes + loto_results đồng bộ
# ===========================================================================
class FetchPrizesForDateTests(unittest.TestCase):
    """Verify ``fetch_prizes_for_date`` gom đúng theo prize_name."""

    def test_fetch_prizes_for_date_groups_correctly(self) -> None:
        conn = get_connection(":memory:")
        try:
            create_schema(conn)
            insert_results(conn, make_sample_results())
            prizes = fetch_prizes_for_date(conn, "01/01/2010")
            self.assertEqual(prizes["special"], ["37754"])
            self.assertEqual(prizes["G1"], ["84983"])
            self.assertEqual(prizes["G2"], ["12345", "67890"])
            self.assertEqual(
                prizes["G3"],
                ["11111", "22222", "33333", "44444", "55555", "66666"],
            )
            self.assertEqual(prizes["G7"], ["12", "34", "56", "78"])
        finally:
            conn.close()

    def test_fetch_prizes_for_unknown_date(self) -> None:
        conn = get_connection(":memory:")
        try:
            create_schema(conn)
            self.assertEqual(fetch_prizes_for_date(conn, "31/12/1999"), {})
        finally:
            conn.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
