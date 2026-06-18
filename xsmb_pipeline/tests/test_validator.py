from __future__ import annotations

import unittest

from xsmb_pipeline.dataset import derive_loto, flatten_numbers, parse_date
from xsmb_pipeline.schema import LotteryResult
from xsmb_pipeline.validator import (
    auto_fill_missing_dates,
    data_quality_report,
    detect_outliers,
    find_missing_dates,
    flag_missing_dates,
    missing_dates_report,
    validate_loto,
)


def _make_result(date: str, special: str = "12345") -> LotteryResult:
    return LotteryResult(
        date=date, region="XSMB", special=special,
        first=["54321"], second=["11111", "22222"],
        third=["33333", "44444", "55555", "66666", "77777", "88888"],
        fourth=["1234", "2345", "3456", "4567"],
        fifth=["1000", "2000", "3000", "4000", "5000", "6000"],
        sixth=["101", "202", "303"], seventh=["12", "23", "34", "45"],
    )


# P0.2
class FindMissingDatesTests(unittest.TestCase):
    def test_detects_single_gap(self):
        r = [_make_result("01/01/2010"), _make_result("03/01/2010")]
        self.assertEqual(find_missing_dates(r), ["02/01/2010"])

    def test_no_gap_returns_empty(self):
        r = [_make_result("01/01/2010"), _make_result("02/01/2010"), _make_result("03/01/2010")]
        self.assertEqual(find_missing_dates(r), [])

    def test_empty_results_returns_empty(self):
        self.assertEqual(find_missing_dates([]), [])


class MissingDatesReportTests(unittest.TestCase):
    def test_report_detects_gap(self):
        r = [_make_result("01/01/2010"), _make_result("03/01/2010")]
        report = missing_dates_report(r)
        self.assertEqual(report["missing_count"], 1)
        self.assertEqual(report["missing_dates"], ["02/01/2010"])
        self.assertEqual(report["dataset_size"], 2)

    def test_report_no_gap(self):
        r = [_make_result("01/01/2010"), _make_result("02/01/2010"), _make_result("03/01/2010")]
        report = missing_dates_report(r)
        self.assertEqual(report["missing_count"], 0)
        self.assertEqual(report["coverage_pct"], 100.0)

    def test_report_empty(self):
        report = missing_dates_report([])
        self.assertEqual(report["missing_count"], 0)


# P0.3
class ValidateLotoTests(unittest.TestCase):
    def test_valid_results_produce_no_issues(self):
        r = [_make_result("01/01/2010"), _make_result("02/01/2010")]
        self.assertEqual(validate_loto(r), [])

    def test_all_loto_numbers_are_2_digits(self):
        loto = derive_loto(flatten_numbers(_make_result("01/01/2010")))
        for lo in loto:
            self.assertEqual(len(lo), 2)
            self.assertTrue(lo.isdigit())


# P0.4
class DetectOutliersTests(unittest.TestCase):
    def test_uniform_data_has_no_outliers(self):
        r = [_make_result(f"0{i}/01/2010") for i in range(1, 4)]
        self.assertEqual(detect_outliers(r), [])

    def test_single_row_has_no_outliers(self):
        self.assertEqual(detect_outliers([_make_result("01/01/2010")]), [])


# P0.5
class DataQualityReportTests(unittest.TestCase):
    def test_quality_report_has_required_keys(self):
        r = [_make_result("01/01/2010"), _make_result("02/01/2010")]
        report = data_quality_report(r)
        for key in ("total_rows", "date_range", "coverage_pct", "missing_dates_count",
                    "invalid_date_format", "invalid_prize_format", "invalid_loto",
                    "loto_outliers", "prize_count_issues", "quality_score", "status"):
            self.assertIn(key, report)

    def test_clean_data_scores_ok(self):
        r = [_make_result("01/01/2010"), _make_result("02/01/2010")]
        report = data_quality_report(r)
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["invalid_prize_format"], 0)
        self.assertEqual(report["invalid_loto"], 0)


# P0.6
class FlagMissingDatesTests(unittest.TestCase):
    def test_flag_marks_march_gap_as_needs_crawl(self):
        r = [_make_result("01/03/2010"), _make_result("03/03/2010")]
        flagged = flag_missing_dates(r)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["date"], "02/03/2010")
        self.assertTrue(flagged[0]["needs_crawl"])

    def test_flag_no_missing_returns_empty(self):
        r = [_make_result("01/01/2010"), _make_result("02/01/2010"), _make_result("03/01/2010")]
        self.assertEqual(flag_missing_dates(r), [])

    def test_flag_january_gap_not_marked_for_crawl(self):
        r = [_make_result("01/01/2010"), _make_result("03/01/2010")]
        flagged = flag_missing_dates(r)
        self.assertEqual(len(flagged), 1)
        self.assertFalse(flagged[0]["needs_crawl"])


class AutoFillMissingDatesTests(unittest.TestCase):
    def test_no_missing_returns_original_results(self):
        r = [_make_result("01/01/2010"), _make_result("02/01/2010"), _make_result("03/01/2010")]
        payload = auto_fill_missing_dates(r)
        self.assertEqual(payload["missing_dates"], [])
        self.assertEqual(payload["filled_dates"], [])
        self.assertEqual(payload["failed_dates"], [])
        self.assertEqual(len(payload["results"]), 3)

    def test_payload_contains_required_keys(self):
        r = [_make_result("01/01/2010"), _make_result("03/01/2010")]
        payload = auto_fill_missing_dates(r)
        for key in ("missing_dates", "filled_dates", "failed_dates", "results"):
            self.assertIn(key, payload)

    def test_missing_dates_listed_correctly(self):
        r = [_make_result("01/01/2010"), _make_result("03/01/2010")]
        payload = auto_fill_missing_dates(r)
        self.assertEqual(payload["missing_dates"], ["02/01/2010"])

    def test_results_is_list_of_lottery_results(self):
        r = [_make_result("01/01/2010"), _make_result("03/01/2010")]
        payload = auto_fill_missing_dates(r)
        for item in payload["results"]:
            self.assertIsInstance(item, LotteryResult)

    def test_filled_plus_failed_equals_missing(self):
        r = [_make_result("01/01/2010"), _make_result("03/01/2010")]
        payload = auto_fill_missing_dates(r)
        total = len(payload["filled_dates"]) + len(payload["failed_dates"])
        self.assertEqual(total, len(payload["missing_dates"]))

    def test_results_are_sorted_by_date(self):
        r = [_make_result("01/01/2010"), _make_result("03/01/2010")]
        payload = auto_fill_missing_dates(r)
        dates = [item.date for item in payload["results"]]
        self.assertEqual(dates, sorted(dates, key=parse_date))


if __name__ == "__main__":
    unittest.main()