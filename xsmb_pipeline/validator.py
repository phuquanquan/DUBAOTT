from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List

from .dataset import DATE_FMT
from .schema import LotteryResult
from .scraper import fetch_results_for_dates

# Số lượng và độ dài chữ số chuẩn theo từng giải XSMB
PRIZE_SPECS = {
    "special": {"count": 1, "digits": 5},
    "first":   {"count": 1, "digits": 5},
    "second":  {"count": 2, "digits": 5},
    "third":   {"count": 6, "digits": 5},
    "fourth":  {"count": 4, "digits": 4},
    "fifth":   {"count": 6, "digits": 4},
    "sixth":   {"count": 3, "digits": 3},
    "seventh": {"count": 4, "digits": 2},
}


@dataclass
class PrizeError:
    prize: str
    index: int
    value: str
    reason: str

    def __str__(self) -> str:
        return f"{self.prize}[{self.index}]={self.value!r}: {self.reason}"


@dataclass
class ValidationResult:
    date: str
    valid: bool
    errors: List[PrizeError] = field(default_factory=list)

    def __str__(self) -> str:
        if self.valid:
            return f"{self.date}: OK"
        lines = [f"{self.date}: {len(self.errors)} lỗi"]
        for error in self.errors:
            lines.append(f"  - {error}")
        return "\n".join(lines)


def validate_prize_list(prize_name: str, values: List[str], expected_count: int, expected_digits: int) -> List[PrizeError]:
    """Validate số lượng và độ dài từng số trong một giải."""
    errors: List[PrizeError] = []
    if len(values) != expected_count:
        errors.append(PrizeError(
            prize=prize_name,
            index=-1,
            value=str(values),
            reason=f"cần {expected_count} số, có {len(values)}",
        ))
    for index, value in enumerate(values):
        if not value.isdigit():
            errors.append(PrizeError(
                prize=prize_name,
                index=index,
                value=value,
                reason="không phải chữ số",
            ))
        elif len(value) != expected_digits:
            errors.append(PrizeError(
                prize=prize_name,
                index=index,
                value=value,
                reason=f"cần {expected_digits} chữ số, có {len(value)}",
            ))
    return errors


def validate_result(result: LotteryResult) -> ValidationResult:
    """Validate toàn bộ format của một kết quả xổ số."""
    errors: List[PrizeError] = []
    prize_values = {
        "special": [result.special],
        "first":   result.first,
        "second":  result.second,
        "third":   result.third,
        "fourth":  result.fourth,
        "fifth":   result.fifth,
        "sixth":   result.sixth,
        "seventh": result.seventh,
    }
    for prize_name, spec in PRIZE_SPECS.items():
        values = prize_values[prize_name]
        errors.extend(validate_prize_list(
            prize_name=prize_name,
            values=values,
            expected_count=spec["count"],
            expected_digits=spec["digits"],
        ))
    return ValidationResult(date=result.date, valid=len(errors) == 0, errors=errors)


def validate_results(results: List[LotteryResult]) -> List[ValidationResult]:
    """Validate danh sách kết quả, trả về tất cả ValidationResult."""
    return [validate_result(result) for result in results]


def filter_invalid(results: List[LotteryResult]) -> List[ValidationResult]:
    """Chỉ trả về các kết quả có lỗi."""
    return [vr for vr in validate_results(results) if not vr.valid]


def assert_all_valid(results: List[LotteryResult]) -> None:
    """Raise ValueError nếu bất kỳ kết quả nào không hợp lệ."""
    invalid = filter_invalid(results)
    if invalid:
        lines = [f"{len(invalid)} ngày có lỗi:"]
        for vr in invalid[:20]:
            lines.append(str(vr))
        if len(invalid) > 20:
            lines.append(f"  ... và {len(invalid) - 20} ngày nữa")
        raise ValueError("\n".join(lines))


def find_missing_dates(results: List[LotteryResult], start_date: str | None = None, end_date: str | None = None) -> List[str]:
    """Tìm các ngày bị thiếu trong dataset giữa start_date và end_date.

    XSMB xổ mỗi ngày kể cả Chủ Nhật — không loại trừ ngày nào.
    Nếu không truyền start_date/end_date thì tự lấy min/max từ dataset.
    """
    if not results:
        return []
    existing: set[str] = {result.date for result in results}
    if start_date is None:
        start_date = min(existing, key=lambda d: _to_iso(d))
    if end_date is None:
        end_date = max(existing, key=lambda d: _to_iso(d))
    current = date.fromisoformat(_to_iso(start_date))
    last = date.fromisoformat(_to_iso(end_date))
    missing: List[str] = []
    while current <= last:
        formatted = current.strftime(DATE_FMT)
        if formatted not in existing:
            missing.append(formatted)
        current += timedelta(days=1)
    return missing


def _to_iso(date_str: str) -> str:
    """Chuyển DD/MM/YYYY sang YYYY-MM-DD để so sánh."""
    day, month, year = date_str.split("/")
    return f"{year}-{month}-{day}"


def missing_dates_report(results: List[LotteryResult]) -> Dict[str, object]:
    """Báo cáo tổng hợp ngày missing, tự lấy min/max từ dataset."""
    if not results:
        return {"missing_count": 0, "missing_dates": [], "coverage_pct": 0.0}
    existing: set[str] = {result.date for result in results}
    start_date = min(existing, key=lambda d: _to_iso(d))
    end_date = max(existing, key=lambda d: _to_iso(d))
    missing = find_missing_dates(results, start_date=start_date, end_date=end_date)
    start = date.fromisoformat(_to_iso(start_date))
    end = date.fromisoformat(_to_iso(end_date))
    total_days = (end - start).days + 1
    coverage_pct = round(len(existing) / total_days * 100, 2) if total_days > 0 else 0.0
    return {
        "start": start_date,
        "end": end_date,
        "dataset_size": len(results),
        "total_days_in_range": total_days,
        "missing_count": len(missing),
        "missing_dates": missing,
        "coverage_pct": coverage_pct,
    }


def normalize_date(date_str: str) -> str:
    parts = date_str.strip().split("/")
    if len(parts) != 3:
        raise ValueError(f"Ngay khong hop le: {date_str}")
    day, month, year = parts
    return f"{int(day):02d}/{int(month):02d}/{year}"


def is_valid_date_format(date_str: str) -> bool:
    import re as _re
    return bool(_re.match(r"^\d{2}/\d{2}/\d{4}$", date_str))


def validate_loto(results: List[LotteryResult]) -> List[Dict[str, object]]:
    from .dataset import derive_loto, flatten_numbers
    issues: List[Dict[str, object]] = []
    for result in results:
        loto = derive_loto(flatten_numbers(result))
        bad = [lo for lo in loto if not lo.isdigit() or len(lo) != 2]
        if bad:
            issues.append({"date": result.date, "invalid_loto": bad})
    return issues


def normalize_results(results: List[LotteryResult]) -> List[LotteryResult]:
    normalized = []
    for result in results:
        normalized.append(LotteryResult(
            date=normalize_date(result.date),
            region=result.region,
            special=result.special,
            first=result.first,
            second=result.second,
            third=result.third,
            fourth=result.fourth,
            fifth=result.fifth,
            sixth=result.sixth,
            seventh=result.seventh,
        ))
    return normalized


def full_quality_report(results: List[LotteryResult]) -> Dict[str, object]:
    bad_dates = [r.date for r in results if not is_valid_date_format(r.date)]
    bad_prizes = filter_invalid(results)
    bad_loto = validate_loto(results)
    missing = missing_dates_report(results)
    quality = round((1 - (len(bad_prizes) + len(bad_loto)) / max(len(results), 1)) * 100, 2)
    start = missing.get("start", "?")
    end = missing.get("end", "?")
    return {
        "total_rows": len(results),
        "invalid_date_format": len(bad_dates),
        "invalid_prize_format": len(bad_prizes),
        "invalid_loto": len(bad_loto),
        "missing_dates": missing["missing_count"],
        "coverage_pct": missing["coverage_pct"],
        "date_range": start + " -> " + end,
        "quality_score": quality,
    }


def detect_outliers(results: List[LotteryResult], zscore_threshold: float = 3.0) -> List[Dict[str, object]]:
    from .dataset import derive_loto, flatten_numbers
    from statistics import mean, stdev
    if len(results) < 2:
        return []
    counts = [(result.date, len(derive_loto(flatten_numbers(result)))) for result in results]
    values = [c for _, c in counts]
    avg = mean(values)
    std = stdev(values) if len(values) > 1 else 0.0
    outliers: List[Dict[str, object]] = []
    for date_str, count in counts:
        zscore = abs(count - avg) / std if std > 0 else 0.0
        if zscore > zscore_threshold:
            outliers.append({
                "date": date_str,
                "loto_count": count,
                "expected": avg,
                "zscore": round(zscore, 2),
                "reason": "qua it lo to" if count < avg else "qua nhieu lo to",
            })
    return outliers


def prize_count_outliers(results: List[LotteryResult]) -> List[Dict[str, object]]:
    issues: List[Dict[str, object]] = []
    for result in results:
        prize_map = {
            "special": ([result.special], 1),
            "first": (result.first, 1),
            "second": (result.second, 2),
            "third": (result.third, 6),
            "fourth": (result.fourth, 4),
            "fifth": (result.fifth, 6),
            "sixth": (result.sixth, 3),
            "seventh": (result.seventh, 4),
        }
        for prize_name, (values, expected) in prize_map.items():
            if len(values) != expected:
                issues.append({
                    "date": result.date,
                    "prize": prize_name,
                    "expected_count": expected,
                    "actual_count": len(values),
                })
    return issues


def data_quality_report(results: List[LotteryResult]) -> Dict[str, object]:
    bad_dates = [r.date for r in results if not is_valid_date_format(r.date)]
    bad_prizes = filter_invalid(results)
    bad_loto = validate_loto(results)
    outliers = detect_outliers(results)
    prize_issues = prize_count_outliers(results)
    missing = missing_dates_report(results)
    quality = round((1 - (len(bad_prizes) + len(bad_loto) + len(outliers)) / max(len(results), 1)) * 100, 2)
    start = missing.get("start", "?")
    end = missing.get("end", "?")
    return {
        "total_rows": len(results),
        "date_range": start + " -> " + end,
        "coverage_pct": missing["coverage_pct"],
        "missing_dates_count": missing["missing_count"],
        "missing_dates": missing["missing_dates"][:20],
        "invalid_date_format": len(bad_dates),
        "invalid_prize_format": len(bad_prizes),
        "invalid_loto": len(bad_loto),
        "loto_outliers": len(outliers),
        "prize_count_issues": len(prize_issues),
        "quality_score": quality,
        "status": "OK" if quality == 100.0 else "NEEDS_REVIEW",
    }


def flag_missing_dates(results: List[LotteryResult]) -> List[Dict[str, object]]:
    missing = find_missing_dates(results)
    flagged: List[Dict[str, object]] = []
    for date_str in missing:
        day, month, year = date_str.split("/")
        month_int = int(month)
        is_tet = month_int == 1 or (month_int == 2 and int(day) <= 20)
        flagged.append({
            "date": date_str,
            "reason": "Tet Nguyen Dan (binh thuong)" if is_tet else "Khong ro nguyen nhan - can kiem tra",
            "needs_crawl": not is_tet,
        })
    return flagged


def auto_fill_missing_dates(results: List[LotteryResult]) -> Dict[str, object]:
    """Thử crawl và bù các ngày thiếu trong dataset.

    Trả về payload mô tả ngày nào được thêm mới, ngày nào lỗi và dataset sau khi ghép.
    """
    missing = find_missing_dates(results)
    if not missing:
        return {
            "missing_dates": [],
            "filled_dates": [],
            "failed_dates": [],
            "results": results,
        }

    fetched = fetch_results_for_dates(missing)
    filled_dates = [item.date for item in fetched]
    failed_dates = [date_str for date_str in missing if date_str not in set(filled_dates)]
    merged = sorted({item.date: item for item in [*results, *fetched]}.values(), key=lambda item: date.fromisoformat(_to_iso(item.date)))
    return {
        "missing_dates": missing,
        "filled_dates": filled_dates,
        "failed_dates": failed_dates,
        "results": merged,
    }
