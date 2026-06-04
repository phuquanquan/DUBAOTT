from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .schema import LotteryResult, result_to_row, row_to_result

DATE_FMT = "%d/%m/%Y"


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, DATE_FMT)


def format_date(value: datetime) -> str:
    return value.strftime(DATE_FMT)


def iter_dates(start: str, end: str) -> Iterable[str]:
    current = parse_date(start)
    final = parse_date(end)
    step = timedelta(days=1)
    while current <= final:
        yield format_date(current)
        current += step


def write_json(path: Path, results: Sequence[LotteryResult]) -> None:
    path.write_text(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, results: Sequence[LotteryResult]) -> None:
    if not results:
        raise ValueError("Không có dữ liệu để ghi CSV")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_to_row(results[0]).keys()))
        writer.writeheader()
        for item in results:
            writer.writerow(result_to_row(item))


def load_csv(path: Path) -> List[LotteryResult]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row_to_result(row) for row in reader]


def sort_results(results: Sequence[LotteryResult]) -> List[LotteryResult]:
    return sorted(results, key=lambda item: parse_date(item.date))


def dedupe_results(results: Sequence[LotteryResult]) -> List[LotteryResult]:
    by_date: Dict[str, LotteryResult] = {}
    for result in results:
        by_date[result.date] = result
    return sort_results(by_date.values())


def merge_results(existing: Sequence[LotteryResult], new: Sequence[LotteryResult]) -> List[LotteryResult]:
    return dedupe_results([*existing, *new])


def flatten_numbers(result: LotteryResult) -> List[str]:
    values = [result.special]
    for group in (result.first, result.second, result.third, result.fourth, result.fifth, result.sixth, result.seventh):
        values.extend(group)
    return values


def derive_loto(numbers: Sequence[str]) -> List[str]:
    return [number[-2:] for number in numbers if len(number) >= 2]
