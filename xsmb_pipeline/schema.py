from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LotteryResult:
    date: str
    region: str
    special: str
    first: List[str]
    second: List[str]
    third: List[str]
    fourth: List[str]
    fifth: List[str]
    sixth: List[str]
    seventh: List[str]


class ParseError(RuntimeError):
    pass


class FetchError(RuntimeError):
    pass


def result_to_row(result: LotteryResult) -> Dict[str, str]:
    return {
        "date": result.date,
        "region": result.region,
        "special": result.special,
        "first": "|".join(result.first),
        "second": "|".join(result.second),
        "third": "|".join(result.third),
        "fourth": "|".join(result.fourth),
        "fifth": "|".join(result.fifth),
        "sixth": "|".join(result.sixth),
        "seventh": "|".join(result.seventh),
    }


def row_to_result(row: Dict[str, str]) -> LotteryResult:
    return LotteryResult(
        date=row["date"],
        region=row["region"],
        special=row["special"],
        first=row["first"].split("|") if row["first"] else [],
        second=row["second"].split("|") if row["second"] else [],
        third=row["third"].split("|") if row["third"] else [],
        fourth=row["fourth"].split("|") if row["fourth"] else [],
        fifth=row["fifth"].split("|") if row["fifth"] else [],
        sixth=row["sixth"].split("|") if row["sixth"] else [],
        seventh=row["seventh"].split("|") if row["seventh"] else [],
    )
