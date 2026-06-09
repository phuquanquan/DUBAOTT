from __future__ import annotations

from typing import List

from .dataset import derive_loto, flatten_numbers
from .schema import LotteryResult


def loto_2d_targets(result: LotteryResult) -> List[str]:
    return sorted({item for item in derive_loto(flatten_numbers(result)) if len(item) == 2})


def actual_targets(result: LotteryResult, target_name: str) -> List[str]:
    if target_name == "loto2":
        return loto_2d_targets(result)
    raise ValueError(f"Target không hỗ trợ: {target_name}")


def target_width(target_name: str) -> int:
    if target_name == "loto2":
        return 2
    raise ValueError(f"Target không hỗ trợ: {target_name}")


def special_items_for_history(result: LotteryResult, number_width: int) -> List[str]:
    return [result.special[-2:]]
