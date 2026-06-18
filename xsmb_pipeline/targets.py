from __future__ import annotations

from functools import lru_cache
from typing import List

from .dataset import derive_loto, flatten_numbers
from .schema import LotteryResult


@lru_cache(maxsize=8192)
def loto_2d_targets(result: LotteryResult) -> tuple[str, ...]:
    return tuple(sorted({item for item in derive_loto(flatten_numbers(result)) if len(item) == 2}))


def dau_targets(result: LotteryResult) -> List[str]:
    return list({item[0] for item in loto_2d_targets(result) if item})


def duoi_targets(result: LotteryResult) -> List[str]:
    return list({item[-1] for item in loto_2d_targets(result) if item})


def cham_targets(result: LotteryResult) -> List[str]:
    return list({digit for item in loto_2d_targets(result) for digit in item})


def tong_targets(result: LotteryResult) -> List[str]:
    return list({str(sum(int(ch) for ch in item) % 10) for item in loto_2d_targets(result) if item})


def so00_99_targets(result: LotteryResult) -> List[str]:
    return list(loto_2d_targets(result))


@lru_cache(maxsize=8192)
def actual_targets(result: LotteryResult, target_name: str) -> tuple[str, ...]:
    if target_name == "loto2":
        return loto_2d_targets(result)
    if target_name == "dau":
        return tuple(dau_targets(result))
    if target_name == "duoi":
        return tuple(duoi_targets(result))
    if target_name == "cham":
        return tuple(cham_targets(result))
    if target_name == "tong":
        return tuple(tong_targets(result))
    if target_name == "so00_99":
        return loto_2d_targets(result)
    raise ValueError(f"Target khong ho tro: {target_name}")


def target_width(target_name: str) -> int:
    if target_name in ("loto2", "so00_99"):
        return 2
    if target_name in ("dau", "duoi", "cham", "tong"):
        return 1
    raise ValueError(f"Target khong ho tro: {target_name}")


def special_items_for_history(result: LotteryResult, number_width: int) -> List[str]:
    return [result.special[-2:]]
