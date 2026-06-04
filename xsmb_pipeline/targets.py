from __future__ import annotations

from typing import List

from .dataset import derive_loto, flatten_numbers
from .schema import LotteryResult


def loto_2d_targets(result: LotteryResult) -> List[str]:
    return sorted({item for item in derive_loto(flatten_numbers(result)) if len(item) == 2})


def loto_3d_targets(result: LotteryResult) -> List[str]:
    return sorted({number[-3:] for number in flatten_numbers(result) if len(number) >= 3})


def special_2d_target(result: LotteryResult) -> List[str]:
    return [result.special[-2:]]


def special_3d_target(result: LotteryResult) -> List[str]:
    return [result.special[-3:]]


def actual_targets(result: LotteryResult, target_name: str) -> List[str]:
    if target_name == "loto2":
        return loto_2d_targets(result)
    if target_name == "loto3":
        return loto_3d_targets(result)
    if target_name == "special2":
        return special_2d_target(result)
    if target_name == "special3":
        return special_3d_target(result)
    raise ValueError(f"Target không hỗ trợ: {target_name}")


def target_width(target_name: str) -> int:
    if target_name in {"loto2", "special2"}:
        return 2
    if target_name in {"loto3", "special3"}:
        return 3
    raise ValueError(f"Target không hỗ trợ: {target_name}")


def special_items_for_history(result: LotteryResult, number_width: int) -> List[str]:
    if number_width == 2:
        return [result.special[-2:]]
    return [result.special[-3:]]


def target_choices() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def compare_target_choices() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def prediction_target_choices() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def feature_enabled_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def weighted_target_choices() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def sklearn_target_choices() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def default_compare_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def prediction_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def cli_target_choices() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def benchmark_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def supported_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def compare_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def prediction_output_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def model_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def default_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def all_target_choices() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def default_compare_target_sequence() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def supported_target_names() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def compare_model_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def cli_target_names() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def prediction_cli_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def enabled_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def target_names() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def compare_defaults() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def prediction_defaults() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def target_cli_choices() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]


def available_targets() -> List[str]:
    return ["loto2", "loto3", "special2", "special3"]
