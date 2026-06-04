from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple

from ..schema import LotteryResult


@dataclass
class RankingPrediction:
    model: str
    target: str
    top_predictions: List[Tuple[str, float]]
    frequency_top_predictions: List[Tuple[str, float]]
    metadata: Dict[str, object]


class RankingStrategy(Protocol):
    def fit(self, results: Sequence[LotteryResult], target_name: str, top_k: int): ...
