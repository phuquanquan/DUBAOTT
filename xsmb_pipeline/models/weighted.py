from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

from ..features import bridge_frequency, bridge_streak, digit_part_frequency, digit_position_frequency, digit_transition_score, gap_since_last_seen, head_frequency, recent_long_term_delta, recency_decay_score, repeated_digit_ratio, rolling_frequency, tail_frequency, unique_digit_ratio
from ..schema import LotteryResult
from ..signals import ensemble_signal_score
from ..targets import actual_targets, special_items_for_history, target_width
from .base import RankingPrediction

ROLLING_WINDOWS = (7, 30, 60, 90, 200)
FEATURE_WEIGHTS = {
    "loto2": {
        "history": 0.14,
        "window_7": 0.30,
        "window_30": 0.22,
        "window_60": 0.10,
        "window_90": 0.06,
        "window_200": 0.03,
        "special_history": 0.01,
        "special_window_30": 0.0,
        "recency": 0.14,
        "recency_decay": 0.12,
        "window_14": 0.12,
        "recent_delta": 0.10,
        "gap_penalty": -0.02,
        "bridge_frequency": 0.0,
        "bridge_streak": 0.0,
        "digit_position": 0.0,
        "digit_parts": 0.0,
        "digit_transition": 0.0,
        "head_frequency": 0.10,
        "tail_frequency": 0.12,
        "repeat_ratio": 0.04,
        "unique_ratio": 0.03,
        "signal_ensemble": 0.14,
    },
    "loto3": {
        "history": 0.08,
        "window_7": 0.18,
        "window_30": 0.14,
        "window_60": 0.10,
        "window_90": 0.08,
        "window_200": 0.05,
        "special_history": 0.06,
        "special_window_30": 0.04,
        "recency": 0.08,
        "bridge_frequency": 0.12,
        "bridge_streak": 0.05,
        "digit_position": 0.06,
        "digit_parts": 0.08,
        "digit_transition": 0.06,
        "signal_ensemble": 0.16,
    },
    "special2": {
        "history": 0.18,
        "window_7": 0.18,
        "window_30": 0.22,
        "window_60": 0.12,
        "window_90": 0.08,
        "window_200": 0.05,
        "special_history": 0.28,
        "special_window_30": 0.12,
        "recency": 0.10,
        "bridge_frequency": 0.08,
        "bridge_streak": 0.03,
        "digit_position": 0.03,
        "digit_parts": 0.04,
        "digit_transition": 0.02,
        "signal_ensemble": 0.10,
    },
    "special3": {
        "history": 0.08,
        "window_7": 0.10,
        "window_30": 0.16,
        "window_60": 0.12,
        "window_90": 0.10,
        "window_200": 0.08,
        "special_history": 0.42,
        "special_window_30": 0.22,
        "recency": 0.08,
        "bridge_frequency": 0.10,
        "bridge_streak": 0.04,
        "digit_position": 0.04,
        "digit_parts": 0.05,
        "digit_transition": 0.03,
        "signal_ensemble": 0.14,
    },
}


@dataclass
class RankingModel:
    target: str
    number_width: int
    top_k: int
    weights: Dict[str, float]
    scores: Dict[str, float]
    counts: Dict[str, int]
    total: int
    baseline_scores: Dict[str, float]

    def predict(self) -> List[Tuple[str, float]]:
        scored = sorted(self.scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return scored[: self.top_k]

    def predict_baseline(self) -> List[Tuple[str, float]]:
        scored = sorted(self.baseline_scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return scored[: self.top_k]


@dataclass
class RankingEvaluation:
    target: str
    model: str
    train_size: int
    test_size: int
    top_k: int
    hit_rate: float
    baseline_hit_rate: float
    frequency_hit_rate: float
    precision_at_k: float
    baseline_precision_at_k: float
    frequency_precision_at_k: float
    hit_rate_pct: float
    baseline_hit_rate_pct: float
    frequency_hit_rate_pct: float
    precision_at_k_pct: float
    baseline_precision_at_k_pct: float
    frequency_precision_at_k_pct: float


def candidate_universe(number_width: int) -> List[str]:
    return [f"{i:0{number_width}d}" for i in range(10 ** number_width)]


def build_target_counts(results: Sequence[LotteryResult], target_name: str) -> Tuple[Counter[str], int, int]:
    width = target_width(target_name)
    counts: Counter[str] = Counter()
    total = 0
    for result in results:
        for item in actual_targets(result, target_name):
            counts[item] += 1
            total += 1
    return counts, total, width


def window_slice(results: Sequence[LotteryResult], days: int) -> Sequence[LotteryResult]:
    return results[-days:] if len(results) > days else results


def recency_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> float:
    history = list(results)
    for offset, result in enumerate(reversed(history), start=1):
        if candidate in actual_targets(result, target_name):
            return 1.0 / offset
    return 0.0


def normalized_frequency(results: Sequence[LotteryResult], candidate: str, target_name: str) -> float:
    occurrences = 0
    total = 0
    for result in results:
        items = actual_targets(result, target_name)
        occurrences += items.count(candidate)
        total += len(items)
    return occurrences / total if total else 0.0


def normalized_special_frequency(results: Sequence[LotteryResult], candidate: str, number_width: int) -> float:
    occurrences = 0
    total = 0
    for result in results:
        items = special_items_for_history(result, number_width)
        occurrences += items.count(candidate)
        total += len(items)
    return occurrences / total if total else 0.0


def score_candidate(results: Sequence[LotteryResult], candidate: str, target_name: str, number_width: int, weights: Dict[str, float]) -> float:
    rows = list(results)
    score = 0.0
    score += weights.get("history", 0.0) * normalized_frequency(rows, candidate, target_name)
    for window in ROLLING_WINDOWS:
        window_results = window_slice(rows, window)
        score += weights.get(f"window_{window}", 0.0) * normalized_frequency(window_results, candidate, target_name)
    score += weights.get("window_14", 0.0) * rolling_frequency(rows, candidate, target_name, 14)
    score += weights.get("special_history", 0.0) * normalized_special_frequency(rows, candidate, number_width)
    score += weights.get("special_window_30", 0.0) * normalized_special_frequency(window_slice(rows, 30), candidate, number_width)
    score += weights.get("recency", 0.0) * recency_score(rows, candidate, target_name)
    score += weights.get("recency_decay", 0.0) * recency_decay_score(rows, candidate, target_name)
    score += weights.get("recent_delta", 0.0) * recent_long_term_delta(rows, candidate, target_name)
    score += weights.get("gap_penalty", 0.0) * gap_since_last_seen(rows, candidate, target_name)
    score += weights.get("bridge_frequency", 0.0) * bridge_frequency(rows, candidate)
    score += weights.get("bridge_streak", 0.0) * bridge_streak(rows, candidate)
    score += weights.get("digit_position", 0.0) * digit_position_frequency(rows, candidate, target_name)
    score += weights.get("digit_parts", 0.0) * digit_part_frequency(rows, candidate, target_name)
    score += weights.get("digit_transition", 0.0) * digit_transition_score(rows, candidate, target_name)
    score += weights.get("head_frequency", 0.0) * head_frequency(rows, candidate, target_name)
    score += weights.get("tail_frequency", 0.0) * tail_frequency(rows, candidate, target_name)
    score += weights.get("repeat_ratio", 0.0) * repeated_digit_ratio(candidate)
    score += weights.get("unique_ratio", 0.0) * unique_digit_ratio(candidate)
    score += weights.get("signal_ensemble", 0.0) * ensemble_signal_score(rows, candidate, target_name)
    return score


def train_ranking_model(results: Sequence[LotteryResult], target_name: str, top_k: int, weights: Dict[str, float] | None = None) -> RankingModel:
    counts, total, width = build_target_counts(results, target_name)
    active_weights = weights or FEATURE_WEIGHTS[target_name]
    baseline_scores = {}
    scores = {}
    for candidate in candidate_universe(width):
        baseline_scores[candidate] = counts.get(candidate, 0) / total if total else 0.0
        scores[candidate] = score_candidate(results, candidate, target_name, width, active_weights)
    return RankingModel(
        target=target_name,
        number_width=width,
        top_k=top_k,
        weights=active_weights,
        scores=scores,
        counts=dict(counts),
        total=total,
        baseline_scores=baseline_scores,
    )


def score_weight_config(results: Sequence[LotteryResult], target_name: str, top_k: int, weights: Dict[str, float], min_train_size: int = 30) -> float:
    if len(results) <= min_train_size:
        return 0.0
    hit_total = 0.0
    precision_total = 0.0
    baseline_hit_total = 0.0
    steps = 0
    width = target_width(target_name)
    universe_size = 10 ** width
    for split_index in range(min_train_size, len(results)):
        train = list(results[:split_index])
        test = results[split_index]
        model = train_ranking_model(train, target_name=target_name, top_k=top_k, weights=weights)
        predicted = [number for number, _ in model.predict()]
        actual = actual_targets(test, target_name)
        hit, baseline_hit, precision, _ = evaluate_prediction_set(predicted, actual, universe_size)
        hit_total += hit
        baseline_hit_total += baseline_hit
        precision_total += precision
        steps += 1
    if not steps:
        return 0.0
    hit_rate = hit_total / steps
    precision_rate = precision_total / steps
    baseline_hit_rate = baseline_hit_total / steps
    return hit_rate * 0.65 + precision_rate * 0.30 + (hit_rate - baseline_hit_rate) * 0.05


def loto2_weight_candidates() -> Dict[str, List[float]]:
    return {
        "history": [0.05, 0.10, 0.15, 0.20, 0.25],
        "window_7": [0.10, 0.15, 0.20, 0.25, 0.30],
        "window_14": [0.0, 0.05, 0.10, 0.15, 0.20],
        "window_30": [0.05, 0.10, 0.15, 0.20, 0.25],
        "window_60": [0.0, 0.03, 0.05, 0.08, 0.10],
        "recency": [0.05, 0.10, 0.15, 0.20],
        "recency_decay": [0.0, 0.05, 0.10, 0.15, 0.20],
        "recent_delta": [0.0, 0.05, 0.10, 0.15],
        "gap_penalty": [-0.08, -0.05, -0.03, -0.01, 0.0],
        "head_frequency": [0.0, 0.05, 0.10, 0.15],
        "tail_frequency": [0.0, 0.05, 0.10, 0.15, 0.20],
        "repeat_ratio": [0.0, 0.02, 0.04, 0.06],
        "unique_ratio": [0.0, 0.02, 0.04, 0.06],
        "signal_ensemble": [0.0, 0.05, 0.10, 0.15, 0.20],
    }


def tune_loto2_weights(results: Sequence[LotteryResult], top_k: int, min_train_size: int = 30) -> Dict[str, float]:
    base = dict(FEATURE_WEIGHTS["loto2"])
    best_weights = dict(base)
    best_score = score_weight_config(results, "loto2", top_k, best_weights, min_train_size=min_train_size)
    for key, values in loto2_weight_candidates().items():
        local_best_weights = dict(best_weights)
        local_best_score = best_score
        for value in values:
            trial = dict(best_weights)
            trial[key] = value
            score = score_weight_config(results, "loto2", top_k, trial, min_train_size=min_train_size)
            if score > local_best_score:
                local_best_score = score
                local_best_weights = trial
        best_weights = local_best_weights
        best_score = local_best_score
    return best_weights


def evaluate_prediction_set(predicted: Sequence[str], actual: Sequence[str], universe_size: int) -> Tuple[int, float, float, float]:
    predicted_set = set(predicted)
    actual_set = set(actual)
    overlap = len(predicted_set & actual_set)
    hit = 1 if overlap > 0 else 0
    actual_count = len(actual_set)
    miss_probability = ((universe_size - actual_count) / universe_size) ** len(predicted_set) if universe_size and predicted_set else 1.0
    baseline_hit = 1.0 - miss_probability
    baseline_precision = actual_count / universe_size if universe_size else 0.0
    precision = overlap / max(1, len(predicted_set))
    return hit, baseline_hit, precision, baseline_precision


def fit_best_loto2_model(results: Sequence[LotteryResult], top_k: int, min_train_size: int = 30) -> RankingModel:
    tuned_weights = tune_loto2_weights(results, top_k=top_k, min_train_size=min_train_size)
    return train_ranking_model(results, target_name="loto2", top_k=top_k, weights=tuned_weights)


def compare_loto2_weight_strategies(results: Sequence[LotteryResult], top_k: int, min_train_size: int = 30) -> Dict[str, object]:
    base_score = score_weight_config(results, "loto2", top_k, FEATURE_WEIGHTS["loto2"], min_train_size=min_train_size)
    tuned_weights = tune_loto2_weights(results, top_k=top_k, min_train_size=min_train_size)
    tuned_score = score_weight_config(results, "loto2", top_k, tuned_weights, min_train_size=min_train_size)
    return {
        "base_score": base_score,
        "tuned_score": tuned_score,
        "weights": tuned_weights,
    }


def score_weight_config_by_grid(results: Sequence[LotteryResult], target_name: str, top_k: int, weights: Dict[str, float], min_train_size: int = 30) -> float:
    return score_weight_config(results, target_name, top_k, weights, min_train_size=min_train_size)


def tune_weights(results: Sequence[LotteryResult], target_name: str, top_k: int, min_train_size: int = 30) -> Dict[str, float]:
    if target_name == "loto2":
        return tune_loto2_weights(results, top_k=top_k, min_train_size=min_train_size)
    if len(results) <= min_train_size:
        return dict(FEATURE_WEIGHTS[target_name])
    base = FEATURE_WEIGHTS[target_name]
    candidate_keys = ["history", "window_7", "window_30", "special_history", "recency"]
    candidate_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    best_weights = dict(base)
    best_score = score_weight_config_by_grid(results, target_name, top_k, best_weights, min_train_size=min_train_size)

    for values in product(candidate_values, repeat=len(candidate_keys)):
        trial = dict(base)
        for key, value in zip(candidate_keys, values):
            trial[key] = value
        score = score_weight_config_by_grid(results, target_name, top_k, trial, min_train_size=min_train_size)
        if score > best_score:
            best_score = score
            best_weights = trial
    return best_weights


def fit_tuned_ranking_model(results: Sequence[LotteryResult], target_name: str, top_k: int, min_train_size: int = 30) -> RankingModel:
    if target_name == "loto2":
        return fit_best_loto2_model(results, top_k=top_k, min_train_size=min_train_size)
    tuned_weights = tune_weights(results, target_name=target_name, top_k=top_k, min_train_size=min_train_size)
    return train_ranking_model(results, target_name=target_name, top_k=top_k, weights=tuned_weights)


def build_prediction(model_name: str, model: RankingModel, metadata: Dict[str, object] | None = None) -> RankingPrediction:
    return RankingPrediction(
        model=model_name,
        target=model.target,
        top_predictions=model.predict(),
        frequency_top_predictions=model.predict_baseline(),
        metadata=metadata or {},
    )


def predict_next_day(results: Sequence[LotteryResult], top_k_loto2: int, top_k_loto3: int, top_k_special2: int, top_k_special3: int, tuned: bool = False) -> Dict[str, object]:
    trainer = fit_tuned_ranking_model if tuned else train_ranking_model
    model_name = "tuned-weighted-ranking" if tuned else "weighted-ranking"
    loto2_model = trainer(results, target_name="loto2", top_k=top_k_loto2, min_train_size=30) if tuned else trainer(results, target_name="loto2", top_k=top_k_loto2)
    loto3_model = trainer(results, target_name="loto3", top_k=top_k_loto3, min_train_size=30) if tuned else trainer(results, target_name="loto3", top_k=top_k_loto3)
    special2_model = trainer(results, target_name="special2", top_k=top_k_special2, min_train_size=30) if tuned else trainer(results, target_name="special2", top_k=top_k_special2)
    special3_model = trainer(results, target_name="special3", top_k=top_k_special3, min_train_size=30) if tuned else trainer(results, target_name="special3", top_k=top_k_special3)
    return {
        "train_size": len(results),
        "model": model_name,
        "loto2_top": loto2_model.predict(),
        "loto3_top": loto3_model.predict(),
        "special2_top": special2_model.predict(),
        "special3_top": special3_model.predict(),
        "loto2_frequency_top": loto2_model.predict_baseline(),
        "loto3_frequency_top": loto3_model.predict_baseline(),
        "special2_frequency_top": special2_model.predict_baseline(),
        "special3_frequency_top": special3_model.predict_baseline(),
        "loto2_weights": loto2_model.weights,
        "loto3_weights": loto3_model.weights,
        "special2_weights": special2_model.weights,
        "special3_weights": special3_model.weights,
    }

