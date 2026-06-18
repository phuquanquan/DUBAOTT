from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple
import multiprocessing as mp

from ..features import (
    bridge_frequency,
    bridge_streak,
    decay_weighted_frequency,
    digit_part_frequency,
    digit_position_frequency,
    digit_transition_score,
    gap_since_last_seen,
    head_frequency,
    recent_long_term_delta,
    recency_cluster_score,
    recency_decay_score,
    recency_gap_ratio,
    recent_peak_frequency,
    repeated_digit_ratio,
    rolling_frequency,
    tail_frequency,
    unique_digit_ratio,
)
from ..schema import LotteryResult
from ..signals import ensemble_signal_score
from ..targets import actual_targets, special_items_for_history, target_width
from .base import RankingPrediction

ROLLING_WINDOWS = (7, 30, 60, 90, 200)
FEATURE_WEIGHTS: Dict[str, Dict[str, float]] = {
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
        "signal_ensemble": 0.22,
    },
}


# =========================================================================
# Candidate universe
# =========================================================================
def candidate_universe(number_width: int) -> List[str]:
    return [f"{i:0{number_width}d}" for i in range(10**number_width)]


# =========================================================================
# Low-level per-candidate feature functions
# =========================================================================
def _norm_freq(rows: Sequence[LotteryResult], candidate: str, target: str) -> float:
    occ, total = 0, 0
    for r in rows:
        items = actual_targets(r, target)
        occ += items.count(candidate)
        total += len(items)
    return occ / total if total else 0.0


def _norm_special_freq(rows: Sequence[LotteryResult], candidate: str, width: int) -> float:
    occ, total = 0, 0
    for r in rows:
        items = special_items_for_history(r, width)
        occ += items.count(candidate)
        total += len(items)
    return occ / total if total else 0.0


def _recency(rows: Sequence[LotteryResult], candidate: str, target: str) -> float:
    for offset, r in enumerate(reversed(list(rows)), start=1):
        if candidate in actual_targets(r, target):
            return 1.0 / offset
    return 0.0


# Number of worker threads


def _compute_one_candidate(args: tuple[Sequence[LotteryResult], str, str, int]) -> tuple[str, Dict[str, float]]:
    """Worker: compute all features for one candidate."""
    rows, cand, target, width = args
    all_rows = list(rows)
    fm: Dict[str, float] = {}
    fm["history"] = _norm_freq(all_rows, cand, target)
    fm["special_history"] = _norm_special_freq(all_rows, cand, width)
    fm["recency"] = _recency(all_rows, cand, target)
    fm["recency_decay"] = recency_decay_score(all_rows, cand, target)
    fm["recent_delta"] = recent_long_term_delta(all_rows, cand, target)
    fm["decay_weighted"] = decay_weighted_frequency(all_rows, cand, target)
    fm["recency_gap_ratio"] = recency_gap_ratio(all_rows, cand, target)
    fm["recent_peak"] = recent_peak_frequency(all_rows, cand, target)
    fm["recency_cluster"] = recency_cluster_score(all_rows, cand, target)
    fm["gap_penalty"] = gap_since_last_seen(all_rows, cand, target)
    fm["bridge_frequency"] = bridge_frequency(all_rows, cand)
    fm["bridge_streak"] = bridge_streak(all_rows, cand)
    fm["digit_position"] = digit_position_frequency(all_rows, cand, target)
    fm["digit_parts"] = digit_part_frequency(all_rows, cand, target)
    fm["digit_transition"] = digit_transition_score(all_rows, cand, target)
    fm["head_frequency"] = head_frequency(all_rows, cand, target)
    fm["tail_frequency"] = tail_frequency(all_rows, cand, target)
    fm["repeat_ratio"] = repeated_digit_ratio(cand)
    fm["unique_ratio"] = unique_digit_ratio(cand)
    fm["signal_ensemble"] = ensemble_signal_score(all_rows, cand, target)
    for window in ROLLING_WINDOWS:
        fm[f"window_{window}"] = _norm_freq(all_rows[-window:], cand, target)
    fm["window_14"] = rolling_frequency(all_rows, cand, target, 14)
    fm["special_window_30"] = _norm_special_freq(all_rows[-30:], cand, width)
    return cand, fm


# =========================================================================
# Precompute all features for ALL candidates in ONE pass per split
# Returns: Dict[candidate -> Dict[feature_name -> float]]
# =========================================================================
def precompute_features(
    rows: Sequence[LotteryResult],
    candidates: List[str],
    target: str,
    width: int,
) -> Dict[str, Dict[str, float]]:
    feature_map: Dict[str, Dict[str, float]] = {c: {} for c in candidates}

    # Sequential is fast enough with current optimizations
    for cand in candidates:
        _, fm = _compute_one_candidate((rows, cand, target, width))
        feature_map[cand] = fm

    return feature_map


# =========================================================================
# Fast scoring: apply weights to precomputed features
# =========================================================================
def fast_score_candidate(
    feature_map: Dict[str, Dict[str, float]],
    candidate: str,
    weights: Dict[str, float],
) -> float:
    fm = feature_map[candidate]
    score = 0.0
    for key, w in weights.items():
        if w == 0.0:
            continue
        score += w * fm.get(key, 0.0)
    return score


def fast_score_all(
    feature_map: Dict[str, Dict[str, float]],
    candidates: List[str],
    weights: Dict[str, float],
) -> Dict[str, float]:
    return {c: fast_score_candidate(feature_map, c, weights) for c in candidates}


# =========================================================================
# RankingModel
# =========================================================================
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
        return sorted(self.scores.items(), key=lambda p: (-p[1], p[0]))[: self.top_k]

    def predict_baseline(self) -> List[Tuple[str, float]]:
        return sorted(self.baseline_scores.items(), key=lambda p: (-p[1], p[0]))[: self.top_k]


# =========================================================================
# Evaluation
# =========================================================================
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


# =========================================================================
# Train / score
# =========================================================================
def build_target_counts(results: Sequence[LotteryResult], target_name: str) -> Tuple[Counter[str], int, int]:
    counts: Counter[str] = Counter()
    total = 0
    for result in results:
        for item in actual_targets(result, target_name):
            counts[item] += 1
            total += 1
    return counts, total, target_width(target_name)


def train_ranking_model(
    results: Sequence[LotteryResult],
    target_name: str,
    top_k: int,
    weights: Optional[Dict[str, float]] = None,
) -> RankingModel:
    counts, total, width = build_target_counts(results, target_name)
    active_weights = weights or FEATURE_WEIGHTS.get(target_name, FEATURE_WEIGHTS["loto2"])
    candidates = candidate_universe(width)

    feature_map = precompute_features(results, candidates, target_name, width)
    scores = fast_score_all(feature_map, candidates, active_weights)
    baseline_scores = {c: counts.get(c, 0) / total if total else 0.0 for c in candidates}

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


def evaluate_prediction_set(
    predicted: Sequence[str],
    actual: Sequence[str],
    universe_size: int,
) -> Tuple[int, float, float, float]:
    pred_set = set(predicted)
    actual_set = set(actual)
    overlap = len(pred_set & actual_set)
    hit = 1 if overlap > 0 else 0
    actual_count = len(actual_set)
    miss_prob = ((universe_size - actual_count) / universe_size) ** len(pred_set) if universe_size and pred_set else 1.0
    baseline_hit = 1.0 - miss_prob
    baseline_precision = actual_count / universe_size if universe_size else 0.0
    precision = overlap / max(1, len(pred_set))
    return hit, baseline_hit, precision, baseline_precision


# =========================================================================
# Precompute features ONCE for all split points used during tuning.
# Returns dict: split_index -> feature_map
# =========================================================================
def precompute_all_split_features(
    results: Sequence[LotteryResult],
    target_name: str,
    min_train_size: int = 30,
    max_tune_steps: int = 10,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    width = target_width(target_name)
    candidates = candidate_universe(width)
    all_indices = list(range(min_train_size, len(results)))
    tune_indices = all_indices[-max_tune_steps:]
    return {
        idx: precompute_features(results[:idx], candidates, target_name, width)
        for idx in tune_indices
    }


# =========================================================================
# Fast weight tuning: score using precomputed feature dict
# =========================================================================
def score_weight_config_with_cache(
    cached: Dict[int, Dict[str, Dict[str, float]]],
    results: Sequence[LotteryResult],
    target_name: str,
    top_k: int,
    weights: Dict[str, float],
    min_train_size: int = 30,
) -> float:
    if len(results) <= min_train_size:
        return 0.0
    width = target_width(target_name)
    universe_size = 10**width
    candidates = candidate_universe(width)

    hit_total, precision_total, baseline_total = 0.0, 0.0, 0.0
    steps = 0

    for split_index, feature_map in cached.items():
        scores = fast_score_all(feature_map, candidates, weights)
        predicted = [c for c, _ in sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:top_k]]
        actual = actual_targets(results[split_index], target_name)
        hit, baseline_hit, precision, _ = evaluate_prediction_set(predicted, actual, universe_size)
        hit_total += hit
        precision_total += precision
        baseline_total += baseline_hit
        steps += 1

    if not steps:
        return 0.0
    hit_rate = hit_total / steps
    precision_rate = precision_total / steps
    baseline_rate = baseline_total / steps
    return hit_rate * 0.65 + precision_rate * 0.30 + (hit_rate - baseline_rate) * 0.05


# =========================================================================
# Original score_weight_config (kept for backward compat)
# =========================================================================
def score_weight_config(
    results: Sequence[LotteryResult],
    target_name: str,
    top_k: int,
    weights: Dict[str, float],
    min_train_size: int = 30,
    max_tune_steps: int = 10,
) -> float:
    if len(results) <= min_train_size:
        return 0.0
    width = target_width(target_name)
    universe_size = 10**width
    candidates = candidate_universe(width)
    all_indices = list(range(min_train_size, len(results)))
    tune_indices = all_indices[-max_tune_steps:]

    hit_total, precision_total, baseline_total = 0.0, 0.0, 0.0
    steps = 0
    cached: Dict[int, Dict[str, Dict[str, float]]] = {}

    for split_index in tune_indices:
        if split_index not in cached:
            cached[split_index] = precompute_features(
                results[:split_index], candidates, target_name, width
            )
        feature_map = cached[split_index]
        scores = fast_score_all(feature_map, candidates, weights)
        predicted = [c for c, _ in sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:top_k]]
        actual = actual_targets(results[split_index], target_name)
        hit, baseline_hit, precision, _ = evaluate_prediction_set(predicted, actual, universe_size)
        hit_total += hit
        precision_total += precision
        baseline_total += baseline_hit
        steps += 1

    if not steps:
        return 0.0
    hit_rate = hit_total / steps
    precision_rate = precision_total / steps
    baseline_rate = baseline_total / steps
    return hit_rate * 0.65 + precision_rate * 0.30 + (hit_rate - baseline_rate) * 0.05


# =========================================================================
# Tuning: incremental grid search (loto2)
# =========================================================================
def loto2_weight_candidates() -> Dict[str, List[float]]:
    return {
        "history": [0.08, 0.14, 0.22],
        "window_7": [0.20, 0.30, 0.38],
        "window_14": [0.05, 0.12, 0.20],
        "window_30": [0.10, 0.22, 0.30],
        "window_60": [0.03, 0.08, 0.12],
        "recency": [0.08, 0.14, 0.20],
        "recency_decay": [0.05, 0.12, 0.18],
        "recent_delta": [0.0, 0.08, 0.14],
        "gap_penalty": [-0.05, -0.02, 0.0],
        "head_frequency": [0.05, 0.10, 0.15],
        "tail_frequency": [0.05, 0.12, 0.18],
        "signal_ensemble": [0.10, 0.18, 0.26],
    }


def tune_loto2_weights(
    results: Sequence[LotteryResult],
    top_k: int,
    min_train_size: int = 30,
) -> Dict[str, float]:
    # Precompute features ONCE for all split points
    cached = precompute_all_split_features(results, "loto2", min_train_size=min_train_size)
    base = dict(FEATURE_WEIGHTS["loto2"])
    best_weights = dict(base)
    best_score = score_weight_config_with_cache(cached, results, "loto2", top_k, best_weights, min_train_size=min_train_size)
    for key, values in loto2_weight_candidates().items():
        local_best_weights = dict(best_weights)
        local_best_score = best_score
        for value in values:
            trial = dict(best_weights)
            trial[key] = value
            score = score_weight_config_with_cache(cached, results, "loto2", top_k, trial, min_train_size=min_train_size)
            if score > local_best_score:
                local_best_score = score
                local_best_weights = trial
        best_weights = local_best_weights
        best_score = local_best_score
    return best_weights


# =========================================================================
# Tuning: grid search with early stop (non-loto2)
# =========================================================================
def tune_weights(
    results: Sequence[LotteryResult],
    target_name: str,
    top_k: int,
    min_train_size: int = 30,
) -> Dict[str, float]:
    if target_name == "loto2":
        return tune_loto2_weights(results, top_k=top_k, min_train_size=min_train_size)
    if len(results) <= min_train_size:
        return dict(FEATURE_WEIGHTS.get(target_name, FEATURE_WEIGHTS["loto2"]))
    cached = precompute_all_split_features(results, target_name, min_train_size=min_train_size)
    base = FEATURE_WEIGHTS.get(target_name, FEATURE_WEIGHTS["loto2"])
    candidate_keys = ["history", "window_7", "window_30", "special_history", "recency"]
    candidate_values = [0.10, 0.20, 0.30, 0.40]
    best_weights = dict(base)
    best_score = score_weight_config_with_cache(cached, results, target_name, top_k, best_weights, min_train_size=min_train_size)
    no_improve = 0
    for values in product(candidate_values, repeat=len(candidate_keys)):
        trial = dict(base)
        for key, value in zip(candidate_keys, values):
            trial[key] = value
        score = score_weight_config_with_cache(cached, results, target_name, top_k, trial, min_train_size=min_train_size)
        if score > best_score:
            best_score = score
            best_weights = trial
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 256:
            break
    return best_weights


# =========================================================================
# Public API
# =========================================================================
def fit_best_loto2_model(
    results: Sequence[LotteryResult],
    top_k: int,
    min_train_size: int = 30,
) -> RankingModel:
    tuned_weights = tune_loto2_weights(results, top_k=top_k, min_train_size=min_train_size)
    return train_ranking_model(results, target_name="loto2", top_k=top_k, weights=tuned_weights)


def compare_loto2_weight_strategies(
    results: Sequence[LotteryResult],
    top_k: int,
    min_train_size: int = 30,
) -> Dict[str, object]:
    cached = precompute_all_split_features(results, "loto2", min_train_size=min_train_size)
    base_score = score_weight_config_with_cache(cached, results, "loto2", top_k, FEATURE_WEIGHTS["loto2"], min_train_size=min_train_size)
    tuned_weights = tune_loto2_weights(results, top_k=top_k, min_train_size=min_train_size)
    tuned_score = score_weight_config_with_cache(cached, results, "loto2", top_k, tuned_weights, min_train_size=min_train_size)
    return {
        "base_score": base_score,
        "tuned_score": tuned_score,
        "weights": tuned_weights,
    }


# Re-export for sklearn_ranker compatibility
normalized_frequency = _norm_freq
normalized_special_frequency = _norm_special_freq
recency_score = _recency


def fit_tuned_ranking_model(
    results: Sequence[LotteryResult],
    target_name: str,
    top_k: int,
    min_train_size: int = 30,
) -> RankingModel:
    if target_name == "loto2":
        return fit_best_loto2_model(results, top_k=top_k, min_train_size=min_train_size)
    tuned_weights = tune_weights(results, target_name=target_name, top_k=top_k, min_train_size=min_train_size)
    return train_ranking_model(results, target_name=target_name, top_k=top_k, weights=tuned_weights)


def build_prediction(
    model_name: str,
    model: RankingModel,
    metadata: Optional[Dict[str, object]] = None,
) -> RankingPrediction:
    return RankingPrediction(
        model=model_name,
        target=model.target,
        top_predictions=model.predict(),
        frequency_top_predictions=model.predict_baseline(),
        metadata=metadata or {},
    )


def predict_next_day(
    results: Sequence[LotteryResult],
    top_k_loto2: int,
    tuned: bool = False,
) -> Dict[str, object]:
    trainer = fit_tuned_ranking_model if tuned else train_ranking_model
    model_name = "tuned-weighted-ranking" if tuned else "weighted-ranking"
    loto2_model = trainer(results, target_name="loto2", top_k=top_k_loto2)
    return {
        "train_size": len(results),
        "model": model_name,
        "loto2_top": loto2_model.predict(),
        "loto2_frequency_top": loto2_model.predict_baseline(),
        "loto2_weights": loto2_model.weights,
    }
