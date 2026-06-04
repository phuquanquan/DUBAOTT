from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Dict, List, Sequence, Tuple

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    ExtraTreesClassifier = None
    RandomForestClassifier = None
    LogisticRegression = None
    MLPClassifier = None
    make_pipeline = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

from ..features import bridge_frequency, bridge_streak, digit_part_frequency, digit_position_frequency, digit_transition_score, gap_since_last_seen, head_frequency, recent_long_term_delta, recency_decay_score, repeated_digit_ratio, rolling_frequency, tail_frequency, unique_digit_ratio
from ..schema import LotteryResult
from ..signals import signal_vector
from ..targets import actual_targets, target_width
from .weighted import ROLLING_WINDOWS, candidate_universe, normalized_frequency, normalized_special_frequency, recency_score


@dataclass
class SklearnRankingModel:
    target: str
    number_width: int
    top_k: int
    model_name: str
    model: Any
    scores: Dict[str, float]

    def predict(self) -> List[Tuple[str, float]]:
        scored = sorted(self.scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return scored[: self.top_k]


def require_sklearn(model_name: str) -> None:
    if not SKLEARN_AVAILABLE:
        raise ModuleNotFoundError(f"scikit-learn is required for model '{model_name}'")


def available_model_names() -> List[str]:
    return ["extra_trees", "logistic", "logistic_l2", "mlp", "random_forest"]


def model_is_available(model_name: str) -> bool:
    return model_name in available_model_names() and SKLEARN_AVAILABLE


def benchmarkable_model_names(target_name: str) -> List[str]:
    if target_name == "loto2":
        return ["logistic", "random_forest", "extra_trees", "mlp"]
    return ["logistic"]


def build_estimator(model_name: str) -> Any:
    require_sklearn(model_name)
    if model_name == "logistic":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if model_name == "logistic_l2":
        return LogisticRegression(max_iter=1500, class_weight="balanced", C=0.7)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, class_weight="balanced_subsample")
    if model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=400, max_depth=10, min_samples_leaf=2, random_state=42, class_weight="balanced_subsample")
    if model_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001, learning_rate_init=0.001, max_iter=800, random_state=42),
        )
    raise ValueError(f"Unsupported sklearn ranking model: {model_name}")


def model_label(model_name: str) -> str:
    return f"sklearn-{model_name}-ranking"


def predict_positive_score(model: Any, vector: List[float]) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba([vector])[0][1])
    if hasattr(model, "decision_function"):
        raw = float(model.decision_function([vector])[0])
        return 1.0 / (1.0 + exp(-raw))
    return float(model.predict([vector])[0])


def build_feature_vector(history: Sequence[LotteryResult], candidate: str, target_name: str, number_width: int, selected_signal_names: Sequence[str] | None = None) -> List[float]:
    rows = list(history)
    features: List[float] = [normalized_frequency(rows, candidate, target_name)]
    for window in ROLLING_WINDOWS:
        window_results = rows[-window:] if len(rows) > window else rows
        features.append(normalized_frequency(window_results, candidate, target_name))
    features.append(rolling_frequency(rows, candidate, target_name, 14))
    features.append(normalized_special_frequency(rows, candidate, number_width))
    special_window = rows[-30:] if len(rows) > 30 else rows
    features.append(normalized_special_frequency(special_window, candidate, number_width))
    features.append(recency_score(rows, candidate, target_name))
    features.append(recency_decay_score(rows, candidate, target_name))
    features.append(gap_since_last_seen(rows, candidate, target_name))
    features.append(recent_long_term_delta(rows, candidate, target_name))
    if number_width >= 3:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        features.append(bridge_frequency(rows, candidate))
        features.append(float(bridge_streak(rows, candidate)))
        features.append(digit_position_frequency(rows, candidate, target_name))
        features.append(digit_part_frequency(rows, candidate, target_name))
        features.append(digit_transition_score(rows, candidate, target_name))
    features.append(head_frequency(rows, candidate, target_name))
    features.append(tail_frequency(rows, candidate, target_name))
    features.append(repeated_digit_ratio(candidate))
    features.append(unique_digit_ratio(candidate))
    features.extend(signal_vector(rows, candidate, target_name, selected_signal_names=selected_signal_names))
    return features


def training_candidates(universe: Sequence[str], actual: set[str], split_index: int, number_width: int) -> List[str]:
    if number_width <= 2:
        return list(universe)
    negative_limit = 40 if number_width == 3 else 60
    negatives = [candidate for candidate in universe if candidate not in actual]
    if len(negatives) <= negative_limit:
        sampled = negatives
    else:
        start = split_index % len(negatives)
        step = 7
        sampled = []
        idx = start
        seen = set()
        while len(sampled) < negative_limit and len(seen) < len(negatives):
            candidate = negatives[idx % len(negatives)]
            if candidate not in seen:
                sampled.append(candidate)
                seen.add(candidate)
            idx += step
    return sorted(actual | set(sampled))


def training_split_indexes(total_results: int, min_train_size: int) -> List[int]:
    indexes = list(range(min_train_size, total_results))
    if len(indexes) <= 18:
        return indexes
    return indexes[-18:]


def build_training_matrix(results: Sequence[LotteryResult], target_name: str, min_train_size: int, selected_signal_names: Sequence[str] | None = None) -> Tuple[int, List[str], List[List[float]], List[int]]:
    width = target_width(target_name)
    universe = candidate_universe(width)
    X: List[List[float]] = []
    y: List[int] = []
    for split_index in training_split_indexes(len(results), min_train_size):
        history = list(results[:split_index])
        actual = set(actual_targets(results[split_index], target_name))
        for candidate in training_candidates(universe, actual, split_index, width):
            X.append(build_feature_vector(history, candidate, target_name, width, selected_signal_names=selected_signal_names))
            y.append(1 if candidate in actual else 0)
    return width, universe, X, y


def score_universe(model: Any, results: Sequence[LotteryResult], target_name: str, width: int, universe: Sequence[str], selected_signal_names: Sequence[str] | None = None) -> Dict[str, float]:
    final_scores: Dict[str, float] = {}
    for candidate in universe:
        vector = build_feature_vector(results, candidate, target_name, width, selected_signal_names=selected_signal_names)
        final_scores[candidate] = predict_positive_score(model, vector)
    return final_scores


def fit_named_sklearn_ranking_model(results: Sequence[LotteryResult], target_name: str, top_k: int, min_train_size: int = 30, model_name: str = "logistic", selected_signal_names: Sequence[str] | None = None) -> SklearnRankingModel:
    width, universe, X, y = build_training_matrix(results, target_name, min_train_size, selected_signal_names=selected_signal_names)
    model = build_estimator(model_name)
    model.fit(X, y)
    final_scores = score_universe(model, results, target_name, width, universe, selected_signal_names=selected_signal_names)
    return SklearnRankingModel(target=target_name, number_width=width, top_k=top_k, model_name=model_name, model=model, scores=final_scores)


def fit_sklearn_ranking_model(results: Sequence[LotteryResult], target_name: str, top_k: int, min_train_size: int = 30) -> SklearnRankingModel:
    return fit_named_sklearn_ranking_model(results, target_name=target_name, top_k=top_k, min_train_size=min_train_size, model_name="logistic")
