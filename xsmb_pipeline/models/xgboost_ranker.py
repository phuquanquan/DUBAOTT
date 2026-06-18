from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import xgboost as xgb

from ..database import fetch_features_for_date, get_connection
from ..schema import LotteryResult
from ..targets import actual_targets, target_width

XGBOOST_AVAILABLE = True

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy is a transitive dep of xgboost/sklearn in practice
    np = None

try:
    import shap
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    shap = None

try:
    from sklearn.feature_selection import RFE, mutual_info_classif
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression

    SKLEARN_SELECTION_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    RFE = None
    mutual_info_classif = None
    permutation_importance = None
    LogisticRegression = None
    SKLEARN_SELECTION_AVAILABLE = False


@dataclass(frozen=True)
class XGBoostRankingEvaluation:
    target: str
    model: str
    train_size: int
    test_size: int
    top_k: int
    hit_rate: float
    precision_at_k: float
    baseline_hit_rate: float
    baseline_precision_at_k: float
    frequency_hit_rate: float
    frequency_precision_at_k: float
    hit_rate_pct: float
    precision_at_k_pct: float
    baseline_hit_rate_pct: float
    baseline_precision_at_k_pct: float
    frequency_hit_rate_pct: float
    frequency_precision_at_k_pct: float
    feature_importance: List[Tuple[str, float]]
    selected_features: List[str]
    shap_importance: List[Tuple[str, float]]
    mutual_info: List[Tuple[str, float]]
    permutation_importance: List[Tuple[str, float]]
    rfe_ranking: List[Tuple[str, int]]


@dataclass(frozen=True)
class XGBoostRankingModel:
    target: str
    number_width: int
    top_k: int
    feature_names: List[str]
    booster: xgb.Booster
    scores: Dict[str, float]
    baseline_scores: Dict[str, float]
    feature_importance: List[Tuple[str, float]]
    selected_features: List[str]
    shap_importance: List[Tuple[str, float]]
    mutual_info: List[Tuple[str, float]]
    permutation_importance: List[Tuple[str, float]]
    rfe_ranking: List[Tuple[str, int]]

    def predict(self) -> List[Tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda item: (-item[1], item[0]))[: self.top_k]

    def predict_baseline(self) -> List[Tuple[str, float]]:
        return sorted(self.baseline_scores.items(), key=lambda item: (-item[1], item[0]))[: self.top_k]

    def to_payload(self) -> Dict[str, object]:
        return {
            "model": "xgboost-ranking",
            "target": self.target,
            "number_width": self.number_width,
            "top_k": self.top_k,
            "feature_names": self.feature_names,
            "selected_features": self.selected_features,
            "feature_importance": self.feature_importance,
            "shap_importance": self.shap_importance,
            "mutual_info": self.mutual_info,
            "permutation_importance": self.permutation_importance,
            "rfe_ranking": self.rfe_ranking,
            "top_predictions": self.predict(),
            "baseline_predictions": self.predict_baseline(),
        }


class _BoosterEstimator:
    def __init__(self, booster: xgb.Booster, feature_names: Sequence[str]) -> None:
        self.booster = booster
        self.feature_names = list(feature_names)

    def predict_proba(self, x_rows: Sequence[Sequence[float]]) -> List[List[float]]:
        matrix = xgb.DMatrix(list(x_rows), feature_names=self.feature_names)
        probabilities = self.booster.predict(matrix)
        return [[1.0 - float(probability), float(probability)] for probability in probabilities]

    def predict(self, x_rows: Sequence[Sequence[float]]) -> List[int]:
        return [1 if row[1] >= 0.5 else 0 for row in self.predict_proba(x_rows)]

    def score(self, x_rows: Sequence[Sequence[float]], y_rows: Sequence[int]) -> float:
        predictions = self.predict(x_rows)
        if not y_rows:
            return 0.0
        correct = sum(1 for prediction, label in zip(predictions, y_rows) if prediction == label)
        return correct / len(y_rows)


def _candidate_vector(candidate: str, feature_map: Dict[str, float]) -> List[float]:
    digits = [int(character) for character in candidate]
    digit_sum = float(sum(digits))
    repeated_digit_count = float(sum(1 for left, right in zip(candidate, candidate[1:]) if left == right))
    return [
        digit_sum,
        repeated_digit_count,
        float(digits[0]) if digits else 0.0,
        float(digits[-1]) if digits else 0.0,
        float(digit_sum / max(1, len(digits))),
        float(feature_map.get("freq_00_3d", 0.0)),
        float(feature_map.get("freq_11_3d", 0.0)),
        float(feature_map.get("freq_22_3d", 0.0)),
        float(feature_map.get("gan_00_current", 0.0)),
        float(feature_map.get("gan_11_current", 0.0)),
        float(feature_map.get("gan_22_current", 0.0)),
        float(feature_map.get("dau_hot_0", 0.0)),
        float(feature_map.get("duoi_hot_0", 0.0)),
        float(feature_map.get("cham_0", 0.0)),
        float(feature_map.get("tong_de", 0.0)),
        float(feature_map.get("tong_lo", 0.0)),
    ]


def _feature_names() -> List[str]:
    return [
        "digit_sum",
        "repeated_digit_count",
        "head_digit",
        "tail_digit",
        "body_mean",
        "freq_00_3d",
        "freq_11_3d",
        "freq_22_3d",
        "gan_00_current",
        "gan_11_current",
        "gan_22_current",
        "dau_hot_0",
        "duoi_hot_0",
        "cham_0",
        "tong_de",
        "tong_lo",
    ]


def _feature_map_for_date(db_path: str, draw_date: str) -> Dict[str, float]:
    read_only = db_path != ":memory:"
    connection = get_connection(db_path, read_only=read_only)
    try:
        return fetch_features_for_date(connection, draw_date) or {}
    except Exception:
        return {}
    finally:
        connection.close()


def _feature_map_for_prediction(db_path: str, draw_date: str, feature_names: Sequence[str]) -> Dict[str, float]:
    feature_map = _feature_map_for_date(db_path, draw_date)
    if not feature_map:
        raise ValueError(f"Không có features_daily cho ngày {draw_date}")
    return {feature_name: float(feature_map.get(feature_name, 0.0)) for feature_name in feature_names}


def _load_training_rows(results: Sequence[LotteryResult], db_path: str, target_name: str, min_train_size: int) -> tuple[List[List[float]], List[int], List[str]]:
    feature_names = _feature_names()
    width = target_width(target_name)
    universe = [f"{number:0{width}d}" for number in range(10 ** width)]
    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    usable_days = 0
    for split_index in range(min_train_size, len(results)):
        current_result = results[split_index]
        feature_map = _feature_map_for_date(db_path, current_result.date)
        if not feature_map:
            continue
        usable_days += 1
        actual = set(actual_targets(current_result, target_name))
        for candidate in universe:
            x_rows.append(_candidate_vector(candidate, feature_map))
            y_rows.append(1 if candidate in actual else 0)
    if usable_days == 0:
        raise ValueError("Không đủ feature_daily để train XGBoost")
    return x_rows, y_rows, feature_names


def _score_universe(feature_map: Dict[str, float], target_name: str, booster: xgb.Booster, feature_names: Sequence[str]) -> Dict[str, float]:
    width = target_width(target_name)
    scores: Dict[str, float] = {}
    for number in range(10 ** width):
        candidate = f"{number:0{width}d}"
        vector = _candidate_vector(candidate, feature_map)
        scores[candidate] = float(booster.predict(xgb.DMatrix([vector], feature_names=list(feature_names)))[0])
    return scores


def _feature_importance(booster: xgb.Booster, feature_names: Sequence[str]) -> List[Tuple[str, float]]:
    importance_map = booster.get_score(importance_type="gain")
    rows = [(feature_name, float(importance_map.get(feature_name, 0.0))) for feature_name in feature_names]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows


def _compute_shap_importance(booster: xgb.Booster, x_rows: Sequence[Sequence[float]], feature_names: Sequence[str]) -> List[Tuple[str, float]]:
    if shap is None or np is None or not x_rows:
        return [(feature_name, 0.0) for feature_name in feature_names]
    sample_size = min(len(x_rows), 200)
    sample_matrix = np.asarray(list(x_rows)[:sample_size], dtype=float)
    try:
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(sample_matrix)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        importance_values = np.abs(np.asarray(shap_values)).mean(axis=0)
        rows = [(feature_names[index], float(importance_values[index])) for index in range(len(feature_names))]
        rows.sort(key=lambda item: (-item[1], item[0]))
        return rows
    except Exception:  # pragma: no cover - shap may fail on some builds
        return [(feature_name, 0.0) for feature_name in feature_names]


def _compute_mutual_information(x_rows: Sequence[Sequence[float]], y_rows: Sequence[int], feature_names: Sequence[str]) -> List[Tuple[str, float]]:
    if not SKLEARN_SELECTION_AVAILABLE or mutual_info_classif is None or np is None or not x_rows:
        return [(feature_name, 0.0) for feature_name in feature_names]
    try:
        x_matrix = np.asarray(list(x_rows), dtype=float)
        y_array = np.asarray(list(y_rows), dtype=int)
        scores = mutual_info_classif(x_matrix, y_array, random_state=42)
        rows = [(feature_names[index], float(scores[index])) for index in range(len(feature_names))]
        rows.sort(key=lambda item: (-item[1], item[0]))
        return rows
    except Exception:  # pragma: no cover - defensive
        return [(feature_name, 0.0) for feature_name in feature_names]


def _compute_permutation_importance(booster: xgb.Booster, x_rows: Sequence[Sequence[float]], y_rows: Sequence[int], feature_names: Sequence[str]) -> List[Tuple[str, float]]:
    if not SKLEARN_SELECTION_AVAILABLE or permutation_importance is None or np is None or not x_rows:
        return [(feature_name, 0.0) for feature_name in feature_names]
    try:
        estimator = _BoosterEstimator(booster, feature_names)
        result = permutation_importance(estimator, np.asarray(list(x_rows), dtype=float), np.asarray(list(y_rows), dtype=int), n_repeats=3, random_state=42, scoring="roc_auc")
        rows = [(feature_names[index], float(result.importances_mean[index])) for index in range(len(feature_names))]
        rows.sort(key=lambda item: (-item[1], item[0]))
        return rows
    except Exception:  # pragma: no cover - defensive
        return [(feature_name, 0.0) for feature_name in feature_names]


def _compute_rfe_ranking(x_rows: Sequence[Sequence[float]], y_rows: Sequence[int], feature_names: Sequence[str]) -> List[Tuple[str, int]]:
    if not SKLEARN_SELECTION_AVAILABLE or RFE is None or LogisticRegression is None or np is None or not x_rows:
        return [(feature_name, index + 1) for index, feature_name in enumerate(feature_names)]
    try:
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
        selector = RFE(estimator=estimator, n_features_to_select=max(1, min(8, len(feature_names))), step=1)
        selector.fit(np.asarray(list(x_rows), dtype=float), np.asarray(list(y_rows), dtype=int))
        ranking = list(selector.ranking_)
        rows = [(feature_names[index], int(ranking[index])) for index in range(len(feature_names))]
        rows.sort(key=lambda item: (item[1], item[0]))
        return rows
    except Exception:  # pragma: no cover - defensive
        return [(feature_name, index + 1) for index, feature_name in enumerate(feature_names)]


def _select_from_rankings(
    feature_names: Sequence[str],
    shap_importance: Sequence[Tuple[str, float]],
    mutual_info: Sequence[Tuple[str, float]],
    permutation_importance_rows: Sequence[Tuple[str, float]],
    rfe_ranking: Sequence[Tuple[str, int]],
    top_k_features: int,
) -> List[str]:
    combined_scores: Dict[str, float] = {feature_name: 0.0 for feature_name in feature_names}
    for rows in (shap_importance, mutual_info, permutation_importance_rows):
        for feature_name, value in rows:
            combined_scores[feature_name] += float(value) if value is not None else 0.0
    for feature_name, rank_value in rfe_ranking:
        combined_scores[feature_name] += 1.0 / max(1, rank_value)
    ordered = sorted(combined_scores.items(), key=lambda item: (-item[1], item[0]))
    selected = [feature_name for feature_name, _score in ordered[:top_k_features]]
    return selected if selected else list(feature_names[:top_k_features])


def _ranking_to_dict_list(rows: Sequence[Tuple[str, float]]) -> List[Dict[str, float | str]]:
    return [{"feature": feature_name, "score": float(score)} for feature_name, score in rows]


def _rfe_to_dict_list(rows: Sequence[Tuple[str, int]]) -> List[Dict[str, int | str]]:
    return [{"feature": feature_name, "rank": int(rank_value)} for feature_name, rank_value in rows]


def build_xgboost_feature_artifact(model: XGBoostRankingModel) -> Dict[str, object]:
    return {
        "model": model.to_payload(),
        "feature_artifacts": {
            "xgboost_gain": _ranking_to_dict_list(model.feature_importance),
            "shap": _ranking_to_dict_list(model.shap_importance),
            "mutual_info": _ranking_to_dict_list(model.mutual_info),
            "permutation_importance": _ranking_to_dict_list(model.permutation_importance),
            "rfe": _rfe_to_dict_list(model.rfe_ranking),
            "selected_features": list(model.selected_features),
        },
    }


def save_xgboost_feature_artifact(model: XGBoostRankingModel, output_path: str, model_name: str = "xgboost-ranking") -> Dict[str, object]:
    import json
    from pathlib import Path

    artifact = build_xgboost_feature_artifact(model)
    artifact["model"]["model"] = model_name
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def _selected_indices(feature_names: Sequence[str], selected_features: Sequence[str]) -> List[int]:
    return [index for index, feature_name in enumerate(feature_names) if feature_name in selected_features]


def _filter_vector(vector: Sequence[float], selected_indices: Sequence[int]) -> List[float]:
    return [value for index, value in enumerate(vector) if index in selected_indices]


def _top_feature_count(feature_names: Sequence[str], limit: int = 8) -> int:
    return min(limit, len(feature_names))


def train_xgboost_ranking_model(results: Sequence[LotteryResult], db_path: str, target_name: str, top_k: int, min_train_size: int = 30) -> XGBoostRankingModel:
    if len(results) <= min_train_size:
        raise ValueError("Không đủ dữ liệu để train XGBoost")
    x_rows, y_rows, feature_names = _load_training_rows(results, db_path, target_name, min_train_size)
    if not x_rows:
        raise ValueError("Không đủ feature_daily để train XGBoost")
    dtrain = xgb.DMatrix(x_rows, label=y_rows, feature_names=feature_names)
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 3,
            "eta": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "seed": 42,
        },
        dtrain,
        num_boost_round=30,
    )
    latest_feature_map = _feature_map_for_prediction(db_path, results[-1].date, feature_names)
    scores = _score_universe(latest_feature_map, target_name, booster, feature_names)
    width = target_width(target_name)
    baseline_scores = {}
    for number in range(10 ** width):
        candidate = f"{number:0{width}d}"
        digit_sum = sum(int(character) for character in candidate)
        baseline_scores[candidate] = 1.0 / (1.0 + abs(digit_sum - float(latest_feature_map.get("tong_lo", 0.0))))

    xgb_importance = _feature_importance(booster, feature_names)
    shap_importance = _compute_shap_importance(booster, x_rows, feature_names)
    mutual_info = _compute_mutual_information(x_rows, y_rows, feature_names)
    permutation_importance_rows = _compute_permutation_importance(booster, x_rows, y_rows, feature_names)
    rfe_ranking = _compute_rfe_ranking(x_rows, y_rows, feature_names)
    selected_features = _select_from_rankings(
        feature_names,
        shap_importance,
        mutual_info,
        permutation_importance_rows,
        rfe_ranking,
        top_k_features=_top_feature_count(feature_names),
    )
    return XGBoostRankingModel(
        target=target_name,
        number_width=width,
        top_k=top_k,
        feature_names=list(feature_names),
        booster=booster,
        scores=scores,
        baseline_scores=baseline_scores,
        feature_importance=xgb_importance,
        selected_features=selected_features,
        shap_importance=shap_importance,
        mutual_info=mutual_info,
        permutation_importance=permutation_importance_rows,
        rfe_ranking=rfe_ranking,
    )


def _train_selected_feature_model(results: Sequence[LotteryResult], db_path: str, target_name: str, top_k: int, min_train_size: int, selected_features: Sequence[str]) -> XGBoostRankingModel:
    base_model = train_xgboost_ranking_model(results, db_path=db_path, target_name=target_name, top_k=top_k, min_train_size=min_train_size)
    feature_names = [feature_name for feature_name in base_model.feature_names if feature_name in selected_features]
    if not feature_names:
        raise ValueError("Không có selected_features hợp lệ để train XGBoost")
    selected_indices = _selected_indices(base_model.feature_names, feature_names)
    width = base_model.number_width
    feature_map = _feature_map_for_prediction(db_path, results[-1].date, base_model.feature_names)
    training_vectors: List[List[float]] = []
    training_labels: List[int] = []
    universe = [f"{number:0{width}d}" for number in range(10 ** width)]
    actual = set(actual_targets(results[-1], target_name))
    for candidate in universe:
        vector = _candidate_vector(candidate, feature_map)
        training_vectors.append(_filter_vector(vector, selected_indices))
        training_labels.append(1 if candidate in actual else 0)
    booster = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.1, "seed": 42},
        xgb.DMatrix(training_vectors, label=training_labels, feature_names=feature_names),
        num_boost_round=10,
    )
    scores: Dict[str, float] = {}
    for candidate in universe:
        vector = _candidate_vector(candidate, feature_map)
        filtered_vector = _filter_vector(vector, selected_indices)
        scores[candidate] = float(booster.predict(xgb.DMatrix([filtered_vector], feature_names=feature_names))[0])
    importance = _feature_importance(booster, feature_names)
    baseline_scores = {candidate: 0.0 for candidate in scores}
    return XGBoostRankingModel(
        target=target_name,
        number_width=width,
        top_k=top_k,
        feature_names=list(feature_names),
        booster=booster,
        scores=scores,
        baseline_scores=baseline_scores,
        feature_importance=importance,
        selected_features=list(feature_names),
        shap_importance=[(name, 0.0) for name in feature_names],
        mutual_info=[(name, 0.0) for name in feature_names],
        permutation_importance=[(name, 0.0) for name in feature_names],
        rfe_ranking=[(name, index + 1) for index, name in enumerate(feature_names)],
    )


def train_xgboost_with_selected_features(results: Sequence[LotteryResult], db_path: str, target_name: str, top_k: int, min_train_size: int = 30, top_k_features: int = 8) -> XGBoostRankingModel:
    base_model = train_xgboost_ranking_model(results, db_path=db_path, target_name=target_name, top_k=top_k, min_train_size=min_train_size)
    selected_features = base_model.selected_features[:top_k_features]
    return _train_selected_feature_model(results, db_path=db_path, target_name=target_name, top_k=top_k, min_train_size=min_train_size, selected_features=selected_features)


def evaluate_xgboost_ranking_backtest(results: Sequence[LotteryResult], db_path: str, target_name: str, top_k: int, min_train_size: int = 30) -> XGBoostRankingEvaluation:
    if len(results) <= min_train_size:
        raise ValueError("Không đủ dữ liệu để backtest XGBoost")
    width = target_width(target_name)
    universe_size = 10 ** width
    hits = 0
    precision_total = 0.0
    baseline_hit_total = 0.0
    baseline_precision_total = 0.0
    test_count = 0
    last_model: XGBoostRankingModel | None = None
    for split_index in range(min_train_size, len(results)):
        train_results = results[:split_index]
        if len(train_results) <= min_train_size:
            continue
        test_result = results[split_index]
        model = train_xgboost_ranking_model(train_results, db_path=db_path, target_name=target_name, top_k=top_k, min_train_size=min_train_size)
        last_model = model
        predicted = [candidate for candidate, _score in model.predict()]
        actual = actual_targets(test_result, target_name)
        overlap = len(set(predicted) & set(actual))
        hits += 1 if overlap else 0
        precision_total += overlap / max(1, len(predicted))
        baseline_hit_total += 1.0 if actual else 0.0
        baseline_precision_total += len(actual) / universe_size
        test_count += 1
    feature_importance = last_model.feature_importance if last_model is not None else []
    if not feature_importance:
        feature_importance = [
            (feature_name, 0.0) for feature_name in _feature_names()
        ]
    selected_features = last_model.selected_features if last_model is not None else []
    shap_importance = last_model.shap_importance if last_model is not None else []
    mutual_info = last_model.mutual_info if last_model is not None else []
    permutation_importance_rows = last_model.permutation_importance if last_model is not None else []
    rfe_ranking = last_model.rfe_ranking if last_model is not None else []
    if not shap_importance:
        shap_importance = [(feature_name, 0.0) for feature_name in _feature_names()]
    if not mutual_info:
        mutual_info = [(feature_name, 0.0) for feature_name in _feature_names()]
    if not permutation_importance_rows:
        permutation_importance_rows = [(feature_name, 0.0) for feature_name in _feature_names()]
    if not rfe_ranking:
        rfe_ranking = [(feature_name, index + 1) for index, feature_name in enumerate(_feature_names())]
    if not selected_features:
        selected_features = [feature_name for feature_name, _score in feature_importance[:_top_feature_count(_feature_names())]]
    return XGBoostRankingEvaluation(
        target=target_name,
        model="xgboost-ranking",
        train_size=min_train_size,
        test_size=test_count,
        top_k=top_k,
        hit_rate=hits / max(1, test_count),
        precision_at_k=precision_total / max(1, test_count),
        baseline_hit_rate=baseline_hit_total / max(1, test_count),
        baseline_precision_at_k=baseline_precision_total / max(1, test_count),
        frequency_hit_rate=0.0,
        frequency_precision_at_k=0.0,
        hit_rate_pct=100.0 * hits / max(1, test_count),
        precision_at_k_pct=100.0 * precision_total / max(1, test_count),
        baseline_hit_rate_pct=100.0 * baseline_hit_total / max(1, test_count),
        baseline_precision_at_k_pct=100.0 * baseline_precision_total / max(1, test_count),
        frequency_hit_rate_pct=0.0,
        frequency_precision_at_k_pct=0.0,
        feature_importance=feature_importance,
        selected_features=selected_features,
        shap_importance=shap_importance,
        mutual_info=mutual_info,
        permutation_importance=permutation_importance_rows,
        rfe_ranking=rfe_ranking,
    )


def export_xgboost_model_payload(model: XGBoostRankingModel, model_name: str = "xgboost-ranking") -> Dict[str, object]:
    payload = model.to_payload()
    payload["model"] = model_name
    return payload


def save_xgboost_model_payload(model: XGBoostRankingModel, output_path: str, model_name: str = "xgboost-ranking") -> Dict[str, object]:
    return save_xgboost_feature_artifact(model, output_path=output_path, model_name=model_name)
