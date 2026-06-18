from .weighted import FEATURE_WEIGHTS, RankingEvaluation, RankingModel, fit_tuned_ranking_model, predict_next_day, train_ranking_model, tune_weights

try:
    from .sklearn_ranker import (
        SKLEARN_AVAILABLE,
        SklearnRankingModel,
        benchmarkable_model_names,
        build_estimator,
        fit_named_sklearn_ranking_model,
        fit_sklearn_ranking_model,
        layer1_model_names,
    )
except ModuleNotFoundError:  # sklearn is optional for signal-only workflows
    SKLEARN_AVAILABLE = False
    SklearnRankingModel = None
    benchmarkable_model_names = None
    build_estimator = None
    fit_named_sklearn_ranking_model = None
    fit_sklearn_ranking_model = None
    layer1_model_names = None

try:
    from .neural_ranker import (
        GRURankerModel,
        LSTMRankerModel,
        TORCH_AVAILABLE,
        TransformerRankerModel,
        build_gru_ranker,
        build_lstm_ranker,
        build_transformer_ranker,
        export_neural_ranker_payload,
    )
except ModuleNotFoundError:  # neural backends are optional placeholders
    GRURankerModel = None
    LSTMRankerModel = None
    TORCH_AVAILABLE = False
    TransformerRankerModel = None
    build_gru_ranker = None
    build_lstm_ranker = None
    build_transformer_ranker = None
    export_neural_ranker_payload = None

try:
    from .xgboost_ranker import (
        XGBOOST_AVAILABLE,
        XGBoostRankingEvaluation,
        XGBoostRankingModel,
        build_xgboost_feature_artifact,
        evaluate_xgboost_ranking_backtest,
        export_xgboost_model_payload,
        save_xgboost_feature_artifact,
        save_xgboost_model_payload,
        train_xgboost_ranking_model,
        train_xgboost_with_selected_features,
    )
except ModuleNotFoundError:  # xgboost is optional for ML workflows
    XGBOOST_AVAILABLE = False
    XGBoostRankingEvaluation = None
    XGBoostRankingModel = None
    build_xgboost_feature_artifact = None
    evaluate_xgboost_ranking_backtest = None
    export_xgboost_model_payload = None
    save_xgboost_feature_artifact = None
    save_xgboost_model_payload = None
    train_xgboost_ranking_model = None
    train_xgboost_with_selected_features = None

from ..signals import phase6_target_names, target_preset
from ..targets import actual_targets, target_width


def provider_model_predictions(*args, **kwargs):
    from ..evaluate import provider_model_predictions as _provider_model_predictions

    return _provider_model_predictions(*args, **kwargs)

__all__ = [
    "FEATURE_WEIGHTS",
    "RankingEvaluation",
    "RankingModel",
    "fit_tuned_ranking_model",
    "predict_next_day",
    "train_ranking_model",
    "tune_weights",
    "SklearnRankingModel",
    "benchmarkable_model_names",
    "build_estimator",
    "fit_named_sklearn_ranking_model",
    "fit_sklearn_ranking_model",
    "layer1_model_names",
    "SKLEARN_AVAILABLE",
    "LSTMRankerModel",
    "GRURankerModel",
    "TransformerRankerModel",
    "build_lstm_ranker",
    "build_gru_ranker",
    "build_transformer_ranker",
    "export_neural_ranker_payload",
    "TORCH_AVAILABLE",
    "provider_model_predictions",
    "phase6_target_names",
    "target_preset",
    "actual_targets",
    "target_width",
    "XGBOOST_AVAILABLE",
    "XGBoostRankingEvaluation",
    "XGBoostRankingModel",
    "build_xgboost_feature_artifact",
    "evaluate_xgboost_ranking_backtest",
    "export_xgboost_model_payload",
    "save_xgboost_feature_artifact",
    "save_xgboost_model_payload",
    "train_xgboost_ranking_model",
    "train_xgboost_with_selected_features",
]
