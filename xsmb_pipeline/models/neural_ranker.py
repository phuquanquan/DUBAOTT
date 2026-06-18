from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional transitive dep
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class NeuralRankerPayload:
    model: str
    target: str
    top_k: int
    top_predictions: List[Tuple[str, float]]
    metadata: Dict[str, Any]


@dataclass
class LSTMRankerModel:
    target: str
    top_k: int
    hidden_size: int = 32
    num_layers: int = 1

    def fit(self, _history: Sequence[object] | None = None) -> "LSTMRankerModel":
        return self

    def predict(self) -> List[Tuple[str, float]]:
        return []

    def summary(self) -> Dict[str, object]:
        return {
            "model": "lstm-ranking",
            "target": self.target,
            "top_k": self.top_k,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }


@dataclass
class GRURankerModel:
    target: str
    top_k: int
    hidden_size: int = 32
    num_layers: int = 1

    def fit(self, _history: Sequence[object] | None = None) -> "GRURankerModel":
        return self

    def predict(self) -> List[Tuple[str, float]]:
        return []

    def summary(self) -> Dict[str, object]:
        return {
            "model": "gru-ranking",
            "target": self.target,
            "top_k": self.top_k,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }


@dataclass
class TransformerRankerModel:
    target: str
    top_k: int
    num_heads: int = 2
    num_layers: int = 1

    def fit(self, _history: Sequence[object] | None = None) -> "TransformerRankerModel":
        return self

    def predict(self) -> List[Tuple[str, float]]:
        return []

    def summary(self) -> Dict[str, object]:
        return {
            "model": "transformer-ranking",
            "target": self.target,
            "top_k": self.top_k,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }


def build_lstm_ranker(target: str = "loto2", top_k: int = 5) -> LSTMRankerModel:
    return LSTMRankerModel(target=target, top_k=top_k)


def build_gru_ranker(target: str = "loto2", top_k: int = 5) -> GRURankerModel:
    return GRURankerModel(target=target, top_k=top_k)


def build_transformer_ranker(target: str = "loto2", top_k: int = 5) -> TransformerRankerModel:
    return TransformerRankerModel(target=target, top_k=top_k)


def export_neural_ranker_payload(model: object, model_name: str) -> Dict[str, object]:
    top_predictions = model.predict() if hasattr(model, "predict") else []
    target = getattr(model, "target", "loto2")
    top_k = getattr(model, "top_k", len(top_predictions))
    return {
        "model": model_name,
        "target": target,
        "top_k": top_k,
        "top_predictions": top_predictions,
        "metadata": {
            "torch_available": TORCH_AVAILABLE,
            "numpy_available": np is not None,
        },
    }
