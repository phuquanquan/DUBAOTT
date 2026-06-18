from __future__ import annotations

"""Online Learning: Predict -> Compare -> Update Weights (Phase 10 - P9.1).

Moi ngay sau 18h30, khi co ket qua thuc te:
1. Load ket qua predict da luu (predictions/YYYY-MM-DD.json)
2. So sanh voi ket qua thuc te (CSV/DuckDB)
3. Tinh metrics: hit, miss, precision
4. Update model weights dua tren ket qua
5. Luu weight moi vao model registry

Usage:
    feedback = PredictionFeedback()
    feedback.process_draw("10/06/2026", actual_loto2_list)
    # -> update weights, log metrics

Integration:
    Duoc goi tu DailyScheduler sau khi co ket qua thuc te.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .dataset import format_date, load_csv, parse_date
from .models.weighted import FEATURE_WEIGHTS, fit_tuned_ranking_model, train_ranking_model
from .schema import LotteryResult
from .targets import actual_targets


# =========================================================================
# Prediction record
# =========================================================================
@dataclass
class DailyPrediction:
    predict_date: str
    data_latest_date: str
    model: str
    top_k: int
    predictions: List[Dict[str, Any]]
    generated_at: str
    actual_result: Optional[str] = None
    actual_loto2: Optional[List[str]] = None
    hit: Optional[bool] = None
    hit_count: Optional[int] = None
    precision: Optional[float] = None


# =========================================================================
# Online feedback engine
# =========================================================================
@dataclass
class FeedbackResult:
    date: str
    predicted: List[str]
    actual: List[str]
    hit: bool
    hit_count: int
    precision: float
    model_before: Dict[str, float]
    model_after: Dict[str, float]
    weight_delta: Dict[str, float]
    model_version: str
    updated: bool


def load_prediction_file(date_str: str, predictions_dir: Path) -> Optional[DailyPrediction]:
    """Load prediction file cho mot ngay.

    Args:
        date_str: Ngay predict (DD/MM/YYYY)
        predictions_dir: Thu muc predictions/

    Returns:
        DailyPrediction hoac None neu file khong ton tai.
    """
    if predictions_dir is None:
        predictions_dir = Path("predictions")

    file_date = date_str.replace("/", "-")
    paths = [
        predictions_dir / f"{file_date}.json",
        predictions_dir / f"{date_str}.json",
    ]
    for path in paths:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return DailyPrediction(
                    predict_date=data.get("predict_date", date_str),
                    data_latest_date=data.get("data_latest_date", ""),
                    model=data.get("model", "unknown"),
                    top_k=data.get("top_k", 5),
                    predictions=data.get("predictions", []),
                    generated_at=data.get("generated_at", ""),
                )
            except (json.JSONDecodeError, TypeError, KeyError):
                return None
    return None


def save_prediction_file(pred: DailyPrediction, predictions_dir: Path) -> Path:
    """Ghi lai prediction file voi actual result da dien."""
    if predictions_dir is None:
        predictions_dir = Path("predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    file_date = pred.predict_date.replace("/", "-")
    path = predictions_dir / f"{file_date}.json"

    data = {
        "predict_date": pred.predict_date,
        "data_latest_date": pred.data_latest_date,
        "model": pred.model,
        "top_k": pred.top_k,
        "predictions": pred.predictions,
        "generated_at": pred.generated_at,
        "actual_result": pred.actual_result,
        "actual_loto2": pred.actual_loto2,
        "hit": pred.hit,
        "hit_count": pred.hit_count,
        "precision": pred.precision,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def compute_hit_metrics(
    predicted: List[str],
    actual: List[str],
    top_k: int = 5,
) -> tuple[bool, int, float]:
    """Tinh hit/precision giua predict va actual.

    Args:
        predicted: Danh sach so du doan (theo thu tu)
        actual: Danh sach so thuc te
        top_k: So luong du doan can tinh

    Returns:
        Tuple (hit, hit_count, precision)
    """
    pred_set = set(predicted[:top_k])
    actual_set = set(actual)
    overlap = len(pred_set & actual_set)
    hit = overlap > 0
    precision = overlap / min(top_k, len(pred_set)) if pred_set else 0.0
    return hit, overlap, precision


def get_actual_loto2_for_date(
    results: Sequence[LotteryResult],
    target_date: str,
    target: str = "loto2",
) -> Optional[List[str]]:
    """Lay danh sach loto2 thuc te cho mot ngay.

    Args:
        results: Toan bo LotteryResult (da sort theo ngay)
        target_date: Ngay can lay (DD/MM/YYYY)
        target: Target name (loto2, dau, duoi, ...)

    Returns:
        List loto2 actuals hoac None neu ngay khong ton tai.
    """
    for result in reversed(list(results)):
        if result.date == target_date:
            items = actual_targets(result, target)
            return sorted(set(items))
    return None


def compute_weight_adjustment(
    current_weights: Dict[str, float],
    hit: bool,
    hit_count: int,
    precision: float,
    learning_rate: float = 0.02,
) -> Dict[str, float]:
    """Tinh dieu chinh weight dua tren ket qua.

    Neu hit -> tang weight cua cac signal thanh cong.
    Neu miss -> giam weight nhung khong zero.

    Args:
        current_weights: Weights hien tai
        hit: True neu co it nhat 1 so trung
        hit_count: So luong so trung
        precision: Ty le precision
        learning_rate: Toc do hoc

    Returns:
        Dict weight moi.
    """
    adjusted = dict(current_weights)

    hit_key = "signal_ensemble"
    if hit_key in adjusted:
        if hit:
            adjusted[hit_key] = min(
                adjusted[hit_key] + learning_rate * precision * 2,
                0.5,
            )
            adjusted["window_7"] = min(
                adjusted.get("window_7", 0.30) + learning_rate * hit_count * 0.05,
                0.5,
            )
        else:
            adjusted[hit_key] = max(
                adjusted[hit_key] - learning_rate * 0.5,
                0.05,
            )
            adjusted["window_7"] = max(
                adjusted.get("window_7", 0.30) - learning_rate * 0.3,
                0.05,
            )

    return adjusted


class OnlineLearningEngine:
    """Engine xu ly online learning moi ngay.

    Flow:
        1. load_prediction(date) -> DailyPrediction
        2. get_actual(date) -> List[str]
        3. compute_hit_metrics() -> FeedbackResult
        4. compute_weight_adjustment() -> Dict[str, float]
        5. Lưu ket qua + weights moi
    """

    def __init__(
        self,
        predictions_dir: Path | str = "predictions",
        csv_path: Path | str = "xsmb_full.csv",
        weights_dir: Path | str = "weights",
        learning_rate: float = 0.02,
    ):
        self.predictions_dir = Path(predictions_dir) if predictions_dir else Path("predictions")
        self.csv_path = Path(csv_path) if csv_path else Path("xsmb_full.csv")
        self.weights_dir = Path(weights_dir) if weights_dir else Path("weights")
        self.learning_rate = learning_rate
        self._results_cache: Optional[List[LotteryResult]] = None
        self._current_weights: Dict[str, float] = {}

    def _load_results(self) -> List[LotteryResult]:
        """Load CSV (cached)."""
        if self._results_cache is not None:
            return self._results_cache
        if not self.csv_path.exists():
            return []
        try:
            self._results_cache = load_csv(self.csv_path)
            from ..dataset import sort_results
            self._results_cache = sort_results(self._results_cache)
        except Exception:
            self._results_cache = []
        return self._results_cache

    def _load_current_weights(self) -> Dict[str, float]:
        """Load weights moi nhat tu file, hoac default."""
        if self._current_weights:
            return self._current_weights

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        latest = self.weights_dir / "latest.json"
        if latest.exists():
            try:
                self._current_weights = json.loads(latest.read_text(encoding="utf-8"))
                return self._current_weights
            except (json.JSONDecodeError, TypeError):
                pass

        self._current_weights = dict(FEATURE_WEIGHTS["loto2"])
        return self._current_weights

    def _save_weights(self, weights: Dict[str, float], date: str) -> Path:
        """Luu weights vao file JSON."""
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        dated_path = self.weights_dir / f"{date.replace('/', '-')}.json"
        latest_path = self.weights_dir / "latest.json"

        dated_path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
        return latest_path

    def process_draw(self, draw_date: str) -> Optional[FeedbackResult]:
        """Xu ly mot ngay quay: load predict -> compare -> update weights.

        Args:
            draw_date: Ngay quay (DD/MM/YYYY) - ket qua nay da co trong CSV

        Returns:
            FeedbackResult hoac None neu khong co prediction truoc do.
        """
        pred = load_prediction_file(draw_date, self.predictions_dir)
        if pred is None:
            return None

        results = self._load_results()
        actual_loto2 = get_actual_loto2_for_date(results, draw_date, "loto2")
        if actual_loto2 is None:
            return None

        predicted = [p["candidate"] for p in pred.predictions]
        hit, hit_count, precision = compute_hit_metrics(
            predicted, actual_loto2, top_k=pred.top_k
        )

        current_weights = self._load_current_weights()
        new_weights = compute_weight_adjustment(
            current_weights, hit, hit_count, precision, self.learning_rate
        )

        updated = new_weights != current_weights
        if updated:
            weight_delta = {
                k: round(new_weights.get(k, 0) - current_weights.get(k, 0), 4)
                for k in set(new_weights) | set(current_weights)
                if new_weights.get(k, 0) != current_weights.get(k, 0)
            }
            self._current_weights = new_weights
            self._save_weights(new_weights, draw_date)
        else:
            weight_delta = {}

        pred.actual_result = draw_date
        pred.actual_loto2 = actual_loto2
        pred.hit = hit
        pred.hit_count = hit_count
        pred.precision = precision
        save_prediction_file(pred, self.predictions_dir)

        model_version = f"online-{draw_date.replace('/', '')}"
        return FeedbackResult(
            date=draw_date,
            predicted=predicted[:pred.top_k],
            actual=actual_loto2,
            hit=hit,
            hit_count=hit_count,
            precision=precision,
            model_before=current_weights,
            model_after=new_weights,
            weight_delta=weight_delta,
            model_version=model_version,
            updated=updated,
        )

    def process_previous_day(self) -> Optional[FeedbackResult]:
        """Process ket qua cua ngay hom qua.

        Goi sau khi co ket qua quay (sau 18h30).
        """
        from ..updater import now_vietnam
        yesterday = now_vietnam() - timedelta(days=1)
        return self.process_draw(format_date(yesterday))

    def process_range(self, start_date: str, end_date: str) -> List[FeedbackResult]:
        """Process nhieu ngay lien tiep.

        Dung de backfill ket qua cho cac ngay chua process.
        """
        results = []
        current = parse_date(start_date)
        final = parse_date(end_date)
        step = timedelta(days=1)
        while current <= final:
            date_str = format_date(current)
            result = self.process_draw(date_str)
            if result is not None:
                results.append(result)
            current += step
        return results

    def get_learning_summary(self, limit: int = 30) -> Dict[str, Any]:
        """Tra ve tom tat ket qua online learning gan nhat."""
        if not self.predictions_dir.exists():
            return {"total_processed": 0, "recent_results": [], "current_weights": self._load_current_weights()}

        pred_files = sorted(self.predictions_dir.glob("*.json"), key=lambda p: p.name, reverse=True)[:limit]
        recent: List[Dict[str, Any]] = []
        for path in pred_files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("hit") is not None:
                    recent.append({
                        "date": data.get("predict_date", ""),
                        "hit": data.get("hit"),
                        "hit_count": data.get("hit_count", 0),
                        "precision": data.get("precision", 0.0),
                        "model": data.get("model", ""),
                    })
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        total = len(recent)
        hit_count = sum(1 for r in recent if r.get("hit"))
        total_precision = sum(r.get("precision", 0.0) for r in recent)

        return {
            "total_processed": total,
            "recent_results": recent,
            "hit_rate": hit_count / total if total else 0.0,
            "avg_precision": total_precision / total if total else 0.0,
            "current_weights": self._load_current_weights(),
        }


# =========================================================================
# Integration: update scheduler to do online learning
# =========================================================================
def run_daily_feedback(
    predictions_dir: Path | str = "predictions",
    csv_path: Path | str = "xsmb_full.csv",
    weights_dir: Path | str = "weights",
    learning_rate: float = 0.02,
) -> Optional[FeedbackResult]:
    """Ham don goi process ngay hom qua.

    Dung trong scheduler hoac cron job hang ngay sau 18h30.
    """
    engine = OnlineLearningEngine(
        predictions_dir=Path(predictions_dir),
        csv_path=Path(csv_path),
        weights_dir=Path(weights_dir),
        learning_rate=learning_rate,
    )
    return engine.process_previous_day()
