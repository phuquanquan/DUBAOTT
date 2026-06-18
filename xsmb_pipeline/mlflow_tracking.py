from __future__ import annotations

"""MLflow tracking cho XSMB system (Phase 9 - P9.2).

Theo thiết kế:
    - Model versioning: mỗi lần train lưu model version + params + metrics
    - Feature versioning: theo dõi feature nào được sinh, bao lâu, hiệu quả thế nào
    - Training runs: log parameters, metrics, artifacts
    - Experiment: so sánh các lần chạy

Usage:
    # Track một training run
    with MLflowTracker("xsmb-loto2", db_path="xsmb.duckdb") as tracker:
        tracker.log_params({"model": "xgboost", "top_k": 5})
        tracker.log_metrics({"hit_rate": 0.28, "precision@5": 0.12})
        tracker.log_artifact("model.json")

    # Feature registry
    mlflow_feature_register(compute_result)

    # Auto-track model training
    tracked_model = mlflow_tracked_train(results, target_name="loto2", ...)
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schema import LotteryResult
from .dataset import format_date


# =========================================================================
# Fallback JSON-based tracking (khi MLflow không có)
# =========================================================================
DEFAULT_TRACK_DIR = Path("mlruns")
METADATA_FILE = DEFAULT_TRACK_DIR / "metadata.jsonl"


@dataclass
class TrackingRun:
    run_id: str
    experiment_name: str
    created_at: str
    status: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    tags: Dict[str, str]
    model_version: Optional[str]
    feature_version: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureVersion:
    feature_id: str
    feature_name: str
    generator: str
    created_at: str
    lookback_days: int
    num_features: int
    hit_rate: Optional[float]
    roi: Optional[float]


@dataclass
class ModelVersion:
    model_id: str
    model_name: str
    model_type: str
    created_at: str
    target: str
    top_k: int
    hit_rate: Optional[float]
    precision_at_k: Optional[float]
    params: Dict[str, Any]
    artifact_path: Optional[str]
    feature_version: Optional[str]


# =========================================================================
# Lightweight JSON tracking backend
# =========================================================================
def _ensure_track_dir() -> Path:
    DEFAULT_TRACK_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.touch(exist_ok=True)
    return DEFAULT_TRACK_DIR


def _load_runs() -> List[TrackingRun]:
    if not METADATA_FILE.exists():
        return []
    runs: List[TrackingRun] = []
    try:
        content = METADATA_FILE.read_text(encoding="utf-8").strip()
        if content:
            for line in content.splitlines():
                data = json.loads(line)
                runs.append(TrackingRun(**data))
    except (json.JSONDecodeError, TypeError):
        pass
    return runs


def _save_run(run: TrackingRun) -> None:
    _ensure_track_dir()
    with METADATA_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(run), ensure_ascii=False) + "\n")


def _generate_run_id(experiment: str, extra: str = "") -> str:
    ts = datetime.utcnow().isoformat()
    raw = f"{experiment}:{ts}:{extra}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


# =========================================================================
# Feature Registry
# =========================================================================
FEATURE_REGISTRY_FILE = DEFAULT_TRACK_DIR / "feature_registry.jsonl"


def mlflow_feature_register(
    generator: str,
    feature_names: Sequence[str],
    lookback_days: int,
    num_features: int,
    hit_rate: Optional[float] = None,
    roi: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FeatureVersion:
    """Đăng ký một phiên bản feature mới.

    Args:
        generator:     Tên generator (vd: "compute_all_features", "FrequencyGenerator")
        feature_names: Danh sách feature được sinh
        lookback_days: Số ngày lookback
        num_features:  Tổng số feature
        hit_rate:      Hit rate đo được (nếu có)
        roi:           ROI đo được (nếu có)
        metadata:      Thông tin bổ sung

    Returns:
        FeatureVersion đã được lưu.
    """
    _ensure_track_dir()

    feature_id = _generate_run_id("feature", generator)
    fv = FeatureVersion(
        feature_id=feature_id,
        feature_name=generator,
        generator=generator,
        created_at=_utc_now(),
        lookback_days=lookback_days,
        num_features=num_features,
        hit_rate=hit_rate,
        roi=roi,
    )

    with FEATURE_REGISTRY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(fv), ensure_ascii=False) + "\n")

    return fv


def get_feature_registry() -> List[FeatureVersion]:
    """Lấy lịch sử feature registry."""
    if not FEATURE_REGISTRY_FILE.exists():
        return []
    versions: List[FeatureVersion] = []
    try:
        content = FEATURE_REGISTRY_FILE.read_text(encoding="utf-8").strip()
        if content:
            for line in content.splitlines():
                data = json.loads(line)
                versions.append(FeatureVersion(**data))
    except (json.JSONDecodeError, TypeError):
        pass
    return versions


def get_latest_feature_version(generator: str) -> Optional[FeatureVersion]:
    """Lấy phiên bản feature mới nhất cho một generator."""
    all_versions = get_feature_registry()
    candidates = [v for v in all_versions if v.generator == generator]
    if not candidates:
        return None
    return max(candidates, key=lambda v: v.created_at)


# =========================================================================
# Model Registry
# =========================================================================
MODEL_REGISTRY_FILE = DEFAULT_TRACK_DIR / "model_registry.jsonl"


def mlflow_model_register(
    model_name: str,
    model_type: str,
    target: str,
    top_k: int,
    params: Optional[Dict[str, Any]] = None,
    artifact_path: Optional[str] = None,
    feature_version: Optional[str] = None,
) -> ModelVersion:
    """Đăng ký một phiên bản model mới.

    Args:
        model_name:     Tên model (vd: "xsmb-xgboost-loto2")
        model_type:     Loại model (xgboost, lightgbm, weighted, neural, ...)
        target:         Target (loto2, dau, duoi, ...)
        top_k:          Top-K dự đoán
        params:         Hyperparameters
        artifact_path:   Đường dẫn artifact
        feature_version: FeatureVersion.feature_id đã dùng

    Returns:
        ModelVersion đã được lưu.
    """
    _ensure_track_dir()

    model_id = _generate_run_id("model", model_name)
    mv = ModelVersion(
        model_id=model_id,
        model_name=model_name,
        model_type=model_type,
        created_at=_utc_now(),
        target=target,
        top_k=top_k,
        hit_rate=None,
        precision_at_k=None,
        params=params or {},
        artifact_path=artifact_path,
        feature_version=feature_version,
    )

    with MODEL_REGISTRY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(mv), ensure_ascii=False) + "\n")

    return mv


def update_model_metrics(
    model_id: str,
    hit_rate: float,
    precision_at_k: float,
) -> None:
    """Cập nhật metrics cho một model đã đăng ký."""
    if not MODEL_REGISTRY_FILE.exists():
        return

    updated_lines: List[str] = []
    try:
        content = MODEL_REGISTRY_FILE.read_text(encoding="utf-8").strip()
        if content:
            for line in content.splitlines():
                data = json.loads(line)
                if data.get("model_id") == model_id:
                    data["hit_rate"] = hit_rate
                    data["precision_at_k"] = precision_at_k
                updated_lines.append(json.dumps(data, ensure_ascii=False))
    except (json.JSONDecodeError, TypeError):
        return

    MODEL_REGISTRY_FILE.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def get_model_registry() -> List[ModelVersion]:
    """Lấy lịch sử model registry."""
    if not MODEL_REGISTRY_FILE.exists():
        return []
    versions: List[ModelVersion] = []
    try:
        content = MODEL_REGISTRY_FILE.read_text(encoding="utf-8").strip()
        if content:
            for line in content.splitlines():
                data = json.loads(line)
                versions.append(ModelVersion(**data))
    except (json.JSONDecodeError, TypeError):
        pass
    return versions


def get_best_model(target: str, by: str = "hit_rate") -> Optional[ModelVersion]:
    """Lấy model tốt nhất cho một target."""
    candidates = [m for m in get_model_registry() if m.target == target]
    if not candidates:
        return None
    return max(candidates, key=lambda m: getattr(m, by, 0.0) or 0.0)


# =========================================================================
# Full MLflowTracker class (wrapper)
# =========================================================================
class MLflowTracker:
    """Context manager cho tracking một experiment run.

    Ưu tiên dùng MLflow thực sự nếu có, fallback về JSON file.

    Usage:
        with MLflowTracker("xsmb-loto2", db_path="xsmb.duckdb") as tracker:
            tracker.log_params({"model": "xgboost", "top_k": 5})
            tracker.log_metrics({"hit_rate": 0.28})
    """

    def __init__(
        self,
        experiment_name: str,
        db_path: str = "xsmb.duckdb",
        use_mlflow: bool = False,
    ):
        self.experiment_name = experiment_name
        self.db_path = db_path
        self.use_mlflow = use_mlflow and self._mlflow_available()
        self.run: Optional[object] = None
        self._params: Dict[str, Any] = {}
        self._metrics: Dict[str, float] = {}
        self._artifacts: List[str] = []
        self._tags: Dict[str, str] = {}
        self._model_version: Optional[str] = None
        self._feature_version: Optional[str] = None
        self._run_id: Optional[str] = None

    @staticmethod
    def _mlflow_available() -> bool:
        try:
            import mlflow
            return True
        except ImportError:
            return False

    def __enter__(self) -> "MLflowTracker":
        if self.use_mlflow:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            self.run = mlflow.start_run()
            self._run_id = self.run.info.run_id
        else:
            self._run_id = _generate_run_id(self.experiment_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        status = "SUCCESS" if exc_type is None else "FAILED"
        run = TrackingRun(
            run_id=self._run_id or _generate_run_id(self.experiment_name),
            experiment_name=self.experiment_name,
            created_at=_utc_now(),
            status=status,
            params=dict(self._params),
            metrics=dict(self._metrics),
            artifacts=list(self._artifacts),
            tags=dict(self._tags),
            model_version=self._model_version,
            feature_version=self._feature_version,
        )
        _save_run(run)

        if self.use_mlflow:
            import mlflow
            mlflow.end_run()

    def log_param(self, key: str, value: Any) -> None:
        self._params[str(key)] = value
        if self.use_mlflow:
            import mlflow
            mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float) -> None:
        self._metrics[str(key)] = float(value)
        if self.use_mlflow:
            import mlflow
            mlflow.log_metric(key, float(value))

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self.log_metric(key, value)

    def log_artifact(self, artifact_path: str) -> None:
        self._artifacts.append(artifact_path)
        if self.use_mlflow:
            import mlflow
            mlflow.log_artifact(artifact_path)

    def set_tag(self, key: str, value: str) -> None:
        self._tags[str(key)] = str(value)
        if self.use_mlflow:
            import mlflow
            mlflow.set_tag(key, value)

    def set_feature_version(self, feature_id: str) -> None:
        self._feature_version = feature_id

    def set_model_version(self, model_id: str) -> None:
        self._model_version = model_id

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id


# =========================================================================
# Auto-tracking wrappers
# =========================================================================
def mlflow_tracked_train(
    results: Sequence[LotteryResult],
    target_name: str,
    model_type: str = "weighted",
    top_k: int = 5,
    use_mlflow: bool = False,
    db_path: str = "xsmb.duckdb",
) -> tuple[Any, ModelVersion]:
    """Train model + auto-track vào MLflow/JSON.

    Args:
        results:     Dữ liệu train
        target_name: Target (loto2, dau, ...)
        model_type:  Loại model
        top_k:       Top-K
        use_mlflow:  Dùng MLflow thực sự hay JSON
        db_path:     Đường dẫn DuckDB

    Returns:
        Tuple (trained_model, ModelVersion)
    """
    experiment_name = f"xsmb-{target_name}"

    with MLflowTracker(experiment_name, db_path=db_path, use_mlflow=use_mlflow) as tracker:
        tracker.log_params({
            "target": target_name,
            "model_type": model_type,
            "top_k": top_k,
            "train_size": len(results),
            "min_train_size": 30,
        })

        if model_type == "weighted":
            from ..models.weighted import fit_tuned_ranking_model, train_ranking_model
            model = train_ranking_model(
                list(results), target_name=target_name, top_k=top_k
            )
        elif model_type == "tuned":
            from ..models.weighted import fit_tuned_ranking_model
            model = fit_tuned_ranking_model(
                list(results), target_name=target_name, top_k=top_k, min_train_size=30
            )
        elif model_type == "xgboost":
            from ..models.xgboost_ranker import train_xgboost_ranker
            model = train_xgboost_ranker(
                list(results), target_name=target_name, top_k=top_k
            )
        else:
            from ..models.weighted import train_ranking_model
            model = train_ranking_model(
                list(results), target_name=target_name, top_k=top_k
            )

        tracker.set_model_version(
            mlflow_model_register(
                model_name=f"xsmb-{model_type}-{target_name}",
                model_type=model_type,
                target=target_name,
                top_k=top_k,
                params={"train_size": len(results)},
            ).model_id
        )

        return model, ModelVersion(
            model_id=tracker._model_version or "",
            model_name=experiment_name,
            model_type=model_type,
            created_at=_utc_now(),
            target=target_name,
            top_k=top_k,
            hit_rate=None,
            precision_at_k=None,
            params={"train_size": len(results)},
            artifact_path=None,
            feature_version=None,
        )


# =========================================================================
# Tracking summary
# =========================================================================
def get_training_runs(
    experiment: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[TrackingRun]:
    """Lấy danh sách training runs, có lọc."""
    runs = _load_runs()
    if experiment:
        runs = [r for r in runs if r.experiment_name == experiment]
    if status:
        runs = [r for r in runs if r.status == status]
    return runs[-limit:]


def print_tracking_summary() -> None:
    """In bảng tổng hợp tracking."""
    runs = _load_runs()
    if not runs:
        print("No training runs tracked yet.")
        return

    print(f"\n{'='*70}")
    print("MLFLOW TRACKING SUMMARY")
    print(f"{'='*70}")

    experiments: Dict[str, List[TrackingRun]] = {}
    for run in runs:
        experiments.setdefault(run.experiment_name, []).append(run)

    for exp_name, exp_runs in sorted(experiments.items()):
        print(f"\nExperiment: {exp_name}")
        print(f"  Total runs: {len(exp_runs)}")
        latest = max(exp_runs, key=lambda r: r.created_at)
        print(f"  Latest: {latest.created_at} [{latest.status}]")
        if latest.metrics:
            for key, val in sorted(latest.metrics.items()):
                print(f"    {key}: {val}")

    print(f"\n{'='*70}")
    print("MODEL REGISTRY")
    print(f"{'='*70}")
    models = get_model_registry()
    if models:
        for mv in sorted(models, key=lambda m: m.created_at, reverse=True)[:10]:
            metrics = []
            if mv.hit_rate is not None:
                metrics.append(f"hit={mv.hit_rate:.3f}")
            if mv.precision_at_k is not None:
                metrics.append(f"p@{mv.top_k}={mv.precision_at_k:.3f}")
            print(f"  {mv.model_name} [{mv.model_type}] {mv.target} top-{mv.top_k} - {', '.join(metrics) if metrics else 'no metrics'}")

    print(f"\n{'='*70}")
    print("FEATURE REGISTRY")
    print(f"{'='*70}")
    features = get_feature_registry()
    if features:
        for fv in sorted(features, key=lambda f: f.created_at, reverse=True)[:10]:
            metrics = []
            if fv.hit_rate is not None:
                metrics.append(f"hit={fv.hit_rate:.3f}")
            if fv.roi is not None:
                metrics.append(f"roi={fv.roi:.3f}")
            print(f"  {fv.feature_name}: {fv.num_features} features, lookback={fv.lookback_days}d - {', '.join(metrics) if metrics else 'no metrics'}")
