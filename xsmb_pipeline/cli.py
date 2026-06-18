from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from .database import (
    DEFAULT_DB_PATH,
    fetch_all_digit_history,
    fetch_loto_history,
    get_connection,
    insert_features_daily,
    integrity_check,
    migrate_csv,
)

from .feature_generators import compute_all_features, feature_summary

from .dataset import DATE_FMT, format_date, iter_dates, load_csv, merge_results, parse_date, write_csv, write_json
from .evaluate import all_signals_backtest, build_full_dashboard_payload, compare_loto2_models, compare_meta_models, compare_models, compare_specialized_models, ensemble_backtest, evaluate_named_ranking_backtest, evaluate_ranking_backtest, evaluate_sklearn_backtest, meta_stack_predictions, signal_backtest, signal_group_backtest, walkforward_ranking_backtest, walkforward_yearly_backtest, yearly_backtest_report
from .models.weighted import fit_tuned_ranking_model, train_ranking_model, predict_next_day
from .plots import plot_walkforward
from .schema import FetchError, LotteryResult, ParseError
from .scraper import RANGE_URLS, bootstrap_history, build_url, crawl_daily, crawl_range, fetch_html, fetch_results_for_dates, parse_result
from .updater import ensure_latest_data, ensure_latest_for_predict, refresh_to_latest


def today_text() -> str:
    return datetime.now().strftime(DATE_FMT)


def default_refresh_end_text(draw_hour: int = 18, draw_minute: int = 30) -> str:
    now = datetime.now()
    draw_time = now.replace(hour=draw_hour, minute=draw_minute, second=0, microsecond=0)
    if now < draw_time:
        return format_date(now - timedelta(days=1))
    return format_date(now)


def next_date_text(date: str) -> str:
    return format_date(parse_date(date) + timedelta(days=1))


def latest_dataset_date(results) -> str | None:
    if not results:
        return None
    return max((item.date for item in results), key=parse_date)


def fetch_results_with_failures(dates: list[str]) -> tuple[list[LotteryResult], list[dict[str, str]]]:
    results: list[LotteryResult] = []
    failures: list[dict[str, str]] = []
    for date in dates:
        try:
            page_html = fetch_html(build_url(date))
            results.append(parse_result(page_html, date))
        except (FetchError, ParseError) as exc:
            failures.append({"date": date, "error": str(exc)})
    return results, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="XSMB lottery data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    day_cmd = sub.add_parser("day", help="Fetch one day")
    day_cmd.add_argument("date", help="Date in DD/MM/YYYY format")
    day_cmd.add_argument("--url", help="Override page URL")
    day_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    range_cmd = sub.add_parser("range", help="Fetch one supported range page and expand to daily pages")
    range_cmd.add_argument("days", type=int, choices=sorted(RANGE_URLS.keys()))
    range_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    range_cmd.add_argument("--json", help="Write JSON output path")
    range_cmd.add_argument("--csv", help="Write CSV output path")

    bootstrap_cmd = sub.add_parser("bootstrap", help="Discover dates from multiple range pages, fetch daily pages, and dedupe")
    bootstrap_cmd.add_argument("windows", nargs="+", type=int, choices=sorted(RANGE_URLS.keys()))
    bootstrap_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    bootstrap_cmd.add_argument("--json", help="Write JSON output path")
    bootstrap_cmd.add_argument("--csv", help="Write CSV output path")

    crawl_cmd = sub.add_parser("crawl", help="Fetch daily pages across a date range")
    crawl_cmd.add_argument("start", help="Start date DD/MM/YYYY")
    crawl_cmd.add_argument("end", help="End date DD/MM/YYYY")
    crawl_cmd.add_argument("--json", help="Write JSON output path")
    crawl_cmd.add_argument("--csv", help="Write CSV output path")

    rebuild_cmd = sub.add_parser("rebuild", help="Crawl a date range and rebuild a deduplicated dataset")
    rebuild_cmd.add_argument("start", help="Start date DD/MM/YYYY")
    rebuild_cmd.add_argument("end", help="End date DD/MM/YYYY")
    rebuild_cmd.add_argument("--input", help="Existing CSV dataset to merge before deduping")
    rebuild_cmd.add_argument("--json", help="Write JSON output path")
    rebuild_cmd.add_argument("--csv", help="Write CSV output path")
    rebuild_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    train_cmd = sub.add_parser("train", help="Train a ranking model from a CSV dataset")
    train_cmd.add_argument("input", help="Input CSV dataset")
    train_cmd.add_argument("--target", choices=["loto2"], required=True)
    train_cmd.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    train_cmd.add_argument("--tuned", action="store_true", help="Tune weights from historical data before training")
    train_cmd.add_argument("--min-train-size", type=int, default=30, help="Minimum history length for tuning")
    train_cmd.add_argument("--model-out", help="Write model JSON")

    eval_cmd = sub.add_parser("evaluate", help="Evaluate a ranking target by time split")
    eval_cmd.add_argument("input", help="Input CSV dataset")
    eval_cmd.add_argument("--target", choices=["loto2"], required=True)
    eval_cmd.add_argument("--split-ratio", type=float, default=0.8)
    eval_cmd.add_argument("--top-k", type=int, default=5)

    walk_cmd = sub.add_parser("walkforward", help="Run walk-forward evaluation")
    walk_cmd.add_argument("input", help="Input CSV dataset")
    walk_cmd.add_argument("--target", choices=["loto2"], required=True)
    walk_cmd.add_argument("--top-k", type=int, default=5)
    walk_cmd.add_argument("--min-train-size", type=int, default=30)

    yearly_cmd = sub.add_parser("walkforward-yearly", help="Run yearly walk-forward summary")
    yearly_cmd.add_argument("input", help="Input CSV dataset")
    yearly_cmd.add_argument("--target", choices=["loto2"], required=True)
    yearly_cmd.add_argument("--top-k", type=int, default=5)
    yearly_cmd.add_argument("--min-train-size", type=int, default=30)

    backtest_report_cmd = sub.add_parser("backtest-report", help="Export yearly backtest report")
    backtest_report_cmd.add_argument("input", help="Input CSV dataset")
    backtest_report_cmd.add_argument("--target", choices=["loto2"], required=True)
    backtest_report_cmd.add_argument("--top-k", type=int, default=5)
    backtest_report_cmd.add_argument("--min-train-size", type=int, default=30)

    plot_cmd = sub.add_parser("plot", help="Export walk-forward visualization or fallback CSV")
    plot_cmd.add_argument("input", help="Input CSV dataset")
    plot_cmd.add_argument("--target", choices=["loto2"], required=True)
    plot_cmd.add_argument("--top-k", type=int, default=5)
    plot_cmd.add_argument("--output", required=True, help="PNG output path")

    predict_cmd = sub.add_parser("predict", help="Predict ranked candidates for the next draw")
    predict_cmd.add_argument("input", help="Input CSV dataset")
    predict_cmd.add_argument("--top-k-loto2", type=int, default=5)
    predict_cmd.add_argument("--tuned", action="store_true", help="Tune weights from historical data before predicting")

    sklearn_cmd = sub.add_parser("sklearn-train", help="Train a tabular ranking model from a CSV dataset")
    sklearn_cmd.add_argument("input", help="Input CSV dataset")
    sklearn_cmd.add_argument("--target", choices=["loto2"], required=True)
    sklearn_cmd.add_argument("--top-k", type=int, default=5)
    sklearn_cmd.add_argument("--min-train-size", type=int, default=30)
    sklearn_cmd.add_argument("--model-name", default="logistic", help="Model backend: logistic, random_forest, extra_trees, mlp, xgboost")
    sklearn_cmd.add_argument("--model-out", help="Write model summary JSON")

    sklearn_eval_cmd = sub.add_parser("sklearn-evaluate", help="Evaluate a tabular ranking model by time split")
    sklearn_eval_cmd.add_argument("input", help="Input CSV dataset")
    sklearn_eval_cmd.add_argument("--target", choices=["loto2"], required=True)
    sklearn_eval_cmd.add_argument("--split-ratio", type=float, default=0.8)
    sklearn_eval_cmd.add_argument("--top-k", type=int, default=5)
    sklearn_eval_cmd.add_argument("--min-train-size", type=int, default=30)
    sklearn_eval_cmd.add_argument("--model-name", default="logistic", help="Model backend: logistic, random_forest, extra_trees, mlp, xgboost")

    xgb_export_cmd = sub.add_parser("xgboost-export", help="Train XGBoost ranking and export importance artifact JSON")
    xgb_export_cmd.add_argument("input", help="Input CSV dataset")
    xgb_export_cmd.add_argument("--target", choices=["loto2"], required=True)
    xgb_export_cmd.add_argument("--top-k", type=int, default=5)
    xgb_export_cmd.add_argument("--min-train-size", type=int, default=30)
    xgb_export_cmd.add_argument("--output", required=True, help="Artifact JSON output path")
    xgb_export_cmd.add_argument("--selected-top-k", type=int, default=8, help="Number of features to keep after selection")

    compare_cmd = sub.add_parser("compare-models", help="Compare weighted and tabular ranking models across targets")
    compare_cmd.add_argument("input", help="Input CSV dataset")
    compare_cmd.add_argument("--split-ratio", type=float, default=0.8)
    compare_cmd.add_argument("--top-k", type=int, default=5)
    compare_cmd.add_argument("--min-train-size", type=int, default=30)

    meta_compare_cmd = sub.add_parser("compare-meta-models", help="Benchmark Layer 2 meta models")
    meta_compare_cmd.add_argument("input", help="Input CSV dataset")
    meta_compare_cmd.add_argument("--split-ratio", type=float, default=0.8)
    meta_compare_cmd.add_argument("--top-k", type=int, default=5)
    meta_compare_cmd.add_argument("--min-train-size", type=int, default=30)

    specialized_compare_cmd = sub.add_parser("compare-specialized-models", help="Benchmark specialized target presets")
    specialized_compare_cmd.add_argument("input", help="Input CSV dataset")
    specialized_compare_cmd.add_argument("--split-ratio", type=float, default=0.8)
    specialized_compare_cmd.add_argument("--top-k", type=int, default=5)
    specialized_compare_cmd.add_argument("--min-train-size", type=int, default=30)

    meta_stack_cmd = sub.add_parser("meta-stack", help="Build a meta stacking ranking from Layer 1 models")
    meta_stack_cmd.add_argument("input", help="Input CSV dataset")
    meta_stack_cmd.add_argument("--top-k", type=int, default=5)
    meta_stack_cmd.add_argument("--min-train-size", type=int, default=30)

    loto2_compare_cmd = sub.add_parser("compare-loto2-models", help="Benchmark weighted, ML, and DL-style loto2 models")
    loto2_compare_cmd.add_argument("input", help="Input CSV dataset")
    loto2_compare_cmd.add_argument("--split-ratio", type=float, default=0.8)
    loto2_compare_cmd.add_argument("--top-k", type=int, default=5)
    loto2_compare_cmd.add_argument("--min-train-size", type=int, default=30)
    loto2_compare_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON")


    dashboard_cmd = sub.add_parser("dashboard-export", help="Export a combined dashboard payload JSON")
    dashboard_cmd.add_argument("input", help="Input CSV dataset")
    dashboard_cmd.add_argument("--output", required=True, help="Dashboard JSON output path")
    dashboard_cmd.add_argument("--split-ratio", type=float, default=0.8)
    dashboard_cmd.add_argument("--top-k", type=int, default=5)
    dashboard_cmd.add_argument("--min-train-size", type=int, default=30)
    dashboard_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    signal_backtest_cmd = sub.add_parser("signal-backtest", help="Run walk-forward backtests for individual signals or the ensemble")
    signal_backtest_cmd.add_argument("input", help="Input CSV dataset")
    signal_backtest_cmd.add_argument("--target", choices=["loto2"], default="loto2")
    signal_backtest_cmd.add_argument("--mode", choices=["single", "all", "groups", "ensemble"], default="single")
    signal_backtest_cmd.add_argument("--signal", help="Signal name required for --mode single")
    signal_backtest_cmd.add_argument("--top-k", type=int, default=5)
    signal_backtest_cmd.add_argument("--min-train-size", type=int, default=30)
    signal_backtest_cmd.add_argument("--recent-rows", type=int, default=10)

    refresh_cmd = sub.add_parser("refresh-dashboard", help="Update missing daily results, save dataset, and export dashboard payload")
    refresh_cmd.add_argument("--csv", default="xsmb_full.csv", help="CSV dataset path to update")
    refresh_cmd.add_argument("--json", default="xsmb_full.json", help="JSON dataset path to update")
    refresh_cmd.add_argument("--dashboard-output", default="dashboard/dashboard-payload.json", help="Dashboard payload output path")
    refresh_cmd.add_argument("--start", help="Optional refresh start date DD/MM/YYYY; default is day after latest dataset date")
    refresh_cmd.add_argument("--end", help="Optional refresh end date DD/MM/YYYY; default is latest likely posted draw date")
    refresh_cmd.add_argument("--draw-hour", type=int, default=18, help="Daily draw cutoff hour used for default refresh end")
    refresh_cmd.add_argument("--draw-minute", type=int, default=30, help="Daily draw cutoff minute used for default refresh end")
    refresh_cmd.add_argument("--split-ratio", type=float, default=0.8)
    refresh_cmd.add_argument("--top-k", type=int, default=3)
    refresh_cmd.add_argument("--min-train-size", type=int, default=30)
    refresh_cmd.add_argument("--skip-dashboard", action="store_true", help="Only update CSV/JSON, do not export dashboard payload")
    refresh_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON summary")

    # Phase 9: Telegram Bot commands
    telegram_cmd = sub.add_parser("telegram-test", help="Test Telegram Bot locally (Phase 9 - P8.1)")
    telegram_cmd.add_argument("--csv", default="xsmb_full.csv", help="Path to xsmb_full.csv")
    telegram_cmd.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to xsmb.duckdb")
    telegram_cmd.add_argument("--top-k", type=int, default=5, help="Top-K for ranking")
    telegram_cmd.add_argument("--top-n", type=int, default=20, help="Top-N for display")

    ensure_cmd = sub.add_parser("ensure-data", help="Auto-update missing days before predict/analyze (18h30 daily check)")
    ensure_cmd.add_argument("--csv", default="xsmb_full.csv", help="CSV dataset path")
    ensure_cmd.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB path")
    ensure_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON summary")

    schedule_cmd = sub.add_parser("schedule", help="Run daily scheduler: auto-update + predict after 18h30 (Phase 9 - P8.x)")
    schedule_cmd.add_argument("--csv", default="xsmb_full.csv", help="CSV dataset path")
    schedule_cmd.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB path")
    schedule_cmd.add_argument("--predictions-dir", default="predictions", help="Predictions output directory")
    schedule_cmd.add_argument("--token", default="", help="Telegram Bot Token (or set TELEGRAM_BOT_TOKEN env)")
    schedule_cmd.add_argument("--top-n", type=int, default=20, help="Number of predictions")
    schedule_cmd.add_argument("--tuned", action="store_true", help="Use tuned model")

    mlflow_summary_cmd = sub.add_parser("mlflow-summary", help="Print MLflow tracking summary (Phase 9 - P9.2)")
    mlflow_summary_cmd.add_argument("--experiment", help="Filter by experiment name")
    mlflow_summary_cmd.add_argument("--status", help="Filter by run status (SUCCESS/FAILED)")
    mlflow_summary_cmd.add_argument("--limit", type=int, default=50, help="Limit number of runs")

    track_model_cmd = sub.add_parser("track-model", help="Register a model version (Phase 9 - P9.2)")
    track_model_cmd.add_argument("model_name", help="Model name (e.g. xsmb-xgboost-loto2)")
    track_model_cmd.add_argument("--type", default="unknown", help="Model type (xgboost, weighted, neural, ...)")
    track_model_cmd.add_argument("--target", default="loto2", help="Target (loto2, dau, duoi, ...)")
    track_model_cmd.add_argument("--top-k", type=int, default=5, help="Top-K")
    track_model_cmd.add_argument("--params", help="JSON params string")
    track_model_cmd.add_argument("--artifact", help="Artifact path")
    track_model_cmd.add_argument("--feature-version", help="Feature version ID")

    track_feature_cmd = sub.add_parser("track-feature", help="Register a feature version (Phase 9 - P9.2)")
    track_feature_cmd.add_argument("generator", help="Feature generator name")
    track_feature_cmd.add_argument("--lookback", type=int, default=400, help="Lookback days")
    track_feature_cmd.add_argument("--num-features", type=int, default=0, help="Number of features")
    track_feature_cmd.add_argument("--hit-rate", type=float, help="Hit rate")
    track_feature_cmd.add_argument("--roi", type=float, help="ROI")

    predict_ensure_cmd = sub.add_parser("predict-ensure", help="Ensure data is current then predict (Phase 9)")
    predict_ensure_cmd.add_argument("--csv", default="xsmb_full.csv", help="CSV dataset path")
    predict_ensure_cmd.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB path")
    predict_ensure_cmd.add_argument("--top-k", type=int, default=5, help="Top-K for ranking")
    predict_ensure_cmd.add_argument("--tuned", action="store_true", help="Use tuned model")
    predict_ensure_cmd.add_argument("--top-n", type=int, default=20, help="Top-N to show")
    predict_ensure_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    migrate_cmd = sub.add_parser(
        "migrate-duckdb",
        help="Migrate CSV vao DuckDB (4 bang nguon) + integrity check (P1.7)",
    )
    migrate_cmd.add_argument("--csv", default="xsmb_full.csv", help="CSV nguon")
    migrate_cmd.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB dich")
    migrate_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    integrity_cmd = sub.add_parser(
        "integrity-check",
        help="Kiem tra tinh nhat quan 4 bang nguon trong DuckDB",
    )
    integrity_cmd.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB can check")
    integrity_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    features_cmd = sub.add_parser(
        "compute-features",
        help="Sinh toan bo feature Phase 3 (P2.1->P2.19) ghi vao features_daily",
    )
    features_cmd.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB target")
    features_cmd.add_argument(
        "--start",
        help="Ngay bat dau tinh feature (DD/MM/YYYY). Mac dinh: ngay cuoi cung trong DuckDB.",
    )
    features_cmd.add_argument(
        "--end",
        help="Ngay ket thuc (DD/MM/YYYY). Mac dinh = --start (chi tinh 1 ngay).",
    )
    features_cmd.add_argument(
        "--lookback",
        type=int,
        default=400,
        help="So ngay lich su lay lam input cho cac generator (mac dinh 400).",
    )
    features_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON")


    args = parser.parse_args()

    if args.command == "day":
        url = args.url or build_url(args.date)
        result = parse_result(fetch_html(url), args.date)
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2 if args.pretty else None))
        return

    if args.command == "range":
        results = crawl_range(args.days)
        if args.json:
            write_json(Path(args.json), results)
        if args.csv:
            write_csv(Path(args.csv), results)
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2 if args.pretty else None))
        return

    if args.command == "bootstrap":
        results = bootstrap_history(args.windows)
        if args.json:
            write_json(Path(args.json), results)
        if args.csv:
            write_csv(Path(args.csv), results)
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2 if args.pretty else None))
        return

    if args.command == "crawl":
        results = crawl_daily(args.start, args.end)
        if args.json:
            write_json(Path(args.json), results)
        if args.csv:
            write_csv(Path(args.csv), results)
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
        return

    if args.command == "rebuild":
        existing = load_csv(Path(args.input)) if args.input else []
        existing_dates = {item.date for item in existing}
        requested_dates = list(iter_dates(args.start, args.end))
        missing_dates = [date for date in requested_dates if date not in existing_dates]
        crawled, failed_dates = fetch_results_with_failures(missing_dates)
        results = merge_results(existing, crawled)
        if args.json:
            write_json(Path(args.json), results)
        if args.csv:
            write_csv(Path(args.csv), results)
        payload = {
            "existing_count": len(existing),
            "requested_count": len(requested_dates),
            "skipped_existing_count": len(requested_dates) - len(missing_dates),
            "missing_count": len(missing_dates),
            "crawled_count": len(crawled),
            "merged_count": len(results),
            "results": [asdict(item) for item in results],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    if args.command == "train":
        results = load_csv(Path(args.input))
        trainer = fit_tuned_ranking_model if args.tuned else train_ranking_model
        model = trainer(results, target_name=args.target, top_k=args.top_k, min_train_size=args.min_train_size) if args.tuned else trainer(results, target_name=args.target, top_k=args.top_k)
        payload = {
            "model": "tuned-weighted-ranking" if args.tuned else "weighted-ranking",
            "target": args.target,
            "number_width": model.number_width,
            "total": model.total,
            "top_k": model.top_k,
            "weights": model.weights,
            "counts": model.counts,
            "top_predictions": model.predict(),
            "frequency_top_predictions": model.predict_baseline(),
        }
        if args.model_out:
            Path(args.model_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate":
        results = load_csv(Path(args.input))
        metrics = evaluate_ranking_backtest(results, split_ratio=args.split_ratio, target_name=args.target, top_k=args.top_k)
        print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
        return

    if args.command == "walkforward":
        results = ensure_latest_for_predict(Path(args.input), DEFAULT_DB_PATH)
        rows = walkforward_ranking_backtest(results, target_name=args.target, top_k=args.top_k, min_train_size=args.min_train_size)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if args.command == "walkforward-yearly":
        results = ensure_latest_for_predict(Path(args.input), DEFAULT_DB_PATH)
        payload = walkforward_yearly_backtest(results, target_name=args.target, top_k=args.top_k, min_train_size=args.min_train_size)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "backtest-report":
        results = ensure_latest_for_predict(Path(args.input), DEFAULT_DB_PATH)
        payload = yearly_backtest_report(results, target_name=args.target, top_k=args.top_k, min_train_size=args.min_train_size)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "plot":
        results = load_csv(Path(args.input))
        message = plot_walkforward(results, target_name=args.target, top_k=args.top_k, output_path=Path(args.output))
        print(json.dumps({"message": message}, ensure_ascii=False, indent=2))
        return

    if args.command == "predict":
        results = ensure_latest_for_predict(Path(args.input), DEFAULT_DB_PATH)
        payload = predict_next_day(
            results,
            top_k_loto2=args.top_k_loto2,
            tuned=args.tuned,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "sklearn-train":
        from .models.sklearn_ranker import fit_named_sklearn_ranking_model

        results = load_csv(Path(args.input))
        model = fit_named_sklearn_ranking_model(results, target_name=args.target, top_k=args.top_k, min_train_size=args.min_train_size, model_name=args.model_name)
        payload = {
            "model": f"sklearn-{model.model_name}-ranking",
            "target": args.target,
            "number_width": model.number_width,
            "top_k": model.top_k,
            "top_predictions": model.predict(),
        }
        if args.model_out:
            Path(args.model_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "sklearn-evaluate":
        results = load_csv(Path(args.input))
        metrics = evaluate_named_ranking_backtest(results, split_ratio=args.split_ratio, target_name=args.target, top_k=args.top_k, min_train_size=args.min_train_size, model_name=args.model_name)
        print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
        return

    if args.command == "xgboost-export":
        from .database import DEFAULT_DB_PATH as _LOCAL_DB, create_schema, get_connection, insert_features_daily, insert_results
        from .feature_generators import compute_all_features
        from .models.xgboost_ranker import save_xgboost_feature_artifact, train_xgboost_with_selected_features

        results = load_csv(Path(args.input))
        connection = get_connection(_LOCAL_DB)
        try:
            create_schema(connection)
            insert_results(connection, results)
            for result in results:
                rows = compute_all_features(
                    loto_history=[(item.date, item.special[-2:]) for item in results if item.date <= result.date],
                    digit_history=[
                        (item.date, "special", 0, position_index, digit)
                        for item in results
                        if item.date <= result.date
                        for position_index, digit in enumerate(item.special)
                    ],
                    as_of_date=result.date,
                )
                insert_features_daily(connection, rows)
        finally:
            connection.close()
        model = train_xgboost_with_selected_features(results, db_path=_LOCAL_DB, target_name=args.target, top_k=args.top_k, min_train_size=args.min_train_size, top_k_features=args.selected_top_k)
        artifact = save_xgboost_feature_artifact(model, args.output)
        print(json.dumps(artifact, ensure_ascii=False, indent=2))
        return

    if args.command == "compare-models":
        results = ensure_latest_for_predict(Path(args.input), DEFAULT_DB_PATH)
        payload = compare_models(results, split_ratio=args.split_ratio, top_k=args.top_k, min_train_size=args.min_train_size)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "compare-meta-models":
        results = ensure_latest_for_predict(Path(args.input), DEFAULT_DB_PATH)
        payload = compare_meta_models(results, split_ratio=args.split_ratio, top_k=args.top_k, min_train_size=args.min_train_size)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "compare-specialized-models":
        results = ensure_latest_for_predict(Path(args.input), DEFAULT_DB_PATH)
        payload = compare_specialized_models(results, split_ratio=args.split_ratio, top_k=args.top_k, min_train_size=args.min_train_size)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "meta-stack":
        results = ensure_latest_for_predict(Path(args.input), DEFAULT_DB_PATH)
        payload = meta_stack_predictions(results, top_k=args.top_k, min_train_size=args.min_train_size)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "compare-loto2-models":
        results = ensure_latest_for_predict(Path(args.input), DEFAULT_DB_PATH)
        payload = compare_loto2_models(results, split_ratio=args.split_ratio, top_k=args.top_k, min_train_size=args.min_train_size)
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    if args.command == "dashboard-export":
        results = ensure_latest_for_predict(Path(args.input), DEFAULT_DB_PATH)
        payload = build_full_dashboard_payload(results, split_ratio=args.split_ratio, top_k=args.top_k, min_train_size=args.min_train_size)
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"message": f"Saved dashboard payload to {args.output}"}, ensure_ascii=False, indent=2))
        return

    if args.command == "signal-backtest":
        results = load_csv(Path(args.input))
        if args.mode == "single":
            if not args.signal:
                raise ValueError("--signal là bắt buộc khi --mode single")
            payload = signal_backtest(
                results,
                target_name=args.target,
                signal_name=args.signal,
                top_k=args.top_k,
                min_train_size=args.min_train_size,
                recent_rows=args.recent_rows,
            )
        elif args.mode == "all":
            payload = all_signals_backtest(
                results,
                target_name=args.target,
                top_k=args.top_k,
                min_train_size=args.min_train_size,
                recent_rows=args.recent_rows,
            )
        elif args.mode == "groups":
            payload = signal_group_backtest(
                results,
                target_name=args.target,
                top_k=args.top_k,
                min_train_size=args.min_train_size,
                recent_rows=args.recent_rows,
            )
        else:
            payload = ensemble_backtest(
                results,
                target_name=args.target,
                top_k=args.top_k,
                min_train_size=args.min_train_size,
                recent_rows=args.recent_rows,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "refresh-dashboard":
        csv_path = Path(args.csv)
        json_path = Path(args.json) if args.json else None
        dashboard_path = Path(args.dashboard_output)
        existing = load_csv(csv_path) if csv_path.exists() else []
        latest_before = latest_dataset_date(existing)
        refresh_end = args.end or default_refresh_end_text(args.draw_hour, args.draw_minute)
        refresh_start = args.start or (next_date_text(latest_before) if latest_before else args.end)
        if latest_before is None and args.start is None:
            refresh_start = refresh_end
        requested_dates = list(iter_dates(refresh_start, refresh_end)) if parse_date(refresh_start) <= parse_date(refresh_end) else []
        existing_dates = {item.date for item in existing}
        missing_dates = [date for date in requested_dates if date not in existing_dates]
        crawled, failed_dates = fetch_results_with_failures(missing_dates)
        results = merge_results(existing, crawled)
        should_write_dataset = bool(crawled) or not csv_path.exists() or (json_path is not None and not json_path.exists())
        if results and should_write_dataset:
            write_csv(csv_path, results)
            if json_path:
                write_json(json_path, results)
        dashboard_exported = False
        if not args.skip_dashboard:
            dashboard_payload = build_full_dashboard_payload(results, split_ratio=args.split_ratio, top_k=args.top_k, min_train_size=args.min_train_size)
            dashboard_path.parent.mkdir(parents=True, exist_ok=True)
            dashboard_path.write_text(json.dumps(dashboard_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            dashboard_exported = True
        payload = {
            "csv": str(csv_path),
            "json": str(json_path) if json_path else None,
            "dashboard_output": str(dashboard_path) if dashboard_exported else None,
            "latest_before": latest_before,
            "latest_after": latest_dataset_date(results),
            "refresh_start": refresh_start,
            "refresh_end": refresh_end,
            "draw_cutoff": f"{args.draw_hour:02d}:{args.draw_minute:02d}",
            "requested_count": len(requested_dates),
            "missing_count": len(missing_dates),
            "crawled_count": len(crawled),
            "failed_count": len(failed_dates),
            "failed_dates": failed_dates,
            "dataset_count": len(results),
            "dataset_written": should_write_dataset,
            "dashboard_exported": dashboard_exported,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    if args.command == "migrate-duckdb":
        summary = migrate_csv(csv_path=args.csv, db_path=args.db)
        conn = get_connection(args.db, read_only=True)
        try:
            integrity = integrity_check(conn)
        finally:
            conn.close()
        payload = {"migrate": summary, "integrity": integrity}
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    if args.command == "integrity-check":
        conn = get_connection(args.db, read_only=True)
        try:
            integrity = integrity_check(conn)
        finally:
            conn.close()
        print(json.dumps(integrity, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    if args.command == "compute-features":
        # Mo connection write-mode de insert_features_daily co the ghi.
        conn = get_connection(args.db, read_only=False)
        try:
            loto_history = fetch_loto_history(conn)
            digit_history = fetch_all_digit_history(conn)


            # Tap hop tat ca distinct date trong loto_history.
            all_dates: list[str] = []
            seen: set[str] = set()
            for date_text, _loto in loto_history:
                if date_text not in seen:
                    seen.add(date_text)
                    all_dates.append(date_text)
            all_dates.sort(key=parse_date)

            if not all_dates:
                print(json.dumps({"error": "No data in DuckDB"}, ensure_ascii=False))
                return

            start_date = args.start or all_dates[-1]
            end_date = args.end or start_date
            target_dates = [
                date_text for date_text in all_dates
                if parse_date(start_date) <= parse_date(date_text) <= parse_date(end_date)
            ]

            written_per_day: dict[str, int] = {}
            last_summary: dict[str, int] = {}
            for as_of_date in target_dates:
                # Cat lookback de generator chay nhanh.
                cutoff = parse_date(as_of_date) - timedelta(days=args.lookback)
                trimmed_loto = [
                    (date_text, loto) for date_text, loto in loto_history
                    if cutoff <= parse_date(date_text) <= parse_date(as_of_date)
                ]
                trimmed_digit = [
                    row for row in digit_history
                    if cutoff <= parse_date(row[0]) <= parse_date(as_of_date)
                ]
                rows = compute_all_features(
                    loto_history=trimmed_loto,
                    digit_history=trimmed_digit,
                    as_of_date=as_of_date,
                )
                inserted = insert_features_daily(conn, rows)
                written_per_day[as_of_date] = inserted
                last_summary = feature_summary(rows)

            payload = {
                "db": args.db,
                "lookback_days": args.lookback,
                "target_dates_count": len(target_dates),
                "first_date": target_dates[0] if target_dates else None,
                "last_date": target_dates[-1] if target_dates else None,
                "total_rows_written": sum(written_per_day.values()),
                "rows_per_day_sample": dict(list(written_per_day.items())[:3]),
                "feature_groups_last_day": last_summary,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        finally:
            conn.close()
        return

    if args.command == "schedule":
        from .scheduler import DailyScheduler
        scheduler = DailyScheduler(
            csv_path=Path(args.csv),
            db_path=args.db,
            predictions_dir=Path(args.predictions_dir),
            telegram_token=args.token or None,
            top_n=args.top_n,
            tuned=args.tuned,
        )
        print(json.dumps({
            "message": "Scheduler started",
            "csv": str(args.csv),
            "db": str(args.db),
            "predictions_dir": str(args.predictions_dir),
            "telegram": "enabled" if args.token else "disabled",
            "top_n": args.top_n,
            "model": "tuned" if args.tuned else "weighted",
        }, ensure_ascii=False, indent=2))
        try:
            scheduler.run()
        except KeyboardInterrupt:
            scheduler.stop()
            print(json.dumps({"message": "Scheduler stopped"}, ensure_ascii=False, indent=2))
        return

    if args.command == "mlflow-summary":
        from .mlflow_tracking import get_training_runs, get_model_registry, get_feature_registry, print_tracking_summary
        runs = get_training_runs(
            experiment=getattr(args, "experiment", None),
            status=getattr(args, "status", None),
            limit=args.limit,
        )
        models = get_model_registry()
        features = get_feature_registry()
        payload = {
            "total_runs": len(runs),
            "total_models": len(models),
            "total_feature_versions": len(features),
            "runs": [asdict(r) for r in runs],
            "latest_models": [asdict(m) for m in sorted(models, key=lambda x: x.created_at, reverse=True)[:10]],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print_tracking_summary()
        return

    if args.command == "track-model":
        from .mlflow_tracking import mlflow_model_register
        import json as _json
        params = {}
        if args.params:
            params = _json.loads(args.params)
        mv = mlflow_model_register(
            model_name=args.model_name,
            model_type=getattr(args, "type", "unknown"),
            target=args.target,
            top_k=args.top_k,
            params=params,
            artifact_path=args.artifact or None,
            feature_version=args.feature_version or None,
        )
        print(json.dumps(asdict(mv), ensure_ascii=False, indent=2))
        return

    if args.command == "track-feature":
        from .mlflow_tracking import mlflow_feature_register
        fv = mlflow_feature_register(
            generator=args.generator,
            feature_names=[],
            lookback_days=args.lookback,
            num_features=args.num_features,
            hit_rate=getattr(args, "hit_rate", None),
            roi=getattr(args, "roi", None),
        )
        print(json.dumps(asdict(fv), ensure_ascii=False, indent=2))
        return

    if args.command == "predict-ensure":
        from .scheduler import compute_top_n
        info = refresh_to_latest(Path(args.csv), args.db)
        results = load_csv(Path(args.csv)) if Path(args.csv).exists() else []
        latest = get_latest_draw_date(Path(args.csv), args.db)
        rows = compute_top_n(results, top_n=args.top_n, tuned=args.tuned)
        payload = {
            "update_info": info,
            "latest_date": latest,
            "dataset_count": len(results),
            "model": "tuned-weighted-ranking" if args.tuned else "weighted-ranking",
            "top_k": args.top_k,
            "top_n": args.top_n,
            "predictions": rows,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    if args.command == "ensure-data":
        from pathlib import Path as P

        info = refresh_to_latest(P(args.csv), args.db)
        print(json.dumps(info, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    if args.command == "telegram-test":
        from .telegram_bot import XSMBBot
        from pathlib import Path as P2

        bot = XSMBBot(csv_path=P2(args.csv), db_path=args.db, top_k=args.top_k, top_n=args.top_n)
        print("=== /start ===")
        print(bot.handle_start())
        print()
        print("=== /status ===")
        print(bot.handle_status())
        print()
        print("=== /predict ===")
        print(bot.handle_predict())
        print()
        print("=== /explain 27 ===")
        print(bot.handle_explain("27"))
        print()
        print("=== TEST DONE ===")
        return

    raise ValueError(f"Unsupported command: {args.command}")


