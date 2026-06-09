from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from .dataset import DATE_FMT, format_date, iter_dates, load_csv, merge_results, parse_date, write_csv, write_json
from .evaluate import all_signals_backtest, build_full_dashboard_payload, compare_loto2_models, compare_models, ensemble_backtest, evaluate_named_ranking_backtest, evaluate_ranking_backtest, evaluate_sklearn_backtest, signal_backtest, signal_group_backtest, walkforward_ranking_backtest
from .models.weighted import fit_tuned_ranking_model, train_ranking_model, predict_next_day
from .plots import plot_walkforward
from .schema import FetchError, LotteryResult, ParseError
from .scraper import RANGE_URLS, bootstrap_history, build_url, crawl_daily, crawl_range, fetch_html, fetch_results_for_dates, parse_result


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
    sklearn_cmd.add_argument("--model-name", default="logistic", help="Model backend: logistic, random_forest, extra_trees, mlp")
    sklearn_cmd.add_argument("--model-out", help="Write model summary JSON")

    sklearn_eval_cmd = sub.add_parser("sklearn-evaluate", help="Evaluate a tabular ranking model by time split")
    sklearn_eval_cmd.add_argument("input", help="Input CSV dataset")
    sklearn_eval_cmd.add_argument("--target", choices=["loto2"], required=True)
    sklearn_eval_cmd.add_argument("--split-ratio", type=float, default=0.8)
    sklearn_eval_cmd.add_argument("--top-k", type=int, default=5)
    sklearn_eval_cmd.add_argument("--min-train-size", type=int, default=30)
    sklearn_eval_cmd.add_argument("--model-name", default="logistic", help="Model backend: logistic, random_forest, extra_trees, mlp")

    compare_cmd = sub.add_parser("compare-models", help="Compare weighted and tabular ranking models across targets")
    compare_cmd.add_argument("input", help="Input CSV dataset")
    compare_cmd.add_argument("--split-ratio", type=float, default=0.8)
    compare_cmd.add_argument("--top-k", type=int, default=5)
    compare_cmd.add_argument("--min-train-size", type=int, default=30)

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
        results = load_csv(Path(args.input))
        rows = walkforward_ranking_backtest(results, target_name=args.target, top_k=args.top_k, min_train_size=args.min_train_size)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if args.command == "plot":
        results = load_csv(Path(args.input))
        message = plot_walkforward(results, target_name=args.target, top_k=args.top_k, output_path=Path(args.output))
        print(json.dumps({"message": message}, ensure_ascii=False, indent=2))
        return

    if args.command == "predict":
        results = load_csv(Path(args.input))
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

    if args.command == "compare-models":
        results = load_csv(Path(args.input))
        payload = compare_models(results, split_ratio=args.split_ratio, top_k=args.top_k, min_train_size=args.min_train_size)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "compare-loto2-models":
        results = load_csv(Path(args.input))
        payload = compare_loto2_models(results, split_ratio=args.split_ratio, top_k=args.top_k, min_train_size=args.min_train_size)
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    if args.command == "dashboard-export":
        results = load_csv(Path(args.input))
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

    raise ValueError(f"Unsupported command: {args.command}")
