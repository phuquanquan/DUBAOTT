"""
Backtest Ensemble - Walk-Forward evaluation cho tat ca signals + ensemble.

Chay: python xsmb_pipeline/backtest_ensemble.py
Output: Bang ranking 33 signals + ensemble evaluation.

Port y tu: junlangzi/Lottery-Predictor (performance.ini)
  - Top-3 accuracy
  - Top-5 accuracy
  - Top-10 accuracy
Nhung dung walk-forward thay vi chi 1 split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.evaluate import (
    all_signals_backtest,
    ensemble_backtest,
)
from xsmb_pipeline.signals import SIGNAL_DEFINITIONS


def run_full_backtest(
    csv_path: str | Path = "xsmb_full.csv",
    top_k: int = 3,
    min_train_size: int = 30,
    recent_rows: int = 10,
) -> dict:
    """Chay walk-forward backtest cho tat ca signals + ensemble."""
    from xsmb_pipeline.dataset import load_csv, sort_results
    from xsmb_pipeline.evaluate import (
        all_signals_backtest,
        ensemble_backtest,
    )
    path = Path(csv_path)
    if not path.exists():
        return {"error": f"Khong tim thay file: {csv_path}"}

    results = sort_results(load_csv(path))
    if len(results) < min_train_size + 5:
        return {"error": f"Khong du du lieu: {len(results)} dong, can it nhat {min_train_size + 5}"}

    print(f"Loaded {len(results)} results: {results[0].date} -> {results[-1].date}")
    print(f"Running walk-forward backtest (top_k={top_k}, min_train={min_train_size})...")

    all_payload = all_signals_backtest(
        results,
        target_name="loto2",
        top_k=top_k,
        min_train_size=min_train_size,
        recent_rows=recent_rows,
    )
    ens_payload = ensemble_backtest(
        results,
        target_name="loto2",
        top_k=top_k,
        min_train_size=min_train_size,
        recent_rows=recent_rows,
    )

    leaderboard = all_payload["evaluations"]
    verdicts = all_payload["verdict_counts"]

    return {
        "dataset_size": len(results),
        "date_range": f"{results[0].date} - {results[-1].date}",
        "top_k": top_k,
        "min_train_size": min_train_size,
        "test_period": f"{all_payload.get('dataset_size', len(results))} days",
        "verdicts": verdicts,
        "leaderboard": leaderboard,
        "ensemble": ens_payload["evaluation"],
        "kept_signals": all_payload["filter_summary"]["kept_signals"],
        "watch_signals": all_payload["filter_summary"]["watch_signals"],
        "dropped_signals": all_payload["filter_summary"]["dropped_signals"],
    }


def print_backtest_report(payload: dict) -> None:
    """In ra bang bao cao."""
    if "error" in payload:
        print(f"Loi: {payload['error']}")
        return

    print(f"\n{'='*70}")
    print(f"  BACKTEST REPORT - Walk-Forward Ensemble")
    print(f"{'='*70}")
    print(f"  Data: {payload['date_range']}  ({payload['dataset_size']} days)")
    print(f"  Config: top_k={payload['top_k']}, min_train={payload['min_train_size']}")
    print(f"{'='*70}")

    print(f"\n[Verdicts]")
    v = payload["verdicts"]
    print(f"  Keep: {v['keep']}  |  Watch: {v['watch']}  |  Drop: {v['drop']}")

    print(f"\n[Ensemble Signal Evaluation]")
    e = payload["ensemble"]
    print(f"  Hit Rate@{payload['top_k']}:    {e['hit_rate_pct']:.2f}%  (baseline: {e['baseline_hit_rate_pct']:.2f}%)")
    print(f"  Precision@{payload['top_k']}:   {e['precision_at_k_pct']:.2f}%  (baseline: {e['baseline_precision_at_pct']:.2f}%)")
    print(f"  vs Frequency:         {e['hit_rate_delta_vs_frequency_pct']:+.2f}pp  (precision: {e['precision_delta_vs_frequency_pct']:+.2f}pp)")
    print(f"  Verdict: {e['research_verdict']}  |  Ranking Score: {e['ranking_score_pct']:.2f}%")

    print(f"\n[Signal Leaderboard - Top 20]")
    print(f"  {'Rank':<4} {'Signal':<35} {'Hit%':>7} {'Prec%':>7} {'Delta%':>8} {'Verdict':<8} {'Score':>6}")
    print(f"  {'-'*4} {'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*6}")
    for i, item in enumerate(payload["leaderboard"][:20], 1):
        delta = item.get("hit_rate_delta_vs_frequency_pct", 0)
        sign = "+" if delta > 0 else ""
        print(
            f"  {i:<4} {item['signal']:<35} "
            f"{item['hit_rate_pct']:>7.2f} {item['precision_at_k_pct']:>7.2f} "
            f"{sign}{delta:>7.2f} {item['research_verdict']:<8} {item['ranking_score_pct']:>6.2f}"
        )

    print(f"\n[Signals by Verdict]")
    if payload["kept_signals"]:
        print(f"  KEEP ({len(payload['kept_signals'])}): {', '.join(sorted(payload['kept_signals']))}")
    if payload["watch_signals"]:
        print(f"  WATCH ({len(payload['watch_signals'])}): {', '.join(sorted(payload['watch_signals']))}")
    if payload["dropped_signals"]:
        print(f"  DROP ({len(payload['dropped_signals'])}): {', '.join(sorted(payload['dropped_signals']))}")

    print(f"\n[Comparison: Ensemble vs Best Individual]")
    best = payload["leaderboard"][0] if payload["leaderboard"] else None
    if best:
        print(f"  Ensemble:  Hit={e['hit_rate_pct']:.2f}%, Prec={e['precision_at_k_pct']:.2f}%")
        print(f"  Best:     Hit={best['hit_rate_pct']:.2f}%, Prec={best['precision_at_k_pct']:.2f}%")
        hit_diff = e['hit_rate_pct'] - best['hit_rate_pct']
        prec_diff = e['precision_at_k_pct'] - best['precision_at_k_pct']
        print(f"  Delta:    Hit={hit_diff:+.2f}pp, Prec={prec_diff:+.2f}pp")

    print(f"\n{'='*70}")


def save_backtest_report(payload: dict, output_path: str | Path = "agents/backtest_report.json") -> None:
    """Luu ket qua ra JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Report saved: {path}")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "xsmb_full.csv"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    payload = run_full_backtest(csv_path=csv, top_k=top_k, min_train_size=30, recent_rows=10)
    print_backtest_report(payload)
    save_backtest_report(payload)
