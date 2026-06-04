from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence, Tuple

from .evaluate import walkforward_ranking_backtest
from .schema import LotteryResult


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def export_walkforward_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("Không có dữ liệu để export")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_walkforward(results: Sequence[LotteryResult], target_name: str, top_k: int, output_path: Path) -> str:
    rows = walkforward_ranking_backtest(results, target_name=target_name, top_k=top_k)
    plt = try_import_matplotlib()
    if plt is None:
        fallback = output_path.with_suffix(".csv")
        export_walkforward_csv(fallback, rows)
        return f"matplotlib is not installed; exported fallback CSV to {fallback}"

    dates = [row["date"] for row in rows]
    weighted_hits = [row["hit"] for row in rows]
    frequency_hits = [row["frequency_hit"] for row in rows]
    baseline_hits = [row["baseline_hit"] for row in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, weighted_hits, label="weighted-hit", linewidth=1.5)
    ax.plot(dates, frequency_hits, label="frequency-hit", linewidth=1.5)
    ax.plot(dates, baseline_hits, label="random-expected-hit", linewidth=1.5)
    ax.set_title(f"Walk-forward hit comparison: {target_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hit")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return f"Saved plot to {output_path}"
