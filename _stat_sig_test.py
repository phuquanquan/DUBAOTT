"""
Statistical Significance Test cho Signal Backtests.

Chay: python _stat_sig_test.py

Kiem tra 3 dong thuat:
1. Binomial test vs theoretical baseline (p0 = 1 - C(99,k)/C(100,k))
2. Bootstrap 95% CI cho hit rate
3. Permutation test vs frequency baseline

Baseline theory: voi top_k=3, p(hit) = 1 - C(99,3)/C(100,3) = 1 - (99*98*97)/(100*99*98) = 1 - 97/100 = 0.03 = 3%
"""
from __future__ import annotations

import math
import random
import statistics
from pathlib import Path
from typing import List, Tuple

from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.evaluate import walkforward_signal_backtest, rank_candidates_by_frequency
from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
from xsmb_pipeline.targets import actual_targets, target_width


def theoretical_baseline(top_k: int, universe_size: int = 100) -> float:
    """Tinh baseline p(hit) = 1 - C(universe-1, top_k) / C(universe, top_k)."""
    n = universe_size
    k = top_k
    if k > n:
        return 0.0
    c_nk = math.comb(n, k)
    c_n_minus_1_k = math.comb(n - 1, k)
    return 1.0 - (c_n_minus_1_k / c_nk)


def binomial_pvalue(n: int, hits: int, p0: float) -> float:
    """One-tailed binomial test: P(X >= hits | n, p0)."""
    if hits > n or hits < 0:
        return 0.0
    total = 0.0
    for k in range(hits, n + 1):
        total += math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
    return min(total, 1.0)


def bootstrap_ci(values: List[int], n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float]:
    """Bootstrap 95% CI cho hit rate (proportion of 1s)."""
    observed = values
    n = len(observed)
    if n == 0:
        return (0.0, 0.0)
    alpha = 1.0 - confidence
    estimates: List[float] = []
    for _ in range(n_bootstrap):
        sample = [random.choice(observed) for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1.0 - alpha / 2) * n_bootstrap)
    lower_idx = max(0, lower_idx)
    upper_idx = min(n_bootstrap - 1, upper_idx)
    return (estimates[lower_idx], estimates[upper_idx])


def permutation_test(
    signal_hits: List[int],
    signal_precision: List[float],
    frequency_hits: List[int],
    frequency_precision: List[float],
    n_permutations: int = 10000,
) -> Tuple[float, float]:
    """Permutation test: is signal > frequency more often than by chance?"""
    n = len(signal_hits)
    if n == 0:
        return (1.0, 1.0)

    # Permutation test on hit rate difference
    signal_wins = sum(1 for s, f in zip(signal_hits, frequency_hits) if s > f)
    freq_wins = sum(1 for s, f in zip(signal_hits, frequency_hits) if s < f)
    ties = sum(1 for s, f in zip(signal_hits, frequency_hits) if s == f)
    observed_diff = signal_wins - freq_wins

    count_extreme = 0
    for _ in range(n_permutations):
        perm_diff = 0
        for s, f in zip(signal_hits, frequency_hits):
            if random.random() < 0.5:
                perm_diff += 1 if f > s else (-1 if f < s else 0)
            else:
                perm_diff += 1 if s > f else (-1 if s < f else 0)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    hit_pvalue = count_extreme / n_permutations

    # Permutation test on precision
    signal_prec_wins = sum(1 for s, f in zip(signal_precision, frequency_precision) if s > f)
    freq_prec_wins = sum(1 for s, f in zip(signal_precision, frequency_precision) if s < f)
    observed_prec_diff = signal_prec_wins - freq_prec_wins

    count_extreme_prec = 0
    for _ in range(n_permutations):
        perm_diff = 0
        for s, f in zip(signal_precision, frequency_precision):
            if random.random() < 0.5:
                perm_diff += 1 if f > s else (-1 if f < s else 0)
            else:
                perm_diff += 1 if s > f else (-1 if s < f else 0)
        if abs(perm_diff) >= abs(observed_prec_diff):
            count_extreme_prec += 1
    prec_pvalue = count_extreme_prec / n_permutations

    return (hit_pvalue, prec_pvalue)


def yearly_breakdown(rows: List[dict]) -> List[dict]:
    """Tinh hit rate moi nam."""
    by_year: dict[str, List[dict]] = {}
    for row in rows:
        year = row["date"][-4:]
        by_year.setdefault(year, []).append(row)

    summaries = []
    for year in sorted(by_year.keys()):
        yr_rows = by_year[year]
        n = len(yr_rows)
        hits = sum(int(r["hit"]) for r in yr_rows)
        freq_hits = sum(int(r["frequency_hit"]) for r in yr_rows)
        hit_rate = hits / n if n else 0.0
        freq_rate = freq_hits / n if n else 0.0
        summaries.append({
            "year": year,
            "test_size": n,
            "hit_rate": hit_rate,
            "hit_rate_pct": hit_rate * 100.0,
            "freq_hit_rate": freq_rate,
            "freq_hit_rate_pct": freq_rate * 100.0,
            "delta": (hit_rate - freq_rate) * 100.0,
            "wins": sum(1 for r in yr_rows if r["hit"] > r["frequency_hit"]),
            "losses": sum(1 for r in yr_rows if r["hit"] < r["frequency_hit"]),
        })
    return summaries


def run_statistical_tests(
    csv_path: str | Path = "xsmb_full.csv",
    top_k: int = 3,
    min_train_size: int = 30,
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
) -> dict:
    """Chay tat ca statistical tests cho tat ca signals."""
    path = Path(csv_path)
    if not path.exists():
        return {"error": f"Khong tim thay: {csv_path}"}

    results = sort_results(load_csv(path))
    if len(results) < min_train_size + 5:
        return {"error": f"Khong du du lieu: {len(results)} dong"}

    universe_size = 10 ** target_width("loto2")
    p0 = theoretical_baseline(top_k, universe_size)

    signal_names = [d.name for d in SIGNAL_DEFINITIONS]
    all_results: List[dict] = []

    print(f"\nStatistical Significance Test")
    print(f"  Dataset: {len(results)} rows ({results[0].date} -> {results[-1].date})")
    print(f"  top_k={top_k}, universe={universe_size}, min_train={min_train_size}")
    print(f"  Theoretical baseline p(hit) = {p0:.4f} ({p0*100:.2f}%)")
    print(f"  Bootstrap CI: {n_bootstrap} reps")
    print(f"  Permutation test: {n_permutations} shuffles")
    print(f"\n{'='*120}")

    for signal_name in signal_names:
        rows = walkforward_signal_backtest(
            results, target_name="loto2", signal_name=signal_name,
            top_k=top_k, min_train_size=min_train_size,
        )
        if not rows:
            continue

        hits = [int(r["hit"]) for r in rows]
        freq_hits = [int(r["frequency_hit"]) for r in rows]
        precisions = [float(r["precision"]) for r in rows]
        freq_precisions = [float(r["frequency_precision"]) for r in rows]

        n = len(hits)
        observed_hits = sum(hits)
        observed_hit_rate = observed_hits / n

        # Binomial test vs theoretical
        binom_pvalue = binomial_pvalue(n, observed_hits, p0)

        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_ci(hits, n_bootstrap=n_bootstrap)

        # Permutation test vs frequency
        perm_hit_pvalue, perm_prec_pvalue = permutation_test(
            hits, precisions, freq_hits, freq_precisions, n_permutations=n_permutations,
        )

        # Yearly breakdown
        yearly = yearly_breakdown(rows)

        # Win rate vs frequency
        wins = sum(1 for s, f in zip(hits, freq_hits) if s > f)
        losses = sum(1 for s, f in zip(hits, freq_hits) if s < f)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

        all_results.append({
            "signal": signal_name,
            "test_size": n,
            "hits": observed_hits,
            "hit_rate": observed_hit_rate,
            "hit_rate_pct": observed_hit_rate * 100.0,
            "freq_hit_rate": sum(freq_hits) / n,
            "freq_hit_rate_pct": (sum(freq_hits) / n) * 100.0,
            "delta_vs_freq": (observed_hit_rate - sum(freq_hits) / n) * 100.0,
            "theoretical_baseline": p0,
            "binom_pvalue": binom_pvalue,
            "binom_significant": binom_pvalue < 0.05,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_lower_pct": ci_lower * 100.0,
            "ci_upper_pct": ci_upper * 100.0,
            "ci_covers_zero": ci_lower <= p0 <= ci_upper,
            "perm_hit_pvalue": perm_hit_pvalue,
            "perm_prec_pvalue": perm_prec_pvalue,
            "perm_hit_significant": perm_hit_pvalue < 0.05,
            "perm_prec_significant": perm_prec_pvalue < 0.05,
            "win_rate_vs_freq": win_rate,
            "win_rate_vs_freq_pct": win_rate * 100.0,
            "wins_vs_freq": wins,
            "losses_vs_freq": losses,
            "yearly": yearly,
        })

    # Sort by combined significance score
    def sig_score(item: dict) -> Tuple[float, float, float, str]:
        return (
            -item["delta_vs_freq"],
            item["binom_pvalue"],
            item["perm_hit_pvalue"],
            item["signal"],
        )

    all_results.sort(key=sig_score)

    # Print summary table
    print(f"\n{'Sig':<3} {'Signal':<28} {'N':>6} {'Hit%':>7} {'Freq%':>7} {'Delta':>7} "
          f"{'Binom-p':>9} {'Perm-p':>8} {'CI-lower':>9} {'CI-upper':>9} {'WinRate':>8}")
    print(f"{'-'*3} {'-'*28} {'-'*6} {'-'*7} {'-'*7} {'-'*7} "
          f"{'-'*9} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")

    for item in all_results:
        sig_binom = "***" if item["binom_pvalue"] < 0.001 else ("**" if item["binom_pvalue"] < 0.01 else ("*" if item["binom_pvalue"] < 0.05 else ""))
        sig_perm = "***" if item["perm_hit_pvalue"] < 0.001 else ("**" if item["perm_hit_pvalue"] < 0.01 else ("*" if item["perm_hit_pvalue"] < 0.05 else ""))
        delta = item["delta_vs_freq"]
        delta_str = f"{delta:+.2f}"
        win_rate = item["win_rate_vs_freq_pct"]
        print(
            f"{sig_binom:<3} {item['signal']:<28} {item['test_size']:>6} "
            f"{item['hit_rate_pct']:>7.2f} {item['freq_hit_rate_pct']:>7.2f} {delta_str:>7} "
            f"{item['binom_pvalue']:>9.4f} {item['perm_hit_pvalue']:>8.4f} "
            f"{item['ci_lower_pct']:>9.2f} {item['ci_upper_pct']:>9.2f} "
            f"{win_rate:>7.1f}%"
        )

    print(f"\n{'='*120}")
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05")
    print("CI covers zero = CI includes theoretical baseline = not significantly above baseline")
    print("Permutation test: signal vs frequency (10k permutations)")
    print(f"Theoretical baseline = {p0*100:.2f}% (top_k={top_k}, universe={universe_size})")

    # Print yearly breakdown for top 5 signals
    print(f"\n{'='*120}")
    print("[Yearly Breakdown - Top 5 Signals by Delta]")
    for item in all_results[:5]:
        print(f"\n  {item['signal']} (Hit={item['hit_rate_pct']:.2f}%, Delta={item['delta_vs_freq']:+.2f}pp, p={item['binom_pvalue']:.4f})")
        for yr in item["yearly"]:
            print(f"    {yr['year']}: N={yr['test_size']:>4}, Hit={yr['hit_rate_pct']:>5.1f}%, "
                  f"Freq={yr['freq_hit_rate_pct']:>5.1f}%, Delta={yr['delta']:>+6.2f}pp, "
                  f"W={yr['wins']:>3}/L={yr['losses']:>3}")

    # Summary of statistically significant signals
    sig_binom_signals = [r for r in all_results if r["binom_significant"]]
    sig_perm_signals = [r for r in all_results if r["perm_hit_significant"]]
    both_sig = [r for r in all_results if r["binom_significant"] and r["perm_hit_significant"]]

    print(f"\n{'='*120}")
    print(f"[Summary]")
    print(f"  Signals beating theoretical baseline (binom p<0.05): {len(sig_binom_signals)}/{len(all_results)}")
    print(f"  Signals beating frequency (perm p<0.05): {len(sig_perm_signals)}/{len(all_results)}")
    print(f"  Signals significant on BOTH tests: {len(both_sig)}/{len(all_results)}")

    if sig_binom_signals:
        print(f"\n  [Binomial Significant - Above Theoretical Baseline]")
        for r in sig_binom_signals:
            print(f"    {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, Delta={r['delta_vs_freq']:+.2f}pp, p={r['binom_pvalue']:.4f}")

    if both_sig:
        print(f"\n  [Both Tests Significant - Real Edge]")
        for r in both_sig:
            print(f"    {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, Delta={r['delta_vs_freq']:+.2f}pp, "
                  f"binom-p={r['binom_pvalue']:.4f}, perm-p={r['perm_hit_pvalue']:.4f}")

    return {
        "dataset_size": len(results),
        "date_range": f"{results[0].date} - {results[-1].date}",
        "config": {"top_k": top_k, "universe_size": universe_size, "min_train_size": min_train_size,
                   "n_bootstrap": n_bootstrap, "n_permutations": n_permutations},
        "theoretical_baseline": p0,
        "theoretical_baseline_pct": p0 * 100.0,
        "total_signals": len(all_results),
        "sig_binomial_count": len(sig_binom_signals),
        "sig_permutation_count": len(sig_perm_signals),
        "sig_both_count": len(both_sig),
        "results": all_results,
        "sig_binomial_signals": sig_binom_signals,
        "sig_permutation_signals": sig_perm_signals,
        "sig_both_signals": both_sig,
    }


if __name__ == "__main__":
    import json
    import sys

    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    csv = sys.argv[1] if len(sys.argv) > 1 else "xsmb_full.csv"

    random.seed(42)
    payload = run_statistical_tests(csv_path=csv, top_k=top_k, n_bootstrap=10000, n_permutations=10000)

    output_path = f"agents/stat_sig_test_topk{top_k}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_path}")
