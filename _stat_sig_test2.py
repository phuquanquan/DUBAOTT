"""
Statistical Significance Test - Optimized Version.

Strategy:
1. Chi chay walkforward 1 lan cho tat ca signals (khong yearly breakdown)
2. Dung sampling moi 5 ngay de giam 5x thoi gian (con du stat)
3. Bootstrap CI + Binomial test + Permutation test

Chay: python _stat_sig_test2.py
Output: agents/stat_sig_test_topk3.json
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.evaluate import evaluate_prediction_set
from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
from xsmb_pipeline.targets import actual_targets, target_width
from xsmb_pipeline.models.weighted import candidate_universe


# ---------- Signal scoring (inline, fast) ----------

def _score_hot_trend(rows: List, candidate: str, width: int, cache: dict) -> float:
    """hot_trend: O(104) per candidate."""
    if "loto2_rolling14" not in cache:
        cache["loto2_rolling14"] = {}
        cache["loto2_rolling90"] = {}
        recent = rows[-14:] if len(rows) > 14 else rows
        full90 = rows[-90:] if len(rows) > 90 else rows
        items14 = []
        for r in recent:
            items14.extend(actual_targets(r, "loto2"))
        items90 = []
        for r in full90:
            items90.extend(actual_targets(r, "loto2"))
        for cand in candidate_universe(width):
            cache["loto2_rolling14"][cand] = items14.count(cand) / max(1, len(items14))
            cache["loto2_rolling90"][cand] = items90.count(cand) / max(1, len(items90))
    r14 = cache["loto2_rolling14"].get(candidate, 0.0)
    r90 = cache["loto2_rolling90"].get(candidate, 0.0)
    return max(0.0, min(1.0, 0.5 + (r14 - r90) * 5.0))


def _score_cold_return(rows: List, candidate: str, width: int, cache: dict) -> float:
    """cold_return: O(n) per candidate."""
    last_seen = None
    for offset, result in enumerate(reversed(rows), start=1):
        if candidate in actual_targets(result, "loto2"):
            last_seen = offset
            break
    val = (last_seen or min(len(rows), 90)) / 90.0
    return max(0.0, min(1.0, val))


def _score_touch(rows: List, candidate: str, width: int, cache: dict) -> float:
    """touch: O(n) per candidate."""
    digits = set(int(ch) for ch in candidate.zfill(width))
    items = []
    for r in rows:
        items.extend(actual_targets(r, "loto2"))
    if not items:
        return 0.0
    matches = sum(1 for item in items if digits & set(int(ch) for ch in item.zfill(width)))
    return matches / len(items)


def _score_inversion(rows: List, candidate: str, width: int, cache: dict) -> float:
    """inversion: O(n)."""
    inverted = candidate[::-1]
    items = []
    for r in rows:
        items.extend(actual_targets(r, "loto2"))
    if not items:
        return 0.0
    return items.count(inverted) / len(items)


def _score_pascal(rows: List, candidate: str, width: int, cache: dict) -> float:
    """pascal: O(1)."""
    digits = [int(ch) for ch in candidate.zfill(width)]
    if width >= 3:
        checks = 2
        matches = int((digits[0] + digits[-1]) % 10 == digits[1])
        matches += int((digits[0] + digits[1]) % 10 == digits[-1])
    else:
        checks = 1
        matches = int(abs(digits[0] - digits[1]) in {0, 1, 9})
    return matches / checks if checks else 0.0


def _score_composition(rows: List, candidate: str, width: int, cache: dict) -> float:
    """composition (digit part freq): O(n) for all candidates cached."""
    if "composition" not in cache:
        cache["composition"] = {}
        items = []
        for r in rows:
            items.extend(actual_targets(r, "loto2"))
        if not items:
            for cand in candidate_universe(width):
                cache["composition"][cand] = 0.0
        else:
            freq = {}
            for item in items:
                suffix = item[-2:] if width == 2 else item[-1:]
                freq[suffix] = freq.get(suffix, 0) + 1
            total = len(items)
            for cand in candidate_universe(width):
                cache["composition"][cand] = freq.get(cand, 0) / total
    return cache["composition"].get(candidate, 0.0)


def _score_frequency(rows: List, candidate: str, width: int, cache: dict) -> float:
    """frequency baseline: O(n) cached."""
    if "freq" not in cache:
        cache["freq"] = {}
        items = []
        for r in rows:
            items.extend(actual_targets(r, "loto2"))
        if not items:
            for cand in candidate_universe(width):
                cache["freq"][cand] = 1.0 / 100.0
        else:
            freq = {}
            for item in items:
                freq[item] = freq.get(item, 0) + 1
            total = len(items)
            for cand in candidate_universe(width):
                cache["freq"][cand] = freq.get(cand, 0) / total
    return cache["freq"].get(candidate, 0.0)


# Mapping signal name -> scoring function
SIGNAL_SCORERS: dict[str, callable] = {
    "hot_trend": _score_hot_trend,
    "cold_return": _score_cold_return,
    "touch": _score_touch,
    "inversion": _score_inversion,
    "pascal": _score_pascal,
    "composition": _score_composition,
    "frequency": _score_frequency,
}


def rank_candidates(rows: List, signal_name: str, top_k: int, width: int) -> List[Tuple[str, float]]:
    """Rank 100 candidates by signal score. Uses caching for O(n) signals."""
    cache: dict = {}
    scorer = SIGNAL_SCORERS.get(signal_name)
    if scorer is None:
        return [(f"{i:02d}", 1.0 / (i + 1)) for i in range(top_k)]
    scored = [(cand, scorer(rows, cand, width, cache)) for cand in candidate_universe(width)]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_k]


def theoretical_baseline(top_k: int, universe_size: int = 100) -> float:
    """p(hit) = 1 - C(99,k)/C(100,k)."""
    n, k = universe_size, top_k
    return 1.0 - (math.comb(n - 1, k) / math.comb(n, k))


def binomial_test(n: int, hits: int, p0: float) -> float:
    """One-tailed binomial: P(X >= hits)."""
    if hits > n or hits < 0:
        return 1.0
    total = 0.0
    for k in range(hits, n + 1):
        total += math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
    return min(total, 1.0)


def wilson_ci(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score CI for proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = hits / n
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def permutation_pvalue(signal_hits: List[int], freq_hits: List[int], n_perm: int = 5000) -> float:
    """Permutation test: is signal hit rate significantly above frequency?"""
    n = len(signal_hits)
    if n == 0:
        return 1.0
    signal_wins = sum(1 for s, f in zip(signal_hits, freq_hits) if s > f)
    freq_wins = sum(1 for s, f in zip(signal_hits, freq_hits) if s < f)
    observed_diff = signal_wins - freq_wins

    # Pool: 1 = signal wins, 0 = freq wins
    pool = [1] * signal_wins + [0] * freq_wins
    count_extreme = 0
    for _ in range(n_perm):
        random.shuffle(pool)
        perm_diff = sum(1 for i in range(n) if pool[i] > 0) - sum(1 for i in range(n) if pool[i] == 0)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    return count_extreme / n_perm


def run_fast_stat_test(
    csv_path: str = "xsmb_full.csv",
    top_k: int = 3,
    min_train_size: int = 30,
    sample_every: int = 5,
    n_perm: int = 5000,
) -> dict:
    """Chay stat test nhanh voi sampling."""
    t_start = time.time()
    path = Path(csv_path)
    results = sort_results(load_csv(path))
    width = target_width("loto2")
    universe = 10 ** width
    p0 = theoretical_baseline(top_k, universe)

    test_indices = list(range(min_train_size, len(results), sample_every))
    print(f"Dataset: {len(results)} rows, {len(test_indices)} test days (sampled every {sample_every})")
    print(f"Baseline p(hit)={p0:.4f} ({p0*100:.2f}%), top_k={top_k}")
    print(f"Permutation reps: {n_perm}")
    print()

    all_results: List[dict] = []
    all_signal_names = [d.name for d in SIGNAL_DEFINITIONS] + ["frequency"]

    # Pre-cache frequency for all test days
    print("Computing frequency baseline for all test days...")
    freq_cache: List[List[Tuple[str, float]]] = []
    for idx in test_indices:
        train = list(results[:idx])
        ranked = rank_candidates(train, "frequency", top_k, width)
        freq_cache.append(ranked)

    for sig_idx, signal_name in enumerate(all_signal_names):
        t_sig = time.time()
        signal_hits: List[int] = []
        freq_hits: List[int] = []
        precisions: List[float] = []
        freq_precisions: List[float] = []
        yearly_hits: dict[str, List[int]] = {}
        yearly_freq_hits: dict[str, List[int]] = {}

        for day_idx, split_idx in enumerate(test_indices):
            train = list(results[:split_idx])
            test_row = results[split_idx]
            actual = actual_targets(test_row, "loto2")

            signal_ranked = rank_candidates(train, signal_name, top_k, width)
            freq_ranked = freq_cache[day_idx]

            predicted = [c for c, _ in signal_ranked]
            freq_predicted = [c for c, _ in freq_ranked]

            hit, _, precision, freq_prec = evaluate_prediction_set(predicted, actual, universe)
            freq_hit, _, _, _ = evaluate_prediction_set(freq_predicted, actual, universe)

            signal_hits.append(int(hit))
            freq_hits.append(int(freq_hit))
            precisions.append(precision)
            freq_precisions.append(freq_prec)

            year = test_row.date[-4:]
            yearly_hits.setdefault(year, []).append(int(hit))
            yearly_freq_hits.setdefault(year, []).append(int(freq_hit))

        n = len(signal_hits)
        signal_sum = sum(signal_hits)
        freq_sum = sum(freq_hits)
        signal_rate = signal_sum / n
        freq_rate = freq_sum / n
        delta = (signal_rate - freq_rate) * 100.0

        binom_p = binomial_test(n, signal_sum, p0)
        ci_lo, ci_hi = wilson_ci(signal_sum, n)
        perm_p = permutation_pvalue(signal_hits, freq_hits, n_perm)

        wins = sum(1 for s, f in zip(signal_hits, freq_hits) if s > f)
        losses = sum(1 for s, f in zip(signal_hits, freq_hits) if s < f)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

        yearly_summary = []
        for year in sorted(yearly_hits.keys()):
            yr_sig = yearly_hits[year]
            yr_freq = yearly_freq_hits[year]
            yr_n = len(yr_sig)
            yr_rate = sum(yr_sig) / yr_n
            yr_freq_rate = sum(yr_freq) / yr_n
            yearly_summary.append({
                "year": year,
                "test_size": yr_n,
                "hit_rate_pct": yr_rate * 100.0,
                "freq_hit_rate_pct": yr_freq_rate * 100.0,
                "delta_pct": (yr_rate - yr_freq_rate) * 100.0,
                "wins": sum(1 for s, f in zip(yr_sig, yr_freq) if s > f),
                "losses": sum(1 for s, f in zip(yr_sig, yr_freq) if s < f),
            })

        all_results.append({
            "signal": signal_name,
            "test_size": n,
            "hits": signal_sum,
            "hit_rate_pct": signal_rate * 100.0,
            "freq_hit_rate_pct": freq_rate * 100.0,
            "delta_pct": delta,
            "theoretical_baseline_pct": p0 * 100.0,
            "binom_pvalue": binom_p,
            "binom_significant": binom_p < 0.05,
            "ci_lower_pct": ci_lo * 100.0,
            "ci_upper_pct": ci_hi * 100.0,
            "ci_covers_baseline": ci_lo <= p0 <= ci_hi,
            "perm_pvalue": perm_p,
            "perm_significant": perm_p < 0.05,
            "win_rate_pct": win_rate * 100.0,
            "wins": wins,
            "losses": losses,
            "yearly": yearly_summary,
        })

        sig_binom = "***" if binom_p < 0.001 else ("**" if binom_p < 0.01 else ("*" if binom_p < 0.05 else ""))
        sig_perm = "***" if perm_p < 0.001 else ("**" if perm_p < 0.01 else ("*" if perm_p < 0.05 else ""))
        print(f"  [{sig_idx+1:2d}/{len(all_signal_names)}] {signal_name:<30} Hit={signal_rate*100:5.2f}% "
              f"Freq={freq_rate*100:5.2f}% Delta={delta:+6.2f}pp "
              f"binom-p={binom_p:.4f}{sig_binom} perm-p={perm_p:.4f}{sig_perm} "
              f"CI=[{ci_lo*100:.2f},{ci_hi*100:.2f}] t={time.time()-t_sig:.1f}s")

    all_results.sort(key=lambda x: (-x["delta_pct"], x["binom_pvalue"], x["perm_pvalue"], x["signal"]))

    sig_binom_all = [r for r in all_results if r["binom_significant"]]
    sig_perm_all = [r for r in all_results if r["perm_significant"]]
    sig_both = [r for r in all_results if r["binom_significant"] and r["perm_significant"]]
    above_baseline = [r for r in all_results if r["ci_lower_pct"] > p0 * 100]

    print(f"\n{'='*110}")
    print(f"[Summary] Total: {len(all_results)} signals, {len(test_indices)} test days (sampled every {sample_every})")
    print(f"  Binomial significant (p<0.05, above theoretical {p0*100:.2f}%): {len(sig_binom_all)}")
    print(f"  Permutation significant (p<0.05, above frequency): {len(sig_perm_all)}")
    print(f"  Significant on BOTH tests: {len(sig_both)}")
    print(f"  CI strictly above baseline: {len(above_baseline)}")

    print(f"\n[Both Significant - Real Edge]")
    for r in sig_both:
        print(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, Delta={r['delta_pct']:+.2f}pp, "
              f"binom-p={r['binom_pvalue']:.4f}, perm-p={r['perm_pvalue']:.4f}")

    print(f"\n[Above Theoretical Baseline]")
    for r in above_baseline:
        print(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, CI=[{r['ci_lower_pct']:.2f},{r['ci_upper_pct']:.2f}], Delta={r['delta_pct']:+.2f}pp")

    print(f"\n[Yearly Delta - Top 5 by Delta]")
    for r in all_results[:5]:
        deltas = [f"{yr['year']}:{yr['delta_pct']:+.1f}" for yr in r["yearly"]]
        print(f"  {r['signal']}: {', '.join(deltas)}")

    print(f"\nTotal time: {time.time()-t_start:.1f}s")

    return {
        "dataset_size": len(results),
        "date_range": f"{results[0].date} - {results[-1].date}",
        "test_days": len(test_indices),
        "sampling_every": sample_every,
        "config": {"top_k": top_k, "universe_size": universe, "min_train_size": min_train_size, "n_permutations": n_perm},
        "theoretical_baseline_pct": p0 * 100.0,
        "total_signals": len(all_results),
        "sig_binomial_count": len(sig_binom_all),
        "sig_permutation_count": len(sig_perm_all),
        "sig_both_count": len(sig_both),
        "above_baseline_count": len(above_baseline),
        "results": all_results,
    }


if __name__ == "__main__":
    import json
    random.seed(42)
    np.random.seed(42)

    csv = sys.argv[1] if len(sys.argv) > 1 else "xsmb_full.csv"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    payload = run_fast_stat_test(csv_path=csv, top_k=top_k, min_train_size=30, sample_every=5, n_perm=5000)

    output_path = Path("agents/stat_sig_test_topk3.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_path}")
