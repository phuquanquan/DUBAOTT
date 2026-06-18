"""
Statistical Significance Test - Incremental version.
Save cache + process in chunks so we never lose progress.
"""
from __future__ import annotations

import json
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


# ---------- Fast signal scorers ----------

def score_hot_trend(rows: List, candidate: str, width: int, cache: dict) -> float:
    if "hot14" not in cache:
        cache["hot14"] = {}
        cache["hot90"] = {}
        recent = rows[-14:] if len(rows) > 14 else rows
        full90 = rows[-90:] if len(rows) > 90 else rows
        items14 = []
        for r in recent:
            items14.extend(actual_targets(r, "loto2"))
        items90 = []
        for r in full90:
            items90.extend(actual_targets(r, "loto2"))
        for cand in candidate_universe(width):
            cache["hot14"][cand] = items14.count(cand) / max(1, len(items14))
            cache["hot90"][cand] = items90.count(cand) / max(1, len(items90))
    r14 = cache["hot14"].get(candidate, 0.0)
    r90 = cache["hot90"].get(candidate, 0.0)
    return max(0.0, min(1.0, 0.5 + (r14 - r90) * 5.0))


def score_cold_return(rows: List, candidate: str, width: int, cache: dict) -> float:
    last_seen = None
    for offset, result in enumerate(reversed(rows), start=1):
        if candidate in actual_targets(result, "loto2"):
            last_seen = offset
            break
    val = (last_seen or min(len(rows), 90)) / 90.0
    return max(0.0, min(1.0, val))


def score_touch(rows: List, candidate: str, width: int, cache: dict) -> float:
    digits = set(int(ch) for ch in candidate.zfill(width))
    items = []
    for r in rows:
        items.extend(actual_targets(r, "loto2"))
    if not items:
        return 0.0
    matches = sum(1 for item in items if digits & set(int(ch) for ch in item.zfill(width)))
    return matches / len(items)


def score_inversion(rows: List, candidate: str, width: int, cache: dict) -> float:
    inverted = candidate[::-1]
    items = []
    for r in rows:
        items.extend(actual_targets(r, "loto2"))
    return items.count(inverted) / len(items) if items else 0.0


def score_pascal(rows: List, candidate: str, width: int, cache: dict) -> float:
    digits = [int(ch) for ch in candidate.zfill(width)]
    if width >= 3:
        checks = 2
        matches = int((digits[0] + digits[-1]) % 10 == digits[1])
        matches += int((digits[0] + digits[1]) % 10 == digits[-1])
    else:
        checks = 1
        matches = int(abs(digits[0] - digits[1]) in {0, 1, 9})
    return matches / checks if checks else 0.0


def score_composition(rows: List, candidate: str, width: int, cache: dict) -> float:
    if "comp" not in cache:
        cache["comp"] = {}
        items = []
        for r in rows:
            items.extend(actual_targets(r, "loto2"))
        freq = {}
        for item in items:
            suffix = item[-2:] if width == 2 else item[-1:]
            freq[suffix] = freq.get(suffix, 0) + 1
        total = len(items) or 1
        for cand in candidate_universe(width):
            cache["comp"][cand] = freq.get(cand, 0) / total
    return cache["comp"].get(candidate, 0.0)


def score_frequency(rows: List, candidate: str, width: int, cache: dict) -> float:
    if "freq" not in cache:
        cache["freq"] = {}
        items = []
        for r in rows:
            items.extend(actual_targets(r, "loto2"))
        freq = {}
        for item in items:
            freq[item] = freq.get(item, 0) + 1
        total = len(items) or 1
        for cand in candidate_universe(width):
            cache["freq"][cand] = freq.get(cand, 0) / total
    return cache["freq"].get(candidate, 0.0)


SCORERS = {
    "hot_trend": score_hot_trend,
    "cold_return": score_cold_return,
    "touch": score_touch,
    "inversion": score_inversion,
    "pascal": score_pascal,
    "composition": score_composition,
    "frequency": score_frequency,
}


def rank_candidates(rows: List, signal_name: str, top_k: int, width: int) -> List[Tuple[str, float]]:
    cache: dict = {}
    scorer = SCORERS.get(signal_name)
    if scorer is None:
        return [(f"{i:02d}", 1.0 / (i + 1)) for i in range(top_k)]
    scored = [(cand, scorer(rows, cand, width, cache)) for cand in candidate_universe(width)]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_k]


def theoretical_baseline(top_k: int, universe: int = 100) -> float:
    return 1.0 - (math.comb(universe - 1, top_k) / math.comb(universe, top_k))


def binomial_test(n: int, hits: int, p0: float) -> float:
    if hits > n or hits < 0:
        return 1.0
    total = 0.0
    for k in range(hits, n + 1):
        total += math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
    return min(total, 1.0)


def wilson_ci(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p_hat = hits / n
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def permutation_pvalue(sig_hits: List[int], freq_hits: List[int], n_perm: int = 5000) -> float:
    n = len(sig_hits)
    if n == 0:
        return 1.0
    wins = sum(1 for s, f in zip(sig_hits, freq_hits) if s > f)
    losses = sum(1 for s, f in zip(sig_hits, freq_hits) if s < f)
    observed = wins - losses
    pool = [1] * wins + [0] * losses
    count_extreme = 0
    for _ in range(n_perm):
        random.shuffle(pool)
        perm_diff = sum(1 for i in range(n) if pool[i] > 0) - sum(1 for i in range(n) if pool[i] == 0)
        if abs(perm_diff) >= abs(observed):
            count_extreme += 1
    return count_extreme / n_perm


def main() -> None:
    t_start = time.time()
    random.seed(42)
    np.random.seed(42)

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "xsmb_full.csv"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    sample_every = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    n_perm = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    chunk = sys.argv[5] if len(sys.argv) > 5 else None  # "freq" or "N" for Nth signal

    cache_path = Path("agents/_stat_cache.json")
    results_path = Path("agents/_stat_results.json")
    output_path = Path("agents/stat_sig_test_topk3.json")

    results = sort_results(load_csv(Path(csv_path)))
    width = target_width("loto2")
    universe = 10 ** width
    p0 = theoretical_baseline(top_k, universe)
    all_signal_names = [d.name for d in SIGNAL_DEFINITIONS] + ["frequency"]
    test_indices = list(range(30, len(results), sample_every))

    if chunk == "freq":
        # Step 1: compute frequency cache
        freq_cache: List[List[Tuple[str, float]]] = []
        for split_idx in test_indices:
            train = list(results[:split_idx])
            freq_cache.append(rank_candidates(train, "frequency", top_k, width))
            if len(freq_cache) % 200 == 0:
                print(f"  freq cache: {len(freq_cache)}/{len(test_indices)}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(freq_cache, f)
        print(f"Freq cache saved: {len(freq_cache)} entries in {time.time()-t_start:.0f}s")
        return

    if chunk == "signal":
        # Step 2: compute all signals (save every 5)
        with open(cache_path, encoding="utf-8") as f:
            freq_cache = json.load(f)
        freq_cache = [[(c, float(s)) for c, s in day] for day in freq_cache]

        signal_results: List[dict] = []
        for sig_idx, signal_name in enumerate(all_signal_names):
            sig_hits, freq_hits_list, precisions, freq_prec_list, yearly_hits, yearly_freq_hits = [], [], [], [], {}, {}
            for day_idx, split_idx in enumerate(test_indices):
                train = list(results[:split_idx])
                actual = actual_targets(results[split_idx], "loto2")
                sig_ranked = rank_candidates(train, signal_name, top_k, width)
                freq_ranked = freq_cache[day_idx]
                pred = [c for c, _ in sig_ranked]
                freq_pred = [c for c, _ in freq_ranked]
                hit, _, prec, freq_prec = evaluate_prediction_set(pred, actual, universe)
                fhit, _, _, _ = evaluate_prediction_set(freq_pred, actual, universe)
                sig_hits.append(int(hit))
                freq_hits_list.append(int(fhit))
                precisions.append(float(prec))
                freq_prec_list.append(float(freq_prec))
                year = results[split_idx].date[-4:]
                yearly_hits.setdefault(year, []).append(int(hit))
                yearly_freq_hits.setdefault(year, []).append(int(fhit))

            n = len(sig_hits)
            ss = sum(sig_hits)
            fs = sum(freq_hits_list)
            srate = ss / n
            frate = fs / n
            delta = (srate - frate) * 100.0
            binom_p = binomial_test(n, ss, p0)
            ci_lo, ci_hi = wilson_ci(ss, n)
            perm_p = permutation_pvalue(sig_hits, freq_hits_list, n_perm)
            wins = sum(1 for s, f in zip(sig_hits, freq_hits_list) if s > f)
            losses = sum(1 for s, f in zip(sig_hits, freq_hits_list) if s < f)

            yearly_summary = []
            for year in sorted(yearly_hits.keys()):
                yr_n = len(yearly_hits[year])
                yearly_summary.append({
                    "year": year, "test_size": yr_n,
                    "hit_rate_pct": sum(yearly_hits[year]) / yr_n * 100.0,
                    "freq_hit_rate_pct": sum(yearly_freq_hits[year]) / yr_n * 100.0,
                    "delta_pct": (sum(yearly_hits[year]) / yr_n - sum(yearly_freq_hits[year]) / yr_n) * 100.0,
                    "wins": sum(1 for s, f in zip(yearly_hits[year], yearly_freq_hits[year]) if s > f),
                    "losses": sum(1 for s, f in zip(yearly_hits[year], yearly_freq_hits[year]) if s < f),
                })

            r = {
                "signal": signal_name, "test_size": n, "hits": ss,
                "hit_rate_pct": srate * 100.0, "freq_hit_rate_pct": frate * 100.0,
                "delta_pct": delta,
                "theoretical_baseline_pct": p0 * 100.0,
                "binom_pvalue": binom_p, "binom_significant": binom_p < 0.05,
                "ci_lower_pct": ci_lo * 100.0, "ci_upper_pct": ci_hi * 100.0,
                "ci_covers_baseline": ci_lo <= p0 <= ci_hi,
                "perm_pvalue": perm_p, "perm_significant": perm_p < 0.05,
                "win_rate_pct": wins / (wins + losses) * 100.0 if (wins + losses) > 0 else 0.0,
                "wins": wins, "losses": losses,
                "yearly": yearly_summary,
            }
            signal_results.append(r)
            sig_binom_s = "***" if binom_p < 0.001 else ("**" if binom_p < 0.01 else ("*" if binom_p < 0.05 else ""))
            sig_perm_s = "***" if perm_p < 0.001 else ("**" if perm_p < 0.01 else ("*" if perm_p < 0.05 else ""))
            print(f"  [{sig_idx+1:2d}/{len(all_signal_names)}] {signal_name:<30} "
                  f"Hit={srate*100:5.2f}% Freq={frate*100:5.2f}% Delta={delta:+6.2f}pp "
                  f"binom={binom_p:.4f}{sig_binom_s} perm={perm_p:.4f}{sig_perm_s}")

            if sig_idx % 5 == 4:
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(signal_results, f)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(signal_results, f)
        print(f"Signals done in {time.time()-t_start:.0f}s")
        return

    # Step 3: assemble and print final results
    with open(results_path, encoding="utf-8") as f:
        signal_results = json.load(f)

    signal_results.sort(key=lambda x: (-x["delta_pct"], x["binom_pvalue"], x["perm_pvalue"], x["signal"]))
    sig_binom_all = [r for r in signal_results if r["binom_significant"]]
    sig_perm_all = [r for r in signal_results if r["perm_significant"]]
    sig_both = [r for r in signal_results if r["binom_significant"] and r["perm_significant"]]
    above_baseline = [r for r in signal_results if r["ci_lower_pct"] > p0 * 100]

    print(f"\n{'='*110}")
    print(f"Dataset: {len(results)} rows ({results[0].date} -> {results[-1].date})")
    print(f"Test days: {len(test_indices)} (sampled every {sample_every})")
    print(f"Baseline p(hit)={p0*100:.2f}%, top_k={top_k}")
    print(f"\n[Full Leaderboard]")
    print(f"  {'Signal':<30} {'Hit%':>7} {'Freq%':>7} {'Delta':>8} {'Binom-p':>9} {'Perm-p':>8} {'CI-lo':>8} {'CI-hi':>8} {'WinRate':>8}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in signal_results:
        sig_b = "***" if r["binom_pvalue"] < 0.001 else ("**" if r["binom_pvalue"] < 0.01 else ("*" if r["binom_pvalue"] < 0.05 else ""))
        sig_p = "***" if r["perm_pvalue"] < 0.001 else ("**" if r["perm_pvalue"] < 0.01 else ("*" if r["perm_pvalue"] < 0.05 else ""))
        print(f"  {r['signal']:<30} {r['hit_rate_pct']:>7.2f} {r['freq_hit_rate_pct']:>7.2f} {r['delta_pct']:>+8.2f} "
              f"{r['binom_pvalue']:>9.4f}{sig_b} {r['perm_pvalue']:>8.4f}{sig_p} "
              f"{r['ci_lower_pct']:>8.2f} {r['ci_upper_pct']:>8.2f} {r['win_rate_pct']:>7.1f}%")

    print(f"\n{'='*110}")
    print(f"[Summary]")
    print(f"  Signals: {len(signal_results)}")
    print(f"  Binomial significant (p<0.05, above {p0*100:.2f}%): {len(sig_binom_all)}")
    print(f"  Permutation significant (p<0.05, above frequency): {len(sig_perm_all)}")
    print(f"  Both significant: {len(sig_both)}")
    print(f"  CI strictly above baseline: {len(above_baseline)}")

    if sig_both:
        print(f"\n[Real Edge - Both Tests Significant]")
        for r in sig_both:
            print(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, Delta={r['delta_pct']:+.2f}pp, "
                  f"binom-p={r['binom_pvalue']:.4f}, perm-p={r['perm_pvalue']:.4f}")

    if above_baseline:
        print(f"\n[Strictly Above Baseline]")
        for r in above_baseline:
            print(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, CI=[{r['ci_lower_pct']:.2f},{r['ci_upper_pct']:.2f}]")

    print(f"\n[Yearly Delta - Top 5]")
    for r in signal_results[:5]:
        deltas = [f"{yr['year']}:{yr['delta_pct']:+.1f}" for yr in r["yearly"]]
        print(f"  {r['signal']}: {', '.join(deltas)}")

    print(f"\n[Significance Key]")
    print(f"  *** p<0.001  ** p<0.01  * p<0.05")
    print(f"  CI covers baseline = not significantly above theoretical")
    print(f"  Total time: {time.time()-t_start:.0f}s")

    # Save final JSON
    payload = {
        "dataset_size": len(results),
        "date_range": f"{results[0].date} - {results[-1].date}",
        "test_days": len(test_indices),
        "sampling_every": sample_every,
        "config": {"top_k": top_k, "universe_size": universe, "min_train_size": 30, "n_permutations": n_perm},
        "theoretical_baseline_pct": p0 * 100.0,
        "total_signals": len(signal_results),
        "sig_binomial_count": len(sig_binom_all),
        "sig_permutation_count": len(sig_perm_all),
        "sig_both_count": len(sig_both),
        "above_baseline_count": len(above_baseline),
        "results": signal_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
