"""
Statistical Significance Test - Clean version.
Uses actual SIGNAL_DEFINITIONS, caches per day, incremental save.
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

LOG = Path("agents/_sig_log.txt")

def log(msg: str) -> None:
    LOG.write_text(LOG.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)


def main() -> None:
    t_start = time.time()
    random.seed(42)
    np.random.seed(42)

    LOG.write_text("", encoding="utf-8")

    # Setup
    from xsmb_pipeline.dataset import load_csv, sort_results
    from xsmb_pipeline.evaluate import evaluate_prediction_set
    from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
    from xsmb_pipeline.targets import actual_targets, target_width
    from xsmb_pipeline.models.weighted import candidate_universe

    log("Imports OK")

    # Load data
    results = sort_results(load_csv(Path("xsmb_full.csv")))
    width = target_width("loto2")
    universe = 10 ** width
    top_k = 3
    sample_every = 10
    min_train = 30
    n_perm = 5000
    p0 = 1.0 - (math.comb(universe - 1, top_k) / math.comb(universe, top_k))

    test_indices = list(range(min_train, len(results), sample_every))
    all_signal_names = [d.name for d in SIGNAL_DEFINITIONS] + ["frequency"]

    log(f"Dataset: {len(results)} rows, {len(test_indices)} test days")
    log(f"Baseline p(hit)={p0:.4f} ({p0*100:.2f}%), top_k={top_k}")

    # Pre-compute frequency cache
    cache_path = Path("agents/_freq_cache2.json")
    if cache_path.exists():
        log("Loading frequency cache...")
        with open(cache_path, encoding="utf-8") as f:
            freq_cache = [[(c, float(s)) for c, s in day] for day in json.load(f)]
        log(f"Loaded {len(freq_cache)} entries")
    else:
        log("Building frequency cache...")
        freq_cache: List[List[Tuple[str, float]]] = []
        for split_idx in test_indices:
            train = list(results[:split_idx])
            items = []
            for r in train:
                items.extend(actual_targets(r, "loto2"))
            freq = {}
            for item in items:
                freq[item] = freq.get(item, 0) + 1
            total = len(items) or 1
            scored = [(cand, freq.get(cand, 0) / total) for cand in candidate_universe(width)]
            scored.sort(key=lambda x: (-x[1], x[0]))
            freq_cache.append(scored)
            if len(freq_cache) % 200 == 0:
                log(f"  freq cache {len(freq_cache)}/{len(test_indices)}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([[(c, s) for c, s in day] for day in freq_cache], f)
        log(f"Freq cache built: {len(freq_cache)} entries")

    # Pre-compute actual targets for all test days
    log("Pre-computing actual targets...")
    actuals: List[List[str]] = []
    for split_idx in test_indices:
        actuals.append(list(actual_targets(results[split_idx], "loto2")))
    log(f"Actuals computed: {len(actuals)} days")

    # Run all signals
    results_path = Path("agents/_sig_results.json")
    output_path = Path("agents/stat_sig_test_topk3.json")

    signal_results: List[dict] = []

    def process_signal(signal_name: str, signal_def_fn) -> dict:
        sig_hits, freq_hits_list = [], []
        precisions, freq_prec_list = [], []
        yearly_hits, yearly_freq_hits = {}, {}

        for day_idx in range(len(test_indices)):
            train = list(results[:test_indices[day_idx]])
            actual = actuals[day_idx]

            # Score all 100 candidates
            scored = []
            for cand in candidate_universe(width):
                score = signal_def_fn(train, cand, "loto2")
                if hasattr(score, "score"):
                    scored.append((cand, score.score))
                else:
                    scored.append((cand, float(score)))
            scored.sort(key=lambda x: (-x[1], x[0]))
            pred = [c for c, _ in scored[:top_k]]
            freq_pred = [c for c, _ in freq_cache[day_idx]]

            hit, _, prec, freq_prec = evaluate_prediction_set(pred, actual, universe)
            fhit, _, _, _ = evaluate_prediction_set(freq_pred, actual, universe)

            sig_hits.append(int(hit))
            freq_hits_list.append(int(fhit))
            precisions.append(float(prec))
            freq_prec_list.append(float(freq_prec))

            year = results[test_indices[day_idx]].date[-4:]
            yearly_hits.setdefault(year, []).append(int(hit))
            yearly_freq_hits.setdefault(year, []).append(int(fhit))

        n = len(sig_hits)
        ss = sum(sig_hits)
        fs = sum(freq_hits_list)
        srate = ss / n
        frate = fs / n
        delta = (srate - frate) * 100.0

        # Binomial test
        binom_p = 1.0
        try:
            total = 0.0
            for k in range(sum(sig_hits), n + 1):
                total += math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
            binom_p = min(total, 1.0)
        except Exception:
            binom_p = 1.0

        # Wilson CI
        p_hat = ss / n
        z = 1.96
        denom = 1.0 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
        ci_lo = max(0.0, center - margin)
        ci_hi = min(1.0, center + margin)

        # Permutation test
        wins = sum(1 for s, f in zip(sig_hits, freq_hits_list) if s > f)
        losses = sum(1 for s, f in zip(sig_hits, freq_hits_list) if s < f)
        pool = [1] * wins + [0] * losses
        observed = wins - losses
        count_extreme = 0
        for _ in range(n_perm):
            random.shuffle(pool)
            pd = sum(1 for i in range(n) if pool[i] > 0) - sum(1 for i in range(n) if pool[i] == 0)
            if abs(pd) >= abs(observed):
                count_extreme += 1
        perm_p = count_extreme / n_perm

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

        return {
            "signal": signal_name,
            "test_size": n,
            "hits": ss,
            "hit_rate_pct": srate * 100.0,
            "freq_hit_rate_pct": frate * 100.0,
            "delta_pct": delta,
            "theoretical_baseline_pct": p0 * 100.0,
            "binom_pvalue": binom_p,
            "binom_significant": binom_p < 0.05,
            "ci_lower_pct": ci_lo * 100.0,
            "ci_upper_pct": ci_hi * 100.0,
            "ci_covers_baseline": ci_lo <= p0 <= ci_hi,
            "perm_pvalue": perm_p,
            "perm_significant": perm_p < 0.05,
            "win_rate_pct": wins / (wins + losses) * 100.0 if (wins + losses) > 0 else 0.0,
            "wins": wins,
            "losses": losses,
            "yearly": yearly_summary,
        }

    log(f"Processing {len(all_signal_names)} signals...")

    # Build name->def map
    name_to_def = {d.name: d for d in SIGNAL_DEFINITIONS}
    name_to_def["frequency"] = None

    for sig_idx, sig_name in enumerate(all_signal_names):
        t_sig = time.time()
        if sig_name == "frequency":
            # Frequency baseline: use cached freq_cache directly
            sig_hits, freq_hits_list = [], []
            yearly_hits, yearly_freq_hits = {}, {}
            for day_idx in range(len(test_indices)):
                actual = actuals[day_idx]
                freq_pred = [c for c, _ in freq_cache[day_idx]]
                fhit, _, _, _ = evaluate_prediction_set(freq_pred, actual, universe)
                sig_hits.append(0)  # not applicable for freq vs freq
                freq_hits_list.append(int(fhit))
                year = results[test_indices[day_idx]].date[-4:]
                yearly_hits.setdefault(year, []).append(0)
                yearly_freq_hits.setdefault(year, []).append(int(fhit))
            n = len(test_indices)
            r = {
                "signal": "frequency", "test_size": n, "hits": 0,
                "hit_rate_pct": 0.0, "freq_hit_rate_pct": sum(freq_hits_list) / n * 100.0,
                "delta_pct": 0.0, "theoretical_baseline_pct": p0 * 100.0,
                "binom_pvalue": 1.0, "binom_significant": False,
                "ci_lower_pct": 0.0, "ci_upper_pct": 0.0,
                "ci_covers_baseline": True,
                "perm_pvalue": 1.0, "perm_significant": False,
                "win_rate_pct": 50.0, "wins": 0, "losses": 0,
                "yearly": [],
            }
        else:
            sig_def = name_to_def[sig_name]
            r = process_signal(sig_name, sig_def.fn)

        signal_results.append(r)

        sig_b = "***" if r["binom_pvalue"] < 0.001 else ("**" if r["binom_pvalue"] < 0.01 else ("*" if r["binom_pvalue"] < 0.05 else ""))
        sig_p = "***" if r["perm_pvalue"] < 0.001 else ("**" if r["perm_pvalue"] < 0.01 else ("*" if r["perm_pvalue"] < 0.05 else ""))
        log(f"  [{sig_idx+1:2d}/{len(all_signal_names)}] {sig_name:<32} "
              f"Hit={r['hit_rate_pct']:5.2f}% Freq={r['freq_hit_rate_pct']:5.2f}% "
              f"Delta={r['delta_pct']:+6.2f}pp binom={r['binom_pvalue']:.4f}{sig_b} "
              f"perm={r['perm_pvalue']:.4f}{sig_p} ({time.time()-t_sig:.1f}s)")

        # Incremental save every 5
        if (sig_idx + 1) % 5 == 0:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(signal_results, f, ensure_ascii=False)
            log(f"  [Checkpoint {sig_idx+1}/{len(all_signal_names)} saved]")

    # Sort and save final
    signal_results.sort(key=lambda x: (-x["delta_pct"], x["binom_pvalue"], x["perm_pvalue"], x["signal"]))

    sig_binom = [r for r in signal_results if r["binom_significant"]]
    sig_perm = [r for r in signal_results if r["perm_significant"]]
    sig_both = [r for r in signal_results if r["binom_significant"] and r["perm_significant"]]
    above_base = [r for r in signal_results if r["ci_lower_pct"] > p0 * 100]

    log(f"\n{'='*110}")
    log(f"[SUMMARY] {len(signal_results)} signals, {len(test_indices)} test days, {time.time()-t_start:.0f}s total")
    log(f"  Binomial sig (above {p0*100:.2f}%): {len(sig_binom)}")
    log(f"  Permutation sig (above freq): {len(sig_perm)}")
    log(f"  Both significant: {len(sig_both)}")
    log(f"  CI strictly above baseline: {len(above_base)}")

    log(f"\n[LEADERBOARD]")
    log(f"  {'Signal':<32} {'Hit%':>7} {'Freq%':>7} {'Delta':>8} {'Binom-p':>9} {'Perm-p':>8} {'CI-lo':>8} {'CI-hi':>8} {'WinRate':>8}")
    for r in signal_results:
        sig_b = "***" if r["binom_pvalue"] < 0.001 else ("**" if r["binom_pvalue"] < 0.01 else ("*" if r["binom_pvalue"] < 0.05 else ""))
        sig_p = "***" if r["perm_pvalue"] < 0.001 else ("**" if r["perm_pvalue"] < 0.01 else ("*" if r["perm_pvalue"] < 0.05 else ""))
        log(f"  {r['signal']:<32} {r['hit_rate_pct']:>7.2f} {r['freq_hit_rate_pct']:>7.2f} {r['delta_pct']:>+8.2f} "
              f"{r['binom_pvalue']:>9.4f}{sig_b} {r['perm_pvalue']:>8.4f}{sig_p} "
              f"{r['ci_lower_pct']:>8.2f} {r['ci_upper_pct']:>8.2f} {r['win_rate_pct']:>7.1f}%")

    if sig_both:
        log(f"\n[REAL EDGE - Both Significant]")
        for r in sig_both:
            log(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, Delta={r['delta_pct']:+.2f}pp, "
                  f"binom-p={r['binom_pvalue']:.4f}, perm-p={r['perm_pvalue']:.4f}")

    if above_base:
        log(f"\n[STRICTLY ABOVE BASELINE]")
        for r in above_base:
            log(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, CI=[{r['ci_lower_pct']:.2f},{r['ci_upper_pct']:.2f}]")

    log(f"\n[YEARLY DELTA - Top 5]")
    for r in signal_results[:5]:
        deltas = [f"{yr['year']}:{yr['delta_pct']:+.1f}" for yr in r["yearly"]]
        log(f"  {r['signal']}: {', '.join(deltas)}")

    # Save JSON
    payload = {
        "dataset_size": len(results),
        "date_range": f"{results[0].date} - {results[-1].date}",
        "test_days": len(test_indices),
        "sampling_every": sample_every,
        "config": {"top_k": top_k, "universe_size": universe, "min_train_size": min_train, "n_permutations": n_perm},
        "theoretical_baseline_pct": p0 * 100.0,
        "total_signals": len(signal_results),
        "sig_binomial_count": len(sig_binom),
        "sig_permutation_count": len(sig_perm),
        "sig_both_count": len(sig_both),
        "above_baseline_count": len(above_base),
        "results": signal_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"\nSaved: {output_path}")
    log(f"TOTAL TIME: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
