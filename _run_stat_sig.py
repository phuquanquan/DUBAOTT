"""
Statistical Significance Test - Runs as background subprocess with persistent log.
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
LOG.write_text("", encoding="utf-8")

def log(msg: str) -> None:
    LOG.write_text(LOG.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)


def main() -> None:
    t_start = time.time()
    random.seed(42)
    np.random.seed(42)

    log("=" * 60)
    log("STATISTICAL SIGNIFICANCE TEST - Starting")
    log("=" * 60)

    # Imports
    from xsmb_pipeline.dataset import load_csv, sort_results
    from xsmb_pipeline.evaluate import evaluate_prediction_set
    from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
    from xsmb_pipeline.targets import actual_targets, target_width
    from xsmb_pipeline.models.weighted import candidate_universe

    log("Step 1: Load data")
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

    log(f"  Data: {len(results)} rows, {len(test_indices)} test days")
    log(f"  Baseline p(hit)={p0:.4f} ({p0*100:.2f}%), top_k={top_k}")

    # Step 2: Pre-compute frequency cache
    log("Step 2: Build frequency cache")
    freq_cache: List[List[Tuple[str, float]]] = []
    t2 = time.time()
    for i, split_idx in enumerate(test_indices):
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
        if (i + 1) % 100 == 0:
            log(f"  freq cache: {i+1}/{len(test_indices)} ({time.time()-t2:.0f}s)")
    log(f"  Freq cache done: {time.time()-t2:.0f}s")

    # Step 3: Pre-compute actuals
    log("Step 3: Pre-compute actual targets")
    t3 = time.time()
    actuals = [list(actual_targets(results[split_idx], "loto2")) for split_idx in test_indices]
    log(f"  Actuals done: {time.time()-t3:.0f}s")

    # Step 4: Process each signal
    log(f"Step 4: Process {len(all_signal_names)} signals")
    signal_results: List[dict] = []
    name_to_def = {d.name: d for d in SIGNAL_DEFINITIONS}
    name_to_def["frequency"] = None

    for sig_idx, sig_name in enumerate(all_signal_names):
        t_sig = time.time()

        sig_hits: List[int] = []
        freq_hits_list: List[int] = []
        yearly_hits: dict = {}
        yearly_freq_hits: dict = {}

        for day_idx in range(len(test_indices)):
            train = list(results[:test_indices[day_idx]])
            actual = actuals[day_idx]

            if sig_name == "frequency":
                scored = freq_cache[day_idx]
            else:
                sig_def = name_to_def[sig_name]
                scored = []
                for cand in candidate_universe(width):
                    score = sig_def.fn(train, cand, "loto2")
                    sc = score.score if hasattr(score, "score") else float(score)
                    scored.append((cand, sc))
                scored.sort(key=lambda x: (-x[1], x[0]))

            pred = [c for c, _ in scored[:top_k]]
            freq_pred = [c for c, _ in freq_cache[day_idx]]
            hit, _, prec, _ = evaluate_prediction_set(pred, actual, universe)
            fhit, _, _, _ = evaluate_prediction_set(freq_pred, actual, universe)
            sig_hits.append(int(hit))
            freq_hits_list.append(int(fhit))

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
        total_prob = 0.0
        for k in range(ss, n + 1):
            total_prob += math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
        binom_p = min(total_prob, 1.0)

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
                "year": year,
                "test_size": yr_n,
                "hit_rate_pct": sum(yearly_hits[year]) / yr_n * 100.0,
                "freq_hit_rate_pct": sum(yearly_freq_hits[year]) / yr_n * 100.0,
                "delta_pct": (sum(yearly_hits[year]) / yr_n - sum(yearly_freq_hits[year]) / yr_n) * 100.0,
                "wins": sum(1 for s, f in zip(yearly_hits[year], yearly_freq_hits[year]) if s > f),
                "losses": sum(1 for s, f in zip(yearly_hits[year], yearly_freq_hits[year]) if s < f),
            })

        r = {
            "signal": sig_name,
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
        signal_results.append(r)

        sig_b = "***" if binom_p < 0.001 else ("**" if binom_p < 0.01 else ("*" if binom_p < 0.05 else ""))
        sig_p = "***" if perm_p < 0.001 else ("**" if perm_p < 0.01 else ("*" if perm_p < 0.05 else ""))
        log(f"  [{sig_idx+1:2d}/{len(all_signal_names)}] {sig_name:<32} "
              f"Hit={srate*100:5.2f}% Freq={frate*100:5.2f}% Delta={delta:+6.2f}pp "
              f"binom={binom_p:.4f}{sig_b} perm={perm_p:.4f}{sig_p} ({time.time()-t_sig:.1f}s)")

        # Checkpoint every 5
        if (sig_idx + 1) % 5 == 0:
            with open(Path("agents/_sig_checkpoint.json"), "w", encoding="utf-8") as f:
                json.dump(signal_results, f, ensure_ascii=False)
            log(f"  [Checkpoint {sig_idx+1}/{len(all_signal_names)}]")

    # Sort and finalize
    log("Step 5: Finalize results")
    signal_results.sort(key=lambda x: (-x["delta_pct"], x["binom_pvalue"], x["perm_pvalue"], x["signal"]))

    sig_binom = [r for r in signal_results if r["binom_significant"]]
    sig_perm = [r for r in signal_results if r["perm_significant"]]
    sig_both = [r for r in signal_results if r["binom_significant"] and r["perm_significant"]]
    above_base = [r for r in signal_results if r["ci_lower_pct"] > p0 * 100]

    log("")
    log("=" * 60)
    log("[SUMMARY]")
    log(f"  Signals: {len(signal_results)}, Test days: {len(test_indices)}")
    log(f"  Binomial sig (above {p0*100:.2f}%): {len(sig_binom)}")
    log(f"  Permutation sig (above freq): {len(sig_perm)}")
    log(f"  Both significant: {len(sig_both)}")
    log(f"  CI strictly above baseline: {len(above_base)}")
    log(f"  Total time: {time.time()-t_start:.0f}s")

    log("")
    log("[LEADERBOARD]")
    for r in signal_results:
        sig_b = "***" if r["binom_pvalue"] < 0.001 else ("**" if r["binom_pvalue"] < 0.01 else ("*" if r["binom_pvalue"] < 0.05 else ""))
        sig_p = "***" if r["perm_pvalue"] < 0.001 else ("**" if r["perm_pvalue"] < 0.01 else ("*" if r["perm_pvalue"] < 0.05 else ""))
        log(f"  {r['signal']:<32} Hit={r['hit_rate_pct']:5.2f}% Freq={r['freq_hit_rate_pct']:5.2f}% "
              f"Delta={r['delta_pct']:+6.2f}pp binom={r['binom_pvalue']:.4f}{sig_b} perm={r['perm_pvalue']:.4f}{sig_p} "
              f"CI=[{r['ci_lower_pct']:.2f},{r['ci_upper_pct']:.2f}] WinRate={r['win_rate_pct']:.1f}%")

    if sig_both:
        log("")
        log("[REAL EDGE]")
        for r in sig_both:
            log(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, Delta={r['delta_pct']:+.2f}pp, "
                  f"binom-p={r['binom_pvalue']:.4f}, perm-p={r['perm_pvalue']:.4f}")

    if above_base:
        log("")
        log("[STRICTLY ABOVE BASELINE]")
        for r in above_base:
            log(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, CI=[{r['ci_lower_pct']:.2f},{r['ci_upper_pct']:.2f}]")

    log("")
    log("[YEARLY DELTA - Top 5]")
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
    output_path = Path("agents/stat_sig_test_topk3.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"\nSaved: {output_path}")
    log(f"ALL DONE in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
