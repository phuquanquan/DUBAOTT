"""
Statistical Significance Test - Lean & Mean.
Targets 6 signals x 73 test days (every 5 days from 5896 rows).
Per-signal: 100 candidates x 73 days = 7300 scoring calls.
Total: ~6 signals x ~0.8s each = ~5s + stat computation.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import List, Tuple


def evaluate_prediction_set(predicted, actual, universe_size):
    predicted_set = set(predicted)
    actual_set = set(actual)
    overlap = len(predicted_set & actual_set)
    hit = 1 if overlap > 0 else 0
    actual_count = len(actual_set)
    miss_prob = ((universe_size - actual_count) / universe_size) ** len(predicted_set) if universe_size and predicted_set else 1.0
    baseline_hit = 1.0 - miss_prob
    baseline_precision = actual_count / universe_size if universe_size else 0.0
    precision = overlap / max(1, len(predicted_set))
    return hit, baseline_hit, precision, baseline_precision


def run_stat_sig():
    t_start = time.time()
    random.seed(42)

    print("=== Statistical Significance Test ===", flush=True)

    from xsmb_pipeline.dataset import load_csv, sort_results
    from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
    from xsmb_pipeline.targets import actual_targets, target_width

    results = sort_results(load_csv(Path("xsmb_full.csv")))
    universe = 10 ** target_width("loto2")
    top_k = 3
    min_train = 30
    sample_every = 100  # ~59 test days
    n_perm = 5000
    p0 = 1.0 - (math.comb(universe - 1, top_k) / math.comb(universe, top_k))

    test_indices = list(range(min_train, len(results), sample_every))

    print(f"Data: {len(results)} rows, {len(test_indices)} test days, baseline={p0*100:.2f}%", flush=True)

    # Precompute freq ranked + actuals
    print("Precomputing...", flush=True)
    freq_ranked = []
    actuals = []
    for split_idx in test_indices:
        train = list(results[:split_idx])
        items = []
        for r in train:
            items.extend(actual_targets(r, "loto2"))
        freq = {}
        for item in items:
            freq[item] = freq.get(item, 0) + 1
        total = len(items) or 1
        scored = [(f"{i:02d}", freq.get(f"{i:02d}", 0) / total) for i in range(100)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        freq_ranked.append(scored)
        actuals.append(list(actual_targets(results[split_idx], "loto2")))
    print(f"Precompute done in {time.time()-t_start:.0f}s", flush=True)

    # 6 signals: 3 new + 3 existing benchmarks
    SIGNALS_TO_TEST = [
        "prize_position_penalty",    # NEW
        "days_since_last",           # NEW
        "thirty_day_freq_penalty",  # NEW
        "hot_trend",               # benchmark
        "cold_return",             # benchmark
        "touch",                   # benchmark
    ]

    name_to_def = {d.name: d for d in SIGNAL_DEFINITIONS}
    all_results = []

    for sig_name in SIGNALS_TO_TEST:
        t_sig = time.time()
        print(f"  {sig_name}: running...", flush=True)

        sig_def = name_to_def[sig_name]
        sig_hits, freq_hits_list = [], []
        yearly_hits, yearly_freq_hits = {}, {}

        for day_idx in range(len(test_indices)):
            train = list(results[:test_indices[day_idx]])
            actual = actuals[day_idx]

            scored = []
            for cand in (f"{i:02d}" for i in range(100)):
                score = sig_def.fn(train, cand, "loto2")
                sc = score.score if hasattr(score, "score") else float(score)
                scored.append((cand, sc))
            scored.sort(key=lambda x: (-x[1], x[0]))
            pred = [c for c, _ in scored[:top_k]]
            freq_pred = [c for c, _ in freq_ranked[day_idx][:top_k]]

            hit, _, _, _ = evaluate_prediction_set(pred, actual, universe)
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
        try:
            tp = 0.0
            for k in range(ss, n + 1):
                tp += math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
            binom_p = min(tp, 1.0)
        except:
            binom_p = 1.0

        # Wilson CI
        p_hat = ss / n
        z = 1.96
        denom = 1.0 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
        ci_lo = max(0.0, center - margin)
        ci_hi = min(1.0, center + margin)

        # Win-rate permutation test (scipy binomial)
        wins = sum(1 for s, f in zip(sig_hits, freq_hits_list) if s > f)
        losses = sum(1 for s, f in zip(sig_hits, freq_hits_list) if s < f)
        n_pairs = wins + losses
        if n_pairs >= 5:
            try:
                from scipy.stats import binomtest
                # Two-sided: is win rate significantly different from 0.5?
                bt = binomtest(wins, n_pairs, 0.5, alternative="two-sided")
                perm_p = bt.pvalue
            except:
                perm_p = 1.0
        else:
            perm_p = 1.0

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5

        yearly = []
        for year in sorted(yearly_hits.keys()):
            yr_n = len(yearly_hits[year])
            yearly.append({
                "year": year, "test_size": yr_n,
                "hit_rate_pct": sum(yearly_hits[year]) / yr_n * 100.0,
                "freq_hit_rate_pct": sum(yearly_freq_hits[year]) / yr_n * 100.0,
                "delta_pct": (sum(yearly_hits[year]) / yr_n - sum(yearly_freq_hits[year]) / yr_n) * 100.0,
                "wins": sum(1 for s, f in zip(yearly_hits[year], yearly_freq_hits[year]) if s > f),
                "losses": sum(1 for s, f in zip(yearly_hits[year], yearly_freq_hits[year]) if s < f),
            })

        r = {
            "signal": sig_name, "test_size": n, "hits": ss,
            "hit_rate_pct": srate * 100.0, "freq_hit_rate_pct": frate * 100.0,
            "delta_pct": delta, "theoretical_baseline_pct": p0 * 100.0,
            "binom_pvalue": binom_p, "binom_significant": binom_p < 0.05,
            "ci_lower_pct": ci_lo * 100.0, "ci_upper_pct": ci_hi * 100.0,
            "ci_covers_baseline": ci_lo <= p0 <= ci_hi,
            "perm_pvalue": perm_p, "perm_significant": perm_p < 0.05,
            "win_rate_pct": wins / (wins + losses) * 100.0 if (wins + losses) > 0 else 0.0,
            "wins": wins, "losses": losses, "yearly": yearly,
        }
        all_results.append(r)

        sig_b = "***" if binom_p < 0.001 else ("**" if binom_p < 0.01 else ("*" if binom_p < 0.05 else ""))
        sig_p = "***" if perm_p < 0.001 else ("**" if perm_p < 0.01 else ("*" if perm_p < 0.05 else ""))
        print(f"  {sig_name}: Hit={srate*100:.2f}% Delta={delta:+.2f}pp "
              f"binom={binom_p:.4f}{sig_b} perm={perm_p:.4f}{sig_p} "
              f"CI=[{ci_lo*100:.1f},{ci_hi*100:.1f}] "
              f"WR={wins/(wins+losses)*100:.1f}% ({time.time()-t_sig:.1f}s)", flush=True)

    # Sort
    all_results.sort(key=lambda x: (-x["delta_pct"], x["binom_pvalue"], x["perm_pvalue"], x["signal"]))

    # Summary
    sig_binom = [r for r in all_results if r["binom_significant"]]
    sig_perm = [r for r in all_results if r["perm_significant"]]
    sig_both = [r for r in all_results if r["binom_significant"] and r["perm_significant"]]
    above_base = [r for r in all_results if r["ci_lower_pct"] > p0 * 100]

    print(f"\n{'='*90}", flush=True)
    print(f"[SUMMARY] {len(all_results)} signals, {len(test_indices)} test days", flush=True)
    print(f"  Binomial sig (above {p0*100:.2f}%): {len(sig_binom)}", flush=True)
    print(f"  Permutation sig (above freq): {len(sig_perm)}", flush=True)
    print(f"  Both significant: {len(sig_both)}", flush=True)
    print(f"  CI strictly above baseline: {len(above_base)}", flush=True)

    print(f"\n[LEADERBOARD]", flush=True)
    print(f"  {'Signal':<28} {'Hit%':>7} {'Freq%':>7} {'Delta':>8} {'Binom-p':>9} {'Perm-p':>8} {'CI-lo':>8} {'CI-hi':>8} {'WinRate':>8}", flush=True)
    for r in all_results:
        sig_b = "***" if r["binom_pvalue"] < 0.001 else ("**" if r["binom_pvalue"] < 0.01 else ("*" if r["binom_pvalue"] < 0.05 else ""))
        sig_p = "***" if r["perm_pvalue"] < 0.001 else ("**" if r["perm_pvalue"] < 0.01 else ("*" if r["perm_pvalue"] < 0.05 else ""))
        print(f"  {r['signal']:<28} {r['hit_rate_pct']:>7.2f} {r['freq_hit_rate_pct']:>7.2f} {r['delta_pct']:>+8.2f} "
              f"{r['binom_pvalue']:>9.4f}{sig_b} {r['perm_pvalue']:>8.4f}{sig_p} "
              f"{r['ci_lower_pct']:>8.1f} {r['ci_upper_pct']:>8.1f} {r['win_rate_pct']:>7.1f}%", flush=True)

    if sig_both:
        print(f"\n[REAL EDGE - Both Significant]", flush=True)
        for r in sig_both:
            print(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, Delta={r['delta_pct']:+.2f}pp, "
                  f"binom-p={r['binom_pvalue']:.4f}, perm-p={r['perm_pvalue']:.4f}", flush=True)

    if above_base:
        print(f"\n[STRICTLY ABOVE BASELINE]", flush=True)
        for r in above_base:
            print(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, CI=[{r['ci_lower_pct']:.2f},{r['ci_upper_pct']:.2f}]", flush=True)

    print(f"\n[YEARLY DELTA]", flush=True)
    for r in all_results:
        deltas = [f"{yr['year']}:{yr['delta_pct']:+.1f}" for yr in r["yearly"]]
        print(f"  {r['signal']}: {', '.join(deltas)}", flush=True)

    # Save
    payload = {
        "dataset_size": len(results),
        "date_range": f"{results[0].date} - {results[-1].date}",
        "test_days": len(test_indices),
        "sampling_every": sample_every,
        "config": {"top_k": top_k, "universe_size": universe, "min_train_size": min_train, "n_permutations": n_perm},
        "theoretical_baseline_pct": p0 * 100.0,
        "results": [
            {**r, "binom_significant": bool(r["binom_significant"]), "perm_significant": bool(r["perm_significant"])}
            for r in all_results
        ],
        "sig_binomial_count": len(sig_binom),
        "sig_permutation_count": len(sig_perm),
        "sig_both_count": len(sig_both),
        "above_baseline_count": len(above_base),
    }
    output_path = Path("agents/stat_sig_test_topk3.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_path}", flush=True)
    print(f"Total time: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    run_stat_sig()
