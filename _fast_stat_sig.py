"""
Statistical Significance Test - Fast version.
Sample 60 test days (every 100 days) -> each signal ~1min, full run ~40min split into ~8 chunks.
Or single-run with sample_every=100 -> ~58 test days.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

def evaluate_prediction_set(
    predicted: Sequence[str], actual: Sequence[str], universe_size: int
) -> Tuple[int, float, float, float]:
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


LOG = Path("agents/_fast_stat_log.txt")
CHECKPOINT = Path("agents/_fast_stat_cp.json")
FINAL = Path("agents/stat_sig_test_topk3.json")


def write_log(msg):
    LOG.write_text(LOG.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)


def run():
    LOG.write_text("", encoding="utf-8")
    t_start = time.time()
    random.seed(42)

    write_log("=== Fast Stat Sig Test ===")

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

    write_log(f"Data: {len(results)} rows, {len(test_indices)} test days")
    write_log(f"Baseline p(hit)={p0*100:.2f}%, top_k={top_k}")

    # Precompute
    write_log("Precomputing...")
    freq_cache = []
    actuals = []
    for i, split_idx in enumerate(test_indices):
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
        freq_cache.append(scored)
        actuals.append(list(actual_targets(results[split_idx], "loto2")))
        if (i + 1) % 20 == 0:
            write_log(f"  {i+1}/{len(test_indices)} ({time.time()-t_start:.0f}s)")
    write_log(f"Precompute done ({time.time()-t_start:.0f}s)")

    # Checkpoint
    cp = {"done": [], "results": []}
    if CHECKPOINT.exists():
        cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        write_log(f"Resuming: {len(cp['done'])} done")

    name_to_def = {d.name: d for d in SIGNAL_DEFINITIONS}
    all_names = [d.name for d in SIGNAL_DEFINITIONS]

    for sig_name in all_names:
        if sig_name in cp["done"]:
            write_log(f"Skipping {sig_name} (done)")
            continue

        sig_def = name_to_def[sig_name]
        t_sig = time.time()
        write_log(f"  {sig_name}: starting...")

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
            freq_pred = [c for c, _ in freq_cache[day_idx]]

            hit, _, prec, _ = evaluate_prediction_set(pred, actual, universe)
            fhit, _, _, _ = evaluate_prediction_set(freq_pred, actual, universe)
            sig_hits.append(int(hit))
            freq_hits_list.append(int(fhit))

            year = results[test_indices[day_idx]].date[-4:]
            yearly_hits.setdefault(year, []).append(int(hit))
            yearly_freq_hits.setdefault(year, []).append(int(fhit))

            if day_idx % 20 == 0 and day_idx > 0:
                write_log(f"    day {day_idx}/{len(test_indices)}")

        n = len(sig_hits)
        ss = sum(sig_hits)
        srate = ss / n
        frate = sum(freq_hits_list) / n
        delta = (srate - frate) * 100.0

        # Binomial
        binom_p = 1.0
        try:
            total_prob = 0.0
            for k in range(ss, n + 1):
                total_prob += math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
            binom_p = min(total_prob, 1.0)
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

        # Permutation
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
        cp["results"].append(r)
        cp["done"].append(sig_name)

        sig_b = "***" if binom_p < 0.001 else ("**" if binom_p < 0.01 else ("*" if binom_p < 0.05 else ""))
        sig_p = "***" if perm_p < 0.001 else ("**" if perm_p < 0.01 else ("*" if perm_p < 0.05 else ""))
        write_log(f"  {sig_name}: Hit={srate*100:.2f}% Delta={delta:+.2f}pp "
                  f"binom={binom_p:.4f}{sig_b} perm={perm_p:.4f}{sig_p} ({time.time()-t_sig:.0f}s)")

        # Checkpoint
        CHECKPOINT.write_text(json.dumps(cp, ensure_ascii=False), encoding="utf-8")

    # Finalize
    write_log("\n=== FINAL ===")
    cp["results"].sort(key=lambda x: (-x["delta_pct"], x["binom_pvalue"], x["perm_pvalue"], x["signal"]))

    sig_binom = [r for r in cp["results"] if r["binom_significant"]]
    sig_perm = [r for r in cp["results"] if r["perm_significant"]]
    sig_both = [r for r in cp["results"] if r["binom_significant"] and r["perm_significant"]]
    above_base = [r for r in cp["results"] if r["ci_lower_pct"] > p0 * 100]

    write_log(f"Signals: {len(cp['results'])}, Test days: {len(test_indices)}")
    write_log(f"Binomial sig: {len(sig_binom)}, Permutation sig: {len(sig_perm)}, Both: {len(sig_both)}")

    write_log(f"\n{'Signal':<32} {'Hit%':>7} {'Freq%':>7} {'Delta':>8} {'Binom-p':>9} {'Perm-p':>8} {'CI-lo':>8} {'CI-hi':>8}")
    for r in cp["results"]:
        sig_b = "***" if r["binom_pvalue"] < 0.001 else ("**" if r["binom_pvalue"] < 0.01 else ("*" if r["binom_pvalue"] < 0.05 else ""))
        sig_p = "***" if r["perm_pvalue"] < 0.001 else ("**" if r["perm_pvalue"] < 0.01 else ("*" if r["perm_pvalue"] < 0.05 else ""))
        write_log(f"{r['signal']:<32} {r['hit_rate_pct']:>7.2f} {r['freq_hit_rate_pct']:>7.2f} {r['delta_pct']:>+8.2f} "
                  f"{r['binom_pvalue']:>9.4f}{sig_b} {r['perm_pvalue']:>8.4f}{sig_p} "
                  f"{r['ci_lower_pct']:>8.2f} {r['ci_upper_pct']:>8.2f}")

    if sig_both:
        write_log("\n[REAL EDGE]")
        for r in sig_both:
            write_log(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, Delta={r['delta_pct']:+.2f}pp, "
                      f"binom-p={r['binom_pvalue']:.4f}, perm-p={r['perm_pvalue']:.4f}")

    if above_base:
        write_log("\n[STRICTLY ABOVE BASELINE]")
        for r in above_base:
            write_log(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, CI=[{r['ci_lower_pct']:.2f},{r['ci_upper_pct']:.2f}]")

    write_log("\n[YEARLY DELTA - Top 5]")
    for r in cp["results"][:5]:
        deltas = [f"{yr['year']}:{yr['delta_pct']:+.1f}" for yr in r["yearly"]]
        write_log(f"  {r['signal']}: {', '.join(deltas)}")

    payload = {
        "dataset_size": len(results),
        "date_range": f"{results[0].date} - {results[-1].date}",
        "test_days": len(test_indices),
        "sampling_every": sample_every,
        "config": {"top_k": top_k, "universe_size": universe, "min_train_size": min_train, "n_permutations": n_perm},
        "theoretical_baseline_pct": p0 * 100.0,
        "results": cp["results"],
        "sig_binomial_count": len(sig_binom),
        "sig_permutation_count": len(sig_perm),
        "sig_both_count": len(sig_both),
        "above_baseline_count": len(above_base),
    }
    with open(FINAL, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_log(f"\nSaved: {FINAL}")
    write_log(f"Total: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    run()
