"""
Quick Stat Sig Test - 5 signals, checkpoint after each, poll until done.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.stats import binomtest

LOG = Path("agents/_qstat_log.txt")
CHECKPOINT = Path("agents/_qstat_checkpoint.json")
FINAL = Path("agents/stat_sig_test_topk3.json")

def log(msg):
    LOG.write_text(LOG.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)

def load_data():
    from xsmb_pipeline.dataset import load_csv, sort_results
    from xsmb_pipeline.targets import target_width
    from xsmb_pipeline.models.weighted import candidate_universe
    results = sort_results(load_csv(Path("xsmb_full.csv")))
    universe = 10 ** target_width("loto2")
    return results, universe

def compute_one_signal(results, universe, sig_name, signal_fn, test_indices, actuals, freq_cache, top_k, p0, n_perm):
    width = 2
    sig_hits = []
    freq_hits_list = []
    yearly_hits = {}
    yearly_freq_hits = {}

    for day_idx in range(len(test_indices)):
        train = list(results[:test_indices[day_idx]])
        actual = actuals[day_idx]

        scored = []
        for cand in candidate_universe(width):
            score = signal_fn(train, cand, "loto2")
            sc = score.score if hasattr(score, "score") else float(score)
            scored.append((cand, sc))
        scored.sort(key=lambda x: (-x[1], x[0]))
        pred = [c for c, _ in scored[:top_k]]
        freq_pred = [c for c, _ in freq_cache[day_idx]]

        from xsmb_pipeline.evaluate import evaluate_prediction_set
        hit, _, _, _ = evaluate_prediction_set(pred, actual, universe)
        fhit, _, _, _ = evaluate_prediction_set(freq_pred, actual, universe)
        sig_hits.append(int(hit))
        freq_hits_list.append(int(fhit))
        year = results[test_indices[day_idx]].date[-4:]
        yearly_hits.setdefault(year, []).append(int(hit))
        yearly_freq_hits.setdefault(year, []).append(int(fhit))

    n = len(sig_hits)
    ss = sum(sig_hits)
    srate = ss / n
    frate = sum(freq_hits_list) / n
    delta = (srate - frate) * 100.0

    try:
        bt = binomtest(int(ss), n, p0, alternative="greater")
        binom_p = bt.pvalue
    except:
        binom_p = 1.0

    p_hat = ss / n
    z = 1.96
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    ci_lo = max(0.0, center - margin)
    ci_hi = min(1.0, center + margin)

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

    return {
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

def run():
    LOG.write_text("", encoding="utf-8")
    t_start = time.time()
    random.seed(42)
    np.random.seed(42)

    log("=== Quick Stat Sig Test ===")

    results, universe = load_data()
    from xsmb_pipeline.models.weighted import candidate_universe
    from xsmb_pipeline.targets import actual_targets

    top_k = 3
    min_train = 30
    sample_every = 5
    n_perm = 5000
    p0 = 1.0 - (math.comb(universe - 1, top_k) / math.comb(universe, top_k))
    test_indices = list(range(min_train, len(results), sample_every))

    log(f"Data: {len(results)} rows, {len(test_indices)} test days")
    log(f"Baseline p(hit)={p0*100:.2f}%")

    # Pre-compute
    from xsmb_pipeline.evaluate import evaluate_prediction_set
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
        scored = [(cand, freq.get(cand, 0) / total) for cand in candidate_universe(2)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        freq_cache.append(scored)
        actuals.append(list(actual_targets(results[split_idx], "loto2")))
        if (i + 1) % 100 == 0:
            log(f"  Precompute: {i+1}/{len(test_indices)} ({time.time()-t_start:.0f}s)")

    log(f"Precompute done ({time.time()-t_start:.0f}s)")

    # Load or start checkpoint
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            completed = json.load(f)
        log(f"Resuming from checkpoint: {len(completed)} signals done")
    else:
        completed = []

    # 5 signals
    from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
    name_to_def = {d.name: d for d in SIGNAL_DEFINITIONS}
    signals_to_test = [
        ("prize_position_penalty", name_to_def["prize_position_penalty"].fn),
        ("days_since_last", name_to_def["days_since_last"].fn),
        ("thirty_day_freq_penalty", name_to_def["thirty_day_freq_penalty"].fn),
        ("hot_trend", name_to_def["hot_trend"].fn),
        ("cold_return", name_to_def["cold_return"].fn),
    ]

    for sig_name, signal_fn in signals_to_test:
        if any(r["signal"] == sig_name for r in completed):
            log(f"Skipping {sig_name} (already done)")
            continue

        log(f"\nProcessing {sig_name}...")
        t_sig = time.time()
        result = compute_one_signal(results, universe, sig_name, signal_fn, test_indices, actuals, freq_cache, top_k, p0, n_perm)
        completed.append(result)

        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(completed, f, ensure_ascii=False)
        log(f"  Done in {time.time()-t_sig:.0f}s. Saved checkpoint.")

    # Finalize
    log("\n=== FINAL RESULTS ===")
    completed.sort(key=lambda x: (-x["delta_pct"], x["binom_pvalue"], x["perm_pvalue"]))
    for r in completed:
        sig_b = "***" if r["binom_pvalue"] < 0.001 else ("**" if r["binom_pvalue"] < 0.01 else ("*" if r["binom_pvalue"] < 0.05 else ""))
        sig_p = "***" if r["perm_pvalue"] < 0.001 else ("**" if r["perm_pvalue"] < 0.01 else ("*" if r["perm_pvalue"] < 0.05 else ""))
        log(f"  {r['signal']:<30} Hit={r['hit_rate_pct']:5.2f}% Delta={r['delta_pct']:+6.2f}pp "
              f"binom={r['binom_pvalue']:.4f}{sig_b} perm={r['perm_pvalue']:.4f}{sig_p} "
              f"CI=[{r['ci_lower_pct']:.2f},{r['ci_upper_pct']:.2f}]")

    payload = {
        "dataset_size": len(results),
        "date_range": f"{results[0].date} - {results[-1].date}",
        "test_days": len(test_indices),
        "config": {"top_k": top_k, "universe_size": universe, "min_train_size": min_train, "n_permutations": n_perm},
        "theoretical_baseline_pct": p0 * 100.0,
        "results": completed,
    }
    with open(FINAL, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"\nSaved: {FINAL}")
    log(f"Total: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    run()
