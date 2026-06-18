"""Statistical Significance Test - Step 2: compute all signals."""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Write ALL output to file
LOG_FILE = Path("agents/_step2_log.txt")
LOG_FILE.write_text("", encoding="utf-8")

def log(msg: str) -> None:
    LOG_FILE.write_text(LOG_FILE.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)

log("Starting step 2...")

from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.evaluate import evaluate_prediction_set
from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
from xsmb_pipeline.targets import actual_targets, target_width
from xsmb_pipeline.models.weighted import candidate_universe

log("Imports done")


# Fast signal scorers
def score_hot_trend(rows, candidate, width, cache):
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

def score_cold_return(rows, candidate, width, cache):
    last_seen = None
    for offset, result in enumerate(reversed(rows), start=1):
        if candidate in actual_targets(result, "loto2"):
            last_seen = offset
            break
    val = (last_seen or min(len(rows), 90)) / 90.0
    return max(0.0, min(1.0, val))

def score_touch(rows, candidate, width, cache):
    digits = set(int(ch) for ch in candidate.zfill(width))
    items = []
    for r in rows:
        items.extend(actual_targets(r, "loto2"))
    if not items:
        return 0.0
    matches = sum(1 for item in items if digits & set(int(ch) for ch in item.zfill(width)))
    return matches / len(items)

def score_inversion(rows, candidate, width, cache):
    inverted = candidate[::-1]
    items = []
    for r in rows:
        items.extend(actual_targets(r, "loto2"))
    return items.count(inverted) / len(items) if items else 0.0

def score_pascal(rows, candidate, width, cache):
    digits = [int(ch) for ch in candidate.zfill(width)]
    if width >= 3:
        checks = 2
        matches = int((digits[0] + digits[-1]) % 10 == digits[1])
        matches += int((digits[0] + digits[1]) % 10 == digits[-1])
    else:
        checks = 1
        matches = int(abs(digits[0] - digits[1]) in {0, 1, 9})
    return matches / checks if checks else 0.0

def score_composition(rows, candidate, width, cache):
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

def score_frequency(rows, candidate, width, cache):
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

def rank_candidates(rows, signal_name, top_k, width):
    cache = {}
    scorer = SCORERS.get(signal_name)
    if scorer is None:
        return [(f"{i:02d}", 1.0 / (i + 1)) for i in range(top_k)]
    scored = [(cand, scorer(rows, cand, width, cache)) for cand in candidate_universe(width)]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_k]

def theoretical_baseline(top_k, universe=100):
    return 1.0 - (math.comb(universe - 1, top_k) / math.comb(universe, top_k))

def binomial_test(n, hits, p0):
    if hits > n or hits < 0:
        return 1.0
    total = 0.0
    for k in range(hits, n + 1):
        total += math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
    return min(total, 1.0)

def wilson_ci(hits, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p_hat = hits / n
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def permutation_pvalue(sig_hits, freq_hits, n_perm=5000):
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

log("Functions defined")

# Load data
results = sort_results(load_csv(Path("xsmb_full.csv")))
log(f"Loaded {len(results)} results")

# Load cache
cache_path = Path("agents/_stat_cache.json")
with open(cache_path, encoding="utf-8") as f:
    freq_cache_raw = json.load(f)
freq_cache = [[(c, float(s)) for c, s in day] for day in freq_cache_raw]
log(f"Loaded freq cache: {len(freq_cache)} entries")

top_k = 3
sample_every = 10
n_perm = 5000
width = target_width("loto2")
universe = 10 ** width
p0 = theoretical_baseline(top_k, universe)
all_signal_names = [d.name for d in SIGNAL_DEFINITIONS] + ["frequency"]
test_indices = list(range(30, len(results), sample_every))
log(f"Test indices: {len(test_indices)}, baseline p={p0:.4f}")

signal_results = []
for sig_idx, signal_name in enumerate(all_signal_names):
    t_sig = time.time()
    sig_hits, freq_hits_list, precisions, freq_prec_list = [], [], [], []
    yearly_hits, yearly_freq_hits = {}, {}

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

    sig_b = "***" if binom_p < 0.001 else ("**" if binom_p < 0.01 else ("*" if binom_p < 0.05 else ""))
    sig_p = "***" if perm_p < 0.001 else ("**" if perm_p < 0.01 else ("*" if perm_p < 0.05 else ""))
    log(f"  [{sig_idx+1:2d}/{len(all_signal_names)}] {signal_name:<30} "
          f"Hit={srate*100:5.2f}% Freq={frate*100:5.2f}% Delta={delta:+6.2f}pp "
          f"binom={binom_p:.4f}{sig_b} perm={perm_p:.4f}{sig_p} ({time.time()-t_sig:.1f}s)")

    if sig_idx % 5 == 4:
        with open(Path("agents/_stat_results.json"), "w", encoding="utf-8") as f:
            json.dump(signal_results, f, ensure_ascii=False)
        log(f"  [Checkpoint saved at signal {sig_idx+1}]")

# Final save
with open(Path("agents/_stat_results.json"), "w", encoding="utf-8") as f:
    json.dump(signal_results, f, ensure_ascii=False)
log(f"DONE - all {len(signal_results)} signals computed. Results in agents/_stat_results.json")
