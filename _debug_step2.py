"""Debug step2 crash."""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.evaluate import evaluate_prediction_set
from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
from xsmb_pipeline.targets import actual_targets, target_width
from xsmb_pipeline.models.weighted import candidate_universe

LOG = Path("agents/_debug2_log.txt")
LOG.write_text("", encoding="utf-8")

def log(msg):
    LOG.write_text(LOG.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)

# Scorers
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

log("Setup complete")

results = sort_results(load_csv(Path("xsmb_full.csv")))
log(f"Results: {len(results)}")

with open(Path("agents/_stat_cache.json"), encoding="utf-8") as f:
    freq_cache_raw = json.load(f)
freq_cache = [[(c, float(s)) for c, s in day] for day in freq_cache_raw]
log(f"Cache: {len(freq_cache)}")

top_k = 3
width = target_width("loto2")
universe = 100
p0 = 1 - (math.comb(99, 3) / math.comb(100, 3))
test_indices = list(range(30, len(results), 10))
all_signal_names = [d.name for d in SIGNAL_DEFINITIONS] + ["frequency"]
log(f"Signals: {len(all_signal_names)}")
log(f"Test indices: {len(test_indices)}")

log("Starting main loop...")

for sig_idx, signal_name in enumerate(all_signal_names[:3]):  # First 3 signals only
    log(f"[{sig_idx}] {signal_name}: START")
    try:
        for day_idx, split_idx in enumerate(test_indices):
            if day_idx == 0:
                log(f"[{sig_idx}] {signal_name}: day 0 START, split_idx={split_idx}")
            train = list(results[:split_idx])
            actual = actual_targets(results[split_idx], "loto2")
            if day_idx == 0:
                log(f"[{sig_idx}] {signal_name}: day 0 got actual={actual}")
            sig_ranked = rank_candidates(train, signal_name, top_k, width)
            if day_idx == 0:
                log(f"[{sig_idx}] {signal_name}: day 0 got ranked={sig_ranked}")
            freq_ranked = freq_cache[day_idx]
            pred = [c for c, _ in sig_ranked]
            freq_pred = [c for c, _ in freq_ranked]
            hit, _, prec, freq_prec = evaluate_prediction_set(pred, actual, universe)
            if day_idx == 0:
                log(f"[{sig_idx}] {signal_name}: day 0 hit={hit}, prec={prec}")
            if day_idx % 100 == 99:
                log(f"[{sig_idx}] {signal_name}: day {day_idx+1} OK")
        log(f"[{sig_idx}] {signal_name}: DONE, total_hits=TBD")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"[{sig_idx}] {signal_name}: CRASH: {e}")
        log(f"[{sig_idx}] Traceback: {tb}")

log("DONE")
