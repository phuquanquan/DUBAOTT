"""Test each signal one-by-one."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.evaluate import evaluate_prediction_set
from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
from xsmb_pipeline.targets import actual_targets, target_width
from xsmb_pipeline.models.weighted import candidate_universe

LOG = Path("agents/_onebyone_log.txt")
LOG.write_text("", encoding="utf-8")

def log(msg):
    LOG.write_text(LOG.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)

random.seed(42)
np.random.seed(42)

results = sort_results(load_csv(Path("xsmb_full.csv")))
width = target_width("loto2")
top_k = 3
min_train = 30
sample_every = 10
n_perm = 5000
universe = 100
p0 = 1 - (math.comb(99, 3) / math.comb(100, 3))

test_indices = list(range(min_train, len(results), sample_every))
log(f"Test indices: {len(test_indices)}")

# Pre-compute
freq_cache = []
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
    scored = [(cand, freq.get(cand, 0) / total) for cand in candidate_universe(width)]
    scored.sort(key=lambda x: (-x[1], x[0]))
    freq_cache.append(scored)
    actuals.append(list(actual_targets(results[split_idx], "loto2")))

log(f"Setup done. Processing {len(SIGNAL_DEFINITIONS)} signals...")

for sig_def in SIGNAL_DEFINITIONS:
    sig_name = sig_def.name
    t_sig = time.time()
    try:
        sig_hits = []
        freq_hits_list = []
        for day_idx in range(len(test_indices)):
            train = list(results[:test_indices[day_idx]])
            actual = actuals[day_idx]

            scored = []
            for cand in candidate_universe(width):
                score = sig_def.fn(train, cand, "loto2")
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

        n = len(sig_hits)
        ss = sum(sig_hits)
        fs = sum(freq_hits_list)
        srate = ss / n
        frate = fs / n
        delta = (srate - frate) * 100.0

        binom_p = 1.0
        total = 0.0
        for k in range(ss, n + 1):
            total += math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
        binom_p = min(total, 1.0)

        wins = sum(1 for s, f in zip(sig_hits, freq_hits_list) if s > f)
        losses = sum(1 for s, f in zip(sig_hits, freq_hits_list) if s < f)

        sig_b = "***" if binom_p < 0.001 else ("**" if binom_p < 0.01 else ("*" if binom_p < 0.05 else ""))
        log(f"  OK  {sig_name:<32} Hit={srate*100:5.2f}% Freq={frate*100:5.2f}% Delta={delta:+6.2f}pp binom={binom_p:.4f}{sig_b} ({time.time()-t_sig:.1f}s)")

    except Exception as e:
        import traceback
        log(f"  CRASH {sig_name}: {e}")
        log(f"    TB: {traceback.format_exc()[-500:]}")
        break

log("ALL DONE")
