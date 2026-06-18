"""Time each step."""
from __future__ import annotations

import time
from pathlib import Path

from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
from xsmb_pipeline.targets import actual_targets, target_width
from xsmb_pipeline.models.weighted import candidate_universe

from xsmb_pipeline.evaluate import evaluate_prediction_set

LOG = Path("agents/_timing_log.txt")
LOG.write_text("", encoding="utf-8")

def log(msg):
    LOG.write_text(LOG.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)

t0 = time.time()
results = sort_results(load_csv(Path("xsmb_full.csv")))
log(f"Load+sort: {time.time()-t0:.1f}s")

width = target_width("loto2")
min_train = 30
sample_every = 10
test_indices = list(range(min_train, len(results), sample_every))
log(f"Test indices: {len(test_indices)}")

t0 = time.time()
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
    if len(freq_cache) % 200 == 0:
        log(f"  {len(freq_cache)}/{len(test_indices)}: {time.time()-t0:.1f}s")
log(f"Setup (freq cache + actuals): {time.time()-t0:.1f}s")

log("Testing signal: touch...")
t0 = time.time()
sig_def = next(d for d in SIGNAL_DEFINITIONS if d.name == "touch")
sig_hits = []
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
    pred = [c for c, _ in scored[:3]]
    freq_pred = [c for c, _ in freq_cache[day_idx]]
    hit, _, _, _ = evaluate_prediction_set(pred, actual, 100)
    sig_hits.append(int(hit))
    if day_idx % 100 == 0:
        log(f"  day {day_idx}/{len(test_indices)}: {time.time()-t0:.1f}s")
log(f"touch: {len(sig_hits)} hits={sum(sig_hits)}, total={time.time()-t0:.1f}s")

log("DONE")
