"""Time setup components separately."""
from __future__ import annotations

import time
from pathlib import Path

from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.targets import actual_targets, target_width
from xsmb_pipeline.models.weighted import candidate_universe

LOG = Path("agents/_timing_log.txt")
LOG.write_text("", encoding="utf-8")

def log(msg):
    LOG.write_text(LOG.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)

results = sort_results(load_csv(Path("xsmb_full.csv")))
width = target_width("loto2")
min_train = 30
sample_every = 10
test_indices = list(range(min_train, len(results), sample_every))
log(f"Indices: {len(test_indices)}")

# Time actuals only
t0 = time.time()
actuals = []
for i, split_idx in enumerate(test_indices):
    actuals.append(list(actual_targets(results[split_idx], "loto2")))
    if i % 100 == 0:
        log(f"  actuals {i}/{len(test_indices)}: {time.time()-t0:.1f}s")
log(f"Actuals: {time.time()-t0:.1f}s")

# Time freq cache only
t0 = time.time()
freq_cache = []
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
    if i % 100 == 0:
        log(f"  freq {i}/{len(test_indices)}: {time.time()-t0:.1f}s")
log(f"Freq cache: {time.time()-t0:.1f}s")
log("DONE")
