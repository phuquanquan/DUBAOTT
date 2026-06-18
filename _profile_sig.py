"""Profile each signal for 1 day."""
from __future__ import annotations

import time
from pathlib import Path

from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
from xsmb_pipeline.targets import actual_targets
from xsmb_pipeline.models.weighted import candidate_universe

LOG = Path("agents/_profile_log.txt")
LOG.write_text("", encoding="utf-8")

def log(msg):
    LOG.write_text(LOG.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)

results = sort_results(load_csv(Path("xsmb_full.csv")))
train = list(results[:100])
candidates = list(candidate_universe(2))

for d in SIGNAL_DEFINITIONS[:5]:
    t0 = time.time()
    for cand in candidates[:5]:  # 5 candidates only
        try:
            _ = d.fn(train, cand, "loto2")
        except Exception as e:
            log(f"  ERROR on {d.name}/{cand}: {e}")
            break
    elapsed = time.time() - t0
    est_full = elapsed / 5 * 100
    log(f"{d.name}: 5 cand={elapsed*1000:.0f}ms, est 100 cand={est_full:.1f}s, est 587 days={est_full*587/60:.0f}min")
