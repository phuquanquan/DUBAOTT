"""Trace exact crash point."""
from pathlib import Path

LOG = Path("agents/_trace_log.txt")
LOG.write_text("", encoding="utf-8")

def log(msg):
    LOG.write_text(LOG.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)

log("1. Starting...")
import time

log("2. Importing xsmb_pipeline.dataset...")
t0 = time.time()
from xsmb_pipeline.dataset import load_csv, sort_results
log(f"   dataset imported in {time.time()-t0:.1f}s")

log("3. load_csv...")
t0 = time.time()
results = sort_results(load_csv(Path("xsmb_full.csv")))
log(f"   load_csv done in {time.time()-t0:.1f}s: {len(results)} rows")

log("4. Importing xsmb_pipeline.signals...")
t0 = time.time()
from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
log(f"   signals imported in {time.time()-t0:.1f}s")

log("5. Importing xsmb_pipeline.targets...")
t0 = time.time()
from xsmb_pipeline.targets import actual_targets, target_width
log(f"   targets imported in {time.time()-t0:.1f}s")

log("6. Importing xsmb_pipeline.evaluate...")
t0 = time.time()
from xsmb_pipeline.evaluate import evaluate_prediction_set
log(f"   evaluate imported in {time.time()-t0:.1f}s")

log("7. candidate_universe...")
t0 = time.time()
from xsmb_pipeline.models.weighted import candidate_universe
log(f"   candidate_universe in {time.time()-t0:.1f}s")

log("8. scipy...")
t0 = time.time()
from scipy.stats import binomtest
log(f"   scipy in {time.time()-t0:.1f}s")

log("ALL DONE")
