"""Quick parallel benchmark."""
import sys, time, os
from pathlib import Path
sys.path.insert(0, '.')
from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.models.weighted import train_ranking_model, fit_tuned_ranking_model

results = sort_results(load_csv(Path('xsmb_full.csv')))
print(f"Data: {len(results)} rows  Workers: {os.cpu_count()} cores", flush=True)

# Test train_ranking_model at various sizes
for size in [30, 100, 500, 1000]:
    t0 = time.time()
    m = train_ranking_model(results[-size:], target_name='loto2', top_k=5)
    t1 = time.time()
    print(f"train ({size} rows): {t1-t0:.1f}s  TOP: {[c for c,_ in m.predict()]}", flush=True)

# Full data
t0 = time.time()
m = train_ranking_model(results, target_name='loto2', top_k=5)
t1 = time.time()
print(f"train (FULL): {t1-t0:.1f}s  TOP: {[c for c,_ in m.predict()]}", flush=True)

# fit_tuned on 500 rows (smaller test)
t0 = time.time()
m = fit_tuned_ranking_model(results[-500:], target_name='loto2', top_k=5)
t1 = time.time()
print(f"fit_tuned (500 rows): {t1-t0:.1f}s  TOP: {[c for c,_ in m.predict()]}", flush=True)

print("DONE", flush=True)
