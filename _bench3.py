"""Focused benchmark."""
import sys, time
from pathlib import Path
sys.path.insert(0, '.')
from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.models.weighted import train_ranking_model, fit_tuned_ranking_model

results = sort_results(load_csv(Path('xsmb_full.csv')))
print(f"Data: {len(results)} rows", flush=True)

# train_ranking_model FULL
t0 = time.time()
m = train_ranking_model(results, target_name='loto2', top_k=5)
t1 = time.time()
print(f"train_ranking_model(FULL): {t1-t0:.1f}s  TOP-5: {[c for c,_ in m.predict()]}", flush=True)

# fit_tuned on 1000 rows
t0 = time.time()
m = fit_tuned_ranking_model(results[-1000:], target_name='loto2', top_k=5)
t1 = time.time()
print(f"fit_tuned_ranking_model(loto2, 1000 rows): {t1-t0:.1f}s  TOP-5: {[c for c,_ in m.predict()]}", flush=True)

# fit_tuned on 2000 rows
t0 = time.time()
m = fit_tuned_ranking_model(results[-2000:], target_name='loto2', top_k=5)
t1 = time.time()
print(f"fit_tuned_ranking_model(loto2, 2000 rows): {t1-t0:.1f}s  TOP-5: {[c for c,_ in m.predict()]}", flush=True)

# non-loto2 on full
for t in ['dau', 'duoi', 'cham']:
    t0 = time.time()
    m = fit_tuned_ranking_model(results, target_name=t, top_k=3)
    t1 = time.time()
    print(f"fit_tuned_ranking_model({t}, full): {t1-t0:.1f}s  TOP-3: {[c for c,_ in m.predict()]}", flush=True)

print("ALL DONE", flush=True)
