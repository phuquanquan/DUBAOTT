"""Quick non-loto2 benchmark on smaller data."""
import sys, time
from pathlib import Path
sys.path.insert(0, '.')
from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.models.weighted import train_ranking_model, fit_tuned_ranking_model

results = sort_results(load_csv(Path('xsmb_full.csv')))
print(f"Data: {len(results)} rows", flush=True)

# non-loto2 targets on 500 rows
for t in ['dau', 'duoi', 'cham']:
    t0 = time.time()
    m = fit_tuned_ranking_model(results[-500:], target_name=t, top_k=3)
    t1 = time.time()
    print(f"fit_tuned ({t}, 500 rows): {t1-t0:.1f}s  TOP-3: {[c for c,_ in m.predict()]}", flush=True)

# train_ranking_model FULL for non-loto2
for t in ['dau', 'duoi', 'cham']:
    t0 = time.time()
    m = train_ranking_model(results, target_name=t, top_k=3)
    t1 = time.time()
    print(f"train ({t}, full): {t1-t0:.1f}s  TOP-3: {[c for c,_ in m.predict()]}", flush=True)

print("ALL DONE", flush=True)
