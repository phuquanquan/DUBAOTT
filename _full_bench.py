"""Full pipeline benchmark with progress output."""
import sys, time
from pathlib import Path
sys.path.insert(0, '.')
from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.models.weighted import train_ranking_model, fit_tuned_ranking_model, compare_loto2_weight_strategies
from xsmb_pipeline.evaluate import walkforward_yearly_backtest

results = sort_results(load_csv(Path('xsmb_full.csv')))
print(f"[DONE] Data loaded: {len(results)} rows ({results[0].date} -> {results[-1].date})", flush=True)

# Step 1: train_ranking_model
for size, label in [(30, '30 rows'), (100, '100 rows'), (500, '500 rows'), (5896, 'FULL')]:
    t0 = time.time()
    m = train_ranking_model(results[-size:], target_name='loto2', top_k=5)
    t1 = time.time()
    top = [c for c,_ in m.predict()]
    print(f"[DONE] train_ranking_model ({label}): {t1-t0:.1f}s  TOP-5: {top}", flush=True)

# Step 2: fit_tuned_ranking_model loto2
print("[RUNNING] fit_tuned_ranking_model (loto2, full data)...", flush=True)
t0 = time.time()
m = fit_tuned_ranking_model(results, target_name='loto2', top_k=5)
t1 = time.time()
print(f"[DONE] fit_tuned loto2: {t1-t0:.1f}s  TOP-5: {[c for c,_ in m.predict()]}", flush=True)

# Step 3: non-loto2
for t in ['dau', 'duoi', 'cham']:
    print(f"[RUNNING] fit_tuned ({t})...", flush=True)
    t0 = time.time()
    m = fit_tuned_ranking_model(results, target_name=t, top_k=3)
    t1 = time.time()
    print(f"[DONE] fit_tuned ({t}): {t1-t0:.1f}s  TOP-3: {[c for c,_ in m.predict()]}", flush=True)

# Step 4: walkforward
print("[RUNNING] walkforward_yearly_backtest...", flush=True)
t0 = time.time()
wf = walkforward_yearly_backtest(results, target_name='loto2', top_k=5)
t1 = time.time()
print(f"[DONE] walkforward: {t1-t0:.1f}s", flush=True)
for year, m in sorted(wf.items()):
    print(f"  {year}: {m}", flush=True)

print("[ALL DONE]", flush=True)
