"""Step-by-step test to find bottleneck."""
import sys, time
sys.path.insert(0, '.')

t0 = time.time()
from xsmb_pipeline.dataset import load_csv, sort_results
from pathlib import Path
results = sort_results(load_csv(Path('xsmb_full.csv')))
print(f"[OK] Data load: {time.time()-t0:.1f}s  {len(results)} rows")

t0 = time.time()
from xsmb_pipeline.models.weighted import train_ranking_model
m = train_ranking_model(results[-30:], target_name='loto2', top_k=5)
print(f"[OK] train_ranking_model(30 rows): {time.time()-t0:.1f}s  TOP: {[c for c,_ in m.predict()]}")

t0 = time.time()
m = train_ranking_model(results[-100:], target_name='loto2', top_k=5)
print(f"[OK] train_ranking_model(100 rows): {time.time()-t0:.1f}s  TOP: {[c for c,_ in m.predict()]}")

t0 = time.time()
m = train_ranking_model(results[-500:], target_name='loto2', top_k=5)
print(f"[OK] train_ranking_model(500 rows): {time.time()-t0:.1f}s  TOP: {[c for c,_ in m.predict()]}")

t0 = time.time()
m = train_ranking_model(results, target_name='loto2', top_k=5)
print(f"[OK] train_ranking_model(full): {time.time()-t0:.1f}s  TOP: {[c for c,_ in m.predict()]}")

print("\n[ALL train_ranking_model steps passed]")
