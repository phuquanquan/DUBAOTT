import sys, time
from pathlib import Path
sys.path.insert(0, '.')
from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.models.weighted import train_ranking_model, fit_tuned_ranking_model

results = sort_results(load_csv(Path('xsmb_full.csv')))
print(f'Data: {len(results)} rows', flush=True)

# test train_ranking_model
t0 = time.time()
m = train_ranking_model(results[-100:], target_name='loto2', top_k=5)
t1 = time.time()
print(f'train_ranking_model (100 rows): {t1-t0:.2f}s  TOP: {[c for c,_ in m.predict()]}', flush=True)

# test fit_tuned_ranking_model (loto2)
t0 = time.time()
m = fit_tuned_ranking_model(results, target_name='loto2', top_k=5)
t1 = time.time()
print(f'fit_tuned_ranking_model (loto2, full): {t1-t0:.1f}s  TOP: {[c for c,_ in m.predict()]}', flush=True)

# test non-loto2
for t in ['dau', 'duoi', 'cham']:
    t0 = time.time()
    m = fit_tuned_ranking_model(results, target_name=t, top_k=3)
    t1 = time.time()
    print(f'fit_tuned_ranking_model ({t}): {t1-t0:.1f}s  TOP: {[c for c,_ in m.predict()]}', flush=True)
