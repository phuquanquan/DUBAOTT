import sys
from pathlib import Path
sys.stdout = open("agents/_stat_timing.txt", "w", buffering=1)

from xsmb_pipeline.dataset import load_csv, sort_results
from xsmb_pipeline.evaluate import walkforward_signal_backtest
import time

results = sort_results(load_csv(Path("xsmb_full.csv")))
print(f"Rows: {len(results)}, {results[0].date} -> {results[-1].date}")
t0 = time.time()
rows = walkforward_signal_backtest(results, target_name="loto2", signal_name="hot_trend", top_k=3, min_train_size=30)
elapsed = time.time() - t0
print(f"1 signal: {len(rows)} test days in {elapsed:.1f}s")
print(f"Est 33 signals: {elapsed * 33:.0f}s = {elapsed * 33 / 60:.1f}min")
sys.stdout.flush()
sys.stdout.close()
