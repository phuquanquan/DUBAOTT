import sys
from pathlib import Path
sys.stdout = open("agents/_debug_stat.txt", "w", buffering=1)
sys.stderr = open("agents/_debug_stat_err.txt", "w", buffering=1)

try:
    import random
    import numpy as np
    from xsmb_pipeline.dataset import load_csv, sort_results
    from xsmb_pipeline.evaluate import evaluate_prediction_set
    from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
    from xsmb_pipeline.targets import actual_targets, target_width
    from xsmb_pipeline.models.weighted import candidate_universe
    print("Imports OK")
    results = sort_results(load_csv(Path("xsmb_full.csv")))
    print(f"Loaded {len(results)} rows")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"ERROR: {e}")

sys.stdout.flush()
sys.stderr.flush()
