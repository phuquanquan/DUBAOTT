import sys
from pathlib import Path
sys.stdout = open("agents/_debug_stat2.txt", "w", buffering=1)
sys.stderr = open("agents/_debug_stat_err2.txt", "w", buffering=1)
import random
import numpy as np
import time

try:
    from xsmb_pipeline.dataset import load_csv, sort_results
    from xsmb_pipeline.evaluate import evaluate_prediction_set
    from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
    from xsmb_pipeline.targets import actual_targets, target_width
    from xsmb_pipeline.models.weighted import candidate_universe

    print("Imports OK")
    results = sort_results(load_csv(Path("xsmb_full.csv")))
    print(f"Loaded {len(results)} rows")
    sys.stdout.flush()

    width = target_width("loto2")
    top_k = 3
    min_train_size = 30
    sample_every = 5

    test_indices = list(range(min_train_size, len(results), sample_every))
    print(f"Test indices: {len(test_indices)}")
    sys.stdout.flush()

    # Test rank_candidates
    def _score_frequency(rows, candidate, width, cache):
        if "freq" not in cache:
            cache["freq"] = {}
            items = []
            for r in rows:
                items.extend(actual_targets(r, "loto2"))
            if not items:
                for cand in candidate_universe(width):
                    cache["freq"][cand] = 1.0 / 100.0
            else:
                freq = {}
                for item in items:
                    freq[item] = freq.get(item, 0) + 1
                total = len(items)
                for cand in candidate_universe(width):
                    cache["freq"][cand] = freq.get(cand, 0) / total
        return cache["freq"].get(candidate, 0.0)

    def rank_candidates_freq(rows, top_k, width):
        cache = {}
        scored = [(cand, _score_frequency(rows, cand, width, cache)) for cand in candidate_universe(width)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]

    # Time 1 ranking
    train = list(results[:100])
    t0 = time.time()
    ranked = rank_candidates_freq(train, top_k, width)
    print(f"1 ranking: {time.time()-t0:.2f}s, result: {[c for c,s in ranked]}")
    sys.stdout.flush()

    # Time full walkforward
    t0 = time.time()
    freq_hits = []
    for i, split_idx in enumerate(test_indices[:20]):
        train = list(results[:split_idx])
        ranked = rank_candidates_freq(train, top_k, width)
        actual = actual_targets(results[split_idx], "loto2")
        universe = 100
        predicted = [c for c, _ in ranked]
        hit, _, _, _ = evaluate_prediction_set(predicted, actual, universe)
        freq_hits.append(int(hit))
        if i % 5 == 0:
            print(f"  day {i}: {time.time()-t0:.2f}s elapsed")
            sys.stdout.flush()
    print(f"20 days: {time.time()-t0:.2f}s")
    sys.stdout.flush()

    # Estimate full
    est = (time.time()-t0) / 20 * len(test_indices)
    print(f"Estimated full: {est:.0f}s = {est/60:.1f}min")
    sys.stdout.flush()

    # Full run
    print("Starting full run...")
    sys.stdout.flush()
    t0 = time.time()
    freq_cache = []
    for split_idx in test_indices:
        train = list(results[:split_idx])
        ranked = rank_candidates_freq(train, top_k, width)
        freq_cache.append(ranked)
        if len(freq_cache) % 100 == 0:
            print(f"  cached {len(freq_cache)}/{len(test_indices)}: {time.time()-t0:.0f}s")
            sys.stdout.flush()
    print(f"Full freq cache: {time.time()-t0:.0f}s")
    sys.stdout.flush()

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"ERROR: {e}")
    sys.stderr.flush()

sys.stdout.flush()
sys.stderr.flush()
