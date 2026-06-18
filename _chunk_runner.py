"""
Statistical Significance Test - Precomputed items cache per day.
No repeated target_items calls across candidates.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import List, Tuple

def evaluate_prediction_set(predicted, actual, universe_size):
    predicted_set = set(predicted)
    actual_set = set(actual)
    overlap = len(predicted_set & actual_set)
    hit = 1 if overlap > 0 else 0
    actual_count = len(actual_set)
    miss_prob = ((universe_size - actual_count) / universe_size) ** len(predicted_set) if universe_size and predicted_set else 1.0
    baseline_hit = 1.0 - miss_prob
    baseline_precision = actual_count / universe_size if universe_size else 0.0
    precision = overlap / max(1, len(predicted_set))
    return hit, baseline_hit, precision, baseline_precision


def run_chunk(chunk_idx: int, signal_names: List[str]) -> None:
    t_start = time.time()
    random.seed(42 + chunk_idx)

    Path("agents/_chunk_log.txt").write_text("", encoding="utf-8")
    def log(msg):
        Path("agents/_chunk_log.txt").write_text(
            Path("agents/_chunk_log.txt").read_text(encoding="utf-8") + msg + "\n",
            encoding="utf-8"
        )

    log(f"[{chunk_idx}] Starting...")

    from xsmb_pipeline.dataset import load_csv, sort_results
    from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
    from xsmb_pipeline.targets import actual_targets, target_width

    results = sort_results(load_csv(Path("xsmb_full.csv")))
    universe = 10 ** target_width("loto2")
    top_k = 3
    min_train = 30
    sample_every = 200
    n_perm = 2000
    p0 = 1.0 - (math.comb(universe - 1, top_k) / math.comb(universe, top_k))
    test_indices = list(range(min_train, len(results), sample_every))

    log(f"[{chunk_idx}] Data: {len(results)} rows, {len(test_indices)} test days")

    # Precompute: for each test day, precompute items list ONCE
    # Then pass items directly to signals instead of full results list
    items_per_day = []
    actuals_per_day = []
    freq_ranked_per_day = []

    for split_idx in test_indices:
        train = list(results[:split_idx])
        items = []
        for r in train:
            items.extend(actual_targets(r, "loto2"))
        items_per_day.append(items)
        actuals_per_day.append(list(actual_targets(results[split_idx], "loto2")))

        # Freq ranked
        freq = {}
        for item in items:
            freq[item] = freq.get(item, 0) + 1
        total = len(items) or 1
        scored = [(f"{i:02d}", freq.get(f"{i:02d}", 0) / total) for i in range(100)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        freq_ranked_per_day.append(scored)

    log(f"[{chunk_idx}] Precompute done ({time.time()-t_start:.0f}s)")

    # Precompute signal scores for all signals, all days
    name_to_def = {d.name: d for d in SIGNAL_DEFINITIONS}
    signal_scores: dict = {s: [] for s in signal_names}

    log(f"[{chunk_idx}] Precomputing signal scores...")
    for day_idx in range(len(test_indices)):
        train = list(results[:test_indices[day_idx]])
        items = items_per_day[day_idx]

        if day_idx % 10 == 0:
            log(f"[{chunk_idx}] day {day_idx}/{len(test_indices)} ({time.time()-t_start:.0f}s)")

        for sig_name in signal_names:
            sig_def = name_to_def[sig_name]
            scored = []
            for cand in (f"{i:02d}" for i in range(100)):
                # Monkey-patch: create a lightweight results-like object
                # that target_items will NOT cache-miss on
                # Instead, just call the signal fn with the train and let it compute
                score = sig_def.fn(train, cand, "loto2")
                sc = score.score if hasattr(score, "score") else float(score)
                scored.append((cand, sc))
            scored.sort(key=lambda x: (-x[1], x[0]))
            signal_scores[sig_name].append(scored)

    log(f"[{chunk_idx}] All scores precomputed ({time.time()-t_start:.0f}s)")

    # Evaluate
    results_list = []
    for sig_name in signal_names:
        t_sig = time.time()
        log(f"[{chunk_idx}] {sig_name}: evaluating...")

        sig_hits, freq_hits_list = [], []
        yearly_hits, yearly_freq_hits = {}, {}

        for day_idx in range(len(test_indices)):
            pred = [c for c, _ in signal_scores[sig_name][day_idx][:top_k]]
            freq_pred = [c for c, _ in freq_ranked_per_day[day_idx][:top_k]]
            actual = actuals_per_day[day_idx]

            hit, _, _, _ = evaluate_prediction_set(pred, actual, universe)
            fhit, _, _, _ = evaluate_prediction_set(freq_pred, actual, universe)
            sig_hits.append(int(hit))
            freq_hits_list.append(int(fhit))

            year = results[test_indices[day_idx]].date[-4:]
            yearly_hits.setdefault(year, []).append(int(hit))
            yearly_freq_hits.setdefault(year, []).append(int(fhit))

        n = len(sig_hits)
        ss = sum(sig_hits)
        srate = ss / n
        frate = sum(freq_hits_list) / n
        delta = (srate - frate) * 100.0

        binom_p = 1.0
        try:
            tp = 0.0
            for k in range(ss, n + 1):
                tp += math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
            binom_p = min(tp, 1.0)
        except:
            binom_p = 1.0

        p_hat = ss / n
        z = 1.96
        denom = 1.0 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
        ci_lo = max(0.0, center - margin)
        ci_hi = min(1.0, center + margin)

        wins = sum(1 for s, f in zip(sig_hits, freq_hits_list) if s > f)
        losses = sum(1 for s, f in zip(sig_hits, freq_hits_list) if s < f)
        pool = [1] * wins + [0] * losses
        observed = wins - losses
        count_extreme = 0
        for _ in range(n_perm):
            random.shuffle(pool)
            pd = sum(1 for i in range(n) if pool[i] > 0) - sum(1 for i in range(n) if pool[i] == 0)
            if abs(pd) >= abs(observed):
                count_extreme += 1
        perm_p = count_extreme / n_perm

        yearly = []
        for year in sorted(yearly_hits.keys()):
            yr_n = len(yearly_hits[year])
            yearly.append({
                "year": year, "test_size": yr_n,
                "hit_rate_pct": sum(yearly_hits[year]) / yr_n * 100.0,
                "freq_hit_rate_pct": sum(yearly_freq_hits[year]) / yr_n * 100.0,
                "delta_pct": (sum(yearly_hits[year]) / yr_n - sum(yearly_freq_hits[year]) / yr_n) * 100.0,
                "wins": sum(1 for s, f in zip(yearly_hits[year], yearly_freq_hits[year]) if s > f),
                "losses": sum(1 for s, f in zip(yearly_hits[year], yearly_freq_hits[year]) if s < f),
            })

        r = {
            "signal": sig_name, "test_size": n, "hits": ss,
            "hit_rate_pct": srate * 100.0, "freq_hit_rate_pct": frate * 100.0,
            "delta_pct": delta, "theoretical_baseline_pct": p0 * 100.0,
            "binom_pvalue": binom_p, "binom_significant": binom_p < 0.05,
            "ci_lower_pct": ci_lo * 100.0, "ci_upper_pct": ci_hi * 100.0,
            "ci_covers_baseline": ci_lo <= p0 <= ci_hi,
            "perm_pvalue": perm_p, "perm_significant": perm_p < 0.05,
            "win_rate_pct": wins / (wins + losses) * 100.0 if (wins + losses) > 0 else 0.0,
            "wins": wins, "losses": losses, "yearly": yearly,
        }
        results_list.append(r)

        sig_b = "***" if binom_p < 0.001 else ("**" if binom_p < 0.01 else ("*" if binom_p < 0.05 else ""))
        sig_p = "***" if perm_p < 0.001 else ("**" if perm_p < 0.01 else ("*" if perm_p < 0.05 else ""))
        log(f"[{chunk_idx}] {sig_name}: Hit={srate*100:.2f}% Delta={delta:+.2f}pp "
              f"binom={binom_p:.4f}{sig_b} perm={perm_p:.4f}{sig_p} ({time.time()-t_sig:.0f}s)")

    chunk_path = Path(f"agents/_chunk_{chunk_idx}.json")
    chunk_path.write_text(json.dumps(results_list, ensure_ascii=False), encoding="utf-8")
    log(f"[{chunk_idx}] Saved {len(results_list)} results to {chunk_path}")
    log(f"[{chunk_idx}] TOTAL: {time.time()-t_start:.0f}s")


def combine() -> None:
    from xsmb_pipeline.dataset import load_csv, sort_results
    from xsmb_pipeline.targets import target_width
    import math

    results = sort_results(load_csv(Path("xsmb_full.csv")))
    universe = 10 ** target_width("loto2")
    top_k = 3
    p0 = 1.0 - (math.comb(universe - 1, top_k) / math.comb(universe, top_k))

    all_results = []
    for chunk_path in sorted(Path("agents").glob("_chunk_*.json")):
        with open(chunk_path, encoding="utf-8") as f:
            all_results.extend(json.load(f))

    all_results.sort(key=lambda x: (-x["delta_pct"], x["binom_pvalue"], x["perm_pvalue"], x["signal"]))

    sig_binom = [r for r in all_results if r["binom_significant"]]
    sig_perm = [r for r in all_results if r["perm_significant"]]
    sig_both = [r for r in all_results if r["binom_significant"] and r["perm_significant"]]
    above_base = [r for r in all_results if r["ci_lower_pct"] > p0 * 100]

    print(f"\n{'='*100}")
    print(f"[SUMMARY] {len(all_results)} signals, baseline={p0*100:.2f}%, top_k={top_k}")
    print(f"  Binomial sig: {len(sig_binom)}, Permutation sig: {len(sig_perm)}, Both: {len(sig_both)}, Above base: {len(above_base)}")

    print(f"\n[LEADERBOARD]")
    for r in all_results:
        sig_b = "***" if r["binom_pvalue"] < 0.001 else ("**" if r["binom_pvalue"] < 0.01 else ("*" if r["binom_pvalue"] < 0.05 else ""))
        sig_p = "***" if r["perm_pvalue"] < 0.001 else ("**" if r["perm_pvalue"] < 0.01 else ("*" if r["perm_pvalue"] < 0.05 else ""))
        print(f"  {r['signal']:<32} Hit={r['hit_rate_pct']:5.2f}% Delta={r['delta_pct']:+6.2f}pp "
              f"binom={r['binom_pvalue']:.4f}{sig_b} perm={r['perm_pvalue']:.4f}{sig_p} "
              f"CI=[{r['ci_lower_pct']:.1f},{r['ci_upper_pct']:.1f}] WR={r['win_rate_pct']:.1f}%")

    if sig_both:
        print(f"\n[REAL EDGE]")
        for r in sig_both:
            print(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, Delta={r['delta_pct']:+.2f}pp, "
                  f"binom-p={r['binom_pvalue']:.4f}, perm-p={r['perm_pvalue']:.4f}")

    if above_base:
        print(f"\n[STRICTLY ABOVE BASELINE]")
        for r in above_base:
            print(f"  {r['signal']}: Hit={r['hit_rate_pct']:.2f}%, CI=[{r['ci_lower_pct']:.2f},{r['ci_upper_pct']:.2f}]")

    print(f"\n[YEARLY DELTA - Top 5]")
    for r in all_results[:5]:
        deltas = [f"{yr['year']}:{yr['delta_pct']:+.1f}" for yr in r["yearly"]]
        print(f"  {r['signal']}: {', '.join(deltas)}")

    payload = {
        "dataset_size": len(results),
        "date_range": f"{results[0].date} - {results[-1].date}",
        "config": {"top_k": top_k, "universe_size": universe, "min_train_size": 30, "n_permutations": 2000},
        "theoretical_baseline_pct": p0 * 100.0,
        "results": all_results,
        "sig_binomial_count": len(sig_binom),
        "sig_permutation_count": len(sig_perm),
        "sig_both_count": len(sig_both),
        "above_baseline_count": len(above_base),
    }
    FINAL = Path("agents/stat_sig_test_topk3.json")
    with open(FINAL, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {FINAL}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python _chunk_runner.py <chunk_idx|combine>")
        sys.exit(1)

    if sys.argv[1] == "combine":
        combine()
    else:
        chunk_idx = int(sys.argv[1])
        from xsmb_pipeline.signals import SIGNAL_DEFINITIONS
        all_names = [d.name for d in SIGNAL_DEFINITIONS]
        chunk_size = (len(all_names) + 3) // 4
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(all_names))
        run_chunk(chunk_idx, all_names[start:end])
