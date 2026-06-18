"""
Quick Statistical Significance Test - Chi 5 signals can thiet.
Uses existing walkforward backtest framework.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

# Write to file immediately
LOG = Path("agents/_quick_stat_log.txt")
LOG.write_text("", encoding="utf-8")

def log(msg):
    LOG.write_text(LOG.read_text(encoding="utf-8") + msg + "\n", encoding="utf-8")
    print(msg, flush=True)


def main() -> None:
    t_start = time.time()
    random.seed(42)
    np.random.seed(42)

    log("=== Quick Stat Sig Test ===")

    from xsmb_pipeline.dataset import load_csv, sort_results
    from xsmb_pipeline.evaluate import walkforward_signal_backtest
    from xsmb_pipeline.targets import target_width

    # Load data
    results = sort_results(load_csv(Path("xsmb_full.csv")))
    universe = 10 ** target_width("loto2")
    top_k = 3
    min_train = 30
    sample_every = 5
    n_perm = 5000
    p0 = 1.0 - (math.comb(universe - 1, top_k) / math.comb(universe, top_k))

    test_indices = list(range(min_train, len(results), sample_every))
    # Sample down to ~120 test days for speed
    test_indices = test_indices[::2]  # every 10 days -> ~590 days total

    log(f"Data: {len(results)} rows, {len(test_indices)} test days")
    log(f"Baseline p(hit)={p0:.4f} ({p0*100:.2f}%)")

    # 5 signals: 3 new + 2 existing benchmarks
    signals_to_test = [
        "prize_position_penalty",  # NEW
        "days_since_last",         # NEW
        "thirty_day_freq_penalty", # NEW
        "hot_trend",              # benchmark
        "cold_return",            # benchmark
    ]

    all_results = []

    for sig_name in signals_to_test:
        t_sig = time.time()
        log(f"\nProcessing {sig_name}...")

        # Use existing walkforward
        rows = walkforward_signal_backtest(
            results, target_name="loto2", signal_name=sig_name,
            top_k=top_k, min_train_size=min_train,
        )

        # Filter to sampled indices
        sampled = [r for r in rows if r["date"] in {results[i].date for i in test_indices}]

        # Actually re-run walkforward to match sampled indices
        # Get hits from the full walkforward but only keep sampled dates
        date_set = {results[i].date for i in test_indices}
        sampled = [r for r in rows if r["date"] in date_set]

        if not sampled:
            log(f"  No sampled data for {sig_name}")
            continue

        hits = [int(r["hit"]) for r in sampled]
        freq_hits = [int(r["frequency_hit"]) for r in sampled]
        precisions = [float(r["precision"]) for r in sampled]
        freq_precisions = [float(r["frequency_precision"]) for r in sampled]

        n = len(hits)
        ss = sum(hits)
        fs = sum(freq_hits)
        srate = ss / n
        frate = fs / n
        delta = (srate - frate) * 100.0

        # Binomial test (scipy)
        try:
            bt = binomtest(int(ss), n, p0, alternative="greater")
            binom_p = bt.pvalue
        except Exception:
            binom_p = 1.0

        # Wilson CI
        p_hat = ss / n
        z = 1.96
        denom = 1.0 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
        ci_lo = max(0.0, center - margin)
        ci_hi = min(1.0, center + margin)

        # Permutation test
        wins = sum(1 for s, f in zip(hits, freq_hits) if s > f)
        losses = sum(1 for s, f in zip(hits, freq_hits) if s < f)
        pool = [1] * wins + [0] * losses
        observed = wins - losses
        count_extreme = 0
        for _ in range(n_perm):
            random.shuffle(pool)
            pd = sum(1 for i in range(n) if pool[i] > 0) - sum(1 for i in range(n) if pool[i] == 0)
            if abs(pd) >= abs(observed):
                count_extreme += 1
        perm_p = count_extreme / n_perm

        wins_vs_freq = sum(1 for s, f in zip(hits, freq_hits) if s > f)
        losses_vs_freq = sum(1 for s, f in zip(hits, freq_hits) if s < f)
        win_rate = wins_vs_freq / (wins_vs_freq + losses_vs_freq) if (wins_vs_freq + losses_vs_freq) > 0 else 0.0

        # Yearly breakdown
        by_year = {}
        for r in sampled:
            year = r["date"][-4:]
            by_year.setdefault(year, []).append(r)

        yearly = []
        for year in sorted(by_year.keys()):
            yr_rows = by_year[year]
            yr_n = len(yr_rows)
            yr_hits = sum(int(r["hit"]) for r in yr_rows)
            yr_freq = sum(int(r["frequency_hit"]) for r in yr_rows)
            yearly.append({
                "year": year,
                "test_size": yr_n,
                "hit_rate_pct": yr_hits / yr_n * 100.0,
                "freq_hit_rate_pct": yr_freq / yr_n * 100.0,
                "delta_pct": (yr_hits / yr_n - yr_freq / yr_n) * 100.0,
                "wins": sum(1 for r in yr_rows if r["hit"] > r["frequency_hit"]),
                "losses": sum(1 for r in yr_rows if r["hit"] < r["frequency_hit"]),
            })

        r = {
            "signal": sig_name,
            "test_size": n,
            "hits": ss,
            "hit_rate_pct": srate * 100.0,
            "freq_hit_rate_pct": frate * 100.0,
            "delta_pct": delta,
            "theoretical_baseline_pct": p0 * 100.0,
            "binom_pvalue": binom_p,
            "binom_significant": binom_p < 0.05,
            "ci_lower_pct": ci_lo * 100.0,
            "ci_upper_pct": ci_hi * 100.0,
            "ci_covers_baseline": ci_lo <= p0 <= ci_hi,
            "perm_pvalue": perm_p,
            "perm_significant": perm_p < 0.05,
            "win_rate_pct": win_rate * 100.0,
            "wins": wins_vs_freq,
            "losses": losses_vs_freq,
            "yearly": yearly,
        }
        all_results.append(r)

        sig_b = "***" if binom_p < 0.001 else ("**" if binom_p < 0.01 else ("*" if binom_p < 0.05 else ""))
        sig_p = "***" if perm_p < 0.001 else ("**" if perm_p < 0.01 else ("*" if perm_p < 0.05 else ""))
        log(f"  Hit={srate*100:5.2f}% Freq={frate*100:5.2f}% Delta={delta:+6.2f}pp "
              f"binom={binom_p:.4f}{sig_b} perm={perm_p:.4f}{sig_p} "
              f"CI=[{ci_lo*100:.2f},{ci_hi*100:.2f}] WinRate={win_rate*100:.1f}% "
              f"({time.time()-t_sig:.0f}s)")

    # Summary
    log(f"\n{'='*70}")
    log("[SUMMARY - 5 Key Signals]")
    log(f"  Baseline: {p0*100:.2f}% (theoretical), top_k={top_k}, {len(test_indices)} test days")
    for r in sorted(all_results, key=lambda x: -x["delta_pct"]):
        sig_b = "***" if r["binom_pvalue"] < 0.001 else ("**" if r["binom_pvalue"] < 0.01 else ("*" if r["binom_pvalue"] < 0.05 else ""))
        sig_p = "***" if r["perm_pvalue"] < 0.001 else ("**" if r["perm_pvalue"] < 0.01 else ("*" if r["perm_pvalue"] < 0.05 else ""))
        log(f"  {r['signal']:<30} Hit={r['hit_rate_pct']:5.2f}% Delta={r['delta_pct']:+6.2f}pp "
              f"binom={r['binom_pvalue']:.4f}{sig_b} perm={r['perm_pvalue']:.4f}{sig_p} "
              f"CI=[{r['ci_lower_pct']:.2f},{r['ci_upper_pct']:.2f}] "
              f"WR={r['win_rate_pct']:.1f}%")

    log(f"\n[YEARLY DELTA]")
    for r in sorted(all_results, key=lambda x: -x["delta_pct"]):
        deltas = [f"{yr['year']}:{yr['delta_pct']:+.1f}" for yr in r["yearly"]]
        log(f"  {r['signal']}: {', '.join(deltas)}")

    # Save
    payload = {
        "dataset_size": len(results),
        "date_range": f"{results[0].date} - {results[-1].date}",
        "test_days": len(test_indices),
        "sampling_every": "every 10 days",
        "config": {"top_k": top_k, "universe_size": universe, "min_train_size": min_train, "n_permutations": n_perm},
        "theoretical_baseline_pct": p0 * 100.0,
        "results": all_results,
    }
    output_path = Path("agents/stat_sig_test_topk3.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"\nSaved: {output_path}")
    log(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
