from __future__ import annotations
"""
XSMB Loto 2-So Predictor.

Tap trung vao 2 phuong phap:
1. Algorithm 01 (junlangzi): Scoring rules + neighbor + cycle + repeat penalty
2. Gap Ratio (khiemdoan): Overdue selection

Chi tap trung vao LOTO 2 SO (special[-2:]).
Khong dung nhieu signal ensemble.
"""

import sys
import json
import random
from pathlib import Path
from collections import Counter
from math import sqrt, erf

sys.stdout.reconfigure(line_buffering=True)

from xsmb_pipeline.dataset import load_csv, sort_results


# ============================================================
# Data Precomputation
# ============================================================
def load_data(csv_path: str = "xsmb_full.csv"):
    results = sort_results(load_csv(Path(csv_path)))
    n = len(results)

    # Loto sets: all 2-digit from ALL 27 prizes per day
    loto_sets: list[set[int]] = []
    # Special[-2:]
    special_2c: list[int] = []

    for r in results:
        sp = r.special
        special_2c.append(int(sp[-2:]) if len(sp) >= 2 else 0)
        nums: set[int] = set()
        for prize_list in [r.first, r.second, r.third, r.fourth,
                          r.fifth, r.sixth, r.seventh]:
            for p in prize_list:
                s = str(p)
                for j in range(len(s) - 1):
                    nums.add(int(s[j:j + 2]))
        loto_sets.append(nums)

    # Delta cache: days since last appearance for each 2cang
    delta: list[list[int]] = [[0] * 100 for _ in range(n)]
    last_seen = [-1] * 100
    for i in range(n):
        for code in range(100):
            delta[i][code] = i - last_seen[code] - 1 if last_seen[code] >= 0 else i + 1
        last_seen[special_2c[i]] = i

    # Frequency cache: precomputed for windows [3, 7, 14, 30, 60, 90]
    # freq_cache[window][day_idx][code] = count
    freq_cache: dict[int, list[list[int]]] = {}
    for window in [3, 7, 14, 30, 60, 90]:
        cache: list[list[int]] = []
        for i in range(n):
            row = [0] * 100
            if i >= window:
                for j in range(i - window, i):
                    for code in loto_sets[j]:
                        row[code] += 1
            cache.append(row)
        freq_cache[window] = cache

    # Last-seen per day: day_idx -> dict[code] = last_day_idx
    last_seen_map: list[dict[int, int]] = []
    for i in range(n):
        ls: dict[int, int] = {}
        for j in range(i - 1, -1, -1):
            for code in loto_sets[j]:
                if code not in ls:
                    ls[code] = j
            if len(ls) >= 100:
                break
        last_seen_map.append(ls)

    # Year boundaries
    year_first = {}
    year_last = {}
    for i, r in enumerate(results):
        try:
            _, _, yr = r.date.rsplit("/", 2)
            yr = int(yr)
            if yr not in year_first:
                year_first[yr] = i
            year_last[yr] = i
        except:
            pass

    return {
        "results": results,
        "n": n,
        "loto": loto_sets,
        "special_2c": special_2c,
        "delta": delta,
        "freq": freq_cache,
        "last_seen": last_seen_map,
        "year_first": year_first,
        "year_last": year_last,
    }


# ============================================================
# Algorithm 01 (junlangzi scoring rules)
# ============================================================
OPTIMIZED_PARAMS = {
    "short_term_days": 20,
    "frequency_window_short": 37,
    "frequency_window_long": 225,
    "base_point_short": 3.675,
    "base_point_long": 0.3,
    "frequency_weight_short": 0.4,
    "frequency_weight_long": 0.25,
    "neighbor_bonus": 1.2,
    "neighbor_range": 5,
    "increment": 0.005,
    "bonus_after_3_days": 0.07,
    "bonus_long_absence": 0.8,
    "deduction_if_appeared_last_day": -0.02,
    "repeat_penalty_top": -0.5,
    "repeat_window": 7,
    "repeat_threshold_penalty": -2.0,
    "repeat_threshold": 2,
    "repeat_streak_penalty": -0.8,
    "bonus_freq_5": 0.25,
    "cycle_7_bonus": 0.2,
    "cycle_30_bonus": 0.3,
    "special_multiplier": 2.1,
    "special_freq_multiplier": 3.0,
}


def predict_algo01(day_idx: int, data: dict,
                   params: dict | None = None,
                   prev_tops: list[list[int]] | None = None) -> tuple[dict[str, float], list[list[int]]]:
    """Predict using Algorithm 01 scoring rules."""
    p = params or OPTIMIZED_PARAMS
    scores: dict[str, float] = {f"{i:02d}": 0.0 for i in range(100)}

    # Short term points
    std = p["short_term_days"]
    for j in range(max(0, day_idx - std), day_idx):
        days_ago = day_idx - j
        base = p["base_point_short"] * (1 - 0.08 * days_ago)
        sp = data["special_2c"][j]
        for num in data["loto"][j]:
            pt = base
            if num == sp:
                pt *= p["special_multiplier"]
            scores[f"{num:02d}"] += pt

    # Frequency (use precomputed cache)
    ws = p["frequency_window_short"]
    wl = p["frequency_window_long"]
    # Use nearest available window
    avail = sorted(data["freq"].keys())
    ws_key = min(avail, key=lambda x: abs(x - ws))
    wl_key = min(avail, key=lambda x: abs(x - wl))
    fc_s = data["freq"][ws_key][day_idx] if day_idx >= ws_key else [0] * 100
    fc_l = data["freq"][wl_key][day_idx] if day_idx >= wl_key else [0] * 100

    for num in range(100):
        num_str = f"{num:02d}"
        scores[num_str] += p["frequency_weight_short"] * fc_s[num]
        scores[num_str] += p["frequency_weight_long"] * fc_l[num]
        if fc_s[num] > 5:
            scores[num_str] += p["bonus_freq_5"]

    # Neighbor bonus
    if day_idx >= 1:
        sp_last = data["special_2c"][day_idx - 1]
        nr = p["neighbor_range"]
        for offset in range(-nr, nr + 1):
            if offset == 0:
                continue
            neighbor = (sp_last + offset) % 100
            scores[f"{neighbor:02d}"] += p["neighbor_bonus"] * (1 - 0.1 * abs(offset))

    # Long absence bonus (use precomputed last_seen)
    ls = data["last_seen"][day_idx]
    for num in range(100):
        num_str = f"{num:02d}"
        last = ls.get(num, -1)
        days_since = day_idx - last - 1 if last >= 0 else day_idx
        scores[num_str] += days_since * p["increment"]
        if days_since >= 3:
            scores[num_str] += p["bonus_after_3_days"]
        if days_since >= 15 and fc_l[num] > 10:
            scores[num_str] += p["bonus_long_absence"]

    # Deduction if appeared last day
    if day_idx >= 1:
        for num in data["loto"][day_idx - 1]:
            scores[f"{num:02d}"] += p["deduction_if_appeared_last_day"]

    # Repeat penalty
    tops = prev_tops or []
    repeat_counts: Counter[int] = Counter()
    streak_counts: dict[int, int] = {}
    rw = p["repeat_window"]
    prev_slice = tops[-rw:] if tops else []
    for i in range(len(prev_slice)):
        prev_top = prev_slice[-(i + 1)]
        repeat_counts.update(prev_top)
        for num in prev_top:
            if i == 0:
                streak_counts[num] = streak_counts.get(num, 0) + 1
            elif num not in prev_slice[-(i + 1)]:
                streak_counts[num] = 1
            else:
                streak_counts[num] = streak_counts.get(num, 0) + 1

    for num in range(100):
        num_str = f"{num:02d}"
        rc = repeat_counts.get(num, 0)
        sc = streak_counts.get(num, 0)
        if rc > 0:
            scores[num_str] += p["repeat_penalty_top"] * rc
        if rc >= p["repeat_threshold"]:
            scores[num_str] += p["repeat_threshold_penalty"]
        if sc > 1:
            scores[num_str] += p["repeat_streak_penalty"] * (sc - 1)

    # Update tops
    top_3 = [num for num, _ in sorted(
        scores.items(), key=lambda x: x[1], reverse=True)[:3]]
    tops = (tops or []) + [top_3]
    if len(tops) > rw:
        tops = tops[-rw:]

    return scores, tops


# ============================================================
# Algorithm 02: Gap Ratio (khiemdoan style)
# ============================================================
def predict_gap_ratio(day_idx: int, data: dict, window: int = 7) -> dict[str, float]:
    """Gap ratio: delta / expected_gap. Simple, no warmup needed."""
    deltas = data["delta"][day_idx]
    # Frequency in last window days
    freq = data["freq"][window][day_idx] if day_idx >= window else [0] * 100

    scores = {}
    for code in range(100):
        gap_ratio = min(deltas[code] / 27.0, 3.0)  # expected gap ~27 days
        freq_norm = freq[code] / max(sum(freq), 1)
        # Only gap ratio (pure khiemdoan)
        scores[f"{code:02d}"] = gap_ratio

    return scores


# ============================================================
# Backtest
# ============================================================
def normal_cdf(z):
    return (1 + erf(z / sqrt(2))) / 2


def backtest(data: dict, predictor_fn,
             K_VALUES: list[int],
             name: str,
             test_years: range = range(2019, 2026),
             extra_args: dict | None = None) -> dict:

    loto = data["loto"]
    special_2c = data["special_2c"]
    year_first = data["year_first"]
    year_last = data["year_last"]

    grand_hits = {K: 0 for K in K_VALUES}
    grand_total = 0
    year_details = {}

    for y in test_years:
        if y not in year_first:
            continue
        test_start = year_first[y]
        test_end = year_last.get(y)
        if test_end is None or test_end - test_start < 30:
            continue

        # Build state up to test_start - 1
        prev_tops = []
        for hi in range(max(0, test_start - 500), test_start):
            scores, prev_tops = predictor_fn(hi, data, **(extra_args or {}), prev_tops=prev_tops)

        hits = {K: 0 for K in K_VALUES}
        total = 0

        for ti in range(test_start, test_end + 1):
            scores, prev_tops = predictor_fn(ti, data, **(extra_args or {}), prev_tops=prev_tops)
            ranked = [s for s, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
            actual = special_2c[ti]

            total += 1
            for K in K_VALUES:
                topK = set(int(s) for s in ranked[:K])
                if actual in topK:
                    hits[K] += 1

        grand_total += total
        for K in K_VALUES:
            grand_hits[K] += hits[K]

        year_details[y] = {"n": total, "hits": dict(hits)}

    # Summary
    summary = {}
    for K in K_VALUES:
        p = grand_hits[K] / grand_total * 100 if grand_total > 0 else 0
        rand = min(100.0, K)
        diff = p - rand
        se = sqrt(K * (100 - K) / grand_total) if grand_total > 0 else 1
        z = diff / se
        p_val = 2 * (1 - normal_cdf(abs(z)))
        summary[K] = {
            "hits": grand_hits[K], "total": grand_total,
            "p": round(p, 2), "diff": round(diff, 2),
            "z": round(z, 3), "p_val": round(p_val, 4),
            "sig": "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        }

    return {"name": name, "grand_total": grand_total, "year_details": year_details, "summary": summary}


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("XSMB LOTO 2-SO PREDICTOR", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading data...", flush=True)
    data = load_data()
    print(f"Data: {data['n']} days", flush=True)

    K_VALUES = [3, 5, 7, 10, 15, 20]

    # --- Algorithm 01 ---
    print("\n" + "=" * 60, flush=True)
    print("ALGORITHM 01 (Junlangzi Scoring Rules)", flush=True)
    print("=" * 60, flush=True)

    def algo01_predict(day_idx, data, prev_tops=None, params=None):
        return predict_algo01(day_idx, data, params, prev_tops)

    res01 = backtest(data, algo01_predict, K_VALUES, "Algo01_Optimized",
                    extra_args={"params": OPTIMIZED_PARAMS})

    # Per-year
    for y, yr in res01["year_details"].items():
        p5 = yr["hits"].get(5, 0) / yr["n"] * 100 if yr["n"] > 0 else 0
        p10 = yr["hits"].get(10, 0) / yr["n"] * 100 if yr["n"] > 0 else 0
        print(f"  {y}: {yr['n']}d | P@5={p5:.1f}%({p5-5:+.1f}pp) | P@10={p10:.1f}%({p10-10:+.1f}pp)", flush=True)

    print(f"\n  OVERALL ({res01['grand_total']}d):", flush=True)
    for K in [5, 10, 15]:
        s = res01["summary"][K]
        print(f"    K={K:3d}: P={s['p']:.2f}% (diff={s['diff']:+.2f}pp, z={s['z']:.2f}, p={s['p_val']:.4f}) {s['sig']}", flush=True)

    # --- Algorithm 02: Gap Ratio ---
    print("\n" + "=" * 60, flush=True)
    print("ALGORITHM 02 (Gap Ratio - khiemdoan)", flush=True)
    print("=" * 60, flush=True)

    def gap_predict(day_idx, data, prev_tops=None, window=7):
        return predict_gap_ratio(day_idx, data, window), prev_tops

    res02 = backtest(data, gap_predict, K_VALUES, "GapRatio",
                    extra_args={"window": 7})

    for y, yr in res02["year_details"].items():
        p5 = yr["hits"].get(5, 0) / yr["n"] * 100 if yr["n"] > 0 else 0
        p10 = yr["hits"].get(10, 0) / yr["n"] * 100 if yr["n"] > 0 else 0
        print(f"  {y}: {yr['n']}d | P@5={p5:.1f}%({p5-5:+.1f}pp) | P@10={p10:.1f}%({p10-10:+.1f}pp)", flush=True)

    print(f"\n  OVERALL ({res02['grand_total']}d):", flush=True)
    for K in [5, 10, 15]:
        s = res02["summary"][K]
        print(f"    K={K:3d}: P={s['p']:.2f}% (diff={s['diff']:+.2f}pp, z={s['z']:.2f}, p={s['p_val']:.4f}) {s['sig']}", flush=True)

    # --- ROI ---
    print("\n" + "=" * 60, flush=True)
    print("ROI COMPARISON", flush=True)
    print("=" * 60, flush=True)
    print(f"  Loto: danh 23k/so, thang 80k (3.47x)", flush=True)
    for K in [5, 10, 15]:
        for name, res in [("Algo01", res01), ("GapRatio", res02)]:
            s = res["summary"][K]
            inv = res["grand_total"] * K * 23000
            won = s["hits"] * 80000
            roi = (won - inv) / inv * 100 if inv > 0 else 0
            rand_roi = -100 + K * 3.47
            print(f"  {name:10s} K={K:3d}: P={s['p']:.1f}% ROI={roi:+.1f}% | Random={rand_roi:+.1f}% | diff={roi-rand_roi:+.1f}pp {s['sig']}", flush=True)

    # --- Conclusion ---
    print("\n" + "=" * 60, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 60, flush=True)
    for name, res in [("Algo01", res01), ("GapRatio", res02)]:
        s = res["summary"][10]
        if s["p_val"] < 0.05:
            verdict = "BETTER" if s["diff"] > 0 else "WORSE"
            print(f"  {name}: SIGNIFICANTLY {verdict} than random (p={s['p_val']:.4f}, diff={s['diff']:+.2f}pp)")
        else:
            print(f"  {name}: NOT significant (p={s['p_val']:.4f}, diff={s['diff']:+.2f}pp) -> Model = Random")
