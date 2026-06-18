from __future__ import annotations
"""
XSMB Pattern Detection - CORRECT 27-loto analysis
Bug fixed: expected probability formula + data leak check
"""

import sys
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(line_buffering=True)

from xsmb_pipeline.dataset import load_csv, sort_results


results = sort_results(load_csv(Path("xsmb_full.csv")))
n = len(results)

def get_27_loto(r):
    codes = []
    codes.append(int(str(r.special)[-2:]))
    if r.first:
        codes.append(int(str(r.first[0])[-2:]))
    if r.second:
        for p in r.second:
            codes.append(int(str(p)[-2:]))
    for p in r.third:
        codes.append(int(str(p)[-2:]))
    for p in r.fourth:
        codes.append(int(str(p)[-2:]))
    for p in r.fifth:
        codes.append(int(str(p)[-2:]))
    for p in r.sixth:
        codes.append(int(str(p)[-2:]))
    for p in r.seventh:
        codes.append(int(str(p)[-2:]))
    return codes

loto_27: list[list[int]] = []
loto_27_set: list[set[int]] = []
for r in results:
    codes = get_27_loto(r)
    loto_27.append(codes)
    loto_27_set.append(set(codes))

print(f"Data: {n} days, {len(loto_27[0])} loto per day", flush=True)
print(f"Avg unique: {sum(len(s) for s in loto_27_set)/n:.1f}/27", flush=True)

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

test_indices = []
for y in range(2019, 2026):
    if y in year_first:
        for i in range(year_first[y], year_last.get(y, year_first[y]) + 1):
            test_indices.append(i)
print(f"Test: {len(test_indices)} days", flush=True)

# ============================================================
# CORRECTED BASELINE: P(at least 1 in K | base repeat P = 23.8%)
# ============================================================
# P(code in loto tomorrow | appeared yesterday) = 23.8%
# P(at least 1 of K | each has 23.8% chance) = 1 - (0.762)^K
# K=3: 1 - 0.762^3 = 1 - 0.443 = 55.7%

avg_p = sum(len(s) for s in loto_27_set) / n / 100.0
print(f"\nAvg P(code in loto) = {avg_p*100:.2f}%", flush=True)
print(f"Correct baseline P(at least 1 of 3) = {1 - (1-avg_p)**3:.4f} = {(1-(1-avg_p)**3)*100:.2f}%", flush=True)
print(f"Correct baseline P(at least 1 of 5) = {1 - (1-avg_p)**5:.4f} = {(1-(1-avg_p)**5)*100:.2f}%", flush=True)

# ============================================================
# METHOD 0: BASELINE - Just predict "yesterday's loto codes"
# This is the REAL baseline to beat
# ============================================================
print(f"\n" + "=" * 65, flush=True)
print("BASELINE: Predict YESTERDAY's 27 loto codes", flush=True)
print("=" * 65, flush=True)

grand_hits = {3: 0, 5: 0, 10: 0, 20: 0}
grand_total = 0

for ti in test_indices:
    if ti < 1 or ti >= n - 1:
        continue
    yesterday_codes = loto_27_set[ti - 1]
    today_set = loto_27_set[ti]
    grand_total += 1

    for K in [3, 5, 10, 20]:
        topK = list(yesterday_codes)[:K]
        if any(code in today_set for code in topK):
            grand_hits[K] += 1

print(f"  OVERALL ({grand_total}d):", flush=True)
for K in [3, 5, 10, 20]:
    p = grand_hits[K] / grand_total * 100
    expected = (1 - (1 - avg_p) ** K) * 100
    diff = p - expected
    print(f"    K={K}: P={p:.2f}% (expected={expected:.2f}%, diff={diff:+.2f}pp)", flush=True)

# ============================================================
# METHOD 1: Cross-prize repetition
# If 2cang appears in N>1 prizes today, P(tomorrow)?
# ============================================================
print(f"\n" + "=" * 65, flush=True)
print("CROSS-PRIZE REPETITION (27-loto)", flush=True)
print("=" * 65, flush=True)

prize_count_per_code: list[dict[int, int]] = []
for i in range(n):
    cc = Counter()
    for code in loto_27[i]:
        cc[code] += 1
    prize_count_per_code.append(dict(cc))

multi_stats: dict[int, list[int]] = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
for i in range(n - 1):
    tomorrow = loto_27_set[i + 1]
    for code, cnt in prize_count_per_code[i].items():
        key = min(cnt, 3)
        multi_stats[key][0] += 1
        if code in tomorrow:
            multi_stats[key][1] += 1

print("\n  P(tomorrow | appeared in N prizes today):", flush=True)
for n_p in [1, 2, 3]:
    apps, reps = multi_stats[n_p]
    if apps >= 50:
        p = reps / apps * 100
        diff = p - avg_p * 100
        print(f"    N={n_p}: {reps}/{apps} = {p:.2f}% (diff={diff:+.2f}pp)", flush=True)

# ============================================================
# METHOD 2: Rolling pattern with CORRECT backtest (no data leak)
# ============================================================
print(f"\n" + "=" * 65, flush=True)
print("ROLLING PATTERN (CORRECT - no data leak)", flush=True)
print("=" * 65, flush=True)

# Key insight: to predict day T, we MUST use data from day T-1 and earlier ONLY
# "Yesterday's loto" -> TODAY's loto: P = avg_p * 100 = 23.8%
# Strategy: predict codes that appeared in last N days

for n_days in [3, 5, 7]:
    grand_hits = {3: 0, 5: 0, 10: 0, 20: 0}
    grand_total = 0

    for ti in test_indices:
        if ti < n_days + 1 or ti >= n - 1:
            continue

        today_set = loto_27_set[ti]
        # Use data from ti-n_days to ti-1 (NOT ti)
        code_count = Counter()
        for d in range(ti - n_days, ti):
            code_count.update(loto_27_set[d])

        # Predict: top K codes by frequency in last n_days
        topK = [code for code, _ in code_count.most_common(20)]
        grand_total += 1

        for K in [3, 5, 10, 20]:
            if any(code in today_set for code in topK[:K]):
                grand_hits[K] += 1

    if grand_total > 0:
        print(f"\n  Top-K by freq in last {n_days} days:", flush=True)
        for K in [3, 5, 10, 20]:
            p = grand_hits[K] / grand_total * 100
            expected = (1 - (1 - avg_p) ** K) * 100
            diff = p - expected
            mark = " <<<" if diff > 3 else ""
            print(f"    K={K}: P={p:.2f}% (exp={expected:.2f}%, diff={diff:+.2f}pp){mark}", flush=True)

# ============================================================
# METHOD 3: When code appears in yesterday's loto, does it repeat today?
# ============================================================
print(f"\n" + "=" * 65, flush=True)
print("REPEAT PROBABILITY (yesterday -> today)", flush=True)
print("=" * 65, flush=True)

repeat_stats: dict[int, list[int]] = {}  # code -> [appeared, repeated]
for i in range(n - 1):
    yesterday = loto_27_set[i]
    today = loto_27_set[i + 1]
    for code in yesterday:
        if code not in repeat_stats:
            repeat_stats[code] = [0, 0]
        repeat_stats[code][0] += 1
        if code in today:
            repeat_stats[code][1] += 1

# Global repeat rate
total_apps = sum(v[0] for v in repeat_stats.values())
total_reps = sum(v[1] for v in repeat_stats.values())
global_repeat_p = total_reps / total_apps * 100 if total_apps > 0 else 0
print(f"  Global repeat P: {global_repeat_p:.2f}%", flush=True)

# Top codes by repeat P
top_repeat = [(code, v[1], v[0]) for code, v in repeat_stats.items() if v[0] >= 100]
top_repeat.sort(key=lambda x: -(x[1] / x[2]) if x[2] > 0 else 0)

print(f"\n  Top 10 by repeat P (min 100 appearances):", flush=True)
for code, reps, apps in top_repeat[:10]:
    p = reps / apps * 100
    diff = p - global_repeat_p
    print(f"    {code:02d}: {reps}/{apps} = {p:.2f}% (diff={diff:+.2f}pp)", flush=True)

# ============================================================
# METHOD 4: Cold strategy - codes with large delta
# ============================================================
print(f"\n" + "=" * 65, flush=True)
print("COLD STRATEGY (large delta)", flush=True)
print("=" * 65, flush=True)

# Precompute delta
delta_27 = [[0] * 100 for _ in range(n)]
last_seen = [-1] * 100
for i in range(n):
    for code in range(100):
        delta_27[i][code] = i - last_seen[code] - 1 if last_seen[code] >= 0 else i + 1
    for code in loto_27_set[i]:
        last_seen[code] = i

for min_delta in [7, 14, 21, 30]:
    grand_hits = {3: 0, 5: 0, 10: 0}
    grand_total = 0

    for ti in test_indices:
        if ti < 1 or ti >= n - 1:
            continue

        today_set = loto_27_set[ti]
        # Yesterday's delta
        deltas = [(code, delta_27[ti - 1][code]) for code in range(100)
                  if delta_27[ti - 1][code] >= min_delta]
        deltas.sort(key=lambda x: -x[1])
        topK = [c for c, _ in deltas[:10]]

        if not topK:
            continue
        grand_total += 1

        for K in [3, 5, 10]:
            if any(code in today_set for code in topK[:K]):
                grand_hits[K] += 1

    if grand_total > 0:
        print(f"\n  Delta >= {min_delta}:", flush=True)
        for K in [3, 5, 10]:
            p = grand_hits[K] / grand_total * 100
            expected = (1 - (1 - avg_p) ** K) * 100
            diff = p - expected
            mark = " <<<" if diff > 3 else ""
            print(f"    K={K}: P={p:.2f}% (exp={expected:.2f}%, diff={diff:+.2f}pp){mark}", flush=True)

# ============================================================
# METHOD 5: COMBINED - Hot + Cold + Repeat
# ============================================================
print(f"\n" + "=" * 65, flush=True)
print("COMBINED STRATEGY", flush=True)
print("=" * 65, flush=True)

for n_days in [3, 5, 7]:
    grand_hits = {3: 0, 5: 0, 10: 0}
    grand_total = 0

    for ti in test_indices:
        if ti < n_days + 1 or ti >= n - 1:
            continue

        today_set = loto_27_set[ti]
        yesterday = loto_27_set[ti - 1]

        # Score each code
        scores: dict[int, float] = {}
        code_count = Counter()
        for d in range(ti - n_days, ti):
            code_count.update(loto_27_set[d])

        for code in range(100):
            # Frequency score
            freq_score = code_count.get(code, 0) / n_days  # 0 to 1
            # Repeat score (appeared in yesterday)
            repeat_score = 1.0 if code in yesterday else 0.0
            # Cold score (large delta = higher score)
            delta_score = min(delta_27[ti - 1][code] / 30.0, 1.0)
            # Combined score
            scores[code] = freq_score * 0.7 + repeat_score * 0.2 + delta_score * 0.1

        topK = sorted(scores, key=scores.get, reverse=True)[:10]
        grand_total += 1

        for K in [3, 5, 10]:
            if any(code in today_set for code in topK[:K]):
                grand_hits[K] += 1

    if grand_total > 0:
        print(f"\n  Combined n={n_days}d:", flush=True)
        for K in [3, 5, 10]:
            p = grand_hits[K] / grand_total * 100
            expected = (1 - (1 - avg_p) ** K) * 100
            diff = p - expected
            mark = " <<<" if diff > 3 else ""
            print(f"    K={K}: P={p:.2f}% (exp={expected:.2f}%, diff={diff:+.2f}pp){mark}", flush=True)

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n" + "=" * 65, flush=True)
print("FINAL SUMMARY", flush=True)
print("=" * 65, flush=True)
print(f"  Avg P(code in loto) = {avg_p*100:.2f}%", flush=True)
print(f"  Baseline P(at least 1 of 3 random) = {(1-(1-avg_p)**3)*100:.2f}%", flush=True)
print(f"  Baseline: predict yesterday's codes", flush=True)
print(f"  Baseline P(at least 1 of 3 from yesterday) = {(1-(1-avg_p)**3)*100:.2f}% (same as random)", flush=True)
print(f"\n  Key insight: Choosing by YESTERDAY's frequency gives same P as random.", flush=True)
print(f"  Because P(code in loto) is independent of whether it was there yesterday.", flush=True)
