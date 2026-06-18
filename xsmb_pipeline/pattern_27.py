from __future__ import annotations
"""
XSMB Pattern Detection - CORRECT TARGET
Target: 27 số lô = 2 số cuối của mỗi giải (đặc biệt, nhất, nhì, ba, tư, năm, sáu, bảy)
27 số duy nhất mỗi ngày.

Câu hỏi: Có nhóm 3 số nào mà P(at least 1) > 63% (base random) không?
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
sys.stdout.reconfigure(line_buffering=True)

from xsmb_pipeline.dataset import load_csv, sort_results


results = sort_results(load_csv(Path("xsmb_full.csv")))
n = len(results)
print(f"Data: {n} days", flush=True)

# ============================================================
# Build correct loto target: 27 unique 2cang = last 2 digits of each prize
# ============================================================

def get_27_loto(r) -> list[int]:
    """Lay 27 so loto = 2 so cuoi cua 27 giai."""
    codes = []
    # Special
    codes.append(int(str(r.special)[-2:]))
    # First
    if r.first:
        codes.append(int(str(r.first[0])[-2:]))
    # Second (2 prizes)
    if r.second:
        for p in r.second:
            codes.append(int(str(p)[-2:]))
    # Third (6 prizes)
    for p in r.third:
        codes.append(int(str(p)[-2:]))
    # Fourth (4 prizes)
    for p in r.fourth:
        codes.append(int(str(p)[-2:]))
    # Fifth (6 prizes)
    for p in r.fifth:
        codes.append(int(str(p)[-2:]))
    # Sixth (3 prizes)
    for p in r.sixth:
        codes.append(int(str(p)[-2:]))
    # Seventh (4 prizes)
    for p in r.seventh:
        codes.append(int(str(p)[-2:]))
    return codes  # should be exactly 27

loto_27: list[list[int]] = []
loto_27_set: list[set[int]] = []

for r in results:
    codes = get_27_loto(r)
    loto_27.append(codes)
    loto_27_set.append(set(codes))

print(f"  Loto per day: {len(loto_27[0])} numbers (expected: 27)", flush=True)

# Verify uniqueness
all_unique = all(len(s) == len(loto_27[i]) for i, s in enumerate(loto_27_set))
print(f"  All days have unique 2cang: {all_unique}", flush=True)

# Base probability
avg_loto = sum(len(s) for s in loto_27_set) / n
print(f"  Avg unique 2cang per day: {avg_loto:.1f}", flush=True)
print(f"  Base P(1 in 100) = {avg_loto:.1f}%", flush=True)
print(f"  Base P(at least 1 of 3) = {1 - (1 - avg_loto/100)**3 * 100:.1f}%", flush=True)

# ============================================================
# Build features
# ============================================================

# Delta: days since last appearance (in loto_27)
delta_27: list[list[int]] = [[0] * 100 for _ in range(n)]
last_seen_27: dict[int, int] = {}
for i in range(n):
    for code in range(100):
        delta_27[i][code] = i - last_seen_27.get(code, -1) - 1 if last_seen_27.get(code, -1) >= 0 else i + 1
    for code in loto_27_set[i]:
        last_seen_27[code] = i

# Appeared today (in loto_27)
in_loto_today: list[list[int]] = [[0] * 100 for _ in range(n)]
for i in range(n):
    for code in loto_27_set[i]:
        in_loto_today[i][code] = 1

# Repeat count (last 3, 5, 7 days)
def get_rc(day_idx: int, code: int, window: int) -> int:
    return sum(1 for j in range(max(0, day_idx - window + 1), day_idx + 1)
                if code in loto_27_set[j])

# Hot rank (last 7 days)
def get_hot_rank(day_idx: int, window: int) -> dict[int, int]:
    freq = Counter()
    for j in range(max(0, day_idx - window + 1), day_idx + 1):
        freq.update(loto_27_set[j])
    if not freq:
        return {}
    sorted_codes = sorted(freq, key=freq.get, reverse=True)
    return {code: sorted_codes.index(code) for code in freq}

# Appear count in last 7 days
def get_appear_count(day_idx: int, window: int, code: int) -> int:
    return sum(1 for j in range(max(0, day_idx - window + 1), day_idx + 1)
               if code in loto_27_set[j])

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

train_end_idx = year_first.get(2019, 0) - 1

# ============================================================
# Method 1: Global Pattern Stats
# ============================================================
print("\n" + "=" * 65, flush=True)
print("METHOD 1: GLOBAL PATTERN STATS (27-loto)", flush=True)
print("=" * 65, flush=True)

# For each 2cang, stats across ALL days
code_stats: dict[int, list[int]] = {}  # code -> [appeared_days, repeat_days]
for i in range(n - 1):
    today = loto_27_set[i]
    tomorrow = loto_27_set[i + 1]
    for code in today:
        if code not in code_stats:
            code_stats[code] = [0, 0]
        code_stats[code][0] += 1
        if code in tomorrow:
            code_stats[code][1] += 1

# Overall repeat rate
total_appear = sum(v[0] for v in code_stats.values())
total_repeat = sum(v[1] for v in code_stats.values())
base_p = total_repeat / total_appear * 100 if total_appear > 0 else 0
print(f"  Base repeat P = {base_p:.2f}% (expected ~{avg_loto:.1f}%)", flush=True)

# Top codes by repeat rate
top_codes = [(code, v[1], v[0]) for code, v in code_stats.items() if v[0] >= 100]
top_codes.sort(key=lambda x: -(x[1] / x[2]) if x[2] > 0 else 0)

print(f"\n  Top 10 by repeat P (min 100 appearances):", flush=True)
for code, reps, apps in top_codes[:10]:
    p = reps / apps * 100
    diff = p - base_p
    print(f"    {code:02d}: {reps}/{apps} = {p:.2f}% (diff={diff:+.2f}pp)", flush=True)

# ============================================================
# Method 2: Short-Window Pattern (3-5 day)
# When code appears in loto 2-3 days in a row, P of day 3?
# ============================================================
print("\n" + "=" * 65, flush=True)
print("METHOD 2: SHORT-WINDOW PATTERN (27-loto)", flush=True)
print("=" * 65, flush=True)

# Streak analysis
for streak_len in [2, 3, 4]:
    total_streaks = 0
    total_hits = 0
    for i in range(streak_len, n - 1):
        # Check if code appeared streak_len consecutive days
        streak_codes = loto_27_set[i - 1]
        valid_codes = []
        for code in streak_codes:
            is_streak = True
            for k in range(2, streak_len + 1):
                if code not in loto_27_set[i - k]:
                    is_streak = False
                    break
            if is_streak:
                valid_codes.append(code)

        if valid_codes:
            tomorrow = loto_27_set[i + 1]
            total_streaks += len(valid_codes)
            for code in valid_codes:
                if code in tomorrow:
                    total_hits += 1

    if total_streaks > 0:
        p = total_hits / total_streaks * 100
        expected = avg_loto
        diff = p - expected
        print(f"  Streak={streak_len}: {total_hits}/{total_streaks} = {p:.2f}% "
              f"(expected={expected:.1f}%, diff={diff:+.2f}pp)", flush=True)

# ============================================================
# Method 3: Cross-Prize Repetition
# If 2cang appears in MULTIPLE prizes today, does it have higher P tomorrow?
# ============================================================
print("\n" + "=" * 65, flush=True)
print("METHOD 3: CROSS-PRIZE REPETITION", flush=True)
print("=" * 65, flush=True)

# For each day, count how many prizes contain each 2cang
prize_count_per_code: list[dict[int, int]] = []
for i in range(n):
    code_count: dict[int, int] = Counter()
    for code in loto_27[i]:
        code_count[code] += 1
    prize_count_per_code.append(dict(code_count))

# Stats: when code appears in N prizes, P of appearing tomorrow
multi_prize_stats: dict[int, list[int]] = defaultdict(lambda: [0, 0])
for i in range(n - 1):
    tomorrow = loto_27_set[i + 1]
    for code, count in prize_count_per_code[i].items():
        multi_prize_stats[count][0] += 1
        if code in tomorrow:
            multi_prize_stats[count][1] += 1

print("\n  P(tomorrow | appeared in N prizes today):", flush=True)
for n_prizes in sorted(multi_prize_stats.keys()):
    apps, reps = multi_prize_stats[n_prizes]
    if apps >= 100:
        p = reps / apps * 100
        expected = avg_loto
        diff = p - expected
        print(f"    N={n_prizes}: {reps}/{apps} = {p:.2f}% "
              f"(exp={expected:.1f}%, diff={diff:+.2f}pp)", flush=True)

# ============================================================
# BACKTEST: Rolling Pattern Strategy (27-loto target)
# ============================================================
print("\n" + "=" * 65, flush=True)
print("BACKTEST: Rolling Pattern (27-loto, P > Base)", flush=True)
print("=" * 65, flush=True)

test_indices = []
for y in range(2019, 2026):
    if y in year_first:
        for i in range(year_first[y], year_last.get(y, year_first[y]) + 1):
            test_indices.append(i)

print(f"Test: {len(test_indices)} days", flush=True)

def rolling_predict(ti: int, n_days: int) -> list[tuple[int, float]]:
    """Predict using rolling stats from last n_days."""
    code_apps: dict[int, int] = Counter()
    code_reps: dict[int, int] = Counter()

    for d in range(ti - n_days, ti):
        if d < 0:
            continue
        today = loto_27_set[d]
        tomorrow = loto_27_set[d + 1]
        for code in today:
            code_apps[code] += 1
            if code in tomorrow:
                code_reps[code] += 1

    predictions: list[tuple[int, float]] = []
    today_set = loto_27_set[ti - 1]  # today's loto
    for code in range(100):
        apps = code_apps.get(code, 0)
        if apps >= n_days - 1:
            p = code_reps.get(code, 0) / apps
            if p > 0:  # Only include codes that appeared at least once
                predictions.append((code, p))

    predictions.sort(key=lambda x: -x[1])
    return predictions[:20]

# Backtest different strategies
for n_days in [3, 5, 7]:
    grand_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    grand_total = 0
    grand_bets = 0

    for ti in test_indices:
        if ti < n_days + 2 or ti >= n - 1:
            continue

        today_codes = loto_27_set[ti - 1]
        actual_set = loto_27_set[ti]
        actual_list = loto_27[ti]  # ordered

        # Predict using rolling window
        preds = rolling_predict(ti, n_days)
        if not preds:
            continue

        grand_total += 1
        grand_bets += 1

        for K in [1, 3, 5, 10]:
            topK_codes = [p[0] for p in preds[:K]]
            # P(at least 1 in topK matches any of the 27 actual)
            if any(code in actual_set for code in topK_codes):
                grand_hits[K] += 1

    if grand_bets > 0:
        print(f"\n  Rolling n={n_days}d:", flush=True)
        for K in [1, 3, 5, 10]:
            p = grand_hits[K] / grand_bets * 100
            # Expected: P(at least 1 in K random from 100 matches any of 27 in actual)
            # Random: K codes from 100, 27/100 chance each
            expected_p = 1 - (1 - avg_loto / 100) ** K * 100
            diff = p - expected_p
            mark = " <<<" if diff > 3 else ""
            print(f"    K={K}: P={p:.1f}% (exp={expected_p:.1f}%, diff={diff:+.1f}pp){mark}", flush=True)

# ============================================================
# BACKTEST: "Hot" Strategy - Predict codes that appeared in last N days
# ============================================================
print(f"\n" + "=" * 65, flush=True)
print("BACKTEST: Hot Strategy (codes that appeared recently)", flush=True)
print("=" * 65, flush=True)

# Strategy: predict codes that appeared in >= M of last N days
for n_days in [3, 5, 7]:
    for min_count in [2, 3]:
        grand_hits = {1: 0, 3: 0, 5: 0, 10: 0}
        grand_total = 0
        grand_bets = 0

        for ti in test_indices:
            if ti < n_days + 1 or ti >= n - 1:
                continue

            today_set = loto_27_set[ti - 1]
            actual_set = loto_27_set[ti]

            # Find codes that appeared >= min_count in last n_days
            code_count = Counter()
            for d in range(ti - n_days, ti):
                code_count.update(loto_27_set[d])

            hot_codes = [(code, count) for code, count in code_count.items()
                        if count >= min_count]
            hot_codes.sort(key=lambda x: -x[1])

            if not hot_codes:
                continue

            grand_total += 1
            grand_bets += 1

            for K in [1, 3, 5, 10]:
                topK = [c for c, _ in hot_codes[:K]]
                if any(code in actual_set for code in topK):
                    grand_hits[K] += 1

        if grand_bets > 0:
            print(f"\n  Hot n={n_days}d, min_count={min_count}:", flush=True)
            for K in [1, 3, 5, 10]:
                p = grand_hits[K] / grand_bets * 100
                # For hot codes, the expected is higher since we filtered
                expected_p = 1 - (1 - avg_loto / 100) ** K * 100
                diff = p - expected_p
                mark = " <<<" if diff > 3 else ""
                print(f"    K={K}: P={p:.1f}% (exp={expected_p:.1f}%, diff={diff:+.1f}pp){mark}", flush=True)

# ============================================================
# BACKTEST: Cold Strategy - Predict codes that HAVEN'T appeared recently
# ============================================================
print(f"\n" + "=" * 65, flush=True)
print("BACKTEST: Cold Strategy (codes that HAVEN'T appeared)", flush=True)
print("=" * 65, flush=True)

for n_days in [7, 14, 21]:
    grand_hits = {3: 0, 5: 0, 10: 0}
    grand_total = 0
    grand_bets = 0

    for ti in test_indices:
        if ti < n_days + 1 or ti >= n - 1:
            continue

        actual_set = loto_27_set[ti]

        # Find codes with delta >= n_days
        cold_codes = [(code, delta_27[ti - 1][code])
                     for code in range(100)
                     if delta_27[ti - 1][code] >= n_days]
        cold_codes.sort(key=lambda x: -x[1])  # coldest first

        if not cold_codes:
            continue

        grand_total += 1
        grand_bets += 1

        for K in [3, 5, 10]:
            topK = [c for c, _ in cold_codes[:K]]
            if any(code in actual_set for code in topK):
                grand_hits[K] += 1

    if grand_bets > 0:
        print(f"\n  Cold delta>={n_days}d:", flush=True)
        for K in [3, 5, 10]:
            p = grand_hits[K] / grand_bets * 100
            expected_p = 1 - (1 - avg_loto / 100) ** K * 100
            diff = p - expected_p
            mark = " <<<" if diff > 3 else ""
            print(f"    K={K}: P={p:.1f}% (exp={expected_p:.1f}%, diff={diff:+.1f}pp){mark}", flush=True)

# ============================================================
# XGBoost ML on 27-loto target
# ============================================================
print(f"\n" + "=" * 65, flush=True)
print("XGBOOST ML (27-loto target)", flush=True)
print("=" * 65, flush=True)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

if HAS_XGB:
    print("Building training set...", flush=True)
    X_train = []
    y_train = []

    for ti in range(30, train_end_idx + 1):
        for code in range(100):
            X_train.append([
                delta_27[ti][code] / 100.0,
                get_rc(ti, code, 3) / 3.0,
                get_rc(ti, code, 7) / 7.0,
                get_appear_count(ti, 7, code) / 7.0,
                in_loto_today[ti - 1][code],
                get_hot_rank(ti, 7).get(code, 99) / 100.0,
            ])
            y_train.append(1 if code in loto_27_set[ti] else 0)

    print(f"  Train: {len(X_train)} samples, {sum(y_train)} positives", flush=True)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        "objective": "binary:logistic",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 30,
        "seed": 42,
        "verbosity": 0,
    }
    model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)

    print("\n  Year-by-year:", flush=True)
    grand_total = 0
    grand_hits = {3: 0, 5: 0, 10: 0}

    for test_year in range(2019, 2026):
        if test_year not in year_first:
            continue
        ts = year_first[test_year]
        te = year_last.get(test_year)
        if te is None or te - ts < 30:
            continue

        year_hits = {3: 0, 5: 0, 10: 0}
        year_total = 0

        for ti in range(ts, te + 1):
            if ti < 30 or ti >= n - 1:
                continue

            X_test = []
            for code in range(100):
                X_test.append([
                    delta_27[ti][code] / 100.0,
                    get_rc(ti, code, 3) / 3.0,
                    get_rc(ti, code, 7) / 7.0,
                    get_appear_count(ti, 7, code) / 7.0,
                    in_loto_today[ti - 1][code],
                    get_hot_rank(ti, 7).get(code, 99) / 100.0,
                ])

            dtest = xgb.DMatrix(X_test)
            probs = model.predict(dtest)
            topK = sorted(range(100), key=lambda i: -probs[i])
            actual_set = loto_27_set[ti]

            year_total += 1
            grand_total += 1

            for K in [3, 5, 10]:
                if any(code in actual_set for code in topK[:K]):
                    year_hits[K] += 1
                    grand_hits[K] += 1

        if year_total > 0:
            print(f"    {test_year}: {year_total}d | "
                  f"P@3={year_hits[3]/year_total*100:.1f}% | "
                  f"P@5={year_hits[5]/year_total*100:.1f}% | "
                  f"P@10={year_hits[10]/year_total*100:.1f}%", flush=True)

    print(f"\n  OVERALL:", flush=True)
    for K in [3, 5, 10]:
        p = grand_hits[K] / grand_total * 100 if grand_total > 0 else 0
        expected_p = 1 - (1 - avg_loto / 100) ** K * 100
        diff = p - expected_p
        print(f"    K={K}: P={p:.2f}% (exp={expected_p:.1f}%, diff={diff:+.2f}pp)", flush=True)

    # Feature importance
    imp = model.get_score(importance_type="gain")
    names = ["delta", "rc3", "rc7", "appear7", "in_loto_yesterday", "hot_rank"]
    print(f"\n  Feature importance:", flush=True)
    for f, s in sorted(imp.items(), key=lambda x: -x[1])[:6]:
        print(f"    {names[int(f[1:])]}: {s:.2f}", flush=True)

# ============================================================
# FINAL QUESTION: Can we find 3 numbers with P(at least 1) > 63%?
# ============================================================
print(f"\n" + "=" * 65, flush=True)
print("FINAL: CAN WE FIND 3 NUMBERS BETTER THAN RANDOM?", flush=True)
print("=" * 65, flush=True)

base_P3 = (1 - (1 - avg_loto / 100) ** 3) * 100
print(f"  Random baseline P(at least 1 of 3): {base_P3:.1f}%", flush=True)
print(f"  We need P > {base_P3:.1f}% with 3 numbers", flush=True)
print(f"\n  XGBoost P@3 = see results above", flush=True)
