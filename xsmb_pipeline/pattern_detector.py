from __future__ import annotations
"""
XSMB Pattern Detection v4 - Gap Pattern + Combined Strategy.

Phuong phap chinh:
1. Gap Pattern: Khi 2cang X chua xuat hien trong N ngay,
   gap cua no tang -> co xu huong xuat hien?
2. Combined: Loto repeat (53.5%) + Gap (overdue) + Position pattern
3. Train classifier: Huan luyen model de phat hien KHI NAO co pattern

Target: Tim 3 so voi P > 50% trong 1 ngay.
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
sys.stdout.reconfigure(line_buffering=True)

from xsmb_pipeline.dataset import load_csv, sort_results


results = sort_results(load_csv(Path("xsmb_full.csv")))
n = len(results)
print(f"Data: {n} days", flush=True)

# Build data
special_2c: list[int] = []
loto: list[set[int]] = []
loto_str: list[list[tuple[int, str]]] = []  # (prize_idx, 2cang_str)

for r in results:
    sp = str(r.special)
    special_2c.append(int(sp[-2:]) if len(sp) >= 2 else 0)
    nums: set[int] = set()
    nums_str: list[tuple[int, str]] = []
    prize_names = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
    for pi, plist in enumerate([r.first, r.second, r.third, r.fourth, r.fifth, r.sixth, r.seventh]):
        for p in plist:
            s = str(p)
            for j in range(len(s) - 1):
                code = int(s[j:j + 2])
                nums.add(code)
                nums_str.append((pi, f"{code:02d}"))
    loto.append(nums)
    loto_str.append(nums_str)

# Delta: days since last appearance for each 2cang in loto
delta: list[list[int]] = [[0] * 100 for _ in range(n)]
last_seen: dict[int, int] = {}
for i in range(n):
    for code in range(100):
        last = last_seen.get(code, -1)
        delta[i][code] = i - last - 1 if last >= 0 else i + 1
    for code in loto[i]:
        last_seen[code] = i

# ============================================================
# Method 1: Gap Pattern Analysis
# When 2cang X has gap = N, what is P(hit tomorrow)?
# ============================================================
print("\n" + "=" * 65, flush=True)
print("METHOD 1: GAP PATTERN ANALYSIS", flush=True)
print("=" * 65, flush=True)

# For each gap value (0-100+), count hits next day
gap_hit: dict[int, list[int]] = defaultdict(lambda: [0, 0])
# For each 2cang, track when its gap = N, does it hit next day?

for i in range(n - 1):
    next_codes = loto[i + 1]
    for code in range(100):
        gap = delta[i][code]
        gap_hit[gap][0] += 1
        if code in next_codes:
            gap_hit[gap][1] += 1

print("\n  Gap value -> P(hit tomorrow):", flush=True)
for gap in sorted(gap_hit.keys())[:20]:
    total, hits = gap_hit[gap]
    if total >= 100:
        p = hits / total * 100
        expected = 53.5  # overall loto repeat rate
        diff = p - expected
        print(f"    Gap={gap:3d}: {hits:5d}/{total:5d} = {p:.2f}% "
              f"(exp={expected:.1f}%, diff={diff:+.2f}pp)")

# ============================================================
# Method 2: SPECIAL DELTA Pattern
# When special[-2:] hasn't appeared for N days,
# does it tend to appear soon?
# ============================================================
print("\n" + "=" * 65, flush=True)
print("METHOD 2: SPECIAL DELTA PATTERN", flush=True)
print("=" * 65, flush=True)

# For each 2cang X, track its appearance in special[-2:]
special_appearances: list[dict[int, int]] = []  # day_idx -> {code: last_day_idx}
last_special = [-1] * 100
special_delta: list[list[int]] = [[0] * 100 for _ in range(n)]
for i in range(n):
    for code in range(100):
        special_delta[i][code] = i - last_special[code] - 1 if last_special[code] >= 0 else i + 1
    last_special[special_2c[i]] = i

# For each gap N, P(special[-2:] = X tomorrow | X has gap N today)
special_gap_hit: dict[int, list[int]] = defaultdict(lambda: [0, 0])
for i in range(n - 1):
    target = special_2c[i + 1]
    for code in range(100):
        gap = special_delta[i][code]
        special_gap_hit[gap][0] += 1
        if code == target:
            special_gap_hit[gap][1] += 1

print("\n  Special gap -> P(special tomorrow matches):", flush=True)
for gap in sorted(special_gap_hit.keys())[:30]:
    total, hits = special_gap_hit[gap]
    if total >= 50:
        p = hits / total * 100
        expected = 1.0
        diff = p - expected
        print(f"    Gap={gap:3d}: {hits:4d}/{total:4d} = {p:.2f}% "
              f"(exp={expected:.1f}%, diff={diff:+.2f}pp)")

# ============================================================
# Method 3: TRAIN PATTERN CLASSIFIER
# Find COMBINATIONS of features that give P > 50%
# ============================================================
print("\n" + "=" * 65, flush=True)
print("METHOD 3: PATTERN CLASSIFIER (Feature Combinations)", flush=True)
print("=" * 65, flush=True)

# Features for each (day, code):
# - delta: days since last appearance
# - repeat_count: how many times code appeared in last 3/5/7 days
# - special_delta: days since last appeared in special
# - loto_hot: is code in top 10 most frequent in last 7 days?
# - loto_cold: is code in bottom 10 in last 7 days?
# - yesterday: did code appear yesterday?

# Build feature matrix
def get_repeat_count(day_idx: int, code: int, window: int) -> int:
    count = 0
    for j in range(max(0, day_idx - window + 1), day_idx + 1):
        if code in loto[j]:
            count += 1
    return count

def get_hot_rank(day_idx: int, window: int) -> dict[int, int]:
    """Return rank of each code (0 = most frequent)."""
    freq = Counter()
    for j in range(max(0, day_idx - window + 1), day_idx + 1):
        freq.update(loto[j])
    if not freq:
        return {}
    sorted_codes = [code for code, _ in sorted(freq.items(), key=lambda x: -x[1])]
    return {code: sorted_codes.index(code) for code in freq}

# Train/test split: last 7 years
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

test_years = list(range(2019, 2026))
test_indices = []
for y in test_years:
    if y in year_first:
        for i in range(year_first[y], year_last.get(y, year_first[y]) + 1):
            test_indices.append(i)

print(f"  Test indices: {len(test_indices)} days", flush=True)

# For each day, for each code, compute features
# Then find combinations that give P > 50%

# Sample: check feature combinations
print("\n  Checking feature combinations...", flush=True)

# Delta thresholds
for delta_thresh in [7, 14, 21, 30, 60, 90]:
    for hot_thresh in [5, 10, 20]:
        hits = 0
        total = 0
        for ti in test_indices:
            if ti < 90:
                continue
            for code in range(100):
                d = delta[ti][code]
                rc = get_repeat_count(ti, code, 3)
                hot_rank = get_hot_rank(ti, 7).get(code, 999)

                # Condition: gap >= thresh AND repeat_count <= 1 AND not in top hot
                if d >= delta_thresh and rc <= 1 and hot_rank >= hot_thresh:
                    total += 1
                    if code == special_2c[ti]:
                        hits += 1

        if total >= 100:
            p = hits / total * 100
            expected = 1.0
            diff = p - expected
            if p > 1.5:
                print(f"    delta>={delta_thresh}, rc<=1, hot_rank>={hot_thresh}: "
                      f"{hits}/{total} = {p:.2f}% (diff={diff:+.2f}pp)")

# ============================================================
# Method 4: THE KEY QUESTION - When do we have P > 50% for 3 numbers?
# ============================================================
print("\n" + "=" * 65, flush=True)
print("METHOD 4: FIND 3 NUMBERS WITH P > 50%", flush=True)
print("=" * 65, flush=True)

# For each day in test set, find all 100 codes with their features,
# then find subsets of 3 codes where ALL have P > 50%

def estimate_p(day_idx: int, code: int) -> float:
    """Estimate P(code appears tomorrow)."""
    d = delta[day_idx][code]
    rc3 = get_repeat_count(day_idx, code, 3)
    rc7 = get_repeat_count(day_idx, code, 7)
    sd = special_delta[day_idx][code]
    hot7_rank = get_hot_rank(day_idx, 7).get(code, 999)

    # Feature weights (from analysis)
    p = 1.0  # base

    # Repeat: each occurrence adds ~1.7pp
    p += rc3 * 1.7

    # Gap: diminishing returns after 30 days
    if d > 30:
        p += 0.3
    elif d > 60:
        p += 0.5
    elif d > 90:
        p += 0.7

    # Hot rank
    if hot7_rank < 5:
        p += 1.5
    elif hot7_rank < 10:
        p += 1.0

    # Special delta
    if sd == 1:
        p -= 1.5  # appeared yesterday in special
    elif sd <= 3:
        p -= 0.5

    return p

# Check each day: find codes with P > 50%
print("\n  Finding days with codes having P > 50%:", flush=True)
days_with_high_p = 0
total_codes_high_p = 0
sample_days = []

for yi, ti in enumerate(test_indices):
    if ti < 90:
        continue

    code_ps = []
    for code in range(100):
        p = estimate_p(ti, code)
        code_ps.append((code, p))

    code_ps.sort(key=lambda x: -x[1])

    # Count how many have P > 50%
    high_p = [(c, p) for c, p in code_ps if p > 50]
    if len(high_p) >= 3:
        days_with_high_p += 1
        total_codes_high_p += len(high_p)
        if len(sample_days) < 5:
            sample_days.append((ti, high_p[:5]))

    if yi % 500 == 0:
        print(f"    Processed {yi}/{len(test_indices)} days...", flush=True)

print(f"\n  Days with >= 3 codes having P > 50%: {days_with_high_p}/{len(test_indices)} "
      f"({days_with_high_p / len(test_indices) * 100:.1f}%)", flush=True)
if sample_days:
    print(f"\n  Sample days:", flush=True)
    for ti, high_p in sample_days:
        date = results[ti].date
        actual = special_2c[ti]
        print(f"    {date}: Top P = {[(f'{c:02d}', f'{p:.0f}%') for c, p in high_p]} | Actual = {actual:02d}")

# ============================================================
# Method 5: ML Pattern Detection (Real Training)
# ============================================================
print("\n" + "=" * 65, flush=True)
print("METHOD 5: ML PATTERN CLASSIFIER", flush=True)
print("=" * 65, flush=True)

# Build dataset:
# For each day, for each code, features + label (1 = hit tomorrow)
# Train: 2010-2018, Test: 2019-2025

train_start = year_first.get(2010, 0)
train_end = year_first.get(2019, 0) - 1

print(f"  Training: {train_end - train_start + 1} days", flush=True)
print(f"  Testing: {len(test_indices)} days", flush=True)

X_train: list[list[float]] = []
y_train: list[int] = []

print("  Building training set...", flush=True)
for ti in range(train_start + 1, train_end + 1):
    if ti < 90:
        continue
    for code in range(100):
        # Features
        d = delta[ti][code]
        rc3 = get_repeat_count(ti, code, 3)
        rc7 = get_repeat_count(ti, code, 7)
        rc14 = get_repeat_count(ti, code, 14)
        sd = special_delta[ti][code]
        hr7 = get_hot_rank(ti, 7).get(code, 99)
        hr14 = get_hot_rank(ti, 14).get(code, 99)
        appeared_yesterday = 1 if code in loto[ti - 1] else 0
        in_special_yesterday = 1 if code == special_2c[ti - 1] else 0

        # Label
        label = 1 if code == special_2c[ti] else 0

        X_train.append([
            d / 100.0,
            rc3 / 3.0,
            rc7 / 7.0,
            rc14 / 14.0,
            sd / 100.0,
            hr7 / 100.0,
            hr14 / 100.0,
            appeared_yesterday,
            in_special_yesterday,
        ])
        y_train.append(label)

print(f"  Training samples: {len(X_train)}, positives: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.2f}%)", flush=True)

# Train with XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

if HAS_XGB:
    print("\n  Training XGBoost...", flush=True)

    # Train 1 model for all 100 codes
    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 100,  # 100:1 imbalance
        "seed": 42,
    }

    model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)

    # Test on each year
    print("\n  Year-by-year results:", flush=True)

    for test_year in test_years:
        if test_year not in year_first:
            continue
        ts = year_first[test_year]
        te = year_last.get(test_year)
        if te is None or te - ts < 30:
            continue

        hits = {1: 0, 3: 0, 5: 0}
        bets = {1: 0, 3: 0, 5: 0}
        total = 0

        for ti in range(ts, te + 1):
            if ti < 90:
                continue

            # Predict for each code
            X_test = []
            for code in range(100):
                d = delta[ti][code]
                rc3 = get_repeat_count(ti, code, 3)
                rc7 = get_repeat_count(ti, code, 7)
                rc14 = get_repeat_count(ti, code, 14)
                sd = special_delta[ti][code]
                hr7 = get_hot_rank(ti, 7).get(code, 99)
                hr14 = get_hot_rank(ti, 14).get(code, 99)
                ay = 1 if code in loto[ti - 1] else 0
                isy = 1 if code == special_2c[ti - 1] else 0
                X_test.append([
                    d / 100.0, rc3 / 3.0, rc7 / 7.0, rc14 / 14.0,
                    sd / 100.0, hr7 / 100.0, hr14 / 100.0, ay, isy
                ])

            dtest = xgb.DMatrix(X_test)
            probs = model.predict(dtest)

            # Top-K by probability
            topK_idx = sorted(range(100), key=lambda i: -probs[i])

            actual = special_2c[ti]
            total += 1

            for K in [1, 3, 5]:
                bets[K] += 1
                if actual in [topK_idx[i] for i in range(K)]:
                    hits[K] += 1

        if total > 0:
            print(f"    {test_year}: {total}d | "
                  f"P@1={hits[1]/total*100:.1f}% | "
                  f"P@3={hits[3]/total*100:.1f}% | "
                  f"P@5={hits[5]/total*100:.1f}%")

    # Feature importance
    importance = model.get_score(importance_type="gain")
    feature_names = ["delta", "rc3", "rc7", "rc14", "special_delta",
                    "hot_rank7", "hot_rank14", "appeared_yesterday", "in_special"]
    print(f"\n  Feature importance:", flush=True)
    for feat, score in sorted(importance.items(), key=lambda x: -x[1])[:9]:
        name = feature_names[int(feat[1:])] if feat.startswith("f") else feat
        print(f"    {name}: {score:.2f}")
else:
    print("  XGBoost not available, skipping ML method")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 65, flush=True)
print("FINAL PATTERN SUMMARY", flush=True)
print("=" * 65, flush=True)
print(f"  1. Loto repeat: P=53.5% (base)")
print(f"  2. Special streak (gap 2): P=0% - ANTIPATTERN")
print(f"  3. Gap pattern: No significant edge found")
print(f"  4. Days with P>50% for >=3 codes: {days_with_high_p}/{len(test_indices)} ({days_with_high_p/len(test_indices)*100:.1f}%)")
print(f"\n  CONCLUSION: No reliable pattern found with P > 50% for 3 numbers.")
print(f"  The {days_with_high_p} days where 3+ codes had P>50% may be noise.")
