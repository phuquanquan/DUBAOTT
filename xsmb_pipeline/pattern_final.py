from __future__ import annotations
"""
XSMB Max Edge - Final with Best Config + Year-by-Year + Stats
"""

import sys
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
try:
    import xgboost as xgb
except ImportError:
    sys.exit(1)

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

loto_27_set: list[set[int]] = []
loto_27_list: list[list[int]] = []
for r in results:
    codes = get_27_loto(r)
    loto_27_list.append(codes)
    loto_27_set.append(set(codes))

avg_p = sum(len(s) for s in loto_27_set) / n / 100.0
print(f"Data: {n} days | Avg P(code) = {avg_p*100:.2f}%")
print(f"Baseline P(at least 1 of 3) = {(1-(1-avg_p)**3)*100:.2f}%")

# Precompute
prize_cnt: list[dict[int, int]] = []
for i in range(n):
    cc = Counter()
    for code in loto_27_list[i]:
        cc[code] += 1
    prize_cnt.append(dict(cc))

delta = [[0] * 100 for _ in range(n)]
last_seen = [-1] * 100
for i in range(n):
    for code in range(100):
        delta[i][code] = i - last_seen[code] - 1 if last_seen[code] >= 0 else i + 1
    for code in loto_27_set[i]:
        last_seen[code] = i

in_loto_yesterday = [[0] * 100 for _ in range(n)]
for i in range(1, n):
    for code in loto_27_set[i - 1]:
        in_loto_yesterday[i][code] = 1

prize_cnt_yest = [{}] + prize_cnt[:-1]

WINDOWS = [3, 5, 7, 14, 30, 60, 90]
appear: dict[int, list[list[int]]] = {}
for w in WINDOWS:
    appear[w] = [[0] * 100 for _ in range(n)]
    running = Counter()
    for i in range(n):
        if i >= w:
            for code in loto_27_set[i - w]:
                running[code] -= 1
        for code in loto_27_set[i]:
            running[code] += 1
        for code in range(100):
            appear[w][i][code] = running.get(code, 0)

ranks: dict[int, list[list[int]]] = {}
for w in [3, 7, 14]:
    ranks[w] = [[99] * 100 for _ in range(n)]
    for i in range(n):
        freq = Counter()
        for d in range(max(0, i - w + 1), i + 1):
            freq.update(loto_27_set[d])
        if not freq:
            continue
        sorted_codes = sorted(freq, key=freq.get, reverse=True)
        for code in freq:
            ranks[w][i][code] = sorted_codes.index(code)

streak = [[0] * 100 for _ in range(n)]
for i in range(n):
    for code in range(100):
        s = 0
        for d in range(i - 1, -1, -1):
            if code in loto_27_set[d]:
                s += 1
            else:
                break
        streak[i][code] = s

prize_rank = [[99] * 100 for _ in range(n)]
for i in range(n):
    if prize_cnt[i]:
        sorted_codes = sorted(prize_cnt[i], key=prize_cnt[i].get, reverse=True)
        for ri, code in enumerate(sorted_codes):
            prize_rank[i][code] = ri

NF = 20
feat: np.ndarray = np.zeros((n, 100, NF), dtype=np.float32)
for i in range(n):
    for code in range(100):
        a3 = appear[3][i][code] / 3.0
        a5 = appear[5][i][code] / 5.0
        a7 = appear[7][i][code] / 7.0
        a14 = appear[14][i][code] / 14.0
        a30 = appear[30][i][code] / 30.0
        a90 = appear[90][i][code] / 90.0
        dt = delta[i][code] / 100.0
        dt_inv = 1.0 / (delta[i][code] + 1)
        sk = min(streak[i][code], 10) / 10.0
        iy = in_loto_yesterday[i][code]
        pc = prize_cnt_yest[i].get(code, 0) / 4.0
        pr = min(prize_rank[i][code], 27) / 27.0
        f_ratio = appear[7][i][code] / max(appear[14][i][code], 1) if appear[14][i][code] > 0 else a7
        freq_mean = a90
        dev = (a7 - freq_mean) / max(freq_mean, 0.01)
        r3 = ranks[3][i][code] / 100.0
        r7 = ranks[7][i][code] / 100.0
        r14 = ranks[14][i][code] / 100.0
        feat[i, code] = [
            dt, a3, a5, a7, a14, a30, a90,
            r3, r7, r14,
            sk, iy, pc, pr,
            f_ratio, dt_inv, dev,
            min(appear[5][i][code], 3) / 3.0,
            appear[60][i][code] / 60.0,
            dev * iy,
        ]

FEAT_NAMES = [
    "delta","a3d","a5d","a7d","a14d","a30d","a90d",
    "r3d","r7d","r14d",
    "streak","in_yest","prize_cnt","prize_rank",
    "f_ratio","inv_delta","dev_mean","a5_cap","a60d","dev_x_iy"
]

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

train_end = year_first.get(2019, 100) - 1

X_train_list = []
y_train_list = []
for ti in range(30, train_end + 1):
    ref = ti - 1
    for code in range(100):
        X_train_list.append(feat[ref, code])
    labels = [1 if c in loto_27_set[ti] else 0 for c in range(100)]
    y_train_list.extend(labels)

X_train_arr = np.array(X_train_list, dtype=np.float32)
y_train_arr = np.array(y_train_list, dtype=np.float32)
print(f"Train: {len(X_train_arr)} samples, pos={int(y_train_arr.sum())}")

dtrain = xgb.DMatrix(X_train_arr, label=y_train_arr)
neg = (1 - y_train_arr).sum()
pos = y_train_arr.sum()
base_spw = float(neg / max(pos, 1))

# BEST CONFIG from search
best_params = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": base_spw,
    "seed": 42,
    "verbosity": 0,
    "tree_method": "hist",
}
best_model = xgb.train(best_params, dtrain, num_boost_round=200, verbose_eval=False)

print("\n" + "=" * 70)
print("YEAR-BY-YEAR (BEST MODEL: d=3, eta=0.05, rounds=200)")
print("=" * 70)

all_hits = {K: 0 for K in [1, 3, 5, 10, 20]}
all_total = 0

for test_year in range(2019, 2026):
    if test_year not in year_first:
        continue
    ts = year_first[test_year]
    te = year_last.get(test_year, ts)
    yh = {K: 0 for K in [1, 3, 5, 10, 20]}
    yt = 0

    for ti in range(ts, te + 1):
        if ti < 31 or ti >= n - 1:
            continue
        ref = ti - 1
        X_day = feat[ref]
        dtest_day = xgb.DMatrix(X_day)
        probs = best_model.predict(dtest_day)
        ranked = np.argsort(-probs)
        actual_set = loto_27_set[ti]
        yt += 1
        all_total += 1

        for K in [1, 3, 5, 10, 20]:
            if any(int(c) in actual_set for c in ranked[:K]):
                yh[K] += 1
                all_hits[K] += 1

    print(f"  {test_year} ({yt:3d}d): ", end="")
    for K in [1, 3, 5, 10, 20]:
        p_obs = yh[K] / yt * 100
        p_exp = (1 - (1 - avg_p) ** K) * 100
        diff = p_obs - p_exp
        print(f"P@{K}={p_obs:.1f}%({diff:+.1f}pp) ", end="")
    print()

print(f"\n  {'OVERALL':>6} ({all_total:4d}d): ", end="")
for K in [1, 3, 5, 10, 20]:
    p_obs = all_hits[K] / all_total * 100
    p_exp = (1 - (1 - avg_p) ** K) * 100
    diff = p_obs - p_exp
    print(f"P@{K}={p_obs:.1f}%({diff:+.1f}pp) ", end="")
print()

print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE")
print("=" * 70)

def normal_pvalue(z):
    if abs(z) > 39:
        return 1e-20 if z > 0 else 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = (t * 0.319381530 + t*t*(-0.356563782) +
            t**3*1.781477937 + t**4*(-1.821255978) + t**5*1.330274429)
    pval = poly * 0.3989422804 * 2
    if z < 0:
        pval = 2 - pval
    return max(0.0, min(1.0, pval))

for K in [1, 3, 5, 10, 20]:
    p_obs = all_hits[K] / all_total
    p_exp = 1 - (1 - avg_p) ** K
    se = np.sqrt(p_exp * (1 - p_exp) / all_total)
    z = (p_obs - p_exp) / max(se, 1e-6)
    pval = normal_pvalue(z)
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    diff_pp = (p_obs - p_exp) * 100
    print(f"  P@{K}: {p_obs*100:.2f}% vs {p_exp*100:.2f}% | "
          f"diff={diff_pp:+.2f}pp | z={z:.2f} | p={pval:.4f} {sig}")

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)
imp = best_model.get_score(importance_type="gain")
for idx, (f, s) in enumerate(sorted(imp.items(), key=lambda x: -x[1])[:15]):
    fi = int(f[1:])
    print(f"  {idx+1:2d}. {FEAT_NAMES[fi]:20s}: {s:.1f}")
