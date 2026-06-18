from __future__ import annotations
"""
XSMB Smart Predictor - Ensemble model khong can TensorFlow.

1. MLP Classifier (sklearn) - sequence embedding
2. Markov Chain (n-gram) 
3. Gap-Based (overdue)
4. Frequency (hot numbers)
5. Ensemble

Muc tieu: Du doan Dau/Duoi (10 class) -> 2cang/3cang.
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)

from xsmb_pipeline.dataset import load_csv, sort_results

results = sort_results(load_csv(Path("xsmb_full.csv")))
n = len(results)
print(f"Loaded {n} rows ({results[0].date} -> {results[-1].date})", flush=True)

# ============================================================
# Build digit sequences
# ============================================================
# special = ABCDE (5 chu so)
# 2cang = DE (duoi)
# 3cang = CDE (3 cuoi)
# Dau = C (hang tram)

special_5 = []
for r in results:
    sp = r.special[-5:]
    if len(sp) == 5:
        special_5.append(sp)
    else:
        special_5.append("00000")

# Sequences
# dau_seq[C]: C = special[-5:-4] (hang tram)
dau_seq = [s[0] for s in special_5]
# d1_seq[D]: D = special[-4:-3] (hang chuc)
d1_seq = [s[1] for s in special_5]
# d2_seq[E]: E = special[-3:-2] (hang don vi)
d2_seq = [s[2] for s in special_5]
# d3_seq[DE dau tien]
d3_seq = [s[3] for s in special_5]
# d4_seq[DE cuoi]
d4_seq = [s[4] for s in special_5]
# 2cang dau = special[-4:-2]
c2_d_seq = [s[1] for s in special_5]
# 2cang cuoi = special[-2:]
c2_e_seq = [s[2:4] for s in special_5]

print(f"Last 10 special: {special_5[-10:]}", flush=True)
print(f"Last 10 dau: {dau_seq[-10:]}", flush=True)

# ============================================================
# Precompute: digit last-seen & frequency
# ============================================================
DIGITS = [str(i) for i in range(10)]

def build_ls(seq: list[str]) -> list[dict[str, int]]:
    cache: list[dict[str, int]] = [{} for _ in range(n)]
    for day_idx in range(1, n):
        prev = dict(cache[day_idx - 1])
        prev[seq[day_idx - 1]] = day_idx - 1
        cache[day_idx] = prev
    return cache

def build_freq(window: int, seq: list[str]) -> list[dict[str, int]]:
    cache: list[dict[str, int]] = [{} for _ in range(n)]
    for day_idx in range(window, n):
        c = Counter()
        for i in range(day_idx - window, day_idx):
            c[seq[i]] += 1
        cache[day_idx] = dict(c)
    return cache

def build_bigram(seq: list[str]) -> list[dict[tuple[str, str], int]]:
    """Bigram: {d_prev: {d_curr: count}}"""
    cache: list[dict[tuple[str, str], int]] = [{} for _ in range(n)]
    for day_idx in range(2, n):
        prev = dict(cache[day_idx - 1])
        key = (seq[day_idx - 2], seq[day_idx - 1])
        prev[key] = prev.get(key, 0) + 1
        cache[day_idx] = prev
    return cache

print("Building caches...", flush=True)

ls_dau = build_ls(dau_seq)
ls_d1 = build_ls(d1_seq)
ls_d2 = build_ls(d2_seq)
ls_d3 = build_ls(d3_seq)
ls_d4 = build_ls(d4_seq)

freq_3 = build_freq(3, dau_seq)
freq_7 = build_freq(7, dau_seq)
freq_14 = build_freq(14, dau_seq)
freq_30 = build_freq(30, dau_seq)

freq_d1_3 = build_freq(3, d1_seq)
freq_d1_7 = build_freq(7, d1_seq)
freq_d2_3 = build_freq(3, d2_seq)
freq_d2_7 = build_freq(7, d2_seq)
freq_d3_3 = build_freq(3, d3_seq)
freq_d4_3 = build_freq(3, d4_seq)

bigram_dau = build_bigram(dau_seq)
bigram_d1 = build_bigram(d1_seq)
bigram_d2 = build_bigram(d2_seq)

print("Caches ready.", flush=True)

# ============================================================
# Year indices
# ============================================================
year_first: dict[int, int] = {}
year_last: dict[int, int] = {}
for i, r in enumerate(results):
    try:
        _, _, yr = r.date.rsplit("/", 2)
        yr = int(yr)
        if yr not in year_first:
            year_first[yr] = i
        year_last[yr] = i
    except:
        pass
print(f"Years: {sorted(year_first.keys())}", flush=True)

# ============================================================
# APPROACH 1: MLP Classifier (sklearn)
# ============================================================
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def build_sequence_features(day_idx: int, seq: list[str], seq_len: int = 10) -> list[float]:
    """Build feature vector from last `seq_len` digits."""
    feats = []
    for offset in range(seq_len):
        idx = day_idx - 1 - offset
        if 0 <= idx < day_idx:
            d = int(seq[idx])
            feats.append(float(d))
        else:
            feats.append(5.0)
    # Add: frequency last 3/7/14 days
    f3 = freq_3[day_idx] if day_idx >= 3 else {}
    f7 = freq_7[day_idx] if day_idx >= 7 else {}
    f14 = freq_14[day_idx] if day_idx >= 14 else {}
    f30 = freq_30[day_idx] if day_idx >= 30 else {}
    for d in range(10):
        feats.append(f3.get(str(d), 0) / 3.0)
        feats.append(f7.get(str(d), 0) / 7.0)
        feats.append(f14.get(str(d), 0) / 14.0)
        feats.append(f30.get(str(d), 0) / 30.0)
    return feats

FEATURE_COUNT = 10 + 10 * 4  # seq(10) + freq(40)

def train_mlp(seq: list[str], train_end: int) -> any:
    """Train MLP classifier for digit sequence."""
    if not SKLEARN_AVAILABLE:
        return None

    X, y = [], []
    for day_idx in range(10, train_end + 1):
        feats = build_sequence_features(day_idx, seq)
        X.append(feats)
        y.append(int(seq[day_idx]))

    if len(X) < 100:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_scaled, y)
    return {"mlp": mlp, "scaler": scaler}


def mlp_predict(model: any, seq: list[str], day_idx: int) -> dict[str, float]:
    """Predict next digit using MLP."""
    if model is None:
        return {}

    feats = build_sequence_features(day_idx, seq)
    X_scaled = model["scaler"].transform([feats])
    probs = model["mlp"].predict_proba(X_scaled)[0]
    scores = {}
    for d in range(10):
        scores[str(d)] = float(probs[d])
    return scores


# ============================================================
# APPROACH 2: Bigram (Markov Chain)
# ============================================================
def bigram_scores(day_idx: int, seq: list[str], bigram_cache: list[dict]) -> dict[str, float]:
    """Markov bigram: P(d_next | d_curr)."""
    if day_idx < 1:
        return {d: 0.1 for d in DIGITS}

    prev_digit = seq[day_idx - 1]
    scores = {}
    total = 0
    for next_digit in DIGITS:
        key = (prev_digit, next_digit)
        count = bigram_cache[day_idx].get(key, 0)
        scores[next_digit] = float(count)
        total += count

    if total == 0:
        return {d: 0.1 for d in DIGITS}

    for d in DIGITS:
        scores[d] /= total
    return scores


# ============================================================
# APPROACH 3: Trigram
# ============================================================
def trigram_scores(day_idx: int, seq: list[str]) -> dict[str, float]:
    """Markov trigram: P(d_next | d_prev2, d_prev1)."""
    if day_idx < 2:
        return bigram_scores(day_idx, seq, bigram_cache)

    d_prev2 = seq[day_idx - 2]
    d_prev1 = seq[day_idx - 1]

    # Count transitions from (d_prev2, d_prev1) -> d_next
    counts: Counter[str] = Counter()
    total = 0
    for i in range(max(0, day_idx - 100), day_idx):
        if i >= 1:
            if seq[i - 1] == d_prev2 and seq[i] == d_prev1:
                counts[seq[i + 1]] += 1
                total += 1

    if total == 0:
        return bigram_scores(day_idx, seq, bigram_cache)

    scores = {d: float(counts.get(d, 0)) / total for d in DIGITS}
    return scores

bigram_cache = bigram_dau  # used by bigram_scores fallback


# ============================================================
# APPROACH 4: Gap/Overdue
# ============================================================
def gap_scores(day_idx: int, seq: list[str], ls: list[dict[str, int]]) -> dict[str, float]:
    """Score by gap ratio (overdue indicator)."""
    scores = {}
    cur_day = day_idx
    for d in DIGITS:
        last = ls[day_idx].get(d, -1)
        gap = cur_day - last - 1
        scores[d] = gap / 10.0  # normalize
    return scores


# ============================================================
# APPROACH 5: Frequency
# ============================================================
def freq_scores(day_idx: int, freq_cache: list[dict[str, int]]) -> dict[str, float]:
    """Score by recent frequency."""
    if day_idx < 1:
        return {d: 0.1 for d in DIGITS}
    fc = freq_cache[day_idx]
    total = sum(fc.values()) if fc else 1
    scores = {d: fc.get(d, 0) / max(total, 1) for d in DIGITS}
    return scores


# ============================================================
# APPROACH 6: Combined Ensemble
# ============================================================
def ensemble(
    mlp_s: dict[str, float],
    bigram_s: dict[str, float],
    gap_s: dict[str, float],
    freq_s: dict[str, float],
    trigram_s: dict[str, float],
    weights: tuple[float, float, float, float, float] = (0.35, 0.20, 0.25, 0.10, 0.10),
) -> dict[str, float]:
    w1, w2, w3, w4, w5 = weights
    scores = {}
    for d in DIGITS:
        mlp_v = mlp_s.get(d, 0.05)
        bigram_v = bigram_s.get(d, 0.05)
        gap_v = gap_s.get(d, 0.05)
        freq_v = freq_s.get(d, 0.05)
        trig_v = trigram_s.get(d, 0.05)
        scores[d] = (mlp_v * w1 + bigram_v * w2 + gap_v * w3 + freq_v * w4 + trig_v * w5)
    return scores


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================
print(f"\n{'='*60}", flush=True)
print("WALK-FORWARD BACKTEST: Ensemble vs Individual Models", flush=True)
print(f"{'='*60}", flush=True)

SEQ_LEN = 10
all_by_year: dict[int, dict] = {}

for test_year in range(2019, 2026):
    if test_year not in year_first:
        continue

    train_end = year_last.get(test_year - 1)
    test_start = year_first[test_year]
    test_end = year_last.get(test_year)

    if train_end is None or test_end is None or test_end - test_start < 30:
        continue

    print(f"\n  {test_year}: train_end={train_end}, test={test_start}..{test_end}", flush=True)

    # Train models
    mlp_dau = train_mlp(dau_seq, train_end)
    mlp_d1 = train_mlp(d1_seq, train_end)
    mlp_d2 = train_mlp(d2_seq, train_end)
    mlp_d3 = train_mlp(d3_seq, train_end)
    mlp_d4 = train_mlp(d4_seq, train_end)

    # Bigra, gap, freq caches for dau position
    ls_dau_c = ls_dau
    ls_d1_c = ls_d1
    ls_d2_c = ls_d2
    ls_d3_c = ls_d3
    ls_d4_c = ls_d4
    freq_dau_c = freq_3  # 3-day freq
    freq_d1_c = freq_d1_3
    freq_d2_c = freq_d2_3
    freq_d3_c = freq_d3_3
    freq_d4_c = freq_d4_3
    bigram_dau_c = bigram_dau
    bigram_d1_c = bigram_d1
    bigram_d2_c = bigram_d2

    # Test
    results_year: dict[str, dict] = {}

    for model_name, mlp_model, seq, ls, freq_c, bigram_c in [
        ("dau", mlp_dau, dau_seq, ls_dau_c, freq_dau_c, bigram_dau_c),
        ("d1", mlp_d1, d1_seq, ls_d1_c, freq_d1_c, bigram_d1_c),
        ("d2", mlp_d2, d2_seq, ls_d2_c, freq_d2_c, bigram_d2_c),
        ("d3", mlp_d3, d3_seq, ls_d3_c, freq_d3_c, None),
        ("d4", mlp_d4, d4_seq, ls_d4_c, freq_d4_c, None),
    ]:
        hits_top1 = hits_top3 = total = 0
        for ti in range(test_start, test_end + 1):
            if ti < SEQ_LEN:
                continue

            mlp_s = mlp_predict(mlp_model, seq, ti) if mlp_model else {d: 0.1 for d in DIGITS}
            bg_s = bigram_scores(ti, seq, bigram_c) if bigram_c else {d: 0.1 for d in DIGITS}
            gp_s = gap_scores(ti, seq, ls)
            fr_s = freq_scores(ti, freq_c)
            tr_s = trigram_scores(ti, seq)  # simple fallback

            ens = ensemble(mlp_s, bg_s, gp_s, fr_s, tr_s)
            ranked = sorted(ens, key=lambda x: -ens[x])
            actual = seq[ti]

            total += 1
            if actual == ranked[0]:
                hits_top1 += 1
            if actual in ranked[:3]:
                hits_top3 += 1

        p1 = hits_top1 / total * 100 if total > 0 else 0
        p3 = hits_top3 / total * 100 if total > 0 else 0
        results_year[model_name] = {"p1": p1, "p3": p3, "hits": hits_top1, "n": total}

    # 2cang (d3+d4)
    hits_2cang_top1 = hits_2cang_top5 = total_2c = 0
    for ti in range(test_start, test_end + 1):
        if ti < SEQ_LEN:
            continue

        # D3
        for model_name, mlp_model, seq, ls, freq_c, bigram_c in [
            ("d3", mlp_d3, d3_seq, ls_d3_c, freq_d3_c, None),
            ("d4", mlp_d4, d4_seq, ls_d4_c, freq_d4_c, None),
        ]:
            mlp_s = mlp_predict(mlp_model, seq, ti) if mlp_model else {d: 0.1 for d in DIGITS}
            bg_s = bigram_scores(ti, seq, bigram_c) if bigram_c else {d: 0.1 for d in DIGITS}
            gp_s = gap_scores(ti, seq, ls)
            fr_s = freq_scores(ti, freq_c)
            tr_s = trigram_scores(ti, seq)
            ens_d3 = ensemble(mlp_s, bg_s, gp_s, fr_s, tr_s)
            ens_d4 = {d: 0.1 for d in DIGITS} if mlp_model is None else {}

            # For D4, use separate model
            mlp_s4 = mlp_predict(mlp_d4, d4_seq, ti) if mlp_d4 else {d: 0.1 for d in DIGITS}
            bg_s4 = bigram_scores(ti, d4_seq, bigram_dau_c) if bigram_dau_c else {d: 0.1 for d in DIGITS}
            gp_s4 = gap_scores(ti, d4_seq, ls_d4_c)
            fr_s4 = freq_scores(ti, freq_d4_c)
            tr_s4 = trigram_scores(ti, d4_seq)
            ens_d4 = ensemble(mlp_s4, bg_s4, gp_s4, fr_s4, tr_s4)

        # Rank 2cang by product of D3 and D4 scores
        c2_scores = {}
        for d3 in DIGITS:
            for d4 in DIGITS:
                c2 = d3 + d4
                c2_scores[c2] = ens_d3.get(d3, 0.05) * ens_d4.get(d4, 0.05)

        ranked_2c = sorted(c2_scores, key=lambda x: -c2_scores[x])
        actual_2c = d3_seq[ti] + d4_seq[ti]
        total_2c += 1
        if actual_2c == ranked_2c[0]:
            hits_2cang_top1 += 1
        if actual_2c in ranked_2c[:5]:
            hits_2cang_top5 += 1

    p2c_1 = hits_2cang_top1 / total_2c * 100 if total_2c > 0 else 0
    p2c_5 = hits_2cang_top5 / total_2c * 100 if total_2c > 0 else 0

    # Print year results
    print(f"    Dau(D0):  P@1={results_year['dau']['p1']:.1f}%  P@3={results_year['dau']['p3']:.1f}% ({results_year['dau']['hits']}/{results_year['dau']['n']})", flush=True)
    print(f"    D1:       P@1={results_year['d1']['p1']:.1f}%  P@3={results_year['d1']['p3']:.1f}%", flush=True)
    print(f"    D2:       P@1={results_year['d2']['p1']:.1f}%  P@3={results_year['d2']['p3']:.1f}%", flush=True)
    print(f"    D3:       P@1={results_year['d3']['p1']:.1f}%  P@3={results_year['d3']['p3']:.1f}%", flush=True)
    print(f"    D4:       P@1={results_year['d4']['p1']:.1f}%  P@3={results_year['d4']['p3']:.1f}%", flush=True)
    print(f"    2cang:    P@1={p2c_1:.2f}%  P@5={p2c_5:.1f}%", flush=True)

    all_by_year[test_year] = {**results_year, "2cang": {"p1": p2c_1, "p5": p2c_5, "hits": hits_2cang_top1, "n": total_2c}}

# Overall summary
print(f"\n{'='*60}", flush=True)
print("OVERALL SUMMARY (2019-2025)", flush=True)

total_n = sum(r["dau"]["n"] for r in all_by_year.values())

for pos in ["dau", "d1", "d2", "d3", "d4"]:
    total_hits = sum(r[pos]["hits"] for r in all_by_year.values())
    total_n_pos = sum(r[pos]["n"] for r in all_by_year.values())
    p1 = total_hits / total_n_pos * 100 if total_n_pos > 0 else 0
    rand = 10.0
    diff = p1 - rand
    print(f"  {pos:<5} P@1={p1:5.1f}%  (vs random {rand:.0f}% | {'+' if diff > 0 else ''}{diff:+.1f}pp)", flush=True)

total_2c = sum(r["2cang"]["n"] for r in all_by_year.values())
hits_2c = sum(r["2cang"]["hits"] for r in all_by_year.values())
p2c = hits_2c / total_2c * 100 if total_2c > 0 else 0
rand2c = 1.0
diff2c = p2c - rand2c
print(f"  {'2cang':<5} P@1={p2c:5.2f}%  (vs random {rand2c:.0f}% | {'+' if diff2c > 0 else ''}{diff2c:+.2f}pp)", flush=True)

print(f"\nRandom baselines: dau=10.0%, 2cang=1.0%, 3cang=0.1%", flush=True)

# Save
output = {
    "model": "XSMB_Ensemble_MLP",
    "description": "MLP + Bigram + Gap + Freq ensemble for digit prediction",
    "by_year": {str(k): v for k, v in all_by_year.items()},
}
out_path = Path("xsmb_pipeline/ensemble_backtest_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved to {out_path}", flush=True)
