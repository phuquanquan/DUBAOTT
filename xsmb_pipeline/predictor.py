from __future__ import annotations
"""
XSMB Predictor - Chi 2 phuong phap tap trung.

Phuong phap 1: LSTM cho Dau/Duoi (10-class per digit)
  - Hoc chuoi digit (10-class)
  - Dung: tan so xuat hien cua cac giai + last-seen
  - Bo cuc nhu 3cang nhung don gian hon

Phuong phap 2: Sparse Delta cho Loto (gap-based)
  - Dua tren sparse matrix (27 giai/ngay) nhu khiemdoan
  - Gap ratio + delta = muc do "qua han" cua tung so
  - Dung cho danh lo 2 so (loto)

KHONG dung: ensemble nhieu signal, regression 1000-class, pattern rules phuc tap.
"""

import json
from pathlib import Path
from collections import Counter

from xsmb_pipeline.dataset import load_csv, sort_results


# ============================================================
# Config
# ============================================================
DATA_PATH = Path("xsmb_full.csv")
DIGITS = [str(i) for i in range(10)]
SEQ_LEN = 10


# ============================================================
# Data Loader - nhu khiemdoan: sparse matrix + delta
# ============================================================
class DataLoader:
    """Sparse matrix + delta cache nhu khiemdoan."""

    def __init__(self, csv_path: Path) -> None:
        self.results = sort_results(load_csv(csv_path))
        self.n = len(self.results)
        self._build()

    def _build(self) -> None:
        n = self.n
        results = self.results

        # DB digits
        self.db: list[list[str]] = []
        for r in results:
            sp = r.special[-5:]
            self.db.append(list(sp) if len(sp) == 5 else ["0"] * 5)

        # Sparse matrix: all 2-digit from ALL 27 prizes per day
        self.sparse: list[list[int]] = []
        for r in results:
            c: Counter[str] = Counter()
            for plist in [r.first, r.second, r.third, r.fourth,
                          r.fifth, r.sixth, r.seventh]:
                for p in plist:
                    s = str(p)
                    for j in range(len(s) - 1):
                        c[s[j:j + 2]] += 1
            row = [0] * 100
            for num, cnt in c.items():
                if len(num) == 2:
                    row[int(num)] = cnt
            self.sparse.append(row)

        # Delta: days since last seen for each 2cang
        self.delta: list[list[int]] = [[0] * 100 for _ in range(n)]
        last_seen = [-1] * 100
        for i in range(n):
            for code in range(100):
                self.delta[i][code] = i - last_seen[code] - 1 if last_seen[code] >= 0 else i + 1
            # Update: special[-2:]
            code = int(self.db[i][3] + self.db[i][4])
            last_seen[code] = i

        # Digit last-seen per position
        self.ls: list[list[dict[str, int]]] = [[{} for _ in range(5)] for _ in range(n)]
        for i in range(1, n):
            for pos in range(5):
                prev = dict(self.ls[i - 1][pos])
                prev[self.db[i - 1][pos]] = i - 1
                self.ls[i][pos] = prev

        # Digit frequency per position
        self.freq: dict[int, list[dict[str, int]]] = {}
        for window in [3, 7, 30]:
            cache: list[dict[str, int]] = [{} for _ in range(n)]
            for i in range(window, n):
                c: Counter[str] = Counter()
                for j in range(i - window, i):
                    for p in range(5):
                        c[self.db[j][p]] += 1
                cache[i] = dict(c)
            self.freq[window] = cache

        # Loto frequency (all prizes)
        self.loto_freq: dict[int, list[dict[str, int]]] = {}
        for window in [3, 7, 14, 30]:
            cache: list[dict[str, int]] = [{} for _ in range(n)]
            for i in range(window, n):
                c: Counter[str] = Counter()
                for j in range(i - window, i):
                    for code, cnt in enumerate(self.sparse[j]):
                        for _ in range(cnt):
                            c[f"{code:02d}"] += 1
                cache[i] = dict(c)
            self.loto_freq[window] = cache

    def summary(self) -> dict:
        """Nhu khiemdoan: top overdue, top hot loto."""
        last_idx = self.n - 1
        deltas = self.delta[last_idx]
        top_overdue = sorted(range(100), key=lambda x: -deltas[x])[:10]
        loto_f = self.loto_freq[3][last_idx]
        top_loto = sorted(loto_f, key=lambda x: -loto_f[x])[:10] if loto_f else []
        return {
            "date": self.results[last_idx].date,
            "special": self.results[last_idx].special,
            "overdue_2cang": [f"{x:02d}" for x in top_overdue],
            "overdue_deltas": [deltas[x] for x in top_overdue],
            "hot_loto": top_loto,
            "hot_loto_freq": [loto_f.get(x, 0) for x in top_loto] if loto_f else [],
        }


# ============================================================
# PHUONG PHAP 1: LSTM Sequence Model (10-class per digit)
# Dua tren 3cang nhung don gian hon
# ============================================================
def build_lstm_features(seq: list[str], day_idx: int,
                        freq_cache: dict[int, list[dict[str, int]]],
                        seq_len: int = 10) -> list[float]:
    """Features cho LSTM: chi chuoi digit + freq, khong nhieu hon."""
    feats = []
    # Last seq_len digits
    for offset in range(seq_len):
        idx = day_idx - seq_len + offset
        feats.append(float(seq[idx]) / 9.0 if 0 <= idx < day_idx else 0.5)

    # Digit frequency (3 windows)
    for window in [3, 7, 14]:
        fc = freq_cache.get(window, [{} for _ in range(len(seq))])
        f = fc[day_idx] if day_idx < len(fc) else {}
        total = sum(f.values()) if f else 1
        for d in range(10):
            feats.append(f.get(str(d), 0) / total)

    return feats


def train_lstm(seq: list[str], train_end: int,
               freq_cache: dict[int, list[dict[str, int]]],
               seq_len: int = 10):
    """Train MLP nhu 3cang: 10-class classifier."""
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    X, y = [], []
    for day_idx in range(seq_len, train_end + 1):
        feats = build_lstm_features(seq, day_idx, freq_cache, seq_len)
        X.append(feats)
        y.append(int(seq[day_idx]))

    if len(X) < 50:
        return None

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_s, y)
    return {"mlp": mlp, "scaler": scaler}


def lstm_predict(model: dict | None, seq: list[str], day_idx: int,
                 freq_cache: dict[int, list[dict[str, int]]],
                 seq_len: int = 10) -> dict[str, float]:
    """Tra ve xac suat cho 10 digits, co temperature scaling nhu 3cang."""
    if model is None:
        return {}

    feats = build_lstm_features(seq, day_idx, freq_cache, seq_len)
    X_s = model["scaler"].transform([feats])
    probs = model["mlp"].predict_proba(X_s)[0]

    # Temperature scaling nhu 3cang
    temperature = 1.5
    import numpy as np
    probs = np.log(probs + 1e-10) / temperature
    probs = np.exp(probs) / np.sum(np.exp(probs))

    scores = {}
    for d in range(10):
        scores[str(d)] = float(probs[d])
    return scores


# ============================================================
# PHUONG PHAP 2: Sparse Delta cho Loto
# Dua tren sparse matrix cua khiemdoan
# ============================================================
def loto_scores(day_idx: int, delta_cache: list[list[int]],
                loto_freq_cache: list[dict[str, int]],
                window: int = 7) -> dict[str, float]:
    """
    Gap ratio + hot frequency cho loto 2 so.
    Khong dung ensemble - chi 1 phuong phap.
    """
    deltas = delta_cache[day_idx]
    freq = loto_freq_cache[day_idx]
    total_freq = sum(freq.values()) if freq else 1

    scores = {}
    for code in range(100):
        s = f"{code:02d}"
        # Gap ratio: delta / expected_gap(27) ~ 2.7% per day
        gap_ratio = min(deltas[code] / 27.0, 3.0)  # cap at 3x
        # Frequency weight
        freq_weight = freq.get(s, 0) / total_freq if total_freq > 0 else 0
        # Chi gap ratio (don phuong phap)
        scores[s] = gap_ratio

    return scores


# ============================================================
# Main Predictor - chi 2 phuong phap
# ============================================================
class XSMBPredictor:
    """
    Su dung:
        p = XSMBPredictor()
        p.load_data()
        p.train()
        r = p.predict()
        print(r)
    """

    def __init__(self) -> None:
        self.data: DataLoader | None = None
        self.lstm_models: dict[str, dict | None] = {}

    def load_data(self, path: Path | None = None) -> "XSMBPredictor":
        p = path or DATA_PATH
        self.data = DataLoader(p)
        print(f"Loaded {self.data.n} days ({self.data.results[0].date} -> {self.data.results[-1].date})")
        return self

    def train(self, train_end: int | None = None) -> "XSMBPredictor":
        """Train LSTM cho tung vi tri digit (chi Dau va Duoi)."""
        if self.data is None:
            raise RuntimeError("Call load_data() first")

        end = train_end if train_end is not None else self.data.n - 1

        # Chi 2 model: Dau (D0) va Duoi (D3, D4)
        seqs = {
            "dau": [d[0] for d in self.data.db],
            "duoi_d3": [d[3] for d in self.data.db],
            "duoi_d4": [d[4] for d in self.data.db],
        }

        for name, seq in seqs.items():
            print(f"  Training {name}...", end=" ", flush=True)
            self.lstm_models[name] = train_lstm(seq, end, self.data.freq)
            print("OK" if self.lstm_models[name] else "SKIP")

        return self

    def predict(self, day_idx: int | None = None,
                k_digit: int = 5, k_loto: int = 5) -> dict:
        """
        Du doan cho ngay cuoi cung.

        Returns:
            dau: TOP-K digit (0-9) voi LSTM score
            loto: TOP-K 2 so (00-99) voi delta score
        """
        if self.data is None:
            raise RuntimeError("Call load_data() first")

        if day_idx is None:
            day_idx = self.data.n - 1

        freq = self.data.freq

        # === DAU: LSTM score ===
        seq_dau = [d[0] for d in self.data.db]
        lstm_dau = lstm_predict(
            self.lstm_models.get("dau"), seq_dau, day_idx, freq)
        ranked_dau = sorted(lstm_dau.items(), key=lambda x: -x[1])[:k_digit]

        # === DUOI: LSTM cho D3 va D4 rieng biet ===
        seq_d3 = [d[3] for d in self.data.db]
        seq_d4 = [d[4] for d in self.data.db]
        lstm_d3 = lstm_predict(
            self.lstm_models.get("duoi_d3"), seq_d3, day_idx, freq)
        lstm_d4 = lstm_predict(
            self.lstm_models.get("duoi_d4"), seq_d4, day_idx, freq)

        # Product of D3 and D4 scores
        duoi_scores: dict[str, float] = {}
        for d3, s3 in lstm_d3.items():
            for d4, s4 in lstm_d4.items():
                code = d3 + d4
                duoi_scores[code] = s3 * s4
        ranked_duoi = sorted(duoi_scores.items(), key=lambda x: -x[1])[:k_loto]

        # === LOTO: Sparse Delta (khong LSTM) ===
        loto = loto_scores(
            day_idx,
            self.data.delta,
            self.data.loto_freq[7],
            window=7
        )
        ranked_loto = sorted(loto.items(), key=lambda x: -x[1])[:k_loto]

        # === SUMMARY ===
        summary = self.data.summary()

        dau_list = [{"digit": d, "score": round(s, 4)} for d, s in ranked_dau]
        duoi_list = [{"number": n, "score": round(s, 4)} for n, s in ranked_duoi]
        loto_list = [{"number": n, "score": round(s, 2)} for n, s in ranked_loto]

        return {
            "date": self.data.results[day_idx].date,
            "previous_special": self.data.results[day_idx].special,
            "dau_lstm": dau_list,
            "duoi_lstm": duoi_list,
            "loto_delta": loto_list,
            "summary": summary,
        }

    def backtest(self, test_start: int, test_end: int,
                 k_digit: int = 5, k_loto: int = 5) -> dict:
        """
        Backtest chi 2 phuong phap, tinh P@K + ROI.
        """
        if self.data is None:
            raise RuntimeError("Call load_data() first")

        # Re-train on data before test_start
        train_end = test_start - 1
        self.train(train_end)

        freq = self.data.freq

        seq_dau = [d[0] for d in self.data.db]
        seq_d3 = [d[3] for d in self.data.db]
        seq_d4 = [d[4] for d in self.data.db]

        dau_hits = {K: 0 for K in [1, 3, 5]}
        duoi_hits = {K: 0 for K in [1, 3, 5]}
        loto_hits = {K: 0 for K in [1, 3, 5]}
        total = 0

        for ti in range(test_start, min(test_end + 1, self.data.n - 1)):
            if ti < SEQ_LEN:
                continue
            total += 1

            # DAU LSTM
            lstm_dau = lstm_predict(self.lstm_models.get("dau"), seq_dau, ti, freq)
            ranked_dau = sorted(lstm_dau, key=lambda x: -lstm_dau[x])
            actual_dau = seq_dau[ti]
            for K in [1, 3, 5]:
                if actual_dau in ranked_dau[:K]:
                    dau_hits[K] += 1

            # DUOI LSTM
            lstm_d3 = lstm_predict(self.lstm_models.get("duoi_d3"), seq_d3, ti, freq)
            lstm_d4 = lstm_predict(self.lstm_models.get("duoi_d4"), seq_d4, ti, freq)
            duoi_scores: dict[str, float] = {}
            for d3, s3 in lstm_d3.items():
                for d4, s4 in lstm_d4.items():
                    duoi_scores[d3 + d4] = s3 * s4
            ranked_duoi = sorted(duoi_scores, key=lambda x: -duoi_scores[x])
            actual_duoi = seq_d3[ti] + seq_d4[ti]
            for K in [1, 3, 5]:
                if actual_duoi in ranked_duoi[:K]:
                    duoi_hits[K] += 1

            # LOTO Delta
            loto = loto_scores(ti, self.data.delta, self.data.loto_freq[7], window=7)
            ranked_loto = sorted(loto, key=lambda x: -loto[x])
            actual_loto = seq_d3[ti] + seq_d4[ti]
            for K in [1, 3, 5]:
                if actual_loto in ranked_loto[:K]:
                    loto_hits[K] += 1

        # Results
        results = {}
        for name, hits in [("dau_lstm", dau_hits), ("duoi_lstm", duoi_hits), ("loto_delta", loto_hits)]:
            results[name] = {}
            for K in [1, 3, 5]:
                p = hits[K] / total * 100 if total > 0 else 0
                rand = {"dau_lstm": 10, "duoi_lstm": 1, "loto_delta": 1}[name] * K
                results[name][K] = {"hits": hits[K], "total": total, "p": round(p, 2), "vs_random": round(p - rand, 2)}

        return results


if __name__ == "__main__":
    print("=" * 60)
    print("XSMB Predictor - 2 Phuong Phap Tap Trung")
    print("=" * 60)

    p = XSMBPredictor()
    p.load_data()
    p.train()

    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)
    r = p.predict()
    print(f"\nDate: {r['date']}")
    print(f"Previous Special: {r['previous_special']}")
    print(f"\n[1] DAU (LSTM): {[x['digit'] for x in r['dau_lstm']]}")
    print(f"    Scores: {[round(x['score'], 4) for x in r['dau_lstm']]}")
    print(f"\n[2] DUOI (LSTM): {[x['number'] for x in r['duoi_lstm']]}")
    print(f"    Scores: {[round(x['score'], 4) for x in r['duoi_lstm']]}")
    print(f"\n[3] LOTO (Delta): {[x['number'] for x in r['loto_delta']]}")
    print(f"    Scores: {[round(x['score'], 2) for x in r['loto_delta']]}")
    print(f"\nSummary (khiemdoan):")
    s = r["summary"]
    print(f"  Overdue 2cang: {s['overdue_2cang']}")
    print(f"  Overdue deltas: {s['overdue_deltas']}")
    print(f"  Hot loto: {s['hot_loto']}")
