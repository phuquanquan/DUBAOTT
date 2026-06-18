from __future__ import annotations
"""
Algorithm 01 - Junlangzi Scoring Rules System.

Ported tu: https://github.com/junlangzi/Lottery-Predictor
Data: xsmb-2-digits.json (khiemdoan)

Tac dung: Tinh diem 00-99 dua tren 20+ scoring rules.
Co 2 bo parameters:
  1. Default: cac gia tri mac dinh
  2. Optimized: da duoc tune grid search

Target: Loto 2 so (special[-2:])
"""

import datetime
from collections import Counter


# ============================================================
# Default parameters (thuat_toan_01.py)
# ============================================================
DEFAULT_PARAMS = {
    "short_term_days": 14,
    "frequency_window_short": 45,
    "frequency_window_long": 180,
    "base_point_short": 3.5,
    "base_point_long": 0.3,
    "frequency_weight_short": 0.5,
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
    "special_multiplier": 2.0,
    "special_freq_multiplier": 3.0,
}

# ============================================================
# Optimized parameters (optimized_thuat_toan_01)
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


# ============================================================
# Algorithm Class
# ============================================================
class HistoryAppearancePointAlgorithm:
    """
    Scoring-based algorithm. Khong dung ML.
    Tinh diem cho 00-99 dua tren 20+ scoring rules.

    Su dung:
        algo = HistoryAppearancePointAlgorithm(params=OPTIMIZED_PARAMS)
        scores = algo.predict(target_date, history_list)
        top_k = algo.top_k(scores, k=5)
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or DEFAULT_PARAMS.copy()
        self.previous_tops: list[list[int]] = []

    def extract_numbers(self, result_dict: dict) -> set[int]:
        """Trich xuat tat ca so 2 chu so cuoi tu result dict."""
        numbers: set[int] = set()
        if not isinstance(result_dict, dict):
            return numbers
        for key, value in result_dict.items():
            if key == "date":
                continue
            try:
                s = str(value).strip()
                if len(s) >= 2:
                    num = int(s[-2:])
                else:
                    num = int(s)
                if 0 <= num <= 99:
                    numbers.add(num)
            except (ValueError, TypeError):
                continue
        return numbers

    def get_special_2cang(self, result_dict: dict) -> int:
        """Lay special[-2:] tu result dict."""
        try:
            sp = str(result_dict.get("special", "00"))
            return int(sp[-2:])
        except (ValueError, TypeError):
            return 0

    def predict(self, date_to_predict: datetime.date,
                historical_results: list[dict]) -> dict[str, float]:
        """
        Tinh diem cho 00-99.

        Args:
            date_to_predict: Ngay can du doan
            historical_results: List of {"date": date, "result": dict}
                               CHI chua data TRUOC ngay can du doan.

        Returns:
            dict "00".."99" -> float score
        """
        p = self.params

        scores: dict[str, float] = {f"{i:02d}": 0.0 for i in range(100)}
        if not historical_results:
            return scores

        # Filter + sort
        history = sorted(
            [r for r in historical_results
             if isinstance(r.get("date"), datetime.date)
             and r["date"] < date_to_predict],
            key=lambda x: x["date"]
        )
        if not history:
            return scores

        # --- Short term points ---
        short_term_start = date_to_predict - datetime.timedelta(days=p["short_term_days"])
        short_term = [h for h in history if h["date"] >= short_term_start]

        for day_data in reversed(short_term):
            days_ago = (date_to_predict - day_data["date"]).days
            numbers = self.extract_numbers(day_data["result"])
            special_num = self.get_special_2cang(day_data["result"])
            for num in numbers:
                point = p["base_point_short"] * (1 - 0.08 * days_ago)
                if num == special_num:
                    point *= p["special_multiplier"]
                scores[f"{num:02d}"] += point

        # --- Frequency ---
        freq_short_start = date_to_predict - datetime.timedelta(days=p["frequency_window_short"])
        freq_long_start = date_to_predict - datetime.timedelta(days=p["frequency_window_long"])

        freq_short: Counter[int] = Counter()
        freq_long: Counter[int] = Counter()
        special_freq_short: Counter[int] = Counter()

        for day_data in history:
            numbers = self.extract_numbers(day_data["result"])
            special_num = self.get_special_2cang(day_data["result"])
            if day_data["date"] >= freq_short_start:
                freq_short.update(numbers)
                special_freq_short[special_num] += 1
            if day_data["date"] >= freq_long_start:
                freq_long.update(numbers)

        for num in range(100):
            num_str = f"{num:02d}"
            scores[num_str] += p["frequency_weight_short"] * freq_short[num]
            scores[num_str] += p["frequency_weight_long"] * freq_long[num]
            if freq_short[num] > 5:
                scores[num_str] += p["bonus_freq_5"]
            if special_freq_short[num] > 5:
                scores[num_str] += p["special_freq_multiplier"]

        # --- Neighbor bonus ---
        if history:
            special_last = self.get_special_2cang(history[-1]["result"])
            for offset in range(-p["neighbor_range"], p["neighbor_range"] + 1):
                if offset == 0:
                    continue
                neighbor = (special_last + offset) % 100
                scores[f"{neighbor:02d}"] += p["neighbor_bonus"] * (1 - 0.1 * abs(offset))

        # --- Cycle bonus ---
        for day_data in history:
            days_ago = (date_to_predict - day_data["date"]).days
            if days_ago % 7 == 0 and days_ago <= 28:
                numbers = self.extract_numbers(day_data["result"])
                for num in numbers:
                    scores[f"{num:02d}"] += p["cycle_7_bonus"]
            if days_ago % 30 == 0 and days_ago <= 180:
                numbers = self.extract_numbers(day_data["result"])
                for num in numbers:
                    scores[f"{num:02d}"] += p["cycle_30_bonus"]

        # --- Long absence bonus ---
        for num in range(100):
            num_str = f"{num:02d}"
            days_since = 0
            for day_data in reversed(history):
                if num in self.extract_numbers(day_data["result"]):
                    break
                days_since += 1
            scores[num_str] += days_since * p["increment"]
            if days_since >= 3:
                scores[num_str] += p["bonus_after_3_days"]
            if days_since >= 15 and freq_long[num] > 10:
                scores[num_str] += p["bonus_long_absence"]

        # --- Deduction if appeared last day ---
        if history and (date_to_predict - history[-1]["date"]).days == 1:
            last_numbers = self.extract_numbers(history[-1]["result"])
            for num in last_numbers:
                scores[f"{num:02d}"] += p["deduction_if_appeared_last_day"]

        # --- Repeat penalty in Top 3 ---
        repeat_counts: Counter[int] = Counter()
        streak_counts: Counter[int] = Counter()
        # --- Repeat penalty in Top 3 ---
        repeat_counts: Counter[int] = Counter()
        # streak_counts: consecutive days in top 3
        streak_counts: dict[int, int] = {}
        prev_tops_slice = self.previous_tops[-p["repeat_window"]:]
        for i in range(len(prev_tops_slice)):
            prev_top = prev_tops_slice[-(i + 1)]  # reversed order
            repeat_counts.update(prev_top)
            for num in prev_top:
                if i == 0:
                    # First (most recent) day
                    streak_counts[num] = streak_counts.get(num, 0) + 1
                elif num not in prev_tops_slice[-(i + 1)]:
                    # Not in previous day's top -> new streak
                    streak_counts[num] = 1
                else:
                    streak_counts[num] = streak_counts.get(num, 0) + 1

        for num in range(100):
            num_str = f"{num:02d}"
            repeat_count = repeat_counts.get(num, 0)
            streak_count = streak_counts.get(num, 0)
            if repeat_count > 0:
                scores[num_str] += p["repeat_penalty_top"] * repeat_count
            if repeat_count >= p["repeat_threshold"]:
                scores[num_str] += p["repeat_threshold_penalty"]
            if streak_count > 1:
                scores[num_str] += p["repeat_streak_penalty"] * (streak_count - 1)

        # --- Update previous tops ---
        top_3 = [num for num, _ in sorted(
            scores.items(), key=lambda x: x[1], reverse=True)[:3]]
        self.previous_tops.append(top_3)
        if len(self.previous_tops) > p["repeat_window"]:
            self.previous_tops.pop(0)

        return scores

    def top_k(self, scores: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
        """Tra ve top K so co diem cao nhat."""
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


# ============================================================
# Fast Version: precomputed loto sets
# ============================================================
def precompute_loto(csv_path: Path | str):
    """Precompute loto sets + special[-2:] per day. Runs once."""
    from xsmb_pipeline.dataset import load_csv, sort_results
    results = sort_results(load_csv(Path(csv_path)))
    loto_sets: list[set[int]] = []
    special_2cang: list[int] = []
    for r in results:
        sp = r.special
        special_2cang.append(int(sp[-2:]) if len(sp) >= 2 else 0)
        nums: set[int] = set()
        for prize_list in [r.first, r.second, r.third, r.fourth,
                          r.fifth, r.sixth, r.seventh]:
            for p in prize_list:
                s = str(p)
                for j in range(len(s) - 1):
                    nums.add(int(s[j:j + 2]))
        loto_sets.append(nums)
    return loto_sets, special_2cang, results


class FastHistoryAlgorithm:
    """
    Fast version: precomputed loto sets + special[-2:].
    Direct array access, no dict parsing.
    """

    def __init__(self, loto_sets: list[set[int]], special_2cang: list[int],
                 params: dict | None = None) -> None:
        self.loto = loto_sets
        self.special = special_2cang
        self.params = params or OPTIMIZED_PARAMS.copy()
        self.previous_tops: list[list[int]] = []

    def predict(self, day_idx: int) -> dict[str, float]:
        """Predict for day_idx using precomputed data."""
        p = self.params
        scores: dict[str, float] = {f"{i:02d}": 0.0 for i in range(100)}

        # Short term
        short_days = p["short_term_days"]
        for j in range(max(0, day_idx - short_days), day_idx):
            days_ago = day_idx - j
            base = p["base_point_short"] * (1 - 0.08 * days_ago)
            sp = self.special[j]
            for num in self.loto[j]:
                pt = base
                if num == sp:
                    pt *= p["special_multiplier"]
                scores[f"{num:02d}"] += pt

        # Frequency
        freq_short_start = max(0, day_idx - p["frequency_window_short"])
        freq_long_start = max(0, day_idx - p["frequency_window_long"])
        freq_short: Counter[int] = Counter()
        freq_long: Counter[int] = Counter()
        special_freq_short: Counter[int] = Counter()

        for j in range(freq_long_start, day_idx):
            for num in self.loto[j]:
                freq_long[num] += 1
                if j >= freq_short_start:
                    freq_short[num] += 1
            if j >= freq_short_start:
                special_freq_short[self.special[j]] += 1

        for num in range(100):
            num_str = f"{num:02d}"
            scores[num_str] += p["frequency_weight_short"] * freq_short[num]
            scores[num_str] += p["frequency_weight_long"] * freq_long[num]
            if freq_short[num] > 5:
                scores[num_str] += p["bonus_freq_5"]
            if special_freq_short[num] > 5:
                scores[num_str] += p["special_freq_multiplier"]

        # Neighbor bonus
        if day_idx >= 1:
            sp_last = self.special[day_idx - 1]
            nr = p["neighbor_range"]
            for offset in range(-nr, nr + 1):
                if offset == 0:
                    continue
                neighbor = (sp_last + offset) % 100
                scores[f"{neighbor:02d}"] += p["neighbor_bonus"] * (1 - 0.1 * abs(offset))

        # Cycle bonus
        for j in range(max(0, day_idx - 180), day_idx):
            days_ago = day_idx - j
            if days_ago % 7 == 0 and days_ago <= 28:
                for num in self.loto[j]:
                    scores[f"{num:02d}"] += p["cycle_7_bonus"]
            if days_ago % 30 == 0 and days_ago <= 180:
                for num in self.loto[j]:
                    scores[f"{num:02d}"] += p["cycle_30_bonus"]

        # Long absence
        last_seen: dict[int, int] = {}
        for j in range(day_idx - 1, -1, -1):
            for num in self.loto[j]:
                if num not in last_seen:
                    last_seen[num] = j
            if len(last_seen) >= 100:
                break

        for num in range(100):
            num_str = f"{num:02d}"
            last = last_seen.get(num, -1)
            days_since = day_idx - last - 1 if last >= 0 else day_idx
            scores[num_str] += days_since * p["increment"]
            if days_since >= 3:
                scores[num_str] += p["bonus_after_3_days"]
            if days_since >= 15 and freq_long[num] > 10:
                scores[num_str] += p["bonus_long_absence"]

        # Deduction if appeared last day
        if day_idx >= 1:
            for num in self.loto[day_idx - 1]:
                scores[f"{num:02d}"] += p["deduction_if_appeared_last_day"]

        # Repeat penalty
        repeat_counts: Counter[int] = Counter()
        streak_counts: dict[int, int] = {}
        rw = p["repeat_window"]
        prev_tops_slice = self.previous_tops[-rw:]
        for i in range(len(prev_tops_slice)):
            prev_top = prev_tops_slice[-(i + 1)]
            repeat_counts.update(prev_top)
            for num in prev_top:
                if i == 0:
                    streak_counts[num] = streak_counts.get(num, 0) + 1
                elif num not in prev_tops_slice[-(i + 1)]:
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
        self.previous_tops.append(top_3)
        if len(self.previous_tops) > rw:
            self.previous_tops.pop(0)

        return scores

    def top_k(self, scores: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
