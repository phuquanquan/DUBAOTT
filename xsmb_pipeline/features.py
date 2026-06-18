from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from math import log2
from statistics import mean, median
from typing import Dict, List, Sequence

from .dataset import flatten_numbers
from .schema import LotteryResult
from .targets import actual_targets, target_width


_target_items_cache: Dict[tuple, List[str]] = {}


def target_items(results: List[LotteryResult], target_name: str) -> List[str]:
    if not results:
        return []
    result_ids = tuple(id(r) for r in results)
    key = (result_ids, target_name)
    cached = _target_items_cache.get(key)
    if cached is not None:
        return cached
    items: List[str] = []
    for result in results:
        items.extend(actual_targets(result, target_name))
    if len(_target_items_cache) > 256:
        _target_items_cache.clear()
    _target_items_cache[key] = items
    return items


@lru_cache(maxsize=4096)
def bridge_pair_strings_from_numbers(numbers: tuple[str, ...]) -> tuple[str, ...]:
    pairs: List[str] = []
    for left in numbers:
        for right in numbers:
            if len(left) < 2 or len(right) < 2:
                continue
            pairs.append(left[-2:] + right[-1])
            if len(left) >= 3 and len(right) >= 3:
                pairs.append(left[-3:] + right[-2:])
    return tuple(pairs)


@lru_cache(maxsize=4096)
def bridge_pair_set_from_numbers(numbers: tuple[str, ...]) -> frozenset[str]:
    return frozenset(bridge_pair_strings_from_numbers(numbers))


def bridge_pair_strings(result: LotteryResult) -> List[str]:
    return list(bridge_pair_strings_from_numbers(tuple(flatten_numbers(result))))


def bridge_parts_for_candidate(candidate: str) -> List[str]:
    if len(candidate) == 2:
        return [candidate]
    if len(candidate) >= 3:
        return [candidate[-2:], candidate[-3:]]
    return [candidate]


def bridge_pair_matches_part(pair: str, part: str) -> bool:
    return pair == part or pair.startswith(part) or pair.endswith(part) or part in pair


def candidate_matches_bridge(result: LotteryResult, candidate: str) -> bool:
    bridge_parts = bridge_pair_set_from_numbers(tuple(flatten_numbers(result)))
    return any(bridge_pair_matches_part(pair, part) for pair in bridge_parts for part in bridge_parts_for_candidate(candidate))


def bridge_streak(results: List[LotteryResult], candidate: str, lookback: int = 15) -> int:
    streak = 0
    for result in reversed(results[-lookback:]):
        if candidate_matches_bridge(result, candidate):
            streak += 1
        else:
            break
    return streak


def bridge_frequency(results: List[LotteryResult], candidate: str, lookback: int = 30) -> float:
    window = results[-lookback:] if len(results) > lookback else results
    if not window:
        return 0.0
    matches = sum(1 for result in window if candidate_matches_bridge(result, candidate))
    return matches / len(window)


def digit_position_frequency(results: List[LotteryResult], candidate: str, target_name: str) -> float:
    if not results:
        return 0.0
    width = target_width(target_name)
    candidate_digits = list(candidate.zfill(width))
    total = 0
    matches = 0
    for result in results:
        for item in actual_targets(result, target_name):
            item_digits = list(item.zfill(width))
            total += len(candidate_digits)
            matches += sum(1 for left, right in zip(candidate_digits, item_digits) if left == right)
    return matches / total if total else 0.0


def digit_part_frequency(results: List[LotteryResult], candidate: str, target_name: str) -> float:
    if not results:
        return 0.0
    parts: List[str] = []
    if len(candidate) >= 1:
        parts.append(candidate[-1])
    if len(candidate) >= 2:
        parts.append(candidate[-2:])
    if len(candidate) >= 3:
        parts.extend([candidate[-2], candidate[-3], candidate[-3:-1], candidate[-3:]])
    total = 0
    matches = 0
    for result in results:
        items = actual_targets(result, target_name)
        total += len(items) * len(parts)
        for item in items:
            matches += sum(1 for part in parts if item.endswith(part))
    return matches / total if total else 0.0


def digit_transition_score(results: List[LotteryResult], candidate: str, target_name: str, lookback: int = 30) -> float:
    window = results[-lookback:] if len(results) > lookback else results
    if len(window) < 2:
        return 0.0
    parts = set(bridge_parts_for_candidate(candidate))
    transitions = 0
    matches = 0
    for previous, current in zip(window, window[1:]):
        previous_parts = bridge_pair_set_from_numbers(tuple(flatten_numbers(previous)))
        current_targets = set(actual_targets(current, target_name))
        transitions += 1
        previous_match = any(bridge_pair_matches_part(pair, part) for pair in previous_parts for part in parts)
        if previous_match and current_targets & parts:
            matches += 1
    return matches / transitions if transitions else 0.0


def rolling_frequency(results: List[LotteryResult], candidate: str, target_name: str, lookback: int) -> float:
    window = results[-lookback:] if len(results) > lookback else results
    if not window:
        return 0.0
    items = target_items(window, target_name)
    return items.count(candidate) / len(items) if items else 0.0


def gap_since_last_seen(results: List[LotteryResult], candidate: str, target_name: str) -> float:
    for offset, result in enumerate(reversed(results), start=1):
        if candidate in actual_targets(result, target_name):
            return float(offset)
    return float(len(results) + 1)


def recency_decay_score(results: List[LotteryResult], candidate: str, target_name: str) -> float:
    gap = gap_since_last_seen(results, candidate, target_name)
    return 1.0 / gap if gap > 0 else 0.0


def recent_long_term_delta(results: List[LotteryResult], candidate: str, target_name: str, recent_window: int = 14, long_window: int = 60) -> float:
    recent = rolling_frequency(results, candidate, target_name, recent_window)
    long_term = rolling_frequency(results, candidate, target_name, long_window)
    return recent - long_term


def decay_weighted_frequency(results: List[LotteryResult], candidate: str, target_name: str, half_life: int = 14) -> float:
    if not results:
        return 0.0
    total_weight = 0.0
    weighted_hits = 0.0
    for offset, result in enumerate(reversed(results), start=1):
        weight = 0.5 ** ((offset - 1) / max(1, half_life))
        total_weight += weight
        if candidate in actual_targets(result, target_name):
            weighted_hits += weight
    return weighted_hits / total_weight if total_weight else 0.0


def recency_gap_ratio(results: List[LotteryResult], candidate: str, target_name: str, short_window: int = 7, long_window: int = 30) -> float:
    short_frequency = rolling_frequency(results, candidate, target_name, short_window)
    long_frequency = rolling_frequency(results, candidate, target_name, long_window)
    if long_frequency == 0.0:
        return short_frequency
    return short_frequency / long_frequency


def recent_peak_frequency(results: List[LotteryResult], candidate: str, target_name: str) -> float:
    return max(
        rolling_frequency(results, candidate, target_name, 7),
        rolling_frequency(results, candidate, target_name, 14),
        rolling_frequency(results, candidate, target_name, 30),
    )


def recency_cluster_score(results: List[LotteryResult], candidate: str, target_name: str) -> float:
    if len(results) < 2:
        return 0.0
    recent_hits = 0
    for result in results[-14:]:
        if candidate in actual_targets(result, target_name):
            recent_hits += 1
    return min(1.0, recent_hits / 3.0)


def digit_sum_score(candidate: str) -> float:
    if not candidate:
        return 0.0
    return sum(int(ch) for ch in candidate) / (9.0 * len(candidate))


def repeated_digit_ratio(candidate: str) -> float:
    if len(candidate) < 2:
        return 0.0
    repeated = sum(1 for left, right in zip(candidate, candidate[1:]) if left == right)
    return repeated / (len(candidate) - 1)


def unique_digit_ratio(candidate: str) -> float:
    if not candidate:
        return 0.0
    return len(set(candidate)) / len(candidate)


def head_frequency(results: List[LotteryResult], candidate: str, target_name: str, lookback: int = 30) -> float:
    if not candidate:
        return 0.0
    window = results[-lookback:] if len(results) > lookback else results
    items = target_items(window, target_name)
    if not items:
        return 0.0
    head = candidate[0]
    matches = sum(1 for item in items if item.zfill(len(candidate))[0] == head)
    return matches / len(items)


def tail_frequency(results: List[LotteryResult], candidate: str, target_name: str, lookback: int = 30) -> float:
    if not candidate:
        return 0.0
    window = results[-lookback:] if len(results) > lookback else results
    items = target_items(window, target_name)
    if not items:
        return 0.0
    tail = candidate[-1]
    matches = sum(1 for item in items if item.zfill(len(candidate))[-1] == tail)
    return matches / len(items)


def falling_score(results: List[LotteryResult], candidate: str, target_name: str, lookback: int = 1) -> float:
    """Lô rơi: candidate xuất hiện ở kỳ trước đó trong cùng target."""
    if len(results) < 2:
        return 0.0
    for i in range(1, min(lookback, len(results)) + 1):
        if candidate in actual_targets(results[-i], target_name):
            return 1.0
    return 0.0


def falling_from_special(results: List[LotteryResult], candidate: str) -> float:
    """Rơi từ ĐB: candidate là 2 số cuối của giải ĐB kỳ trước."""
    if len(results) < 2:
        return 0.0
    prev_special = results[-2].special[-2:]
    return 1.0 if candidate == prev_special else 0.0


def falling_from_first(results: List[LotteryResult], candidate: str) -> float:
    """Rơi từ G1: candidate là 2 số cuối của giải nhất kỳ trước."""
    if len(results) < 2:
        return 0.0
    prev_first = results[-2].first[0][-2:] if results[-2].first else ""
    return 1.0 if candidate == prev_first else 0.0


def cham_frequency(results: List[LotteryResult], digit: str, target_name: str, lookback: int = 30) -> float:
    """Tần suất chạm (chữ số) xuất hiện trong target."""
    window = results[-lookback:] if len(results) > lookback else results
    items = target_items(window, target_name)
    if not items:
        return 0.0
    matches = sum(1 for item in items if digit in item)
    return matches / len(items)


def cham_match_score(results: List[LotteryResult], candidate: str, target_name: str, lookback: int = 30) -> float:
    """Điểm chạm: trung bình tần suất các chữ số trong candidate."""
    if not candidate:
        return 0.0
    scores = []
    for ch in candidate:
        scores.append(cham_frequency(results, ch, target_name, lookback))
    return sum(scores) / len(scores) if scores else 0.0


def tong_de_frequency(results: List[LotteryResult], total: int, lookback: int = 30) -> float:
    """Tần suất tổng đề (tổng 2 số cuối giải ĐB) xuất hiện."""
    window = results[-lookback:] if len(results) > lookback else results
    if not window:
        return 0.0
    matches = 0
    for result in window:
        s = result.special[-2:]
        if sum(int(ch) for ch in s) == total:
            matches += 1
    return matches / len(window)


def tong_de_match_score(results: List[LotteryResult], candidate: str, lookback: int = 30) -> float:
    """Candidate match với tổng đề phổ biến."""
    if not candidate or len(candidate) < 2:
        return 0.0
    total = sum(int(ch) for ch in candidate[-2:])
    return tong_de_frequency(results, total, lookback)


def tong_lo_frequency(results: List[LotteryResult], total: int, target_name: str, lookback: int = 30) -> float:
    """Tần suất tổng lô (tổng 2 số loto2) xuất hiện."""
    window = results[-lookback:] if len(results) > lookback else results
    items = target_items(window, target_name)
    if not items:
        return 0.0
    matches = sum(1 for item in items if sum(int(ch) for ch in item) == total)
    return matches / len(items) if items else 0.0


def tong_lo_match_score(results: List[LotteryResult], candidate: str, target_name: str, lookback: int = 30) -> float:
    """Candidate match với tổng lô phổ biến."""
    if not candidate:
        return 0.0
    total = sum(int(ch) for ch in candidate)
    return tong_lo_frequency(results, total, target_name, lookback)


def gan_stats(results: List[LotteryResult], candidate: str, target_name: str) -> Dict[str, float]:
    """Thống kê gan: độ dài các chu kỳ vắng mặt."""
    gaps: List[int] = []
    last_pos = -1
    for i, result in enumerate(results):
        if candidate in actual_targets(result, target_name):
            if last_pos >= 0:
                gaps.append(i - last_pos - 1)
            last_pos = i
    if not gaps:
        return {"current_gap": len(results) - last_pos - 1 if last_pos >= 0 else len(results), "mean_gap": float("inf"), "max_gap": float("inf"), "min_gap": float("inf")}
    return {
        "current_gap": len(results) - last_pos - 1 if last_pos >= 0 else len(results) + 1,
        "mean_gap": mean(gaps) if gaps else float("inf"),
        "max_gap": max(gaps),
        "min_gap": min(gaps),
    }


def gan_mean_score(results: List[LotteryResult], candidate: str, target_name: str) -> float:
    """Điểm dựa trên gan trung bình: giá trị càng cao nếu gan TB càng lớn."""
    stats = gan_stats(results, candidate, target_name)
    mg = stats["mean_gap"]
    if mg == float("inf") or mg <= 0:
        return 0.0
    return min(1.0, mg / 30.0)


def gan_max_ratio(results: List[LotteryResult], candidate: str, target_name: str) -> float:
    """Tỷ lệ gan max / gan hiện tại: gần 1.0 nghĩa là sắp chạm max."""
    stats = gan_stats(results, candidate, target_name)
    mx = stats["max_gap"]
    if mx == float("inf") or mx <= 0:
        return 0.0
    return min(1.0, stats["current_gap"] / mx)


def special_position_digits(result: LotteryResult, position: str) -> str:
    """Trả về chữ số tại vị trí trong giải ĐB. VD: DB_V1 -> chữ số đầu, DB_V5 -> chữ số cuối."""
    if position.startswith("DB_V"):
        idx = int(position[4:]) - 1
        if 0 <= idx < len(result.special):
            return result.special[idx]
    if position.startswith("G1_V"):
        idx = int(position[4:]) - 1
        if result.first and 0 <= idx < len(result.first[0]):
            return result.first[0][idx]
    return ""


def position_frequency(results: List[LotteryResult], candidate: str, position: str, lookback: int = 30) -> float:
    """Tần suất candidate có chứa chữ số xuất hiện tại vị trí cụ thể trong các giải."""
    window = results[-lookback:] if len(results) > lookback else results
    if not window:
        return 0.0
    matches = 0
    for result in window:
        pos_digit = special_position_digits(result, position)
        if pos_digit and pos_digit in candidate:
            matches += 1
    return matches / len(window)


def db_position_score(results: List[LotteryResult], candidate: str, target_name: str) -> float:
    """Điểm tổng hợp từ tất cả vị trí trong ĐB."""
    if not candidate:
        return 0.0
    score = 0.0
    for i in range(1, 6):
        freq = position_frequency(results, candidate, f"DB_V{i}")
        score += freq
    return score / 5.0


def g1_position_score(results: List[LotteryResult], candidate: str, target_name: str) -> float:
    """Điểm tổng hợp từ tất cả vị trí trong giải nhất."""
    if not candidate:
        return 0.0
    score = 0.0
    for i in range(1, 6):
        freq = position_frequency(results, candidate, f"G1_V{i}")
        score += freq
    return score / 5.0


# ============================================================
# Cross-Position Bridge (Soi cau bang cach ghep chu so tu 2 vi tri)
# VD: Hang tram giai 2 + Hang don vi giai 7 thu 3 = 1 cap loto
# ============================================================

from typing import NamedTuple, Optional


class PrizePosition(NamedTuple):
    """Mot vi tri trong ket qua xo so: ten giai + chi so (neu co) + chi so chu so."""
    prize: str          # special, first, second, third, fourth, fifth, sixth, seventh
    index: int          # 0-based index trong danh sach giai (VD: seventh[2] -> index=2)
    digit: int          # vi tri chu so: 0 = hang don vi (rightmost), -1 = hang chuc, -2 = hang tram

    def label(self) -> str:
        idx_str = f"_{self.index}" if self.index > 0 else ""
        dig_str = ["V1", "V2", "V3", "V4", "V5", "V6"][self.digit] if self.digit >= 0 else ""
        return f"{self.prize}{idx_str}_{dig_str}"


def _get_digit_at(result: LotteryResult, pos: PrizePosition) -> Optional[str]:
    """Trich xuat 1 chu so tai vi tri pos tu result. None neu khong co."""
    attr_map = {
        "special": result.special,
        "first":   result.first[0] if result.first else "",
        "second":  result.second[pos.index] if pos.index < len(result.second) else "",
        "third":   result.third[pos.index]  if pos.index < len(result.third)  else "",
        "fourth":  result.fourth[pos.index] if pos.index < len(result.fourth) else "",
        "fifth":   result.fifth[pos.index]   if pos.index < len(result.fifth)   else "",
        "sixth":   result.sixth[pos.index]   if pos.index < len(result.sixth)   else "",
        "seventh": result.seventh[pos.index] if pos.index < len(result.seventh) else "",
    }
    prize_str = attr_map.get(pos.prize, "")
    if not prize_str:
        return None
    digits = [ch for ch in prize_str if ch.isdigit()]
    if pos.digit < 0:
        idx = len(digits) + pos.digit
    else:
        idx = pos.digit
    if idx < 0 or idx >= len(digits):
        return None
    return digits[idx]


def _build_cross_combo(result: LotteryResult, left: PrizePosition, right: PrizePosition) -> Optional[str]:
    """Ghep 2 chu so tu 2 vi tri thanh 1 cap 2 chu so. None neu khong trich xuat duoc."""
    d_left  = _get_digit_at(result, left)
    d_right = _get_digit_at(result, right)
    if d_left is None or d_right is None:
        return None
    return d_left + d_right


def _all_prize_positions() -> List[PrizePosition]:
    """Tao tat ca cac vi tri co the co trong 1 ket qua XSMB."""
    positions: List[PrizePosition] = []

    # DB: 5 chu so -> 5 vi tri
    for d in range(5):
        positions.append(PrizePosition("special", 0, d))

    # First: 5 chu so -> 5 vi tri
    for d in range(5):
        positions.append(PrizePosition("first", 0, d))

    # Second: 2 giai, moi giai 5 chu so
    for idx in range(2):
        for d in range(5):
            positions.append(PrizePosition("second", idx, d))

    # Third: 6 giai, moi giai 5 chu so
    for idx in range(6):
        for d in range(5):
            positions.append(PrizePosition("third", idx, d))

    # Fourth: 4 giai, moi giai 5 chu so
    for idx in range(4):
        for d in range(5):
            positions.append(PrizePosition("fourth", idx, d))

    # Fifth: 6 giai, moi giai 4 chu so
    for idx in range(6):
        for d in range(4):
            positions.append(PrizePosition("fifth", idx, d))

    # Sixth: 3 giai, moi giai 4 chu so
    for idx in range(3):
        for d in range(4):
            positions.append(PrizePosition("sixth", idx, d))

    # Seventh: 4 giai, moi giai 3 chu so
    for idx in range(4):
        for d in range(3):
            positions.append(PrizePosition("seventh", idx, d))

    return positions


ALL_PRIZE_POSITIONS: List[PrizePosition] = _all_prize_positions()

# Chi dinh truoc 1 so cap hay dung (thu cong)
CURATED_POSITION_PAIRS: List[tuple[PrizePosition, PrizePosition]] = [
    # DB + DB
    (PrizePosition("special", 0, 1), PrizePosition("special", 0, 3)),
    (PrizePosition("special", 0, 2), PrizePosition("special", 0, 4)),
    (PrizePosition("special", 0, 0), PrizePosition("special", 0, 3)),
    (PrizePosition("special", 0, 1), PrizePosition("special", 0, 4)),
    # DB + First
    (PrizePosition("special", 0, 1), PrizePosition("first", 0, 3)),
    (PrizePosition("special", 0, 0), PrizePosition("first", 0, 4)),
    (PrizePosition("special", 0, 2), PrizePosition("first", 0, 0)),
    # DB + Second
    (PrizePosition("special", 0, 1), PrizePosition("second", 0, 3)),
    (PrizePosition("special", 0, 0), PrizePosition("second", 1, 4)),
    # DB + Third
    (PrizePosition("special", 0, 1), PrizePosition("third", 0, 3)),
    (PrizePosition("special", 0, 0), PrizePosition("third", 2, 4)),
    # DB + Fourth
    (PrizePosition("special", 0, 1), PrizePosition("fourth", 0, 3)),
    (PrizePosition("special", 0, 0), PrizePosition("fourth", 2, 4)),
    # DB + Fifth
    (PrizePosition("special", 0, 1), PrizePosition("fifth", 0, 3)),
    (PrizePosition("special", 0, 0), PrizePosition("fifth", 3, 3)),
    # DB + Sixth
    (PrizePosition("special", 0, 1), PrizePosition("sixth", 0, 3)),
    (PrizePosition("special", 0, 0), PrizePosition("sixth", 2, 3)),
    # DB + Seventh
    (PrizePosition("special", 0, 2), PrizePosition("seventh", 2, 2)),   # Hang tram DB + DV G7_3
    (PrizePosition("special", 0, 1), PrizePosition("seventh", 2, 2)),
    (PrizePosition("special", 0, 0), PrizePosition("seventh", 0, 2)),
    # First + Second
    (PrizePosition("first", 0, 1), PrizePosition("second", 0, 3)),
    (PrizePosition("first", 0, 0), PrizePosition("second", 1, 4)),
    # First + Third
    (PrizePosition("first", 0, 1), PrizePosition("third", 0, 3)),
    (PrizePosition("first", 0, 0), PrizePosition("third", 2, 4)),
    # Second + Third
    (PrizePosition("second", 0, 1), PrizePosition("third", 0, 3)),
    (PrizePosition("second", 1, 0), PrizePosition("third", 2, 4)),
    # Second + Seventh
    (PrizePosition("second", 0, 1), PrizePosition("seventh", 2, 2)),
    (PrizePosition("second", 0, 0), PrizePosition("seventh", 0, 2)),
    # Third + Seventh
    (PrizePosition("third", 0, 1), PrizePosition("seventh", 2, 2)),
    (PrizePosition("third", 1, 0), PrizePosition("seventh", 0, 2)),
    # Third + Fourth
    (PrizePosition("third", 0, 1), PrizePosition("fourth", 1, 3)),
    (PrizePosition("third", 2, 0), PrizePosition("fourth", 3, 4)),
    # Fourth + Seventh
    (PrizePosition("fourth", 0, 1), PrizePosition("seventh", 2, 2)),
    # Fifth + Seventh
    (PrizePosition("fifth", 0, 1), PrizePosition("seventh", 2, 2)),
    (PrizePosition("fifth", 3, 0), PrizePosition("seventh", 0, 2)),
    # Sixth + Seventh
    (PrizePosition("sixth", 0, 1), PrizePosition("seventh", 2, 2)),
    (PrizePosition("sixth", 1, 0), PrizePosition("seventh", 0, 2)),
    # Cross giữa các hàng chữ số (trong cùng 1 giải)
    (PrizePosition("special", 0, 0), PrizePosition("special", 0, 4)),
    (PrizePosition("special", 0, 1), PrizePosition("special", 0, 2)),
    (PrizePosition("first",    0, 0), PrizePosition("first",    0, 4)),
    (PrizePosition("second",   0, 0), PrizePosition("second",   0, 4)),
    (PrizePosition("third",    0, 0), PrizePosition("third",    0, 4)),
    (PrizePosition("seventh",  0, 0), PrizePosition("seventh",  2, 2)),
    (PrizePosition("seventh",  1, 0), PrizePosition("seventh",  3, 2)),
    # Gap giữa 2 giải cùng loại
    (PrizePosition("second", 0, 2), PrizePosition("second", 1, 2)),
    (PrizePosition("third",  0, 2), PrizePosition("third",  3, 2)),
    (PrizePosition("fifth",  0, 1), PrizePosition("fifth",  3, 1)),
    (PrizePosition("seventh", 0, 0), PrizePosition("seventh", 3, 2)),
]


def cross_position_combo_score(
    results: List[LotteryResult],
    candidate: str,
    target_name: str,
    lookback: int = 90,
) -> float:
    """Diem cross-position: candidate co the la ket qua ghep 2 vi tri tu 1 ket qua cu.

    Tung ket qua trong lookback, lay 2 vi tri (trai + phai) tu CURATED_POSITION_PAIRS,
    tao cap 2 chu so, kiem tra cap nay co xuat hien trong loto2 cua ngay hom sau hay khong.

    Diem = so lan candidate xuat hien / tong so cap co the trong lookback.
    """
    if target_name != "loto2" or len(candidate) != 2:
        return 0.0
    if len(results) < 3:
        return 0.0

    window = results[-(lookback + 2):]
    hits = 0
    total = 0

    for i in range(len(window) - 1):
        source_row = window[i]
        target_row = window[i + 1]
        target_items_set = set(actual_targets(target_row, target_name))

        for left_pos, right_pos in CURATED_POSITION_PAIRS:
            combo = _build_cross_combo(source_row, left_pos, right_pos)
            total += 1
            if combo == candidate:
                hits += 1

    return hits / total if total > 0 else 0.0


def cross_position_combo_lead_score(
    results: List[LotteryResult],
    candidate: str,
    target_name: str,
    lookback: int = 90,
    delay: int = 1,
) -> float:
    """Diem cross-position voi do tre delay ngay (1 = hom sau, 2 = 2 ngay sau)."""
    if target_name != "loto2" or len(candidate) != 2:
        return 0.0
    if len(results) < delay + 1:
        return 0.0

    window = results[-(lookback + delay + 1):]
    hits = 0
    total = 0

    for i in range(len(window) - delay):
        source_row = window[i]
        target_row = window[i + delay]
        target_items_set = set(actual_targets(target_row, target_name))

        for left_pos, right_pos in CURATED_POSITION_PAIRS:
            combo = _build_cross_combo(source_row, left_pos, right_pos)
            total += 1
            if combo == candidate:
                hits += 1

    return hits / total if total > 0 else 0.0
