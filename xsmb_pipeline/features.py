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
    key = (id(results[0]) if results else 0, len(results), target_name)
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
