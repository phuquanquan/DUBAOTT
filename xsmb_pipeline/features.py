from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from math import log2
from typing import List

from .dataset import flatten_numbers
from .schema import LotteryResult
from .targets import actual_targets, target_width


def target_items(results: List[LotteryResult], target_name: str) -> List[str]:
    items: List[str] = []
    for result in results:
        items.extend(actual_targets(result, target_name))
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
