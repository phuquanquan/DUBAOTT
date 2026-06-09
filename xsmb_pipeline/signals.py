from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Dict, List, Sequence

from .features import (
    bridge_frequency, bridge_streak, cham_match_score, db_position_score, digit_part_frequency, digit_position_frequency, digit_transition_score, falling_from_first, falling_from_special, falling_score, g1_position_score, gan_max_ratio, gan_mean_score, rolling_frequency, target_items, tong_de_match_score, tong_lo_match_score)
from .schema import LotteryResult
from .targets import actual_targets, target_width


SignalFn = Callable[[Sequence[LotteryResult], str, str], "SignalScore"]


@dataclass(frozen=True)
class SignalScore:
    name: str
    score: float
    details: Dict[str, object]


@dataclass(frozen=True)
class SignalDefinition:
    name: str
    label: str
    group: str
    weight: float
    fn: SignalFn


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def candidate_digits(candidate: str, width: int) -> List[int]:
    return [int(ch) for ch in candidate.zfill(width)]


def candidate_universe_for_target(target_name: str) -> List[str]:
    width = target_width(target_name)
    return [f"{number:0{width}d}" for number in range(10**width)]


def rolling_frequency(results: Sequence[LotteryResult], candidate: str, target_name: str, window: int) -> float:
    rows = list(results[-window:]) if len(results) > window else list(results)
    items = target_items(rows, target_name)
    return items.count(candidate) / len(items) if items else 0.0


def pattern_history_score(results: Sequence[LotteryResult], target_name: str, pattern_fn: Callable[[str], bool], recent_window: int = 30, full_weight: float = 0.35, recent_weight: float = 0.65) -> float:
    rows = list(results)
    recent_rows = rows[-recent_window:] if len(rows) > recent_window else rows
    full_items = target_items(rows, target_name)
    recent_items = target_items(recent_rows, target_name)
    full_score = sum(1 for item in full_items if pattern_fn(item)) / len(full_items) if full_items else 0.0
    recent_score = sum(1 for item in recent_items if pattern_fn(item)) / len(recent_items) if recent_items else 0.0
    return full_score * full_weight + recent_score * recent_weight


def touch_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    width = target_width(target_name)
    digits = set(candidate_digits(candidate, width))
    items = target_items(results, target_name)
    matches = sum(1 for item in items if digits & set(candidate_digits(item, width)))
    score = matches / len(items) if items else 0.0
    return SignalScore("touch", score, {"matches": matches, "total": len(items)})


def inversion_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    inverted = candidate[::-1]
    items = target_items(results, target_name)
    matches = items.count(inverted)
    score = matches / len(items) if items else 0.0
    return SignalScore("inversion", score, {"inverted": inverted, "matches": matches, "total": len(items)})


def fibonacci_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    digits = candidate_digits(candidate, target_width(target_name))
    checks = 0
    matches = 0
    if len(digits) >= 3:
        checks += 1
        matches += int((digits[0] + digits[1]) % 10 == digits[2])
        matches += int((digits[1] + digits[2]) % 10 == digits[0])
        checks += 1
    elif len(digits) == 2:
        checks += 1
        matches += int((digits[0] + digits[1]) % 10 in digits)
    score = matches / checks if checks else 0.0
    return SignalScore("fibonacci", score, {"digits": digits, "checks": checks})


def pascal_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    digits = candidate_digits(candidate, target_width(target_name))
    checks = 0
    matches = 0
    if len(digits) >= 3:
        checks += 2
        matches += int((digits[0] + digits[-1]) % 10 == digits[1])
        matches += int((digits[0] + digits[1]) % 10 == digits[-1])
    elif len(digits) == 2:
        checks += 1
        matches += int(abs(digits[0] - digits[1]) in {0, 1, 9})
    score = matches / checks if checks else 0.0
    return SignalScore("pascal", score, {"digits": digits, "checks": checks})


def composition_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = digit_part_frequency(list(results), candidate, target_name)
    return SignalScore("composition", score, {"mode": "suffix-parts"})


def shape_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    digits = candidate_digits(candidate, target_width(target_name))
    unique_count = len(set(digits))
    repeated = len(digits) - unique_count
    shape_history = pattern_history_score(results, target_name, lambda item: len(set(candidate_digits(item, target_width(target_name)))) == unique_count)
    score = clamp((0.25 + repeated / max(1, len(digits)) * 0.45) + shape_history * 0.30)
    return SignalScore("shape", score, {"unique_digits": unique_count, "repeated_digits": repeated, "shape_history": shape_history})


def bridge_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    rows = list(results)
    freq = bridge_frequency(rows, candidate)
    streak = bridge_streak(rows, candidate)
    transition = digit_transition_score(rows, candidate, target_name)
    score = freq * 0.45 + min(streak / 10.0, 1.0) * 0.20 + transition * 0.35
    return SignalScore("bridge", score, {"frequency": freq, "streak": streak, "transition": transition})


def position_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = digit_position_frequency(list(results), candidate, target_name)
    return SignalScore("position", score, {"target": target_name})


def hot_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    recent = rolling_frequency(results, candidate, target_name, 14)
    base = rolling_frequency(results, candidate, target_name, 90)
    score = clamp(0.5 + (recent - base) * 5.0)
    return SignalScore("hot_trend", score, {"window_14": recent, "window_90": base})


def cold_return_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    rows = list(results)
    last_seen = None
    for offset, result in enumerate(reversed(rows), start=1):
        if candidate in actual_targets(result, target_name):
            last_seen = offset
            break
    score = clamp((last_seen or min(len(rows), 90)) / 90.0)
    return SignalScore("cold_return", score, {"last_seen_days": last_seen})


def symmetry_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    width = target_width(target_name)
    digits = candidate_digits(candidate, width)
    complements = sum(1 for digit in digits if (9 - digit) in digits)
    mirror_gap = sum(abs(left - (9 - right)) for left, right in zip(digits, reversed(digits)))
    structural_score = complements / len(digits) if digits else 0.0
    history_score = pattern_history_score(results, target_name, lambda item: sum(1 for digit in candidate_digits(item, width) if (9 - digit) in candidate_digits(item, width)) >= complements)
    gap_bonus = 1.0 / (1.0 + mirror_gap)
    score = clamp(structural_score * 0.45 + history_score * 0.40 + gap_bonus * 0.15)
    return SignalScore("symmetry", score, {"complement_matches": complements, "history_score": history_score, "mirror_gap": mirror_gap})


def head_tail_link_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    width = target_width(target_name)
    digits = candidate_digits(candidate, width)
    rows = list(results[-30:]) if len(results) > 30 else list(results)
    if not rows:
        return SignalScore("head_tail_link", 0.0, {"matches": 0, "total": 0})
    heads = Counter()
    tails = Counter()
    for item in target_items(rows, target_name):
        padded = item.zfill(width)
        heads[padded[0]] += 1
        tails[padded[-1]] += 1
    head_score = heads[str(digits[0])] / max(1, sum(heads.values()))
    tail_score = tails[str(digits[-1])] / max(1, sum(tails.values()))
    return SignalScore("head_tail_link", (head_score + tail_score) / 2.0, {"head": head_score, "tail": tail_score})


def repeat_pair_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    width = target_width(target_name)
    digits = candidate_digits(candidate, width)
    pair_count = sum(1 for left, right in zip(digits, digits[1:]) if left == right)
    history_score = pattern_history_score(results, target_name, lambda item: sum(1 for left, right in zip(candidate_digits(item, width), candidate_digits(item, width)[1:]) if left == right) >= pair_count)
    score = clamp((pair_count / max(1, len(digits) - 1)) * 0.60 + history_score * 0.40)
    return SignalScore("repeat_pair", score, {"adjacent_repeats": pair_count, "history_score": history_score})


def mod10_balance_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    width = target_width(target_name)
    digits = candidate_digits(candidate, width)
    sums = {(left + right) % 10 for left in digits for right in digits}
    structural_score = sum(1 for digit in digits if digit in sums) / max(1, len(digits))
    history_score = pattern_history_score(results, target_name, lambda item: sum(1 for digit in candidate_digits(item, width) if digit in {(left + right) % 10 for left in candidate_digits(item, width) for right in candidate_digits(item, width)}) / max(1, width) >= structural_score)
    score = clamp(structural_score * 0.55 + history_score * 0.45)
    return SignalScore("mod10_balance", score, {"mod_sums": sorted(sums), "history_score": history_score})


def digit_band_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    width = target_width(target_name)
    digits = candidate_digits(candidate, width)
    recent = target_items(list(results[-30:]), target_name)
    if not recent:
        return SignalScore("digit_band", 0.0, {"matches": 0, "total": 0})
    band_matches = 0
    for item in recent:
        other = candidate_digits(item, width)
        band_matches += sum(1 for left, right in zip(digits, other) if left // 3 == right // 3)
    total = len(recent) * width
    return SignalScore("digit_band", band_matches / total if total else 0.0, {"matches": band_matches, "total": total})


def day_cycle_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    rows = list(results)
    if not rows:
        return SignalScore("day_cycle", 0.0, {"weekday": None, "matches": 0, "total": 0})
    try:
        latest = datetime.strptime(rows[-1].date, "%d/%m/%Y")
    except ValueError:
        return SignalScore("day_cycle", 0.0, {"weekday": None, "matches": 0, "total": 0})
    weekday = latest.weekday()
    same_weekday = []
    for result in rows:
        try:
            if datetime.strptime(result.date, "%d/%m/%Y").weekday() == weekday:
                same_weekday.append(result)
        except ValueError:
            continue
    items = target_items(same_weekday, target_name)
    score = items.count(candidate) / len(items) if items else 0.0
    return SignalScore("day_cycle", score, {"weekday": weekday, "total": len(items)})


def falling_1d_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = falling_score(list(results), candidate, target_name, lookback=1)
    return SignalScore("falling_1d", score, {"lookback": 1})


def falling_2d_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = falling_score(list(results), candidate, target_name, lookback=2)
    return SignalScore("falling_2d", score, {"lookback": 2})


def falling_3d_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = falling_score(list(results), candidate, target_name, lookback=3)
    return SignalScore("falling_3d", score, {"lookback": 3})


def falling_from_db_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = falling_from_special(list(results), candidate)
    return SignalScore("falling_from_db", score, {})


def falling_from_g1_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = falling_from_first(list(results), candidate)
    return SignalScore("falling_from_g1", score, {})


def cham_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = clamp(cham_match_score(list(results), candidate, target_name))
    return SignalScore("cham", score, {})


def tong_de_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = clamp(tong_de_match_score(list(results), candidate))
    return SignalScore("tong_de", score, {})


def tong_lo_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = clamp(tong_lo_match_score(list(results), candidate, target_name))
    return SignalScore("tong_lo", score, {})


def db_pos_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = clamp(db_position_score(list(results), candidate, target_name))
    return SignalScore("db_position", score, {})


def g1_pos_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = clamp(g1_position_score(list(results), candidate, target_name))
    return SignalScore("g1_position", score, {})


def gan_mean_signal(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = clamp(gan_mean_score(list(results), candidate, target_name))
    return SignalScore("gan_mean", score, {})


def gan_max_signal(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = clamp(gan_max_ratio(list(results), candidate, target_name))
    return SignalScore("gan_max_ratio", score, {})


def freq_3d_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = rolling_frequency(list(results), candidate, target_name, 3)
    return SignalScore("freq_3d", score, {"window": 3})


def freq_5d_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> SignalScore:
    score = rolling_frequency(list(results), candidate, target_name, 5)
    return SignalScore("freq_5d", score, {"window": 5})


SIGNAL_DEFINITIONS: List[SignalDefinition] = [
    SignalDefinition("touch", "Cham", "cau", 0.08, touch_score),
    SignalDefinition("inversion", "Dao so", "cau", 0.06, inversion_score),
    SignalDefinition("fibonacci", "Fibonacci mod 10", "math", 0.04, fibonacci_score),
    SignalDefinition("pascal", "Pascal mod 10", "math", 0.04, pascal_score),
    SignalDefinition("composition", "Ghep chu so", "digit", 0.08, composition_score),
    SignalDefinition("shape", "Cau hinh", "digit", 0.04, shape_score),
    SignalDefinition("bridge", "Bridge", "cau", 0.12, bridge_score),
    SignalDefinition("position", "Vi tri chu so", "digit", 0.08, position_score),
    SignalDefinition("hot_trend", "Cau nong", "trend", 0.10, hot_score),
    SignalDefinition("cold_return", "Cau lanh", "trend", 0.08, cold_return_score),
    SignalDefinition("symmetry", "Doi xung", "math", 0.04, symmetry_score),
    SignalDefinition("head_tail_link", "Dau duoi lien ket", "digit", 0.08, head_tail_link_score),
    SignalDefinition("repeat_pair", "Lap cap so", "digit", 0.04, repeat_pair_score),
    SignalDefinition("mod10_balance", "Bu tru mod 10", "math", 0.05, mod10_balance_score),
    SignalDefinition("digit_band", "Hang chuc/tram", "digit", 0.05, digit_band_score),
    SignalDefinition("day_cycle", "Chu ky ngay", "cycle", 0.04, day_cycle_score),
    SignalDefinition("falling_1d", "Lo roi 1 ngay", "roi", 0.06, falling_1d_score),
    SignalDefinition("falling_2d", "Lo roi 2 ngay", "roi", 0.04, falling_2d_score),
    SignalDefinition("falling_3d", "Lo roi 3 ngay", "roi", 0.03, falling_3d_score),
    SignalDefinition("falling_from_db", "Roi tu DB", "roi", 0.05, falling_from_db_score),
    SignalDefinition("falling_from_g1", "Roi tu G1", "roi", 0.04, falling_from_g1_score),
    SignalDefinition("cham", "Cham chu so", "digit", 0.06, cham_score),
    SignalDefinition("tong_de", "Tong de", "tong", 0.04, tong_de_score),
    SignalDefinition("tong_lo", "Tong lo", "tong", 0.04, tong_lo_score),
    SignalDefinition("db_position", "Vi tri DB", "vitri", 0.05, db_pos_score),
    SignalDefinition("g1_position", "Vi tri G1", "vitri", 0.05, g1_pos_score),
    SignalDefinition("gan_mean", "Gan trung binh", "gan", 0.04, gan_mean_signal),
    SignalDefinition("gan_max_ratio", "Gan max ratio", "gan", 0.04, gan_max_signal),
    SignalDefinition("freq_3d", "Tan suat 3 ngay", "freq", 0.05, freq_3d_score),
    SignalDefinition("freq_5d", "Tan suat 5 ngay", "freq", 0.04, freq_5d_score),
]


def evaluate_signals(results: Sequence[LotteryResult], candidate: str, target_name: str) -> List[SignalScore]:
    scores: List[SignalScore] = []
    for definition in SIGNAL_DEFINITIONS:
        raw = definition.fn(results, candidate, target_name)
        scores.append(SignalScore(raw.name, clamp(raw.score), raw.details))
    return scores


MODEL_SIGNAL_NAMES = {
    "symmetry",
}


TARGET_MODEL_SIGNAL_NAMES = {
    "loto2": MODEL_SIGNAL_NAMES | {"touch", "inversion", "position", "head_tail_link", "cham", "db_position", "g1_position"},
}


TARGET_ENSEMBLE_SIGNAL_NAMES = {
    "loto2": {
        "touch",
        "inversion",
        "composition",
        "shape",
        "bridge",
        "position",
        "hot_trend",
        "cold_return",
        "head_tail_link",
        "digit_band",
        "day_cycle",
        "falling_1d",
        "falling_2d",
        "falling_3d",
        "falling_from_db",
        "falling_from_g1",
        "cham",
        "tong_de",
        "tong_lo",
        "db_position",
        "g1_position",
        "gan_mean",
        "gan_max_ratio",
        "freq_3d",
        "freq_5d",
    },
}


def active_model_signal_names(target_name: str) -> set[str]:
    return TARGET_MODEL_SIGNAL_NAMES.get(target_name, MODEL_SIGNAL_NAMES)


def active_ensemble_signal_names(target_name: str) -> set[str]:
    return TARGET_ENSEMBLE_SIGNAL_NAMES.get(target_name, MODEL_ENSEMBLE_SIGNAL_NAMES)


def loto2_signal_filter_summary() -> Dict[str, object]:
    return {
        "kept_model_signals": sorted(TARGET_MODEL_SIGNAL_NAMES["loto2"]),
        "kept_ensemble_signals": sorted(TARGET_ENSEMBLE_SIGNAL_NAMES["loto2"]),
        "dropped_signals": sorted({definition.name for definition in SIGNAL_DEFINITIONS if definition.name not in TARGET_ENSEMBLE_SIGNAL_NAMES["loto2"]}),
    }


MODEL_ENSEMBLE_SIGNAL_NAMES = {
    "touch",
    "inversion",
    "fibonacci",
    "pascal",
    "shape",
    "hot_trend",
    "cold_return",
    "symmetry",
    "head_tail_link",
    "repeat_pair",
    "mod10_balance",
    "digit_band",
    "day_cycle",
    "falling_1d",
    "falling_2d",
    "falling_3d",
    "falling_from_db",
    "falling_from_g1",
    "cham",
    "tong_de",
    "tong_lo",
    "db_position",
    "g1_position",
    "gan_mean",
    "gan_max_ratio",
    "freq_3d",
    "freq_5d",
}


def signal_vector(results: Sequence[LotteryResult], candidate: str, target_name: str, selected_signal_names: Sequence[str] | None = None) -> List[float]:
    scores = []
    active_names = set(resolved_model_signal_names(target_name, selected_signal_names))
    for definition in SIGNAL_DEFINITIONS:
        if definition.name in active_names:
            scores.append(clamp(definition.fn(results, candidate, target_name).score))
    return scores


def ensemble_signal_score(results: Sequence[LotteryResult], candidate: str, target_name: str) -> float:
    active_names = active_ensemble_signal_names(target_name)
    active_definitions = [definition for definition in SIGNAL_DEFINITIONS if definition.name in active_names]
    scores = {definition.name: clamp(definition.fn(results, candidate, target_name).score) for definition in active_definitions}
    total_weight = sum(definition.weight for definition in active_definitions)
    if not total_weight:
        return 0.0
    return sum(scores.get(definition.name, 0.0) * definition.weight for definition in active_definitions) / total_weight


def signal_catalog() -> List[Dict[str, object]]:
    return [
        {
            "name": definition.name,
            "label": definition.label,
            "group": definition.group,
            "weight": definition.weight,
        }
        for definition in SIGNAL_DEFINITIONS
    ]


def signal_definitions_by_group() -> Dict[str, List[SignalDefinition]]:
    grouped: Dict[str, List[SignalDefinition]] = {}
    for definition in SIGNAL_DEFINITIONS:
        grouped.setdefault(definition.group, []).append(definition)
    return grouped


def signal_group_names() -> List[str]:
    return sorted(signal_definitions_by_group())


def signal_names_for_group(group_name: str) -> List[str]:
    return [definition.name for definition in signal_definitions_by_group().get(group_name, [])]


def signal_names_for_groups(group_names: Sequence[str]) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for group_name in group_names:
        for signal_name in signal_names_for_group(group_name):
            if signal_name not in seen:
                names.append(signal_name)
                seen.add(signal_name)
    return names


def resolved_model_signal_names(target_name: str, selected_signal_names: Sequence[str] | None = None) -> List[str]:
    if selected_signal_names is None:
        selected = active_model_signal_names(target_name)
    else:
        selected = set(selected_signal_names)
    return [definition.name for definition in SIGNAL_DEFINITIONS if definition.name in selected]


def signal_group_catalog() -> List[Dict[str, object]]:
    grouped = signal_definitions_by_group()
    return [
        {
            "group": group_name,
            "signals": [definition.name for definition in definitions],
            "labels": [definition.label for definition in definitions],
            "weight": sum(definition.weight for definition in definitions),
        }
        for group_name, definitions in sorted(grouped.items())
    ]


def rank_signal(results: Sequence[LotteryResult], target_name: str, signal_name: str, top_n: int = 20) -> List[Dict[str, object]]:
    definition = next((item for item in SIGNAL_DEFINITIONS if item.name == signal_name), None)
    if definition is None:
        raise ValueError(f"Unsupported signal: {signal_name}")
    rows = []
    for candidate in candidate_universe_for_target(target_name):
        signal = definition.fn(results, candidate, target_name)
        rows.append(
            {
                "candidate": candidate,
                "score": clamp(signal.score),
                "details": signal.details,
            }
        )
    rows.sort(key=lambda item: (-float(item["score"]), str(item["candidate"])))
    return rows[:top_n]


def build_signal_rankings(results: Sequence[LotteryResult], top_n: int = 20) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for definition in SIGNAL_DEFINITIONS:
        payload[definition.name] = rank_signal(results, "loto2", definition.name, top_n=top_n)
    return payload


def summarize_signals(results: Sequence[LotteryResult], candidate: str, target_name: str) -> Dict[str, object]:
    signal_rows = evaluate_signals(results, candidate, target_name)
    score_by_name = {signal.name: signal.score for signal in signal_rows}
    total_weight = sum(definition.weight for definition in SIGNAL_DEFINITIONS)
    weighted_score = sum(score_by_name.get(definition.name, 0.0) * definition.weight for definition in SIGNAL_DEFINITIONS)
    return {
        "candidate": candidate,
        "target": target_name,
        "signal_score": weighted_score / total_weight if total_weight else 0.0,
        "signals": [asdict(signal) for signal in signal_rows],
    }


def build_signal_payload(results: Sequence[LotteryResult], predictions: Dict[str, object]) -> Dict[str, object]:
    loto2_rows = predictions.get("loto2_top", [])
    payload: Dict[str, object] = {"catalog": signal_catalog(), "targets": {}, "rankings": build_signal_rankings(results)}
    payload["targets"]["loto2"] = [summarize_signals(results, number, "loto2") for number, _ in loto2_rows]
    return payload
