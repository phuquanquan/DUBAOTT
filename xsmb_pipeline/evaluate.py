from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from .dataset import parse_date
from .models.weighted import RankingEvaluation, candidate_universe, compare_loto2_weight_strategies, fit_tuned_ranking_model, normalized_frequency, predict_next_day, train_ranking_model
from .schema import LotteryResult
from .signals import SIGNAL_DEFINITIONS, build_signal_payload, clamp, ensemble_signal_score, resolved_model_signal_names, signal_definitions_by_group, signal_group_catalog, signal_group_names, signal_names_for_group, signal_names_for_groups
from .targets import actual_targets, target_width


def evaluate_prediction_set(predicted: Sequence[str], actual: Sequence[str], universe_size: int) -> Tuple[int, float, float, float]:
    predicted_set = set(predicted)
    actual_set = set(actual)
    overlap = len(predicted_set & actual_set)
    hit = 1 if overlap > 0 else 0
    actual_count = len(actual_set)
    miss_probability = ((universe_size - actual_count) / universe_size) ** len(predicted_set) if universe_size and predicted_set else 1.0
    baseline_hit = 1.0 - miss_probability
    baseline_precision = actual_count / universe_size if universe_size else 0.0
    precision = overlap / max(1, len(predicted_set))
    return hit, baseline_hit, precision, baseline_precision


def filter_results_by_date_range(results: Sequence[LotteryResult], start_date: str | None = None, end_date: str | None = None) -> List[LotteryResult]:
    rows = list(results)
    if start_date is not None:
        start = parse_date(start_date)
        rows = [item for item in rows if parse_date(item.date) >= start]
    if end_date is not None:
        end = parse_date(end_date)
        rows = [item for item in rows if parse_date(item.date) <= end]
    return rows


def loto2_data_windows(results: Sequence[LotteryResult]) -> List[Dict[str, object]]:
    rows = list(results)
    recent_rows = rows[-120:] if len(rows) > 120 else rows
    return [
        {
            "name": "recent",
            "label": "Recent window",
            "results": recent_rows,
        },
        {
            "name": "y2025_2026",
            "label": "2025-2026 window",
            "results": filter_results_by_date_range(rows, start_date="01/01/2025", end_date="31/12/2026"),
        },
        {
            "name": "y2024_2026",
            "label": "2024-2026 window",
            "results": filter_results_by_date_range(rows, start_date="01/01/2024", end_date="31/12/2026"),
        },
        {
            "name": "full",
            "label": "Full available history",
            "results": rows,
        },
    ]


def benchmark_loto2_data_windows(results: Sequence[LotteryResult], split_ratio: float, top_k: int, min_train_size: int = 30) -> Dict[str, object]:
    windows = loto2_data_windows(results)
    evaluations: List[Dict[str, object]] = []
    for window in windows:
        window_results = list(window["results"])
        if len(window_results) <= min_train_size:
            continue
        payload = compare_loto2_models(window_results, split_ratio=split_ratio, top_k=top_k, min_train_size=min_train_size, include_window_benchmark=False)
        evaluations.append(
            {
                "window": window["name"],
                "label": window["label"],
                "dataset_size": len(window_results),
                "date_start": window_results[0].date if window_results else None,
                "date_end": window_results[-1].date if window_results else None,
                "best_model": payload.get("best_model"),
                "best_input_configuration": payload.get("input_benchmark", {}).get("best_configuration"),
                "summary": payload,
            }
        )
    evaluations.sort(key=lambda item: (-float(item["summary"]["evaluations"][0]["hit_rate"] if item.get("summary") and item["summary"].get("evaluations") else 0.0), -float(item["summary"]["evaluations"][0]["precision_at_k"] if item.get("summary") and item["summary"].get("evaluations") else 0.0), str(item["window"])))
    return {
        "target": "loto2",
        "split_ratio": split_ratio,
        "top_k": top_k,
        "min_train_size": min_train_size,
        "windows": evaluations,
        "best_window": evaluations[0] if evaluations else None,
    }


def rolling_loto2_walkforward_benchmark(results: Sequence[LotteryResult], target_name: str = "loto2", top_k: int = 3, min_train_size: int = 30, train_window_sizes: Sequence[int] = (45, 60, 90, 120)) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for train_window in train_window_sizes:
        if len(results) <= max(min_train_size, train_window):
            continue
        hits = 0
        precision_total = 0.0
        test_count = 0
        for split_index in range(max(min_train_size, train_window), len(results)):
            train = list(results[max(0, split_index - train_window):split_index])
            test_result = results[split_index]
            model = train_ranking_model(train, target_name=target_name, top_k=top_k)
            predicted = [number for number, _ in model.predict()]
            actual = actual_targets(test_result, target_name)
            universe_size = 10 ** model.number_width
            hit, _, precision, _ = evaluate_prediction_set(predicted, actual, universe_size)
            hits += hit
            precision_total += precision
            test_count += 1
        if test_count:
            rows.append(
                {
                    "train_window": train_window,
                    "test_size": test_count,
                    "hit_rate": hits / test_count,
                    "precision_at_k": precision_total / test_count,
                }
            )
    rows.sort(key=lambda item: (-float(item["hit_rate"]), -float(item["precision_at_k"]), int(item["train_window"])))
    return {
        "target": target_name,
        "top_k": top_k,
        "min_train_size": min_train_size,
        "windows": rows,
        "best_window": rows[0] if rows else None,
    }


def blend_model_rankings(rankings: Sequence[Sequence[Tuple[str, float]]]) -> List[Tuple[str, float]]:
    combined: Dict[str, float] = {}
    for ranking in rankings:
        for rank, (candidate, _) in enumerate(ranking, start=1):
            combined[candidate] = combined.get(candidate, 0.0) + 1.0 / rank
    return sorted(combined.items(), key=lambda pair: (-pair[1], pair[0]))


def ensemble_loto2_predictions(results: Sequence[LotteryResult], top_k: int, min_train_size: int = 30) -> Dict[str, object]:
    train = list(results[:-1]) if len(results) > 1 else list(results)
    weighted = train_ranking_model(train, target_name="loto2", top_k=top_k)
    tuned = fit_tuned_ranking_model(train, target_name="loto2", top_k=top_k, min_train_size=min_train_size)
    rankings = [weighted.predict(), tuned.predict()]
    blended = blend_model_rankings(rankings)[:top_k]
    return {
        "target": "loto2",
        "top_k": top_k,
        "model": "rank-blend",
        "top_predictions": blended,
        "components": [weighted.predict(), tuned.predict()],
    }


def build_evaluation(model_name: str, target_name: str, train_size: int, test_size: int, top_k: int, hit_rate: float, baseline_hit_rate: float, frequency_hit_rate: float, precision_at_k: float, baseline_precision_at_k: float, frequency_precision_at_k: float) -> RankingEvaluation:
    return RankingEvaluation(
        target=target_name,
        model=model_name,
        train_size=train_size,
        test_size=test_size,
        top_k=top_k,
        hit_rate=hit_rate,
        baseline_hit_rate=baseline_hit_rate,
        frequency_hit_rate=frequency_hit_rate,
        precision_at_k=precision_at_k,
        baseline_precision_at_k=baseline_precision_at_k,
        frequency_precision_at_k=frequency_precision_at_k,
        hit_rate_pct=hit_rate * 100.0,
        baseline_hit_rate_pct=baseline_hit_rate * 100.0,
        frequency_hit_rate_pct=frequency_hit_rate * 100.0,
        precision_at_k_pct=precision_at_k * 100.0,
        baseline_precision_at_k_pct=baseline_precision_at_k * 100.0,
        frequency_precision_at_k_pct=frequency_precision_at_k * 100.0,
    )


@dataclass(frozen=True)
class SignalBacktestEvaluation:
    target: str
    mode: str
    signal: str
    top_k: int
    train_size: int
    test_size: int
    hit_rate: float
    baseline_hit_rate: float
    frequency_hit_rate: float
    precision_at_k: float
    baseline_precision_at_k: float
    frequency_precision_at_k: float
    hit_rate_pct: float
    baseline_hit_rate_pct: float
    frequency_hit_rate_pct: float
    precision_at_k_pct: float
    baseline_precision_at_k_pct: float
    frequency_precision_at_k_pct: float
    hit_rate_delta_vs_baseline_pct: float
    hit_rate_delta_vs_frequency_pct: float
    precision_delta_vs_baseline_pct: float
    precision_delta_vs_frequency_pct: float
    recent_hit_rate: float
    recent_hit_rate_pct: float
    recent_precision_at_k: float
    recent_precision_at_k_pct: float
    win_rate_vs_frequency: float
    win_rate_vs_frequency_pct: float
    unique_prediction_rate: float
    unique_prediction_rate_pct: float
    avg_score_gap: float
    ranking_score: float
    ranking_score_pct: float
    repeated_prediction_rate: float
    repeated_prediction_rate_pct: float
    research_verdict: str
    research_notes: List[str]
    bridge_signal: bool


def prediction_signature(row: Dict[str, object]) -> Tuple[str, ...]:
    return tuple(str(item) for item in row.get("predicted", []))


def score_gap(row: Dict[str, object]) -> float:
    predicted_scores = row.get("predicted_scores", [])
    if len(predicted_scores) < 2:
        return 0.0
    first = float(predicted_scores[0].get("score", 0.0))
    last = float(predicted_scores[-1].get("score", 0.0))
    return first - last


def build_signal_ranking_score(hit_delta_vs_frequency: float, precision_delta_vs_frequency: float, recent_hit_rate: float, win_rate_vs_frequency: float, unique_prediction_rate: float, avg_score_gap: float) -> float:
    return (
        hit_delta_vs_frequency * 0.35
        + precision_delta_vs_frequency * 0.20
        + recent_hit_rate * 0.20
        + win_rate_vs_frequency * 0.15
        + unique_prediction_rate * 0.07
        + min(avg_score_gap * 5.0, 1.0) * 0.03
    )


def signal_research_verdict(hit_delta_vs_frequency: float, precision_delta_vs_frequency: float, recent_hit_rate: float, win_rate_vs_frequency: float, repeated_prediction_rate: float) -> tuple[str, List[str]]:
    notes: List[str] = []
    if hit_delta_vs_frequency > 0:
        notes.append("beats frequency hit rate")
    elif hit_delta_vs_frequency < 0:
        notes.append("loses to frequency hit rate")
    else:
        notes.append("matches frequency hit rate")
    if precision_delta_vs_frequency > 0:
        notes.append("improves precision")
    elif precision_delta_vs_frequency < 0:
        notes.append("hurts precision")
    if recent_hit_rate >= 0.6:
        notes.append("recent form is strong")
    elif recent_hit_rate <= 0.4:
        notes.append("recent form is weak")
    if repeated_prediction_rate >= 0.5:
        notes.append("repeats predictions often")
    if hit_delta_vs_frequency > 0 and precision_delta_vs_frequency >= 0 and win_rate_vs_frequency >= 0.5:
        return "keep", notes
    if hit_delta_vs_frequency < 0 and precision_delta_vs_frequency <= 0:
        return "drop", notes
    if repeated_prediction_rate >= 0.7 and precision_delta_vs_frequency <= 0:
        return "drop", notes
    return "watch", notes


def is_bridge_signal(signal_name: str) -> bool:
    return signal_name in {"bridge", "composition", "position", "head_tail_link"}


def all_signal_names() -> List[str]:
    return [definition.name for definition in SIGNAL_DEFINITIONS]


def recent_window_size(test_size: int) -> int:
    return max(5, min(20, test_size // 4 if test_size >= 8 else test_size))


def signal_leaderboard_sort_key(item: Dict[str, object]) -> Tuple[float, float, float, float, float, float, str]:
    hit_delta = float(item.get("hit_rate_delta_vs_frequency_pct", 0.0))
    precision_delta = float(item.get("precision_delta_vs_frequency_pct", 0.0))
    return (
        -float(item.get("ranking_score", 0.0)),
        -(1.0 if hit_delta > 0 else 0.0),
        -(1.0 if precision_delta > 0 else 0.0),
        -float(item.get("win_rate_vs_frequency_pct", 0.0)),
        -float(item.get("recent_hit_rate_pct", 0.0)),
        -float(item.get("hit_rate_pct", 0.0)),
        str(item.get("signal", "")),
    )


def score_pattern_frequency(results: Sequence[LotteryResult], target_name: str, pattern_fn: Callable[[str], bool]) -> float:
    items = []
    for result in results:
        items.extend(actual_targets(result, target_name))
    if not items:
        return 0.0
    matches = sum(1 for item in items if pattern_fn(item))
    return matches / len(items)


def signal_recent_rows(rows: Sequence[Dict[str, object]], recent_rows: int) -> List[Dict[str, object]]:
    return list(rows[-recent_rows:]) if recent_rows > 0 else []


def signal_prediction_uniqueness(rows: Sequence[Dict[str, object]]) -> Tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    signatures = [prediction_signature(row) for row in rows]
    unique_rate = len(set(signatures)) / len(signatures)
    repeated_rate = 1.0 - unique_rate
    return unique_rate, repeated_rate


def signal_average_score_gap(rows: Sequence[Dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(score_gap(row) for row in rows) / len(rows)


def signal_win_rate_vs_frequency(rows: Sequence[Dict[str, object]]) -> float:
    if not rows:
        return 0.0
    wins = 0.0
    for row in rows:
        signal_pair = (float(row["hit"]), float(row["precision"]))
        freq_pair = (float(row["frequency_hit"]), float(row["frequency_precision"]))
        wins += 1.0 if signal_pair > freq_pair else 0.5 if signal_pair == freq_pair else 0.0
    return wins / len(rows)


def signal_recent_metrics(rows: Sequence[Dict[str, object]]) -> Tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    window = recent_window_size(len(rows))
    recent_rows = rows[-window:]
    recent_hit_rate = sum(int(row["hit"]) for row in recent_rows) / len(recent_rows)
    recent_precision = sum(float(row["precision"]) for row in recent_rows) / len(recent_rows)
    return recent_hit_rate, recent_precision


def signal_pattern_history_score(results: Sequence[LotteryResult], target_name: str, pattern_fn: Callable[[str], bool], recent_window: int = 30, full_weight: float = 0.35, recent_weight: float = 0.65) -> float:
    rows = list(results)
    recent_rows = rows[-recent_window:] if len(rows) > recent_window else rows
    full_score = score_pattern_frequency(rows, target_name, pattern_fn)
    recent_score = score_pattern_frequency(recent_rows, target_name, pattern_fn)
    return full_score * full_weight + recent_score * recent_weight


SignalScoreFn = Callable[[Sequence[LotteryResult], str, str], float]


def signal_score_fn(signal_name: str) -> SignalScoreFn:
    if signal_name == "ensemble":
        return lambda rows, candidate, target_name: clamp(ensemble_signal_score(rows, candidate, target_name))
    definition = next((item for item in SIGNAL_DEFINITIONS if item.name == signal_name), None)
    if definition is None:
        raise ValueError(f"Unsupported signal: {signal_name}")
    return lambda rows, candidate, target_name: clamp(definition.fn(rows, candidate, target_name).score)


def rank_candidates_by_signal(results: Sequence[LotteryResult], target_name: str, signal_name: str, top_k: int) -> List[Tuple[str, float]]:
    width = target_width(target_name)
    scorer = signal_score_fn(signal_name)
    rows = []
    for candidate in candidate_universe(width):
        rows.append((candidate, scorer(results, candidate, target_name)))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows[:top_k]


def rank_candidates_by_frequency(results: Sequence[LotteryResult], target_name: str, top_k: int) -> List[Tuple[str, float]]:
    width = target_width(target_name)
    rows = []
    for candidate in candidate_universe(width):
        rows.append((candidate, normalized_frequency(results, candidate, target_name)))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows[:top_k]


def build_signal_backtest_evaluation(mode: str, signal_name: str, target_name: str, top_k: int, train_size: int, rows: Sequence[Dict[str, object]]) -> SignalBacktestEvaluation:
    test_size = len(rows)
    if not test_size:
        raise ValueError("Không có dữ liệu kiểm định cho signal backtest")
    hit_rate = sum(int(row["hit"]) for row in rows) / test_size
    baseline_hit_rate = sum(float(row["baseline_hit"]) for row in rows) / test_size
    frequency_hit_rate = sum(int(row["frequency_hit"]) for row in rows) / test_size
    precision_at_k = sum(float(row["precision"]) for row in rows) / test_size
    baseline_precision_at_k = sum(float(row["baseline_precision"]) for row in rows) / test_size
    frequency_precision_at_k = sum(float(row["frequency_precision"]) for row in rows) / test_size
    recent_hit_rate, recent_precision_at_k = signal_recent_metrics(rows)
    win_rate_vs_frequency = signal_win_rate_vs_frequency(rows)
    unique_prediction_rate, repeated_prediction_rate = signal_prediction_uniqueness(rows)
    avg_score_gap = signal_average_score_gap(rows)
    hit_rate_delta_vs_frequency = hit_rate - frequency_hit_rate
    precision_delta_vs_frequency = precision_at_k - frequency_precision_at_k
    ranking_score = build_signal_ranking_score(
        hit_delta_vs_frequency=hit_rate_delta_vs_frequency,
        precision_delta_vs_frequency=precision_delta_vs_frequency,
        recent_hit_rate=recent_hit_rate,
        win_rate_vs_frequency=win_rate_vs_frequency,
        unique_prediction_rate=unique_prediction_rate,
        avg_score_gap=avg_score_gap,
    )
    research_verdict, research_notes = signal_research_verdict(
        hit_delta_vs_frequency=hit_rate_delta_vs_frequency,
        precision_delta_vs_frequency=precision_delta_vs_frequency,
        recent_hit_rate=recent_hit_rate,
        win_rate_vs_frequency=win_rate_vs_frequency,
        repeated_prediction_rate=repeated_prediction_rate,
    )
    return SignalBacktestEvaluation(
        target=target_name,
        mode=mode,
        signal=signal_name,
        top_k=top_k,
        train_size=train_size,
        test_size=test_size,
        hit_rate=hit_rate,
        baseline_hit_rate=baseline_hit_rate,
        frequency_hit_rate=frequency_hit_rate,
        precision_at_k=precision_at_k,
        baseline_precision_at_k=baseline_precision_at_k,
        frequency_precision_at_k=frequency_precision_at_k,
        hit_rate_pct=hit_rate * 100.0,
        baseline_hit_rate_pct=baseline_hit_rate * 100.0,
        frequency_hit_rate_pct=frequency_hit_rate * 100.0,
        precision_at_k_pct=precision_at_k * 100.0,
        baseline_precision_at_k_pct=baseline_precision_at_k * 100.0,
        frequency_precision_at_k_pct=frequency_precision_at_k * 100.0,
        hit_rate_delta_vs_baseline_pct=(hit_rate - baseline_hit_rate) * 100.0,
        hit_rate_delta_vs_frequency_pct=hit_rate_delta_vs_frequency * 100.0,
        precision_delta_vs_baseline_pct=(precision_at_k - baseline_precision_at_k) * 100.0,
        precision_delta_vs_frequency_pct=precision_delta_vs_frequency * 100.0,
        recent_hit_rate=recent_hit_rate,
        recent_hit_rate_pct=recent_hit_rate * 100.0,
        recent_precision_at_k=recent_precision_at_k,
        recent_precision_at_k_pct=recent_precision_at_k * 100.0,
        win_rate_vs_frequency=win_rate_vs_frequency,
        win_rate_vs_frequency_pct=win_rate_vs_frequency * 100.0,
        unique_prediction_rate=unique_prediction_rate,
        unique_prediction_rate_pct=unique_prediction_rate * 100.0,
        avg_score_gap=avg_score_gap,
        ranking_score=ranking_score,
        ranking_score_pct=ranking_score * 100.0,
        repeated_prediction_rate=repeated_prediction_rate,
        repeated_prediction_rate_pct=repeated_prediction_rate * 100.0,
        research_verdict=research_verdict,
        research_notes=research_notes,
        bridge_signal=is_bridge_signal(signal_name),
    )

def walkforward_signal_backtest(results: Sequence[LotteryResult], target_name: str, signal_name: str, top_k: int, min_train_size: int = 30) -> List[Dict[str, object]]:
    if len(results) <= min_train_size:
        raise ValueError("Không đủ dữ liệu cho signal walk-forward evaluation")
    width = target_width(target_name)
    rows: List[Dict[str, object]] = []
    for split_index in range(min_train_size, len(results)):
        train = list(results[:split_index])
        test_result = results[split_index]
        predicted_rows = rank_candidates_by_signal(train, target_name=target_name, signal_name=signal_name, top_k=top_k)
        frequency_rows = rank_candidates_by_frequency(train, target_name=target_name, top_k=top_k)
        predicted = [candidate for candidate, _ in predicted_rows]
        frequency_predicted = [candidate for candidate, _ in frequency_rows]
        actual = actual_targets(test_result, target_name)
        universe_size = 10 ** width
        hit, baseline_hit, precision, baseline_precision = evaluate_prediction_set(predicted, actual, universe_size)
        freq_hit, _, freq_precision, _ = evaluate_prediction_set(frequency_predicted, actual, universe_size)
        rows.append(
            {
                "date": test_result.date,
                "target": target_name,
                "signal": signal_name,
                "top_k": top_k,
                "hit": hit,
                "baseline_hit": baseline_hit,
                "frequency_hit": freq_hit,
                "precision": precision,
                "baseline_precision": baseline_precision,
                "frequency_precision": freq_precision,
                "predicted": predicted,
                "predicted_scores": [{"candidate": candidate, "score": score} for candidate, score in predicted_rows],
                "frequency_predicted": frequency_predicted,
                "actual": actual,
            }
        )
    return rows


def signal_backtest(results: Sequence[LotteryResult], target_name: str, signal_name: str, top_k: int, min_train_size: int = 30, recent_rows: int = 10) -> Dict[str, object]:
    rows = walkforward_signal_backtest(results, target_name=target_name, signal_name=signal_name, top_k=top_k, min_train_size=min_train_size)
    evaluation = build_signal_backtest_evaluation("single", signal_name, target_name, top_k, min_train_size, rows)
    return {
        "mode": "single",
        "target": target_name,
        "signal": signal_name,
        "top_k": top_k,
        "min_train_size": min_train_size,
        "dataset_size": len(results),
        "evaluation": asdict(evaluation),
        "rows": rows[-recent_rows:],
    }


def build_group_backtest_summary(evaluations: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    definition_by_name = {definition.name: definition for definition in SIGNAL_DEFINITIONS}
    for item in evaluations:
        signal_name = str(item.get("signal", ""))
        definition = definition_by_name.get(signal_name)
        if definition is None:
            continue
        grouped.setdefault(definition.group, []).append(item)
    summary_rows: List[Dict[str, object]] = []
    for group_name in signal_group_names():
        items = grouped.get(group_name, [])
        if not items:
            continue
        keep_count = sum(1 for item in items if item.get("research_verdict") == "keep")
        watch_count = sum(1 for item in items if item.get("research_verdict") == "watch")
        drop_count = sum(1 for item in items if item.get("research_verdict") == "drop")
        mean_hit_rate = sum(float(item.get("hit_rate", 0.0)) for item in items) / len(items)
        mean_precision = sum(float(item.get("precision_at_k", 0.0)) for item in items) / len(items)
        mean_hit_delta = sum(float(item.get("hit_rate_delta_vs_frequency_pct", 0.0)) for item in items) / len(items)
        mean_precision_delta = sum(float(item.get("precision_delta_vs_frequency_pct", 0.0)) for item in items) / len(items)
        mean_ranking_score = sum(float(item.get("ranking_score", 0.0)) for item in items) / len(items)
        if keep_count > 0 and mean_hit_delta >= 0:
            verdict = "keep"
        elif drop_count == len(items) or (mean_hit_delta < 0 and mean_precision_delta <= 0):
            verdict = "drop"
        else:
            verdict = "watch"
        summary_rows.append(
            {
                "group": group_name,
                "signals": [item["signal"] for item in items],
                "signal_count": len(items),
                "keep_count": keep_count,
                "watch_count": watch_count,
                "drop_count": drop_count,
                "mean_hit_rate": mean_hit_rate,
                "mean_hit_rate_pct": mean_hit_rate * 100.0,
                "mean_precision_at_k": mean_precision,
                "mean_precision_at_k_pct": mean_precision * 100.0,
                "mean_hit_delta_vs_frequency_pct": mean_hit_delta,
                "mean_precision_delta_vs_frequency_pct": mean_precision_delta,
                "mean_ranking_score": mean_ranking_score,
                "mean_ranking_score_pct": mean_ranking_score * 100.0,
                "research_verdict": verdict,
            }
        )
    summary_rows.sort(key=lambda item: (-float(item["mean_ranking_score"]), -float(item["mean_hit_rate"]), str(item["group"])))
    verdict_counts = {
        "keep": sum(1 for item in summary_rows if item.get("research_verdict") == "keep"),
        "watch": sum(1 for item in summary_rows if item.get("research_verdict") == "watch"),
        "drop": sum(1 for item in summary_rows if item.get("research_verdict") == "drop"),
    }
    return {
        "catalog": signal_group_catalog(),
        "groups": summary_rows,
        "verdict_counts": verdict_counts,
        "kept_groups": [item["group"] for item in summary_rows if item.get("research_verdict") == "keep"],
        "dropped_groups": [item["group"] for item in summary_rows if item.get("research_verdict") == "drop"],
    }


def all_signals_backtest(results: Sequence[LotteryResult], target_name: str, top_k: int, min_train_size: int = 30, recent_rows: int = 5) -> Dict[str, object]:
    evaluations: List[Dict[str, object]] = []
    best_signal_rows: Dict[str, List[Dict[str, object]]] = {}
    for signal_name in all_signal_names():
        rows = walkforward_signal_backtest(results, target_name=target_name, signal_name=signal_name, top_k=top_k, min_train_size=min_train_size)
        evaluation = build_signal_backtest_evaluation("all", signal_name, target_name, top_k, min_train_size, rows)
        evaluations.append(asdict(evaluation))
        best_signal_rows[signal_name] = signal_recent_rows(rows, recent_rows)
    evaluations.sort(key=signal_leaderboard_sort_key)
    best_signal = evaluations[0]["signal"] if evaluations else None
    verdict_counts = {
        "keep": sum(1 for item in evaluations if item.get("research_verdict") == "keep"),
        "watch": sum(1 for item in evaluations if item.get("research_verdict") == "watch"),
        "drop": sum(1 for item in evaluations if item.get("research_verdict") == "drop"),
    }
    bridge_focus = [item for item in evaluations if item.get("bridge_signal")]
    group_summary = build_group_backtest_summary(evaluations)
    return {
        "mode": "all",
        "target": target_name,
        "top_k": top_k,
        "min_train_size": min_train_size,
        "dataset_size": len(results),
        "evaluations": evaluations,
        "best_signal": best_signal,
        "rows": best_signal_rows.get(best_signal, []),
        "verdict_counts": verdict_counts,
        "bridge_focus": bridge_focus,
        "group_summary": group_summary,
    }


def signal_group_backtest(results: Sequence[LotteryResult], target_name: str, top_k: int, min_train_size: int = 30, recent_rows: int = 5) -> Dict[str, object]:
    payload = all_signals_backtest(results, target_name=target_name, top_k=top_k, min_train_size=min_train_size, recent_rows=recent_rows)
    return {
        "mode": "groups",
        "target": target_name,
        "top_k": top_k,
        "min_train_size": min_train_size,
        "dataset_size": len(results),
        "group_summary": payload["group_summary"],
        "evaluations": payload["evaluations"],
    }


def signals_for_research_verdict(payload: Dict[str, object], verdict: str) -> List[str]:
    evaluations = payload.get("evaluations", [])
    return [str(item.get("signal")) for item in evaluations if item.get("research_verdict") == verdict]


def signal_groups_for_research_verdict(payload: Dict[str, object], verdict: str) -> List[str]:
    group_summary = payload.get("group_summary", {})
    groups = group_summary.get("groups", []) if isinstance(group_summary, dict) else []
    return [str(item.get("group")) for item in groups if item.get("research_verdict") == verdict]


def kept_signal_names_by_group(payload: Dict[str, object]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    definition_groups = {definition.name: definition.group for definition in SIGNAL_DEFINITIONS}
    for signal_name in signals_for_research_verdict(payload, "keep"):
        group_name = definition_groups.get(signal_name)
        if group_name is None:
            continue
        grouped.setdefault(group_name, []).append(signal_name)
    return {group_name: sorted(names) for group_name, names in sorted(grouped.items())}


def kept_signal_filter_summary(payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "kept_signals": signals_for_research_verdict(payload, "keep"),
        "watch_signals": signals_for_research_verdict(payload, "watch"),
        "dropped_signals": signals_for_research_verdict(payload, "drop"),
        "kept_groups": signal_groups_for_research_verdict(payload, "keep"),
        "watch_groups": signal_groups_for_research_verdict(payload, "watch"),
        "dropped_groups": signal_groups_for_research_verdict(payload, "drop"),
        "kept_signals_by_group": kept_signal_names_by_group(payload),
    }



def all_signals_backtest(results: Sequence[LotteryResult], target_name: str, top_k: int, min_train_size: int = 30, recent_rows: int = 5) -> Dict[str, object]:
    evaluations: List[Dict[str, object]] = []
    best_signal_rows: Dict[str, List[Dict[str, object]]] = {}
    for signal_name in all_signal_names():
        rows = walkforward_signal_backtest(results, target_name=target_name, signal_name=signal_name, top_k=top_k, min_train_size=min_train_size)
        evaluation = build_signal_backtest_evaluation("all", signal_name, target_name, top_k, min_train_size, rows)
        evaluations.append(asdict(evaluation))
        best_signal_rows[signal_name] = signal_recent_rows(rows, recent_rows)
    evaluations.sort(key=signal_leaderboard_sort_key)
    best_signal = evaluations[0]["signal"] if evaluations else None
    verdict_counts = {
        "keep": sum(1 for item in evaluations if item.get("research_verdict") == "keep"),
        "watch": sum(1 for item in evaluations if item.get("research_verdict") == "watch"),
        "drop": sum(1 for item in evaluations if item.get("research_verdict") == "drop"),
    }
    bridge_focus = [item for item in evaluations if item.get("bridge_signal")]
    group_summary = build_group_backtest_summary(evaluations)
    return {
        "mode": "all",
        "target": target_name,
        "top_k": top_k,
        "min_train_size": min_train_size,
        "dataset_size": len(results),
        "evaluations": evaluations,
        "best_signal": best_signal,
        "rows": best_signal_rows.get(best_signal, []),
        "verdict_counts": verdict_counts,
        "bridge_focus": bridge_focus,
        "group_summary": group_summary,
        "filter_summary": kept_signal_filter_summary({
            "evaluations": evaluations,
            "group_summary": group_summary,
        }),
    }




def ensemble_backtest(results: Sequence[LotteryResult], target_name: str, top_k: int, min_train_size: int = 30, recent_rows: int = 10) -> Dict[str, object]:
    rows = walkforward_signal_backtest(results, target_name=target_name, signal_name="ensemble", top_k=top_k, min_train_size=min_train_size)
    evaluation = build_signal_backtest_evaluation("ensemble", "ensemble", target_name, top_k, min_train_size, rows)
    return {
        "mode": "ensemble",
        "target": target_name,
        "signal": "ensemble",
        "top_k": top_k,
        "min_train_size": min_train_size,
        "dataset_size": len(results),
        "evaluation": asdict(evaluation),
        "rows": rows[-recent_rows:],
    }


def evaluate_ranking_backtest(results: Sequence[LotteryResult], split_ratio: float, target_name: str, top_k: int) -> RankingEvaluation:
    if len(results) < 5:
        raise ValueError("Cần ít nhất 5 kỳ quay để đánh giá")
    split_index = max(1, int(len(results) * split_ratio))
    if split_index >= len(results):
        split_index = len(results) - 1
    train, test = list(results[:split_index]), list(results[split_index:])
    model = train_ranking_model(train, target_name=target_name, top_k=top_k)
    weighted_predicted = [number for number, _ in model.predict()]
    baseline_predicted = [number for number, _ in model.predict_baseline()]
    universe_size = 10 ** model.number_width

    hits = 0
    baseline_hit_rate_total = 0.0
    frequency_hits = 0
    precision_total = 0.0
    baseline_precision_total = 0.0
    frequency_precision_total = 0.0
    for result in test:
        actual = actual_targets(result, target_name)
        hit, baseline_hit, precision, baseline_precision = evaluate_prediction_set(weighted_predicted, actual, universe_size)
        freq_hit, _, freq_precision, _ = evaluate_prediction_set(baseline_predicted, actual, universe_size)
        hits += hit
        baseline_hit_rate_total += baseline_hit
        frequency_hits += freq_hit
        precision_total += precision
        baseline_precision_total += baseline_precision
        frequency_precision_total += freq_precision

    return build_evaluation(
        model_name="weighted-ranking",
        target_name=target_name,
        train_size=len(train),
        test_size=len(test),
        top_k=top_k,
        hit_rate=hits / len(test),
        baseline_hit_rate=baseline_hit_rate_total / len(test),
        frequency_hit_rate=frequency_hits / len(test),
        precision_at_k=precision_total / len(test),
        baseline_precision_at_k=baseline_precision_total / len(test),
        frequency_precision_at_k=frequency_precision_total / len(test),
    )


def evaluate_named_ranking_backtest(results: Sequence[LotteryResult], split_ratio: float, target_name: str, top_k: int, min_train_size: int = 30, model_name: str = "logistic", selected_signal_names: Sequence[str] | None = None) -> RankingEvaluation:
    from .models.sklearn_ranker import fit_named_sklearn_ranking_model, model_label

    if len(results) < 5:
        raise ValueError("Cần ít nhất 5 kỳ quay để đánh giá")
    split_index = max(min_train_size, int(len(results) * split_ratio))
    if split_index >= len(results):
        split_index = len(results) - 1
    train, test = list(results[:split_index]), list(results[split_index:])
    sklearn_model = fit_named_sklearn_ranking_model(
        train,
        target_name=target_name,
        top_k=top_k,
        min_train_size=min_train_size,
        model_name=model_name,
        selected_signal_names=selected_signal_names,
    )
    weighted_model = train_ranking_model(train, target_name=target_name, top_k=top_k)
    predicted = [number for number, _ in sklearn_model.predict()]
    frequency_predicted = [number for number, _ in weighted_model.predict_baseline()]
    universe_size = 10 ** sklearn_model.number_width

    hits = 0
    baseline_hit_rate_total = 0.0
    frequency_hits = 0
    precision_total = 0.0
    baseline_precision_total = 0.0
    frequency_precision_total = 0.0
    for result in test:
        actual = actual_targets(result, target_name)
        hit, baseline_hit, precision, baseline_precision = evaluate_prediction_set(predicted, actual, universe_size)
        freq_hit, _, freq_precision, _ = evaluate_prediction_set(frequency_predicted, actual, universe_size)
        hits += hit
        baseline_hit_rate_total += baseline_hit
        frequency_hits += freq_hit
        precision_total += precision
        baseline_precision_total += baseline_precision
        frequency_precision_total += freq_precision

    return build_evaluation(
        model_name=model_label(model_name),
        target_name=target_name,
        train_size=len(train),
        test_size=len(test),
        top_k=top_k,
        hit_rate=hits / len(test),
        baseline_hit_rate=baseline_hit_rate_total / len(test),
        frequency_hit_rate=frequency_hits / len(test),
        precision_at_k=precision_total / len(test),
        baseline_precision_at_k=baseline_precision_total / len(test),
        frequency_precision_at_k=frequency_precision_total / len(test),
    )


def evaluate_sklearn_backtest(results: Sequence[LotteryResult], split_ratio: float, target_name: str, top_k: int, min_train_size: int = 30) -> RankingEvaluation:
    return evaluate_named_ranking_backtest(results, split_ratio=split_ratio, target_name=target_name, top_k=top_k, min_train_size=min_train_size, model_name="logistic")


def evaluate_tuned_weighted_backtest(results: Sequence[LotteryResult], split_ratio: float, target_name: str, top_k: int, min_train_size: int = 30) -> RankingEvaluation:
    if len(results) < 5:
        raise ValueError("Cần ít nhất 5 kỳ quay để đánh giá")
    split_index = max(min_train_size, int(len(results) * split_ratio))
    if split_index >= len(results):
        split_index = len(results) - 1
    train, test = list(results[:split_index]), list(results[split_index:])
    model = fit_tuned_ranking_model(train, target_name=target_name, top_k=top_k, min_train_size=min_train_size)
    predicted = [number for number, _ in model.predict()]
    baseline_predicted = [number for number, _ in model.predict_baseline()]
    universe_size = 10 ** model.number_width

    hits = 0
    baseline_hit_rate_total = 0.0
    frequency_hits = 0
    precision_total = 0.0
    baseline_precision_total = 0.0
    frequency_precision_total = 0.0
    for result in test:
        actual = actual_targets(result, target_name)
        hit, baseline_hit, precision, baseline_precision = evaluate_prediction_set(predicted, actual, universe_size)
        freq_hit, _, freq_precision, _ = evaluate_prediction_set(baseline_predicted, actual, universe_size)
        hits += hit
        baseline_hit_rate_total += baseline_hit
        frequency_hits += freq_hit
        precision_total += precision
        baseline_precision_total += baseline_precision
        frequency_precision_total += freq_precision

    return build_evaluation(
        model_name="tuned-weighted-ranking",
        target_name=target_name,
        train_size=len(train),
        test_size=len(test),
        top_k=top_k,
        hit_rate=hits / len(test),
        baseline_hit_rate=baseline_hit_rate_total / len(test),
        frequency_hit_rate=frequency_hits / len(test),
        precision_at_k=precision_total / len(test),
        baseline_precision_at_k=baseline_precision_total / len(test),
        frequency_precision_at_k=frequency_precision_total / len(test),
    )


def loto2_input_configurations(results: Sequence[LotteryResult], top_k: int, min_train_size: int) -> List[Dict[str, object]]:
    payload = all_signals_backtest(results, target_name="loto2", top_k=top_k, min_train_size=min_train_size, recent_rows=5)
    filter_summary = payload.get("filter_summary", {}) if isinstance(payload, dict) else {}
    kept_groups = list(filter_summary.get("kept_groups", [])) if isinstance(filter_summary, dict) else []
    kept_signals = list(filter_summary.get("kept_signals", [])) if isinstance(filter_summary, dict) else []
    top_signals = [str(item.get("signal")) for item in payload.get("evaluations", [])[:3]]
    configs = [
        {
            "name": "no_signals",
            "label": "No signal features",
            "selected_signal_names": [],
            "groups": [],
        },
        {
            "name": "default_signals",
            "label": "Default active signal features",
            "selected_signal_names": None,
            "groups": [],
        },
        {
            "name": "kept_groups",
            "label": "Signals from kept groups",
            "selected_signal_names": signal_names_for_groups(kept_groups),
            "groups": kept_groups,
        },
        {
            "name": "kept_signals",
            "label": "Only kept signals",
            "selected_signal_names": resolved_model_signal_names("loto2", kept_signals),
            "groups": sorted({definition.group for definition in SIGNAL_DEFINITIONS if definition.name in kept_signals}),
        },
        {
            "name": "top_signals",
            "label": "Top-ranked signal shortlist",
            "selected_signal_names": resolved_model_signal_names("loto2", top_signals),
            "groups": sorted({definition.group for definition in SIGNAL_DEFINITIONS if definition.name in top_signals}),
        },
    ]
    unique_configs: List[Dict[str, object]] = []
    seen: set[tuple[str, ...] | None] = set()
    for config in configs:
        selected = config["selected_signal_names"]
        key = None if selected is None else tuple(selected)
        if key in seen:
            continue
        seen.add(key)
        unique_configs.append(config)
    return unique_configs


def compare_loto2_input_models(results: Sequence[LotteryResult], split_ratio: float, top_k: int, min_train_size: int = 30) -> Dict[str, object]:
    from .models.sklearn_ranker import benchmarkable_model_names

    configurations = loto2_input_configurations(results, top_k=top_k, min_train_size=min_train_size)
    evaluations: List[Dict[str, object]] = []
    for config in configurations:
        for model_name in benchmarkable_model_names("loto2"):
            metrics = evaluate_named_ranking_backtest(
                results,
                split_ratio=split_ratio,
                target_name="loto2",
                top_k=top_k,
                min_train_size=min_train_size,
                model_name=model_name,
                selected_signal_names=config["selected_signal_names"],
            )
            row = asdict(metrics)
            row["input_config"] = config["name"]
            row["input_label"] = config["label"]
            row["signal_names"] = config["selected_signal_names"]
            row["group_names"] = config["groups"]
            evaluations.append(row)
    evaluations.sort(key=lambda item: (-float(item["hit_rate"]), -float(item["precision_at_k"]), str(item["input_config"]), str(item["model"])))
    return {
        "target": "loto2",
        "dataset_size": len(results),
        "split_ratio": split_ratio,
        "top_k": top_k,
        "min_train_size": min_train_size,
        "configurations": configurations,
        "evaluations": evaluations,
        "best_configuration": evaluations[0] if evaluations else None,
    }


def compare_loto2_models(results: Sequence[LotteryResult], split_ratio: float, top_k: int, min_train_size: int = 30, include_window_benchmark: bool = True) -> Dict[str, object]:
    from .models.sklearn_ranker import benchmarkable_model_names

    target_name = "loto2"
    weighted = evaluate_ranking_backtest(results, split_ratio=split_ratio, target_name=target_name, top_k=top_k)
    tuned_weighted = evaluate_tuned_weighted_backtest(results, split_ratio=split_ratio, target_name=target_name, top_k=top_k, min_train_size=min_train_size)
    candidates = [asdict(weighted), asdict(tuned_weighted)]
    for name in benchmarkable_model_names(target_name):
        candidates.append(asdict(evaluate_named_ranking_backtest(results, split_ratio=split_ratio, target_name=target_name, top_k=top_k, min_train_size=min_train_size, model_name=name)))
    candidates.sort(key=lambda item: (-float(item["hit_rate"]), -float(item["precision_at_k"]), str(item["model"])))
    input_benchmark = compare_loto2_input_models(results, split_ratio=split_ratio, top_k=top_k, min_train_size=min_train_size)
    return {
        "target": target_name,
        "dataset_size": len(results),
        "split_ratio": split_ratio,
        "top_k": top_k,
        "min_train_size": min_train_size,
        "weight_tuning": compare_loto2_weight_strategies(results, top_k=top_k, min_train_size=min_train_size),
        "evaluations": candidates,
        "best_model": candidates[0]["model"] if candidates else None,
        "input_benchmark": input_benchmark,
        "window_benchmark": benchmark_loto2_data_windows(results, split_ratio=split_ratio, top_k=top_k, min_train_size=min_train_size) if include_window_benchmark and len(results) > 160 else None,
    }


def compare_models(results: Sequence[LotteryResult], split_ratio: float, top_k: int, min_train_size: int = 30, targets: Sequence[str] = ("loto2", "loto3", "special2", "special3")) -> Dict[str, object]:
    comparisons: List[Dict[str, object]] = []
    for target_name in targets:
        weighted = evaluate_ranking_backtest(results, split_ratio=split_ratio, target_name=target_name, top_k=top_k)
        sklearn = evaluate_sklearn_backtest(
            results,
            split_ratio=split_ratio,
            target_name=target_name,
            top_k=top_k,
            min_train_size=min_train_size,
        )
        best_hit_rate_model = "tie" if weighted.hit_rate == sklearn.hit_rate else (weighted.model if weighted.hit_rate > sklearn.hit_rate else sklearn.model)
        best_precision_model = "tie" if weighted.precision_at_k == sklearn.precision_at_k else (weighted.model if weighted.precision_at_k > sklearn.precision_at_k else sklearn.model)
        row = {
            "target": target_name,
            "weighted": asdict(weighted),
            "sklearn": asdict(sklearn),
            "best_hit_rate_model": best_hit_rate_model,
            "best_precision_model": best_precision_model,
        }
        if target_name == "loto2":
            row["loto2_benchmark"] = compare_loto2_models(results, split_ratio=split_ratio, top_k=top_k, min_train_size=min_train_size)
        comparisons.append(row)
    return {
        "dataset_size": len(results),
        "split_ratio": split_ratio,
        "top_k": top_k,
        "min_train_size": min_train_size,
        "comparisons": comparisons,
    }




def walkforward_ranking_backtest(results: Sequence[LotteryResult], target_name: str, top_k: int, min_train_size: int = 30) -> List[Dict[str, object]]:
    if len(results) <= min_train_size:
        raise ValueError("Không đủ dữ liệu cho walk-forward evaluation")
    rows: List[Dict[str, object]] = []
    for split_index in range(min_train_size, len(results)):
        train = list(results[:split_index])
        test_result = results[split_index]
        model = train_ranking_model(train, target_name=target_name, top_k=top_k)
        weighted_predicted = [number for number, _ in model.predict()]
        baseline_predicted = [number for number, _ in model.predict_baseline()]
        universe_size = 10 ** model.number_width
        actual = actual_targets(test_result, target_name)
        hit, baseline_hit, precision, baseline_precision = evaluate_prediction_set(weighted_predicted, actual, universe_size)
        freq_hit, _, freq_precision, _ = evaluate_prediction_set(baseline_predicted, actual, universe_size)
        rows.append(
            {
                "date": test_result.date,
                "target": target_name,
                "top_k": top_k,
                "hit": hit,
                "baseline_hit": baseline_hit,
                "frequency_hit": freq_hit,
                "precision": precision,
                "baseline_precision": baseline_precision,
                "frequency_precision": freq_precision,
                "predicted": weighted_predicted,
                "frequency_predicted": baseline_predicted,
                "actual": actual,
            }
        )
    return rows



def build_dashboard_payload(results: Sequence[LotteryResult], split_ratio: float = 0.8, top_k: int = 5, min_train_size: int = 30) -> Dict[str, object]:
    predict_payload = predict_next_day(
        results,
        top_k_loto2=top_k,
        top_k_loto3=top_k,
        top_k_special2=top_k,
        top_k_special3=top_k,
    )
    compare_payload = compare_models(results, split_ratio=split_ratio, top_k=top_k, min_train_size=min_train_size)
    walkforward_min_train_size = max(min_train_size, len(results) - 40)
    walkforward_rows = walkforward_ranking_backtest(results, target_name="loto2", top_k=top_k, min_train_size=walkforward_min_train_size)
    latest_date = results[-1].date if results else None
    return {
        "overview": {
            "dataset_size": len(results),
            "latest_date": latest_date,
            "top_k": top_k,
            "min_train_size": min_train_size,
            "walkforward_min_train_size": walkforward_min_train_size,
            "split_ratio": split_ratio,
        },
        "predictions": predict_payload,
        "compare": compare_payload,
        "walkforward": {
            "target": "loto2",
            "rows": walkforward_rows,
        },
    }


def build_bridge_payload(results: Sequence[LotteryResult], candidates: Sequence[str]) -> List[Dict[str, object]]:
    from .features import bridge_frequency, bridge_streak, digit_part_frequency, digit_position_frequency, digit_transition_score

    payload: List[Dict[str, object]] = []
    rows = list(results)
    for candidate in candidates:
        target_name = "loto3" if len(candidate) >= 3 else "loto2"
        payload.append(
            {
                "candidate": candidate,
                "bridge_frequency": bridge_frequency(rows, candidate),
                "bridge_streak": bridge_streak(rows, candidate),
                "digit_position_frequency": digit_position_frequency(rows, candidate, target_name),
                "digit_part_frequency": digit_part_frequency(rows, candidate, target_name),
                "digit_transition_score": digit_transition_score(rows, candidate, target_name),
            }
        )
    return payload


def enrich_dashboard_payload(results: Sequence[LotteryResult], payload: Dict[str, object]) -> Dict[str, object]:
    loto2_candidates = [number for number, _ in payload["predictions"]["loto2_top"]]
    loto3_candidates = [number for number, _ in payload["predictions"]["loto3_top"]]
    payload["bridge_signals"] = {
        "loto2": build_bridge_payload(results, loto2_candidates),
        "loto3": build_bridge_payload(results, loto3_candidates),
    }
    payload["signal_engine"] = build_signal_payload(results, payload["predictions"])
    return payload


def build_signal_backtests_payload(results: Sequence[LotteryResult], top_k: int = 5, min_train_size: int = 30, targets: Sequence[str] = ("loto2", "loto3", "special2", "special3")) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for target_name in targets:
        all_payload = all_signals_backtest(results, target_name=target_name, top_k=top_k, min_train_size=min_train_size, recent_rows=5)
        ensemble_payload = ensemble_backtest(results, target_name=target_name, top_k=top_k, min_train_size=min_train_size, recent_rows=5)
        payload[target_name] = {
            "all": all_payload,
            "ensemble": ensemble_payload,
        }
    return payload


def build_full_dashboard_payload(results: Sequence[LotteryResult], split_ratio: float = 0.8, top_k: int = 5, min_train_size: int = 30) -> Dict[str, object]:
    payload = enrich_dashboard_payload(results, build_dashboard_payload(results, split_ratio=split_ratio, top_k=top_k, min_train_size=min_train_size))
    payload["signal_backtests"] = build_signal_backtests_payload(results, top_k=top_k, min_train_size=min_train_size)
    return payload
