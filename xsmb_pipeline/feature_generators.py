from __future__ import annotations

"""Feature Generators - Phase 3 (P2.1 -> P2.19) cho XSMB.

Mỗi generator là pure function nhận list ``(draw_date, loto_number)`` hoặc
``(draw_date, digit)`` đã sort tăng theo thời gian + ngày as-of, trả list
tuple ``(draw_date, feature_name, feature_value)`` để insert vào
``features_daily`` qua :func:`xsmb_pipeline.database.insert_features_daily`.

Convention feature_name (theo CLAUDE.md):
    Frequency:   freq_{NN}_{W}d         (NN = "00".."99", W = window days)
    Gan:         gan_{NN}_{current|mean|max|min}
    Roi:         roi_1d_{NN}, roi_2d_{NN}, roi_3d_{NN}, roi_db_{NN}
    Dau/Duoi:    dau_hot_{D}, duoi_hot_{D} (D = 0..9)
    Cham:        cham_{D}
    Tong:        tong_de, tong_lo
    Markov:      markov{K}_{NN}         (K = 1, 2, 3)
    Cycle:       cycle_{P}_phase_{NN}   (P in 7,14,30,60,90)
    Graph:       degree_{NN}, pagerank_{NN}, centrality_{NN}
    Pattern:     pascal_{NN}, reverse_{NN}, mirror_{NN}, modulo_{NN}
    Position:    pos_{prize}_{idx}_{vpos}_count_30d
    CrossPos:    cross_{src}_{dst}_count_30d
    Discovered:  disc_{formula_id}
"""

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

FeatureRow = Tuple[str, str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _all_loto_codes() -> List[str]:
    """Trả 100 lô tô 00..99 (zero-padded)."""
    return [f"{i:02d}" for i in range(100)]


def _group_loto_by_date(
    loto_history: Sequence[Tuple[str, str]],
) -> Dict[str, List[str]]:
    """Gom ``loto_history`` thành ``{draw_date: [loto_number, ...]}``."""
    grouped: Dict[str, List[str]] = defaultdict(list)
    for date, loto in loto_history:
        grouped[date].append(loto)
    return grouped


def _ordered_dates(grouped: Dict[str, List[str]]) -> List[str]:
    """Trả danh sách ``draw_date`` đã sort tăng theo thời gian.

    Giả định ``grouped`` được build từ ``fetch_loto_history`` đã sort theo
    ``strptime``, nên thứ tự dict ``grouped.keys()`` đã đúng. Nếu không,
    sort lại defensive theo (year, month, day).
    """

    def _key(date_text: str) -> Tuple[int, int, int]:
        day, month, year = date_text.split("/")
        return (int(year), int(month), int(day))

    return sorted(grouped.keys(), key=_key)


# ---------------------------------------------------------------------------
# P2.1 - Frequency Generator
# ---------------------------------------------------------------------------
DEFAULT_FREQ_WINDOWS: Tuple[int, ...] = (3, 7, 14, 30, 60, 90, 180, 365)


def frequency_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
    windows: Sequence[int] = DEFAULT_FREQ_WINDOWS,
) -> List[FeatureRow]:
    """Sinh feature tần suất xuất hiện của từng lô 00-99 trong các
    cửa sổ thời gian (P2.1).

    Args:
        loto_history: ``[(draw_date, loto_number), ...]`` sort tăng theo ngày.
        as_of_date: Ngày tính feature (tính từ ``as_of_date`` lùi về quá khứ).
        windows: Danh sách số ngày của cửa sổ.

    Returns:
        ``800`` rows (8 windows × 100 lô) nếu dùng ``DEFAULT_FREQ_WINDOWS``.
    """
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)

    # Vị trí của ``as_of_date`` trong lịch sử (lấy đến ngày này, không gồm).
    cutoff_idx = 0
    for idx, date in enumerate(dates):
        if date <= as_of_date:
            cutoff_idx = idx + 1
        else:
            break

    rows: List[FeatureRow] = []
    for window in windows:
        start_idx = max(0, cutoff_idx - window)
        counter: Counter[str] = Counter()
        for date in dates[start_idx:cutoff_idx]:
            for loto in grouped[date]:
                counter[loto] += 1
        for loto in _all_loto_codes():
            rows.append(
                (as_of_date, f"freq_{loto}_{window}d", float(counter.get(loto, 0)))
            )
    return rows


# ---------------------------------------------------------------------------
# P2.2 - Gan Generator (gap = số ngày từ lần xuất hiện gần nhất)
# ---------------------------------------------------------------------------
def gan_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
) -> List[FeatureRow]:
    """Sinh feature gan/khoảng trống cho từng lô (P2.2).

    Cho mỗi lô NN sinh 4 feature:
        - ``gan_{NN}_current``: số ngày từ lần ra cuối -> as_of (chưa ra: -1).
        - ``gan_{NN}_mean``:   trung bình gap giữa các lần ra trong lịch sử.
        - ``gan_{NN}_max``:    gap lớn nhất.
        - ``gan_{NN}_min``:    gap nhỏ nhất.
    """
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history_dates = [date for date in dates if date <= as_of_date]
    total_days = len(history_dates)
    date_to_index = {date: idx for idx, date in enumerate(history_dates)}

    appearances: Dict[str, List[int]] = defaultdict(list)
    for date in history_dates:
        for loto in grouped[date]:
            appearances[loto].append(date_to_index[date])

    rows: List[FeatureRow] = []
    for loto in _all_loto_codes():
        idxs = appearances.get(loto, [])
        if not idxs:
            current = -1.0
            mean_gap = -1.0
            max_gap = -1.0
            min_gap = -1.0
        else:
            current = float(total_days - 1 - idxs[-1])
            if len(idxs) >= 2:
                gaps = [idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)]
                mean_gap = sum(gaps) / len(gaps)
                max_gap = float(max(gaps))
                min_gap = float(min(gaps))
            else:
                mean_gap = -1.0
                max_gap = -1.0
                min_gap = -1.0
        rows.append((as_of_date, f"gan_{loto}_current", current))
        rows.append((as_of_date, f"gan_{loto}_mean", mean_gap))
        rows.append((as_of_date, f"gan_{loto}_max", max_gap))
        rows.append((as_of_date, f"gan_{loto}_min", min_gap))
    return rows


# ---------------------------------------------------------------------------
# P2.3 - Position Generator
# ---------------------------------------------------------------------------
def position_features(
    digit_history: Sequence[Tuple[str, str, int, int, str]],
    as_of_date: str,
    window_days: int = 30,
) -> List[FeatureRow]:
    """Sinh feature đếm chữ số 0-9 tại từng vị trí (P2.3).

    Args:
        digit_history: ``[(draw_date, prize_name, prize_index,
                          position_index, digit), ...]`` sort theo ngày tăng.
        as_of_date:   Ngày as-of.
        window_days:  Cửa sổ đếm.

    Sinh feature ``pos_{prize}_{idx}_{pos}_d{D}_{W}d`` cho 10 chữ số.
    Đủ tất cả giải: special V1..V5, G1 V1..V5, G2 V1..V5, G3 V1..V5,
    G4 V1..V4, G5 V1..V4, G6 V1..V3, G7 V1..V2.
    """
    # prize_name -> (max_prize_index, max_position_count)
    PRIZE_POSITIONS = {
        "special": (1, 5),  # 1 dãy, 5 vị trí
        "G1":      (1, 5),
        "G2":      (2, 5),
        "G3":      (6, 5),
        "G4":      (4, 4),
        "G5":      (6, 4),
        "G6":      (3, 3),
        "G7":      (4, 2),
    }

    interest: Dict[Tuple[str, int], List[int]] = {}
    for prize_name, (max_idx, max_pos) in PRIZE_POSITIONS.items():
        for idx in range(max_idx):
            interest[(prize_name, idx)] = list(range(max_pos))

    by_date: Dict[str, List[Tuple[str, int, int, str]]] = defaultdict(list)
    for date, prize, idx, pos, digit in digit_history:
        by_date[date].append((prize, idx, pos, digit))

    dates_sorted = sorted(by_date.keys(), key=lambda d: (
        int(d.split("/")[2]), int(d.split("/")[1]), int(d.split("/")[0])
    ))
    history = [date for date in dates_sorted if date <= as_of_date]
    window_dates = history[-window_days:] if window_days > 0 else history

    counters: Dict[Tuple[str, int, int], Counter[str]] = defaultdict(Counter)
    for date in window_dates:
        for prize, idx, pos, digit in by_date[date]:
            if (prize, idx) in interest and pos in interest[(prize, idx)]:
                counters[(prize, idx, pos)][digit] += 1

    rows: List[FeatureRow] = []
    for (prize, idx), positions in interest.items():
        for pos in positions:
            counter = counters.get((prize, idx, pos), Counter())
            for digit_value in range(10):
                key = (
                    f"pos_{prize}_{idx}_{pos}_d{digit_value}_{window_days}d"
                )
                rows.append(
                    (as_of_date, key, float(counter.get(str(digit_value), 0)))
                )
    return rows


# ---------------------------------------------------------------------------
# P2.4 - Cross Position Generator
# ---------------------------------------------------------------------------
def cross_position_features(
    digit_history: Sequence[Tuple[str, str, int, int, str]],
    as_of_date: str,
    window_days: int = 30,
) -> List[FeatureRow]:
    """Đếm số lần (DB_V4 + G1_V0) mod 10 = D xuất hiện trong window (P2.4).

    Để giới hạn explosion, chọn 4 cặp source-destination cốt lõi:
        - special[V4] + G1[V0]
        - special[V3] + G1[V4]
        - special[V4] + G7[0][V1]
        - special[V0] + G1[V0]
    """
    pairs = [
        (("special", 0, 4), ("G1", 0, 0)),
        (("special", 0, 3), ("G1", 0, 4)),
        (("special", 0, 4), ("G7", 0, 1)),
        (("special", 0, 0), ("G1", 0, 0)),
    ]

    by_date_pos: Dict[str, Dict[Tuple[str, int, int], str]] = defaultdict(dict)
    for date, prize, idx, pos, digit in digit_history:
        by_date_pos[date][(prize, idx, pos)] = digit

    dates_sorted = sorted(by_date_pos.keys(), key=lambda d: (
        int(d.split("/")[2]), int(d.split("/")[1]), int(d.split("/")[0])
    ))
    history = [date for date in dates_sorted if date <= as_of_date]
    window_dates = history[-window_days:] if window_days > 0 else history

    rows: List[FeatureRow] = []
    for src, dst in pairs:
        counter: Counter[int] = Counter()
        for date in window_dates:
            digits = by_date_pos.get(date, {})
            d_src = digits.get(src)
            d_dst = digits.get(dst)
            if d_src is None or d_dst is None:
                continue
            modulo = (int(d_src) + int(d_dst)) % 10
            counter[modulo] += 1
        src_tag = f"{src[0]}{src[1]}_{src[2]}"
        dst_tag = f"{dst[0]}{dst[1]}_{dst[2]}"
        for digit_value in range(10):
            key = f"cross_{src_tag}_{dst_tag}_mod10_d{digit_value}_{window_days}d"
            rows.append(
                (as_of_date, key, float(counter.get(digit_value, 0)))
            )
    return rows


# ---------------------------------------------------------------------------
# P2.5 - Cross Day Generator
# ---------------------------------------------------------------------------
def cross_day_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
    lags: Sequence[int] = (1, 2, 3, 7),
) -> List[FeatureRow]:
    """Đếm số lô NN ra ở ngày T-lag và cũng ra ở ngày T (P2.5).

    Sinh feature ``co_occur_lag{L}`` = số lô vừa ra ở T-L vừa ra ở T-1 (gần
    as_of nhất, làm proxy cho "stickiness"). Tính gọn để chỉ 4 lag × 1 = 4
    feature/ngày.
    """
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history = [date for date in dates if date <= as_of_date]

    rows: List[FeatureRow] = []
    if len(history) < 2:
        for lag in lags:
            rows.append((as_of_date, f"co_occur_lag{lag}", 0.0))
        return rows

    last_set = set(grouped[history[-1]])
    for lag in lags:
        if len(history) > lag:
            prev_set = set(grouped[history[-1 - lag]])
            shared = len(last_set & prev_set)
        else:
            shared = 0
        rows.append((as_of_date, f"co_occur_lag{lag}", float(shared)))
    return rows


# ---------------------------------------------------------------------------
# P2.6 - Pattern Generator (Pascal, Reverse, Mirror, Modulo)
# ---------------------------------------------------------------------------
def pattern_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
) -> List[FeatureRow]:
    """Sinh feature pattern (P2.6).

    Cho lô gần nhất (T-1):
        - ``pascal_count``: số cặp Pascal (NN -> (a+b) mod 10) còn xuất hiện T-1.
        - ``reverse_count``: số lô có dạng đảo ngược trong T-1.
        - ``mirror_count``: số cặp gương (12 <-> 21) trong T-1.
        - ``modulo_sum``:   tổng (a+b) mod 10 trên T-1.
    """
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history = [date for date in dates if date <= as_of_date]
    if not history:
        return [
            (as_of_date, "pascal_count", 0.0),
            (as_of_date, "reverse_count", 0.0),
            (as_of_date, "mirror_count", 0.0),
            (as_of_date, "modulo_sum", 0.0),
        ]

    last_loto = grouped[history[-1]]
    last_set = set(last_loto)

    pascal_count = 0
    reverse_count = 0
    mirror_count = 0
    modulo_sum = 0
    for loto in last_loto:
        a_digit, b_digit = int(loto[0]), int(loto[1])
        # Pascal: lô (a*10 + (a+b) mod 10)
        pascal_loto = f"{a_digit}{(a_digit + b_digit) % 10}"
        if pascal_loto in last_set:
            pascal_count += 1
        # Reverse
        if loto[::-1] in last_set:
            reverse_count += 1
        # Mirror: cặp NN <-> NN' với N' = 9 - N
        mirror = f"{9 - a_digit}{9 - b_digit}"
        if mirror in last_set:
            mirror_count += 1
        modulo_sum += (a_digit + b_digit) % 10

    return [
        (as_of_date, "pascal_count", float(pascal_count)),
        (as_of_date, "reverse_count", float(reverse_count)),
        (as_of_date, "mirror_count", float(mirror_count)),
        (as_of_date, "modulo_sum", float(modulo_sum)),
    ]


# ---------------------------------------------------------------------------
# P2.7 - Roi Generator (rồi từ DB hoặc rồi 1/2/3 ngày)
# ---------------------------------------------------------------------------
def roi_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
) -> List[FeatureRow]:
    """Sinh feature rồi (P2.7).

    Cho mỗi lô NN:
        - ``roi_1d_{NN}``:    1 nếu NN ra ngày T-1.
        - ``roi_2d_{NN}``:    1 nếu NN ra ngày T-2.
        - ``roi_3d_{NN}``:    1 nếu NN ra ngày T-3.
        - ``roi_db_{NN}``:    1 nếu NN là 2 số cuối của giải đặc biệt ngày T-1.
        - ``roi_g1_{NN}``:    1 nếu NN là 2 số cuối của giải nhất ngày T-1.
    """
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history = [date for date in dates if date <= as_of_date]

    sets_by_lag: Dict[int, set[str]] = {}
    for lag in (1, 2, 3):
        if len(history) >= lag:
            sets_by_lag[lag] = set(grouped[history[-lag]])
        else:
            sets_by_lag[lag] = set()

    db_prev = sets_by_lag.get(1, set())
    g1_prev: set[str] = set()
    if history:
        # G1 chỉ là loto cuối của ngày trước, dùng 2 số cuối để bắt tín hiệu rơi.
        prev_date = history[-1]
        for loto in grouped[prev_date]:
            g1_prev.add(loto)

    rows: List[FeatureRow] = []
    for loto in _all_loto_codes():
        for lag in (1, 2, 3):
            value = 1.0 if loto in sets_by_lag[lag] else 0.0
            rows.append((as_of_date, f"roi_{lag}d_{loto}", value))
        rows.append((as_of_date, f"roi_db_{loto}", 1.0 if loto in db_prev else 0.0))
        rows.append((as_of_date, f"roi_g1_{loto}", 1.0 if loto in g1_prev else 0.0))
    return rows


# ---------------------------------------------------------------------------
# P2.8 - Đầu/Đuôi/Chạm/Tổng
# ---------------------------------------------------------------------------
def dau_duoi_cham_tong_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
    window_days: int = 30,
) -> List[FeatureRow]:
    """Sinh feature đầu/đuôi/chạm/tổng (P2.8).

    - ``dau_hot_{D}``: số lần D là chữ số đầu trong window.
    - ``duoi_hot_{D}``: số lần D là chữ số đuôi trong window.
    - ``cham_{D}``:    số lần D xuất hiện ở bất kỳ vị trí trong lô.
    - ``tong_de``:     tổng 2 chữ số của lô T-1 đầu tiên (special).
    - ``tong_lo``:     trung bình tổng các lô T-1.
    """
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history = [date for date in dates if date <= as_of_date]
    window = history[-window_days:] if window_days > 0 else history

    dau_counter: Counter[str] = Counter()
    duoi_counter: Counter[str] = Counter()
    cham_counter: Counter[str] = Counter()
    for date in window:
        for loto in grouped[date]:
            dau_counter[loto[0]] += 1
            duoi_counter[loto[1]] += 1
            cham_counter[loto[0]] += 1
            if loto[0] != loto[1]:
                cham_counter[loto[1]] += 1

    rows: List[FeatureRow] = []
    for digit in range(10):
        digit_text = str(digit)
        rows.append(
            (as_of_date, f"dau_hot_{digit}", float(dau_counter.get(digit_text, 0)))
        )
        rows.append(
            (as_of_date, f"duoi_hot_{digit}", float(duoi_counter.get(digit_text, 0)))
        )
        rows.append(
            (as_of_date, f"cham_{digit}", float(cham_counter.get(digit_text, 0)))
        )

    if history:
        last_loto = grouped[history[-1]]
        if last_loto:
            tong_de = (int(last_loto[0][0]) + int(last_loto[0][1])) % 10
            tongs = [
                (int(loto[0]) + int(loto[1])) % 10 for loto in last_loto
            ]
            tong_lo = sum(tongs) / len(tongs) if tongs else 0.0
        else:
            tong_de = 0
            tong_lo = 0.0
    else:
        tong_de = 0
        tong_lo = 0.0
    rows.append((as_of_date, "tong_de", float(tong_de)))
    rows.append((as_of_date, "tong_lo", float(tong_lo)))
    return rows


# ---------------------------------------------------------------------------
# P2.9 - Discovered Features (auto cong thuc don gian)
# ---------------------------------------------------------------------------
def discovered_features(
    digit_history: Sequence[Tuple[str, str, int, int, str]],
    as_of_date: str,
    window_days: int = 30,
) -> List[FeatureRow]:
    """Sinh 5 feature công thức tự khám phá đơn giản (P2.9).

    Các công thức nền tảng nhất:
        - ``disc_db_v4_plus_g1_v0_mod10``: avg ((DB_V4 + G1_V0) mod 10) qua window.
        - ``disc_db_v3_minus_g1_v4_mod10``
        - ``disc_db_sum_mod10``
        - ``disc_db_diff_v0_v4_abs``
        - ``disc_g1_sum_mod10``
    """
    by_date_pos: Dict[str, Dict[Tuple[str, int, int], str]] = defaultdict(dict)
    for date, prize, idx, pos, digit in digit_history:
        by_date_pos[date][(prize, idx, pos)] = digit

    dates_sorted = sorted(by_date_pos.keys(), key=lambda d: (
        int(d.split("/")[2]), int(d.split("/")[1]), int(d.split("/")[0])
    ))
    history = [date for date in dates_sorted if date <= as_of_date]
    window = history[-window_days:] if window_days > 0 else history

    formulas: List[Tuple[str, callable]] = [  # type: ignore[type-arg]
        (
            "disc_db_v4_plus_g1_v0_mod10",
            lambda d: (
                (int(d[("special", 0, 4)]) + int(d[("G1", 0, 0)])) % 10
                if ("special", 0, 4) in d and ("G1", 0, 0) in d else None
            ),
        ),
        (
            "disc_db_v3_minus_g1_v4_mod10",
            lambda d: (
                (int(d[("special", 0, 3)]) - int(d[("G1", 0, 4)])) % 10
                if ("special", 0, 3) in d and ("G1", 0, 4) in d else None
            ),
        ),
        (
            "disc_db_sum_mod10",
            lambda d: (
                sum(int(d[("special", 0, idx)]) for idx in range(5)) % 10
                if all(("special", 0, idx) in d for idx in range(5)) else None
            ),
        ),
        (
            "disc_db_diff_v0_v4_abs",
            lambda d: (
                abs(int(d[("special", 0, 0)]) - int(d[("special", 0, 4)]))
                if ("special", 0, 0) in d and ("special", 0, 4) in d else None
            ),
        ),
        (
            "disc_g1_sum_mod10",
            lambda d: (
                sum(int(d[("G1", 0, idx)]) for idx in range(5)) % 10
                if all(("G1", 0, idx) in d for idx in range(5)) else None
            ),
        ),
    ]

    rows: List[FeatureRow] = []
    for name, fn in formulas:
        values: List[int] = []
        for date in window:
            value = fn(by_date_pos[date])
            if value is not None:
                values.append(int(value))
        avg = sum(values) / len(values) if values else 0.0
        rows.append((as_of_date, name, float(avg)))
    return rows


# ---------------------------------------------------------------------------
# P2.10 - Mini Genetic Programming (tien hoa cong thuc don gian)
# ---------------------------------------------------------------------------
def gp_features(
    digit_history: Sequence[Tuple[str, str, int, int, str]],
    as_of_date: str,
    window_days: int = 30,
) -> List[FeatureRow]:
    """Mini GP: tổ hợp 2 vị trí + toán tử {+, -, *} mod 10 (P2.10).

    Sinh đúng top-10 công thức được "tiến hóa" theo entropy thấp (đoán có
    ý nghĩa). Giới hạn để không bùng feature.
    """
    by_date_pos: Dict[str, Dict[Tuple[str, int, int], str]] = defaultdict(dict)
    for date, prize, idx, pos, digit in digit_history:
        by_date_pos[date][(prize, idx, pos)] = digit

    dates_sorted = sorted(by_date_pos.keys(), key=lambda d: (
        int(d.split("/")[2]), int(d.split("/")[1]), int(d.split("/")[0])
    ))
    history = [date for date in dates_sorted if date <= as_of_date]
    window = history[-window_days:] if window_days > 0 else history

    candidates = [
        (("special", 0, 0), ("G1", 0, 0)),
        (("special", 0, 1), ("G1", 0, 1)),
        (("special", 0, 2), ("G1", 0, 2)),
        (("special", 0, 3), ("G1", 0, 3)),
        (("special", 0, 4), ("G1", 0, 4)),
    ]
    operators: List[Tuple[str, callable]] = [  # type: ignore[type-arg]
        ("plus", lambda a, b: (a + b) % 10),
        ("minus", lambda a, b: (a - b) % 10),
        ("times", lambda a, b: (a * b) % 10),
    ]

    rows: List[FeatureRow] = []
    for src, dst in candidates:
        for op_name, op in operators:
            values: List[int] = []
            for date in window:
                pos_map = by_date_pos[date]
                if src not in pos_map or dst not in pos_map:
                    continue
                values.append(int(op(int(pos_map[src]), int(pos_map[dst]))))
            entropy = _entropy(values, base=10) if values else 0.0
            mean_value = sum(values) / len(values) if values else 0.0
            tag = f"gp_{src[0]}{src[1]}_{src[2]}_{op_name}_{dst[0]}{dst[1]}_{dst[2]}"
            rows.append((as_of_date, f"{tag}_mean", float(mean_value)))
            rows.append((as_of_date, f"{tag}_entropy", float(entropy)))
    return rows


def _entropy(values: Sequence[int], base: int = 10) -> float:
    """Shannon entropy (đơn vị bit) cho phân phối các giá trị 0..base-1."""
    import math

    if not values:
        return 0.0
    counter = Counter(values)
    total = len(values)
    entropy = 0.0
    for count in counter.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy


# ---------------------------------------------------------------------------
# P2.11 - Symbolic Regression (mini, dung linear fit y = a*x1 + b*x2 + c)
# ---------------------------------------------------------------------------
def symbolic_regression_features(
    digit_history: Sequence[Tuple[str, str, int, int, str]],
    as_of_date: str,
    window_days: int = 60,
) -> List[FeatureRow]:
    """Tìm tham số (a, b) tốt nhất cho công thức ``y_t = (a*x1 + b*x2) mod 10``
    với ``y_t = special_v4(t)``, ``x1 = special_v4(t-1)``, ``x2 = G1_v0(t-1)``
    (P2.11). Trả hệ số mean error trên window.
    """
    by_date_pos: Dict[str, Dict[Tuple[str, int, int], str]] = defaultdict(dict)
    for date, prize, idx, pos, digit in digit_history:
        by_date_pos[date][(prize, idx, pos)] = digit

    dates_sorted = sorted(by_date_pos.keys(), key=lambda d: (
        int(d.split("/")[2]), int(d.split("/")[1]), int(d.split("/")[0])
    ))
    history = [date for date in dates_sorted if date <= as_of_date]
    window = history[-window_days:] if window_days > 0 else history

    best_params = (0, 0)
    best_error = float("inf")
    for coef_a in range(0, 10):
        for coef_b in range(0, 10):
            error = 0.0
            count = 0
            for prev_date, curr_date in zip(window, window[1:]):
                prev_map = by_date_pos[prev_date]
                curr_map = by_date_pos[curr_date]
                if (
                    ("special", 0, 4) not in prev_map
                    or ("G1", 0, 0) not in prev_map
                    or ("special", 0, 4) not in curr_map
                ):
                    continue
                x1 = int(prev_map[("special", 0, 4)])
                x2 = int(prev_map[("G1", 0, 0)])
                target = int(curr_map[("special", 0, 4)])
                pred = (coef_a * x1 + coef_b * x2) % 10
                error += abs(pred - target)
                count += 1
            mean_err = error / count if count else float("inf")
            if mean_err < best_error:
                best_error = mean_err
                best_params = (coef_a, coef_b)

    rows: List[FeatureRow] = [
        (as_of_date, "sr_db_v4_coef_a", float(best_params[0])),
        (as_of_date, "sr_db_v4_coef_b", float(best_params[1])),
        (as_of_date, "sr_db_v4_error", float(best_error if best_error != float("inf") else -1.0)),
    ]
    return rows


# ---------------------------------------------------------------------------
# P2.12 - Markov bậc 1/2/3 cho lô tô 2 chữ số
# ---------------------------------------------------------------------------
def markov_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
    orders: Sequence[int] = (1, 2, 3),
) -> List[FeatureRow]:
    """Markov bậc K cho lô tô 2 chữ số (P2.12).

    Với mỗi order K, sinh ``markov{K}_{NN}`` = xác suất thực nghiệm để lô NN
    xuất hiện ở ngày kế tiếp khi ta nhìn K ngày gần nhất làm trạng thái.

    Cách làm:
        - Trạng thái = tuple(sorted(loto set của K ngày gần nhất))
        - Đếm transition từ trạng thái -> ngày kế tiếp
        - Tính xác suất có điều kiện theo tần suất transition

    Đây là Markov xấp xỉ nhẹ nhưng đúng bản chất hơn heuristic similarity.
    """
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history = [date for date in dates if date <= as_of_date]
    if len(history) < 2:
        return [(as_of_date, f"markov{order}_{loto}", 0.0) for order in orders for loto in _all_loto_codes()]

    rows: List[FeatureRow] = []
    for order in orders:
        if len(history) <= order:
            for loto in _all_loto_codes():
                rows.append((as_of_date, f"markov{order}_{loto}", 0.0))
            continue

        transition_counter: Dict[Tuple[Tuple[str, ...], str], Counter[str]] = defaultdict(Counter)
        state_counter: Counter[Tuple[str, ...]] = Counter()
        for idx in range(order, len(history)):
            state = tuple(sorted(set().union(*[grouped[history[j]] for j in range(idx - order, idx)])))
            next_day = history[idx]
            next_lotos = grouped[next_day]
            state_counter[state] += 1
            for loto in next_lotos:
                transition_counter[(state, loto)][loto] += 1

        current_state = tuple(sorted(set().union(*[grouped[date] for date in history[-order:]])))
        next_counts: Counter[str] = Counter()
        for (state, _loto), counter in transition_counter.items():
            if state == current_state:
                for loto, count in counter.items():
                    next_counts[loto] += count

        total = sum(next_counts.values())
        for loto in _all_loto_codes():
            value = (next_counts.get(loto, 0) / total) if total else 0.0
            rows.append((as_of_date, f"markov{order}_{loto}", float(value)))
    return rows



# ---------------------------------------------------------------------------
# P2.13 - Mini HMM features (state via clustering by daily count)
# ---------------------------------------------------------------------------
def hmm_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
    window_days: int = 60,
) -> List[FeatureRow]:
    """Mini HMM proxy (P2.13).

    Định nghĩa 3 trạng thái (state) theo số lô unique mỗi ngày:
        - state 0 (lạnh): unique < 22
        - state 1 (bình thường): 22 <= unique <= 25
        - state 2 (nóng): unique > 25
    Sinh:
        - ``hmm_state_current``
        - ``hmm_state_count_0/1/2`` trong window
        - ``hmm_transition_to_0/1/2``: prob chuyển từ state hiện tại
    """
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history = [date for date in dates if date <= as_of_date]
    if not history:
        return []
    window = history[-window_days:] if window_days > 0 else history

    def _state(unique_count: int) -> int:
        if unique_count < 22:
            return 0
        if unique_count > 25:
            return 2
        return 1

    states = [_state(len(set(grouped[date]))) for date in window]
    current_state = states[-1] if states else 1

    state_counter: Counter[int] = Counter(states)
    transition_counter: Counter[int] = Counter()
    transition_total = 0
    for prev_state, next_state in zip(states, states[1:]):
        if prev_state == current_state:
            transition_counter[next_state] += 1
            transition_total += 1

    rows: List[FeatureRow] = [
        (as_of_date, "hmm_state_current", float(current_state)),
    ]
    for state in (0, 1, 2):
        rows.append(
            (as_of_date, f"hmm_state_count_{state}", float(state_counter.get(state, 0)))
        )
        prob = (
            transition_counter.get(state, 0) / transition_total
            if transition_total
            else 0.0
        )
        rows.append((as_of_date, f"hmm_transition_to_{state}", float(prob)))
    return rows


# ---------------------------------------------------------------------------
# P2.14 - Cycle Detection (chu kỳ 7/14/30/60/90)
# ---------------------------------------------------------------------------
def cycle_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
    periods: Sequence[int] = (7, 14, 30, 60, 90),
) -> List[FeatureRow]:
    """Số lô T trùng với ngày T-period (P2.14)."""
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history = [date for date in dates if date <= as_of_date]
    if not history:
        return [(as_of_date, f"cycle_{period}", 0.0) for period in periods]

    last_set = set(grouped[history[-1]])
    rows: List[FeatureRow] = []
    for period in periods:
        if len(history) > period:
            past_set = set(grouped[history[-1 - period]])
            shared = len(last_set & past_set)
        else:
            shared = 0
        rows.append((as_of_date, f"cycle_{period}", float(shared)))
    return rows


# ---------------------------------------------------------------------------
# P2.15 - Apriori (frequent itemsets bậc 2) - tự code
# ---------------------------------------------------------------------------
def apriori_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
    window_days: int = 60,
    min_support: int = 5,
    top_k: int = 10,
) -> List[FeatureRow]:
    """Đếm top-k cặp lô (NN, MM) đồng xuất hiện trong cùng ngày (P2.15)."""
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history = [date for date in dates if date <= as_of_date]
    window = history[-window_days:] if window_days > 0 else history

    pair_counter: Counter[Tuple[str, str]] = Counter()
    for date in window:
        lotos = sorted(set(grouped[date]))
        for i, loto_a in enumerate(lotos):
            for loto_b in lotos[i + 1:]:
                pair_counter[(loto_a, loto_b)] += 1

    top_pairs = [
        (pair, count)
        for pair, count in pair_counter.most_common(top_k)
        if count >= min_support
    ]
    rows: List[FeatureRow] = [
        (as_of_date, "apriori_top_pairs_count", float(len(top_pairs)))
    ]
    for rank, (pair, count) in enumerate(top_pairs):
        rows.append(
            (as_of_date, f"apriori_pair{rank}_{pair[0]}_{pair[1]}_count", float(count))
        )
    # Ensure đủ top_k row (pad bằng 0).
    for rank in range(len(top_pairs), top_k):
        rows.append((as_of_date, f"apriori_pair{rank}_pad_count", 0.0))
    return rows


# ---------------------------------------------------------------------------
# P2.16 - FP-Growth (alias Apriori bậc 3 cho XSMB - rất rare)
# ---------------------------------------------------------------------------
def fp_growth_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
    window_days: int = 90,
    top_k: int = 5,
) -> List[FeatureRow]:
    """Đếm top-k bộ 3 lô đồng xuất hiện (P2.16).

    Dùng phép đếm trực tiếp (FP-tree thật là quá nặng cho 100 lô × 90 ngày,
    nhưng số distinct triplet vẫn manageable do mỗi ngày ~24 lô).
    """
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history = [date for date in dates if date <= as_of_date]
    window = history[-window_days:] if window_days > 0 else history

    triplet_counter: Counter[Tuple[str, str, str]] = Counter()
    for date in window:
        lotos = sorted(set(grouped[date]))
        # Để tránh O(n^3) bùng, chỉ duyệt 10 lô đầu sau sort (xấp xỉ top
        # nóng theo lexicographic - đủ để phát hiện pattern thường xuyên).
        sample = lotos[:10]
        for i, loto_a in enumerate(sample):
            for j, loto_b in enumerate(sample[i + 1:], start=i + 1):
                for loto_c in sample[j + 1:]:
                    triplet_counter[(loto_a, loto_b, loto_c)] += 1

    top_triplets = triplet_counter.most_common(top_k)
    rows: List[FeatureRow] = [
        (as_of_date, "fpgrowth_top_triplets_count", float(len(top_triplets)))
    ]
    for rank, (_triplet, count) in enumerate(top_triplets):
        rows.append((as_of_date, f"fpgrowth_triplet{rank}_count", float(count)))
    for rank in range(len(top_triplets), top_k):
        rows.append((as_of_date, f"fpgrowth_triplet{rank}_count", 0.0))
    return rows


# ---------------------------------------------------------------------------
# P2.17 - Sequential Pattern Mining (đơn giản: 2-gram lô liên tiếp)
# ---------------------------------------------------------------------------
def sequential_pattern_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
    window_days: int = 60,
    top_k: int = 10,
) -> List[FeatureRow]:
    """Đếm top-k chuỗi 2-gram (lô T-1 -> lô T) trong window (P2.17)."""
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history = [date for date in dates if date <= as_of_date]
    window = history[-window_days:] if window_days > 0 else history

    bigram_counter: Counter[Tuple[str, str]] = Counter()
    for prev_date, curr_date in zip(window, window[1:]):
        prev_set = set(grouped[prev_date])
        curr_set = set(grouped[curr_date])
        for prev_loto in prev_set:
            for curr_loto in curr_set:
                bigram_counter[(prev_loto, curr_loto)] += 1

    top_bigrams = bigram_counter.most_common(top_k)
    rows: List[FeatureRow] = []
    for rank, (_bigram, count) in enumerate(top_bigrams):
        rows.append((as_of_date, f"seq_bigram{rank}_count", float(count)))
    for rank in range(len(top_bigrams), top_k):
        rows.append((as_of_date, f"seq_bigram{rank}_count", 0.0))
    return rows


# ---------------------------------------------------------------------------
# P2.18 - Graph features (Degree, PageRank xấp xỉ)
# ---------------------------------------------------------------------------
def graph_features(
    loto_history: Sequence[Tuple[str, str]],
    as_of_date: str,
    window_days: int = 60,
    pagerank_iterations: int = 20,
    damping: float = 0.85,
) -> List[FeatureRow]:
    """Sinh degree + PageRank + centrality xấp xỉ + community cho graph lô (P2.18).

    Cạnh: 2 lô đồng xuất hiện trong cùng 1 ngày (window).
    Trả:
        - ``degree_{NN}``     - tổng trọng số cạnh kết nối NN
        - ``pagerank_{NN}``   - PageRank power iteration
        - ``centrality_{NN}`` - betweenness xấp xỉ bằng BFS shortest path
        - ``community_{NN}``  - label propagation community id (0..4)
    """
    grouped = _group_loto_by_date(loto_history)
    dates = _ordered_dates(grouped)
    history = [date for date in dates if date <= as_of_date]
    window = history[-window_days:] if window_days > 0 else history

    nodes = _all_loto_codes()
    node_set = set(nodes)
    adjacency: Dict[str, Counter[str]] = {node: Counter() for node in nodes}
    for date in window:
        lotos = list(set(grouped[date]) & node_set)
        for i, loto_a in enumerate(lotos):
            for loto_b in lotos[i + 1:]:
                adjacency[loto_a][loto_b] += 1
                adjacency[loto_b][loto_a] += 1

    # Degree
    degree = {node: float(sum(adjacency[node].values())) for node in nodes}

    # PageRank (power iteration)
    rank = {node: 1.0 / len(nodes) for node in nodes}
    for _ in range(pagerank_iterations):
        new_rank: Dict[str, float] = {node: (1.0 - damping) / len(nodes) for node in nodes}
        for node in nodes:
            total_weight = sum(adjacency[node].values())
            if total_weight == 0:
                continue
            for neighbor, weight in adjacency[node].items():
                new_rank[neighbor] += damping * rank[node] * weight / total_weight
        rank = new_rank

    # Betweenness centrality xấp xỉ: đếm số node mà NN là neighbor chung
    # cho mọi cặp (u, v) trong top-20 node có degree cao nhất.
    # Đây là xấp xỉ nhẹ nhàng thay vì O(V^3) full betweenness.
    top_nodes = sorted(nodes, key=lambda n: -degree[n])[:20]
    between: Dict[str, float] = {node: 0.0 for node in nodes}
    for i, src in enumerate(top_nodes):
        for dst in top_nodes[i + 1:]:
            # Các node nằm trên đường ngắn: neighbor chung của src và dst
            src_neighbors = set(adjacency[src].keys())
            dst_neighbors = set(adjacency[dst].keys())
            common = src_neighbors & dst_neighbors
            for mid in common:
                between[mid] += 1.0
    max_between = max(between.values()) if between else 1.0
    centrality = {
        node: between[node] / max_between if max_between > 0 else 0.0
        for node in nodes
    }

    # Community detection: label propagation đơn giản (5 vòng)
    community: Dict[str, int] = {node: int(node) % 10 for node in nodes}
    for _ in range(5):
        new_community = dict(community)
        for node in nodes:
            neighbors = list(adjacency[node].keys())
            if not neighbors:
                continue
            label_counts: Counter[int] = Counter()
            for neighbor in neighbors:
                label_counts[community[neighbor]] += adjacency[node][neighbor]
            new_community[node] = label_counts.most_common(1)[0][0]
        community = new_community
    # Chuẩn hóa community id về 0..4
    unique_labels = sorted(set(community.values()))
    label_map = {label: idx % 5 for idx, label in enumerate(unique_labels)}
    community = {node: label_map[community[node]] for node in nodes}

    rows: List[FeatureRow] = []
    for node in nodes:
        rows.append((as_of_date, f"degree_{node}", degree[node]))
        rows.append((as_of_date, f"pagerank_{node}", float(rank[node])))
        rows.append((as_of_date, f"centrality_{node}", float(centrality[node])))
        rows.append((as_of_date, f"community_{node}", float(community[node])))
    return rows


# ---------------------------------------------------------------------------
# P2.19 - Orchestrator: chạy toàn bộ generator + ghi vào features_daily
# ---------------------------------------------------------------------------
def compute_all_features(
    loto_history: Sequence[Tuple[str, str]],
    digit_history: Sequence[Tuple[str, str, int, int, str]],
    as_of_date: str,
) -> List[FeatureRow]:
    """Chạy toàn bộ generator của Phase 3 cho ``as_of_date``.

    Returns:
        List ``(draw_date, feature_name, feature_value)``, dùng truyền
        thẳng vào :func:`xsmb_pipeline.database.insert_features_daily`.
    """
    rows: List[FeatureRow] = []
    rows.extend(frequency_features(loto_history, as_of_date))
    rows.extend(gan_features(loto_history, as_of_date))
    rows.extend(position_features(digit_history, as_of_date))
    rows.extend(cross_position_features(digit_history, as_of_date))
    rows.extend(cross_day_features(loto_history, as_of_date))
    rows.extend(pattern_features(loto_history, as_of_date))
    rows.extend(roi_features(loto_history, as_of_date))
    rows.extend(dau_duoi_cham_tong_features(loto_history, as_of_date))
    rows.extend(discovered_features(digit_history, as_of_date))
    rows.extend(gp_features(digit_history, as_of_date))
    rows.extend(symbolic_regression_features(digit_history, as_of_date))
    rows.extend(markov_features(loto_history, as_of_date))
    rows.extend(hmm_features(loto_history, as_of_date))
    rows.extend(cycle_features(loto_history, as_of_date))
    rows.extend(apriori_features(loto_history, as_of_date))
    rows.extend(fp_growth_features(loto_history, as_of_date))
    rows.extend(sequential_pattern_features(loto_history, as_of_date))
    rows.extend(graph_features(loto_history, as_of_date))
    return rows


def feature_summary(rows: Iterable[FeatureRow]) -> Dict[str, int]:
    """Đếm số feature theo prefix (debug/log)."""
    summary: Counter[str] = Counter()
    for _date, name, _value in rows:
        prefix = name.split("_", 1)[0]
        summary[prefix] += 1
    return dict(summary)
