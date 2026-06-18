from __future__ import annotations

"""Lớp truy cập DuckDB cho hệ thống XSMB (đã chốt: dùng DuckDB).

Thiết kế hệ thống Phase 2 (theo CLAUDE.md / agent_plan.md):
    - draws            (draw_date, region, special, first_prize)         <- P1.1
    - prizes           (draw_date, prize_name, prize_index, prize_value) <- P1.2
    - loto_results     (draw_date, loto_number)                          <- P1.3
    - digit_positions  (draw_date, prize_name, prize_index,
                        position_index, digit)                           <- P1.4
    - features_daily   (draw_date, feature_name, feature_value)          <- P1.5
    - pattern_scores   (pattern_id, hit_rate, roi, precision,
                        stability)                                       <- P1.6

Vì sao DuckDB (không dùng PostgreSQL):
    - Workload XSMB là OLAP/analytics (feature engineering,
      backtest walk-forward, ML training) - đúng sở trường DuckDB.
    - Single-user, single-process, không cần concurrent writers.
    - Zero-ops: 1 file ``.duckdb``, không cần server.
    - Tích hợp pandas/pyarrow zero-copy, đọc CSV/Parquet trực tiếp.

Module này cung cấp 3 nhóm hàm cho mọi kiểu soi cầu (Dau/Duoi/Cham/Tong/
Loto2/3, Pascal, Markov, Roi, ...):

1. Schema:        :func:`create_schema` + 6 hàm bảng riêng.
2. Insert:        ``insert_draws``, ``insert_prizes``, ``insert_loto_results``,
                  ``insert_digit_positions``, ``insert_features_daily``,
                  ``insert_pattern_scores``, :func:`insert_results` (4 bảng nguồn).
3. Fetch (đọc):   ``fetch_draw``, ``fetch_prizes_for_date``,
                  ``fetch_loto_for_date``, ``fetch_digits_for_date``,
                  ``fetch_features_for_date``, ``fetch_pattern_scores``,
                  ``fetch_loto_history``, ``fetch_digit_history``,
                  ``count_rows``, :func:`integrity_check`.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import duckdb

from .dataset import derive_loto, flatten_numbers, load_csv
from .schema import LotteryResult

DEFAULT_DB_PATH = "xsmb.duckdb"

# Tên các giải G1..G7 ánh xạ sang field trong :class:`LotteryResult`.
PRIZE_FIELDS: List[tuple[str, int]] = [
    ("first", 1),
    ("second", 2),
    ("third", 3),
    ("fourth", 4),
    ("fifth", 5),
    ("sixth", 6),
    ("seventh", 7),
]


# ---------------------------------------------------------------------------
# Kết nối
# ---------------------------------------------------------------------------
def get_connection(
    db_path: str = DEFAULT_DB_PATH,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Mở kết nối DuckDB, tự tạo file nếu chưa có.

    Args:
        db_path: Đường dẫn file ``.duckdb`` (mặc định ``xsmb.duckdb``).
        read_only: ``True`` nếu chỉ đọc (an toàn khi nhiều process).

    Returns:
        ``duckdb.DuckDBPyConnection`` đã sẵn sàng dùng.
    """
    return duckdb.connect(db_path, read_only=read_only)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def create_draws_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Tạo bảng ``draws`` - kỳ quay XSMB theo ngày (P1.1).

    Cột:
        - ``draw_date``  VARCHAR PRIMARY KEY (định dạng ``DD/MM/YYYY``)
        - ``region``     VARCHAR (mặc định ``XSMB``)
        - ``special``    VARCHAR (giải đặc biệt 5 chữ số)
        - ``first_prize`` VARCHAR (giải nhất 5 chữ số)
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS draws (
            draw_date   VARCHAR PRIMARY KEY,
            region      VARCHAR,
            special     VARCHAR,
            first_prize VARCHAR
        )
        """
    )


def create_prizes_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Tạo bảng ``prizes`` - chi tiết từng giải / từng dãy số (P1.2)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prizes (
            draw_date   VARCHAR,
            prize_name  VARCHAR,
            prize_index INTEGER,
            prize_value VARCHAR,
            PRIMARY KEY (draw_date, prize_name, prize_index)
        )
        """
    )


def create_loto_results_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Tạo bảng ``loto_results`` - lô tô 2 chữ số 00-99 theo ngày (P1.3)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS loto_results (
            draw_date   VARCHAR,
            loto_number VARCHAR,
            PRIMARY KEY (draw_date, loto_number)
        )
        """
    )


def create_digit_positions_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Tạo bảng ``digit_positions`` - từng chữ số tại từng vị trí (P1.4)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS digit_positions (
            draw_date      VARCHAR,
            prize_name     VARCHAR,
            prize_index    INTEGER,
            position_index INTEGER,
            digit          VARCHAR,
            PRIMARY KEY (draw_date, prize_name, prize_index, position_index)
        )
        """
    )


def create_features_daily_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Tạo bảng ``features_daily`` - feature store theo ngày (P1.5)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS features_daily (
            draw_date     VARCHAR,
            feature_name  VARCHAR,
            feature_value DOUBLE,
            PRIMARY KEY (draw_date, feature_name)
        )
        """
    )


def create_pattern_scores_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Tạo bảng ``pattern_scores`` - điểm số của từng pattern (P1.6)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pattern_scores (
            pattern_id  VARCHAR PRIMARY KEY,
            hit_rate    DOUBLE,
            roi         DOUBLE,
            precision   DOUBLE,
            stability   DOUBLE
        )
        """
    )


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Tạo toàn bộ schema XSMB (gọi lần lượt 6 hàm bảng).

    An toàn để gọi nhiều lần - mỗi bảng dùng ``IF NOT EXISTS``.
    """
    create_draws_table(conn)
    create_prizes_table(conn)
    create_loto_results_table(conn)
    create_digit_positions_table(conn)
    create_features_daily_table(conn)
    create_pattern_scores_table(conn)


# ---------------------------------------------------------------------------
# Insert / Upsert
# ---------------------------------------------------------------------------
def insert_draws(
    conn: duckdb.DuckDBPyConnection,
    results: Sequence[LotteryResult],
) -> int:
    """Ghi danh sách kỳ quay vào bảng ``draws`` (P1.1).

    Args:
        conn: Kết nối DuckDB đang mở.
        results: Danh sách :class:`LotteryResult` cần ghi.

    Returns:
        Số bản ghi đã được truyền vào. DuckDB sẽ bỏ qua các ``draw_date``
        đã tồn tại nhờ ``ON CONFLICT DO NOTHING``.
    """
    if not results:
        return 0

    rows: List[tuple[str, str, str, str]] = []
    for result in results:
        first_value = result.first[0] if result.first else ""
        rows.append((result.date, result.region, result.special, first_value))

    conn.executemany(
        """
        INSERT INTO draws (draw_date, region, special, first_prize)
        VALUES (?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    return len(rows)


def insert_prizes(
    conn: duckdb.DuckDBPyConnection,
    results: Sequence[LotteryResult],
) -> int:
    """Ghi chi tiết các giải vào bảng ``prizes``."""
    if not results:
        return 0

    rows: List[tuple[str, str, int, str]] = []
    for result in results:
        rows.append((result.date, "special", 0, result.special))
        for field_name, prize_num in PRIZE_FIELDS:
            values: List[str] = getattr(result, field_name)
            for prize_index, value in enumerate(values):
                rows.append((result.date, f"G{prize_num}", prize_index, value))

    conn.executemany(
        """
        INSERT INTO prizes (draw_date, prize_name, prize_index, prize_value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    return len(rows)


def insert_loto_results(
    conn: duckdb.DuckDBPyConnection,
    results: Sequence[LotteryResult],
) -> int:
    """Ghi lô tô 2 chữ số vào ``loto_results`` (đã dedupe trong ngày)."""
    if not results:
        return 0

    rows: List[tuple[str, str]] = []
    for result in results:
        loto_numbers = derive_loto(flatten_numbers(result))
        for loto_number in sorted(set(loto_numbers)):
            rows.append((result.date, loto_number))

    conn.executemany(
        """
        INSERT INTO loto_results (draw_date, loto_number)
        VALUES (?, ?)
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    return len(rows)


def insert_digit_positions(
    conn: duckdb.DuckDBPyConnection,
    results: Sequence[LotteryResult],
) -> int:
    """Ghi từng chữ số tại từng vị trí vào ``digit_positions``."""
    if not results:
        return 0

    rows: List[tuple[str, str, int, int, str]] = []
    for result in results:
        for position_index, digit in enumerate(result.special):
            rows.append((result.date, "special", 0, position_index, digit))

        for field_name, prize_num in PRIZE_FIELDS:
            values: List[str] = getattr(result, field_name)
            for prize_index, value in enumerate(values):
                for position_index, digit in enumerate(value):
                    rows.append(
                        (
                            result.date,
                            f"G{prize_num}",
                            prize_index,
                            position_index,
                            digit,
                        )
                    )

    conn.executemany(
        """
        INSERT INTO digit_positions
            (draw_date, prize_name, prize_index, position_index, digit)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    return len(rows)


def insert_results(
    conn: duckdb.DuckDBPyConnection,
    results: Sequence[LotteryResult],
) -> int:
    """Ghi đầy đủ 4 bảng nguồn (draws, prizes, loto_results,
    digit_positions) trong cùng một transaction.

    Returns:
        Số :class:`LotteryResult` đã xử lý.
    """
    if not results:
        return 0

    insert_draws(conn, results)
    insert_prizes(conn, results)
    insert_loto_results(conn, results)
    insert_digit_positions(conn, results)
    return len(results)


def insert_features_daily(
    conn: duckdb.DuckDBPyConnection,
    rows: Sequence[tuple[str, str, float]],
) -> int:
    """Ghi feature theo ngày vào ``features_daily`` (P1.5).

    Args:
        conn: Kết nối DuckDB.
        rows: Tuple ``(draw_date, feature_name, feature_value)``. Trùng
            ``(draw_date, feature_name)`` -> ghi đè ``feature_value``
            (UPSERT) để Feature Engine có thể chạy lại idempotent.

    Returns:
        Số dòng đã truyền vào.
    """
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO features_daily (draw_date, feature_name, feature_value)
        VALUES (?, ?, ?)
        ON CONFLICT (draw_date, feature_name) DO UPDATE
            SET feature_value = EXCLUDED.feature_value
        """,
        list(rows),
    )
    return len(rows)


def insert_pattern_scores(
    conn: duckdb.DuckDBPyConnection,
    rows: Sequence[tuple[str, float, float, float, float]],
) -> int:
    """Ghi/UPSERT điểm pattern vào ``pattern_scores`` (P1.6).

    Args:
        rows: ``(pattern_id, hit_rate, roi, precision, stability)``.
    """
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO pattern_scores
            (pattern_id, hit_rate, roi, precision, stability)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (pattern_id) DO UPDATE SET
            hit_rate  = EXCLUDED.hit_rate,
            roi       = EXCLUDED.roi,
            precision = EXCLUDED.precision,
            stability = EXCLUDED.stability
        """,
        list(rows),
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Migrate
# ---------------------------------------------------------------------------
def migrate_csv(
    csv_path: str = "xsmb_full.csv",
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, object]:
    """Đọc CSV gốc rồi nạp đầy đủ vào DuckDB.

    Args:
        csv_path: File CSV nguồn (mặc định ``xsmb_full.csv``).
        db_path:  File DuckDB đích (mặc định ``xsmb.duckdb``).

    Returns:
        Dict gồm:
            - ``csv_rows``: số dòng đọc từ CSV.
            - ``inserted``: số :class:`LotteryResult` đã ghi.
            - ``table_counts``: dict ``{table_name: row_count}``.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Không tìm thấy CSV: {csv_file.resolve()}")

    results = load_csv(csv_file)

    conn = get_connection(db_path)
    try:
        create_schema(conn)
        inserted = insert_results(conn, results)
        table_counts = count_rows(
            conn,
            ("draws", "prizes", "loto_results", "digit_positions"),
        )
    finally:
        conn.close()

    return {
        "csv_rows": len(results),
        "inserted": inserted,
        "table_counts": table_counts,
    }


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def count_rows(
    conn: duckdb.DuckDBPyConnection,
    tables: Sequence[str],
) -> Dict[str, int]:
    """Đếm số dòng cho danh sách bảng (dùng cho test/log)."""
    counts: Dict[str, int] = {}
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = int(row[0]) if row else 0
    return counts


def fetch_draw(
    conn: duckdb.DuckDBPyConnection,
    draw_date: str,
) -> Optional[Dict[str, str]]:
    """Lấy 1 kỳ quay theo ``draw_date`` (định dạng ``DD/MM/YYYY``)."""
    row = conn.execute(
        """
        SELECT draw_date, region, special, first_prize
        FROM draws
        WHERE draw_date = ?
        """,
        [draw_date],
    ).fetchone()
    if row is None:
        return None
    return {
        "draw_date": row[0],
        "region": row[1],
        "special": row[2],
        "first_prize": row[3],
    }


def fetch_prizes_for_date(
    conn: duckdb.DuckDBPyConnection,
    draw_date: str,
) -> Dict[str, List[str]]:
    """Lấy toàn bộ giải của 1 ngày, gom theo ``prize_name``.

    Trả về dict ``{prize_name: [prize_value theo prize_index tăng dần]}``,
    ví dụ ``{"special": ["37754"], "G1": ["84983"], "G3": [...6 dãy...]}``.
    Hữu ích cho các kiểu soi cầu cần truy xuất nguyên ngày.
    """
    rows = conn.execute(
        """
        SELECT prize_name, prize_index, prize_value
        FROM prizes
        WHERE draw_date = ?
        ORDER BY prize_name, prize_index
        """,
        [draw_date],
    ).fetchall()
    grouped: Dict[str, List[str]] = {}
    for prize_name, _prize_index, prize_value in rows:
        grouped.setdefault(prize_name, []).append(prize_value)
    return grouped


def fetch_loto_for_date(
    conn: duckdb.DuckDBPyConnection,
    draw_date: str,
) -> List[str]:
    """Lấy danh sách lô tô 2 chữ số (đã sort tăng) của 1 ngày.

    Dùng cho các kiểu soi cầu Loto2, Pascal, Markov, Roi, ...
    """
    rows = conn.execute(
        """
        SELECT loto_number
        FROM loto_results
        WHERE draw_date = ?
        ORDER BY loto_number
        """,
        [draw_date],
    ).fetchall()
    return [row[0] for row in rows]


def fetch_loto_history(
    conn: duckdb.DuckDBPyConnection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[tuple[str, str]]:
    """Lấy chuỗi ``(draw_date, loto_number)`` theo thứ tự thời gian tăng.

    Tham số ``start_date`` / ``end_date`` định dạng ``DD/MM/YYYY`` đều
    optional. DuckDB sort theo ``strptime`` để chuỗi đúng theo thời gian
    chứ không phải lexicographic.
    """
    where: List[str] = []
    params: List[str] = []
    if start_date is not None:
        where.append("strptime(draw_date, '%d/%m/%Y') >= strptime(?, '%d/%m/%Y')")
        params.append(start_date)
    if end_date is not None:
        where.append("strptime(draw_date, '%d/%m/%Y') <= strptime(?, '%d/%m/%Y')")
        params.append(end_date)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""
        SELECT draw_date, loto_number
        FROM loto_results
        {where_sql}
        ORDER BY strptime(draw_date, '%d/%m/%Y') ASC, loto_number ASC
        """,
        params,
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def fetch_digits_for_date(
    conn: duckdb.DuckDBPyConnection,
    draw_date: str,
) -> List[Dict[str, object]]:
    """Lấy mọi chữ số tại mọi vị trí của 1 ngày.

    Trả list dict ``{prize_name, prize_index, position_index, digit}``,
    sort theo ``prize_name, prize_index, position_index``.
    Dùng cho Position / Cross Position generator (DB_V1..V5, G1_V1..V5, ...).
    """
    rows = conn.execute(
        """
        SELECT prize_name, prize_index, position_index, digit
        FROM digit_positions
        WHERE draw_date = ?
        ORDER BY prize_name, prize_index, position_index
        """,
        [draw_date],
    ).fetchall()
    return [
        {
            "prize_name": row[0],
            "prize_index": int(row[1]),
            "position_index": int(row[2]),
            "digit": row[3],
        }
        for row in rows
    ]


def fetch_digit_history(
    conn: duckdb.DuckDBPyConnection,
    prize_name: str,
    prize_index: int,
    position_index: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[tuple[str, str]]:
    """Lấy chuỗi ``(draw_date, digit)`` cho 1 vị trí cố định theo thời gian.

    Ví dụ: ``fetch_digit_history(conn, "special", 0, 4)`` -> chữ số cuối
    của giải đặc biệt theo từng ngày, dùng cho mọi pattern Markov / chu kỳ.
    """
    where: List[str] = [
        "prize_name = ?",
        "prize_index = ?",
        "position_index = ?",
    ]
    params: List[object] = [prize_name, prize_index, position_index]
    if start_date is not None:
        where.append("strptime(draw_date, '%d/%m/%Y') >= strptime(?, '%d/%m/%Y')")
        params.append(start_date)
    if end_date is not None:
        where.append("strptime(draw_date, '%d/%m/%Y') <= strptime(?, '%d/%m/%Y')")
        params.append(end_date)
    rows = conn.execute(
        f"""
        SELECT draw_date, digit
        FROM digit_positions
        WHERE {' AND '.join(where)}
        ORDER BY strptime(draw_date, '%d/%m/%Y') ASC
        """,
        params,
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def fetch_all_digit_history(
    conn: duckdb.DuckDBPyConnection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[tuple[str, str, int, int, str]]:
    """Lấy toàn bộ ``(draw_date, prize_name, prize_index, position_index,
    digit)`` từ ``digit_positions``, sort theo ngày tăng.

    Dùng cho :func:`xsmb_pipeline.feature_generators.compute_all_features`
    cần truy cập mọi vị trí của mọi giải.
    """
    where: List[str] = []
    params: List[object] = []
    if start_date is not None:
        where.append("strptime(draw_date, '%d/%m/%Y') >= strptime(?, '%d/%m/%Y')")
        params.append(start_date)
    if end_date is not None:
        where.append("strptime(draw_date, '%d/%m/%Y') <= strptime(?, '%d/%m/%Y')")
        params.append(end_date)
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""
        SELECT draw_date, prize_name, prize_index, position_index, digit
        FROM digit_positions
        {where_clause}
        ORDER BY strptime(draw_date, '%d/%m/%Y') ASC,
                 prize_name ASC, prize_index ASC, position_index ASC
        """,
        params,
    ).fetchall()
    return [
        (row[0], row[1], int(row[2]), int(row[3]), row[4]) for row in rows
    ]


def fetch_features_for_date(
    conn: duckdb.DuckDBPyConnection,
    draw_date: str,
) -> Dict[str, float]:

    """Lấy toàn bộ feature đã tính cho 1 ngày từ ``features_daily``.

    Trả dict ``{feature_name: feature_value}``. Rỗng nếu chưa có feature.
    """
    rows = conn.execute(
        """
        SELECT feature_name, feature_value
        FROM features_daily
        WHERE draw_date = ?
        """,
        [draw_date],
    ).fetchall()
    return {row[0]: float(row[1]) for row in rows}


def fetch_pattern_scores(
    conn: duckdb.DuckDBPyConnection,
    pattern_id: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Đọc bảng ``pattern_scores``.

    Nếu ``pattern_id`` truyền vào -> chỉ lấy 1 pattern.
    Không truyền -> trả tất cả, sort theo ``hit_rate DESC``.
    """
    if pattern_id is not None:
        rows = conn.execute(
            """
            SELECT pattern_id, hit_rate, roi, precision, stability
            FROM pattern_scores
            WHERE pattern_id = ?
            """,
            [pattern_id],
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT pattern_id, hit_rate, roi, precision, stability
            FROM pattern_scores
            ORDER BY hit_rate DESC
            """
        ).fetchall()
    return [
        {
            "pattern_id": row[0],
            "hit_rate": float(row[1]) if row[1] is not None else None,
            "roi": float(row[2]) if row[2] is not None else None,
            "precision": float(row[3]) if row[3] is not None else None,
            "stability": float(row[4]) if row[4] is not None else None,
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Integrity check (P1.7) - dùng cho migrate verify + CI
# ---------------------------------------------------------------------------
def integrity_check(
    conn: duckdb.DuckDBPyConnection,
) -> Dict[str, object]:
    """Kiểm tra tính nhất quán giữa 4 bảng nguồn (draws / prizes /
    loto_results / digit_positions).

    Các invariant chính:
        - 4 bảng phải có cùng tập ``draw_date`` (cùng số distinct days).
        - Mỗi ``draw_date`` trong ``prizes`` luôn có 27 dòng (DB=1, G1=1,
          G2=2, G3=6, G4=4, G5=6, G6=3, G7=4) cho XSMB chuẩn.
        - ``loto_number`` luôn dài 2 ký tự, là chữ số 0-9.
        - ``digit`` luôn 1 ký tự là chữ số 0-9.
        - ``loto_results`` của mỗi ngày = set 2 ký tự cuối của 27 giá trị
          ``prizes`` cùng ngày (cross-check sang nhau).

    Returns:
        Dict với:
            - ``ok`` (bool): tổng quát có pass không.
            - ``counts`` (dict): row count + distinct days mỗi bảng.
            - ``errors`` (list[str]): danh sách lỗi cụ thể (rỗng nếu OK).
    """
    errors: List[str] = []

    counts: Dict[str, Dict[str, int]] = {}
    for table in ("draws", "prizes", "loto_results", "digit_positions"):
        row = conn.execute(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT draw_date) FROM {table}
            """
        ).fetchone()
        if row is None:
            errors.append(f"{table}: query không trả kết quả")
            continue
        counts[table] = {"rows": int(row[0]), "days": int(row[1])}

    days_by_table = {table: stats["days"] for table, stats in counts.items()}
    if len(set(days_by_table.values())) > 1:
        errors.append(
            f"distinct draw_date không khớp giữa các bảng: {days_by_table}"
        )

    bad_prize_days = conn.execute(
        """
        SELECT draw_date, COUNT(*) AS c
        FROM prizes
        GROUP BY draw_date
        HAVING c <> 27
        ORDER BY draw_date
        LIMIT 5
        """
    ).fetchall()
    if bad_prize_days:
        errors.append(
            "prizes có ngày khác 27 dòng (5 mẫu): "
            + ", ".join(f"{date}={count}" for date, count in bad_prize_days)
        )

    bad_loto = conn.execute(
        """
        SELECT COUNT(*) FROM loto_results
        WHERE NOT regexp_full_match(loto_number, '^[0-9]{2}$')
        """
    ).fetchone()
    if bad_loto and int(bad_loto[0]) > 0:
        errors.append(f"loto_results có {bad_loto[0]} loto_number sai format")

    bad_digit = conn.execute(
        """
        SELECT COUNT(*) FROM digit_positions
        WHERE NOT regexp_full_match(digit, '^[0-9]$')
        """
    ).fetchone()
    if bad_digit and int(bad_digit[0]) > 0:
        errors.append(f"digit_positions có {bad_digit[0]} digit sai format")

    cross = conn.execute(
        """
        WITH expected AS (
            SELECT
                draw_date,
                COUNT(DISTINCT RIGHT(prize_value, 2)) AS expected_loto
            FROM prizes
            WHERE LENGTH(prize_value) >= 2
            GROUP BY draw_date
        ),
        actual AS (
            SELECT draw_date, COUNT(*) AS actual_loto
            FROM loto_results
            GROUP BY draw_date
        )
        SELECT COUNT(*)
        FROM expected e
        JOIN actual a USING (draw_date)
        WHERE e.expected_loto <> a.actual_loto
        """
    ).fetchone()
    if cross and int(cross[0]) > 0:
        errors.append(
            f"loto_results không khớp với 2 chữ số cuối của prizes "
            f"({cross[0]} ngày lệch)"
        )

    return {
        "ok": len(errors) == 0,
        "counts": counts,
        "errors": errors,
    }
