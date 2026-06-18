from __future__ import annotations

"""Module auto-update du lieu XSMB truoc moi lan predict/analyze.

Vi quay so 18h30 hang ngay, can kiem tra du lieu co duyen (den
kỳ quay truoc hom nay) truoc khi bat dau phan tich / du doan.
Neu co ngay missing -> tu dong crawl va cap nhat.

Usage:
    # Auto-update neu can, tra ve tap ket qua da cap nhat
    results = ensure_latest_data()

    # Chi kiem tra ngay cuoi cung
    latest = get_latest_draw_date()

    # Force refresh (vi du truoc khi predict)
    refreshed = refresh_to_latest()
"""

import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence

from .dataset import (
    DATE_FMT,
    format_date,
    iter_dates,
    load_csv,
    merge_results,
    parse_date,
    write_csv,
    write_json,
)
from .scraper import build_url, fetch_html, parse_result
from .schema import FetchError, LotteryResult, ParseError


# =========================================================================
# Thoi gian quay so
# =========================================================================
DRAW_HOUR = 18
DRAW_MINUTE = 30


def now_vietnam() -> datetime:
    """Tra ve thoi gian hien tai (UTC+7)."""
    return datetime.utcnow() + timedelta(hours=7)


def is_before_draw() -> bool:
    """Kiem tra xem hien tai co truoc gio quay so khong."""
    now = now_vietnam()
    draw_time = now.replace(hour=DRAW_HOUR, minute=DRAW_MINUTE, second=0, microsecond=0)
    return now < draw_time


def get_latest_draw_date_csv(csv_path: Path) -> Optional[str]:
    """Doc ngay cuoi cung tu file CSV.

    Args:
        csv_path: Duong dan toi file xsmb_full.csv

    Returns:
        Ngay cuoi cung dinh dang DD/MM/YYYY, hoac None neu CSV rong/chua co.
    """
    if not csv_path.exists():
        return None
    try:
        results = load_csv(csv_path)
        if not results:
            return None
        return max((item.date for item in results), key=parse_date)
    except Exception:
        return None


def get_latest_draw_date_db(db_path: str) -> Optional[str]:
    """Doc ngay cuoi cung tu Bang draws trong DuckDB.

    Args:
        db_path: Duong dan toi file xsmb.duckdb

    Returns:
        Ngay cuoi cung dinh dang DD/MM/YYYY, hoac None neu DB rong/chua co.
    """
    from .database import get_connection

    db_file = Path(db_path)
    if not db_file.exists():
        return None
    try:
        conn = get_connection(db_path, read_only=True)
        try:
            row = conn.execute(
                """
                SELECT draw_date
                FROM draws
                ORDER BY strptime(draw_date, '%d/%m/%Y') DESC
                LIMIT 1
                """
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except Exception:
        return None


def get_latest_draw_date(csv_path: Path, db_path: str) -> Optional[str]:
    """Lay ngay cuoi cung tu CSV hoac DB, lay max cua 2 nguon.

    Args:
        csv_path: Duong dan toi file CSV
        db_path: Duong dan toi file DuckDB

    Returns:
        Ngay cuoi cung hon trong 2 nguon, hoac None neu ca 2 deu rong.
    """
    csv_date = get_latest_draw_date_csv(csv_path)
    db_date = get_latest_draw_date_db(db_path)

    candidates: List[str] = []
    if csv_date:
        candidates.append(csv_date)
    if db_date:
        candidates.append(db_date)

    if not candidates:
        return None
    return max(candidates, key=parse_date)


def get_target_date() -> str:
    """Ngay ma du lieu can co toi (ngay truoc hom nay neu chua qua gio quay)."""
    now = now_vietnam()
    draw_time = now.replace(hour=DRAW_HOUR, minute=DRAW_MINUTE, second=0, microsecond=0)
    if now < draw_time:
        target = now - timedelta(days=1)
    else:
        target = now
    return format_date(target)


def get_start_date(latest_date: Optional[str]) -> str:
    """Ngay bat dau fetch - ngay sau ngay cuoi cung da co, hoac target date."""
    if latest_date is None:
        return get_target_date()
    next_day = parse_date(latest_date) + timedelta(days=1)
    target = parse_date(get_target_date())
    if next_day > target:
        return latest_date
    return format_date(next_day)


# =========================================================================
# Crawl ngay don le
# =========================================================================
def fetch_single_date(date: str) -> Optional[LotteryResult]:
    """Fetch 1 ngay tu website, tra ve None neu that bai."""
    try:
        page_html = fetch_html(build_url(date))
        return parse_result(page_html, date)
    except (FetchError, ParseError):
        return None


def fetch_missing_dates(start: str, end: str) -> tuple[List[LotteryResult], List[dict[str, str]]]:
    """Fetch 1 khoang ngay, tra ve ket qua + danh sach that bai.

    Args:
        start: Ngay bat dau (DD/MM/YYYY)
        end: Ngay ket thuc (DD/MM/YYYY)

    Returns:
        Tuple (danh sach LotteryResult, danh sach loi {date, error})
    """
    results: List[LotteryResult] = []
    failures: List[dict[str, str]] = []
    for date in iter_dates(start, end):
        result = fetch_single_date(date)
        if result:
            results.append(result)
        else:
            failures.append({"date": date, "error": "fetch_failed"})
    return results, failures


# =========================================================================
# Update DuckDB
# =========================================================================
def update_duckdb_with_results(
    db_path: str,
    results: Sequence[LotteryResult],
) -> dict[str, int]:
    """Ghi kq vao DuckDB, chi insert nhung ngay chua co.

    Args:
        db_path: Duong dan DuckDB
        results: Danh sach LotteryResult can insert

    Returns:
        Dict voi so ban ghi da insert cho tung bang.
    """
    from .database import create_schema, get_connection, insert_results

    if not results:
        return {"draws": 0, "prizes": 0, "loto_results": 0, "digit_positions": 0, "results": 0}

    conn = get_connection(db_path, read_only=False)
    try:
        create_schema(conn)
        inserted = insert_results(conn, results)
        return {
            "draws": inserted,
            "prizes": inserted,
            "loto_results": inserted,
            "digit_positions": inserted,
            "results": inserted,
        }
    finally:
        conn.close()


# =========================================================================
# Main auto-update functions
# =========================================================================
def refresh_to_latest(
    csv_path: Path,
    db_path: str,
) -> dict[str, object]:
    """Ham chinh: kiem tra + auto-update du lieu.

    Thu tu uu tien:
    1. Doc ngay cuoi cung tu CSV (nguon chinh)
    2. So sanh voi target date (ngay truoc hom nay hoac hom nay)
    3. Neu con thieu ngay -> crawl
    4. Ghi CSV + DuckDB
    5. Tra ve thong bao chi tiet

    Args:
        csv_path: Duong dan xsmb_full.csv
        db_path: Duong dan xsmb.duckdb

    Returns:
        Dict chua:
        - latest_before: ngay cuoi truoc update
        - latest_after: ngay cuoi sau update
        - target_date: ngay can co
        - is_up_to_date: True neu khong thieu ngay nao
        - missing_count: so ngay thieu
        - crawled_count: so ngay da crawl
        - results: danh sach LotteryResult moi
    """
    latest_before = get_latest_draw_date(csv_path, db_path)
    target_date = get_target_date()

    if latest_before and parse_date(latest_before) >= parse_date(target_date):
        return {
            "latest_before": latest_before,
            "latest_after": latest_before,
            "target_date": target_date,
            "is_up_to_date": True,
            "missing_count": 0,
            "crawled_count": 0,
            "failed_count": 0,
            "failed_dates": [],
            "results": [],
        }

    start_date = get_start_date(latest_before)
    end_date = target_date

    if parse_date(start_date) > parse_date(end_date):
        return {
            "latest_before": latest_before,
            "latest_after": latest_before,
            "target_date": target_date,
            "is_up_to_date": True,
            "missing_count": 0,
            "crawled_count": 0,
            "failed_count": 0,
            "failed_dates": [],
            "results": [],
        }

    existing: List[LotteryResult] = []
    if csv_path.exists():
        try:
            existing = load_csv(csv_path)
        except Exception:
            existing = []

    missing_dates_list = list(iter_dates(start_date, end_date))
    existing_dates = {item.date for item in existing}
    to_fetch = [date for date in missing_dates_list if date not in existing_dates]

    if not to_fetch:
        return {
            "latest_before": latest_before,
            "latest_after": latest_before,
            "target_date": target_date,
            "is_up_to_date": True,
            "missing_count": len(missing_dates_list),
            "crawled_count": 0,
            "failed_count": 0,
            "failed_dates": [],
            "results": [],
        }

    crawled, failed_dates = fetch_missing_dates(to_fetch[0], to_fetch[-1])

    merged = merge_results(existing, crawled)
    should_write = bool(crawled) or not csv_path.exists()

    if should_write:
        write_csv(csv_path, merged)
        json_path = csv_path.with_suffix(".json")
        write_json(json_path, merged)

    db_counts = update_duckdb_with_results(db_path, crawled)

    latest_after = get_latest_draw_date(csv_path, db_path)

    return {
        "latest_before": latest_before,
        "latest_after": latest_after,
        "target_date": target_date,
        "is_up_to_date": latest_after == target_date if latest_after else False,
        "missing_count": len(to_fetch),
        "crawled_count": len(crawled),
        "failed_count": len(failed_dates),
        "failed_dates": failed_dates,
        "results": [asdict(r) for r in crawled],
        "db_counts": db_counts,
    }


def ensure_latest_data(
    csv_path: Path = Path("xsmb_full.csv"),
    db_path: str = "xsmb.duckdb",
) -> tuple[List[LotteryResult], dict[str, object]]:
    """Ham chinh de goi truoc moi predict/analyze.

    Auto-update neu thieu ngay, tra ve:
    - Danh sach LotteryResult da cap nhat
    - Thong bao chi tiet ve qua trinh update

    Args:
        csv_path: Duong dan file CSV
        db_path: Duong dan file DuckDB

    Returns:
        Tuple (results, update_info)
    """
    info = refresh_to_latest(csv_path, db_path)

    if csv_path.exists():
        try:
            results = load_csv(csv_path)
        except Exception:
            results = []
    else:
        results = []

    if not results and info.get("results"):
        results = [LotteryResult(**r) for r in info["results"]]

    return results, info


def ensure_latest_for_predict(
    csv_path: Path = Path("xsmb_full.csv"),
    db_path: str = "xsmb.duckdb",
) -> List[LotteryResult]:
    """Wrapper don gian: chi tra ve danh sach results, an thong bao.

    Dung trong predict/analyze flow.
    """
    results, _ = ensure_latest_data(csv_path, db_path)
    return results
