from __future__ import annotations

import re
from html import unescape
from typing import List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .dataset import dedupe_results, iter_dates
from .schema import FetchError, LotteryResult, ParseError

BASE_URL = "https://xosodaiphat.com/xsmb-{day}-{month}-{year}.html"
RANGE_URLS = {
    30: "https://xosodaiphat.com/xsmb-30-ngay.html",
    60: "https://xosodaiphat.com/xsmb-60-ngay.html",
    90: "https://xosodaiphat.com/xsmb-90-ngay.html",
    100: "https://xosodaiphat.com/xsmb-100-ngay.html",
    200: "https://xosodaiphat.com/xsmb-200-ngay.html",
}


def fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=25) as response:
            return response.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError) as exc:
        raise FetchError(f"Khong tai duoc URL: {url}") from exc


def clean_html(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_page_date(text: str) -> Optional[str]:
    match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
    if match:
        return match.group(1)
    return None


def parse_result(html: str, fallback_date: str, region: str = "XSMB") -> LotteryResult:
    text = clean_html(html)
    date = extract_page_date(text) or fallback_date

    # Parse tuan tu theo thu tu giai: DB, G1, G2, G3, G4, G5, G6, G7
    # Tim vi tri bat dau cua tung giai
    db_pattern = r"G\.(?:DB|ĐB|ÐB|DB)\s+(\d{5})"
    g1_pattern = r"G\.1\s+(\d{5})"
    g2_pattern = r"G\.2\s+(\d{5})\s+(\d{5})"
    g3_pattern = r"G\.3\s+(\d{5})\s+(\d{5})\s+(\d{5})\s+(\d{5})\s+(\d{5})\s+(\d{5})"
    g4_pattern = r"G\.4\s+(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})"
    g5_pattern = r"G\.5\s+(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})"
    g6_pattern = r"G\.6\s+(\d{3})\s+(\d{3})\s+(\d{3})"
    g7_pattern = r"G\.7\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})"

    def match_group(pattern: str, count: int, name: str) -> List[str]:
        m = re.search(pattern, text)
        if not m:
            raise ParseError(f"Khong tim thay giai {name}")
        return list(m.groups())

    special = match_group(db_pattern, 1, "DB")[0]
    first = match_group(g1_pattern, 1, "G1")
    second = match_group(g2_pattern, 2, "G2")
    third = match_group(g3_pattern, 6, "G3")
    fourth = match_group(g4_pattern, 4, "G4")
    fifth = match_group(g5_pattern, 6, "G5")
    sixth = match_group(g6_pattern, 3, "G6")
    seventh = match_group(g7_pattern, 4, "G7")

    return LotteryResult(
        date=date,
        region=region,
        special=special,
        first=first,
        second=second,
        third=third,
        fourth=fourth,
        fifth=fifth,
        sixth=sixth,
        seventh=seventh,
    )


def build_url(date: str) -> str:
    day, month, year = date.split("/")
    return BASE_URL.format(day=day, month=month, year=year)


def extract_daily_urls(html: str) -> List[str]:
    urls = re.findall(r"(?:https://xosodaiphat\.com)?/xsmb-(\d{2})-(\d{2})-(\d{4})\.html", html)
    result = []
    for day, month, year in urls:
        result.append(day + chr(47) + month + chr(47) + year)
    return result


def discover_dates_from_range(days: int) -> List[str]:
    if days not in RANGE_URLS:
        raise ValueError("days chi ho tro cac moc co san trong RANGE_URLS")
    html = fetch_html(RANGE_URLS[days])
    dates = []
    seen: set[str] = set()
    for date in extract_daily_urls(html):
        if date not in seen:
            seen.add(date)
            dates.append(date)
    return dates


def fetch_results_for_dates(dates: Sequence[str]) -> List[LotteryResult]:
    results: List[LotteryResult] = []
    for date in dates:
        try:
            page_html = fetch_html(build_url(date))
            results.append(parse_result(page_html, date))
        except (FetchError, ParseError):
            continue
    return dedupe_results(results)


def crawl_range(days: int) -> List[LotteryResult]:
    return fetch_results_for_dates(discover_dates_from_range(days))


def bootstrap_history(range_windows: Sequence[int]) -> List[LotteryResult]:
    dates: List[str] = []
    seen: set[str] = set()
    for window in range_windows:
        for date in discover_dates_from_range(window):
            if date not in seen:
                seen.add(date)
                dates.append(date)
    return fetch_results_for_dates(dates)


def crawl_daily(start: str, end: str) -> List[LotteryResult]:
    return fetch_results_for_dates(list(iter_dates(start, end)))
