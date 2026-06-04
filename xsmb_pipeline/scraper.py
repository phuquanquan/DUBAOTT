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
        raise FetchError(f"Không tải được URL: {url}") from exc


def clean_html(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_page_date(html: str) -> Optional[str]:
    match = re.search(r"(\d{2}/\d{2}/\d{4})", html)
    if match:
        return match.group(1)
    match = re.search(r"xsmb-(\d{2})-(\d{2})-(\d{4})\.html", html)
    if match:
        return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
    return None


def extract_section_numbers(text: str, section: str, count: int, width: int) -> List[str]:
    match = re.search(section + r"(.*?)(?:G\.[1-7]|Loto|Đầu|Chạm|$)", text)
    if not match:
        raise ParseError(f"Không tìm thấy khu vực giải {section}")
    chunk = match.group(1)
    numbers = re.findall(rf"\d{{{width}}}", chunk)
    if len(numbers) < count:
        raise ParseError(f"Thiếu dữ liệu cho giải {section}: cần {count}, có {len(numbers)}")
    return numbers[:count]


def parse_result(html: str, fallback_date: str, region: str = "XSMB") -> LotteryResult:
    text = clean_html(html)
    date = extract_page_date(text) or fallback_date
    return LotteryResult(
        date=date,
        region=region,
        special=extract_section_numbers(text, "G.ĐB", 1, 5)[0],
        first=extract_section_numbers(text, "G.1", 1, 5),
        second=extract_section_numbers(text, "G.2", 2, 5),
        third=extract_section_numbers(text, "G.3", 6, 5),
        fourth=extract_section_numbers(text, "G.4", 4, 4),
        fifth=extract_section_numbers(text, "G.5", 6, 4),
        sixth=extract_section_numbers(text, "G.6", 3, 3),
        seventh=extract_section_numbers(text, "G.7", 4, 2),
    )


def build_url(date: str) -> str:
    day, month, year = date.split("/")
    return BASE_URL.format(day=day, month=month, year=year)


def extract_daily_urls(html: str) -> List[str]:
    urls = re.findall(r"(?:https://xosodaiphat\.com)?/xsmb-(\d{2})-(\d{2})-(\d{4})\.html", html)
    return [f"{day}/{month}/{year}" for day, month, year in urls]


def discover_dates_from_range(days: int) -> List[str]:
    if days not in RANGE_URLS:
        raise ValueError("days chỉ hỗ trợ các mốc có sẵn trong RANGE_URLS")
    html = fetch_html(RANGE_URLS[days])
    dates = []
    seen = set()
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
    seen = set()
    for window in range_windows:
        for date in discover_dates_from_range(window):
            if date not in seen:
                seen.add(date)
                dates.append(date)
    return fetch_results_for_dates(dates)


def crawl_daily(start: str, end: str) -> List[LotteryResult]:
    return fetch_results_for_dates(list(iter_dates(start, end)))
