from __future__ import annotations

"""Telegram Bot cho he thong XSMB (Phase 9 - Giai doan 9).

Bot ho tro cac lenh:
    /start          - Gioi thieu he thong
    /predict        - Du doan hom nay (TOP 20 so + Score)
    /top10          - TOP 10 so hom nay
    /top20          - TOP 20 so hom nay
    /explain [so]   - Giai thich tai sao so do duoc chon
    /status         - Trang thai he thong (ngay cuoi, so ngay, next draw)
    /refresh        - Force refresh du lieu

Su dung:
    python agents/telegram_run.py --token YOUR_BOT_TOKEN

Integration voi auto-update (18h30 hang ngay):
    - Moi lenh deu tu dong goi ensure_latest_data() truoc phan tich,
      dam bao du lieu luon la nhat (toi ky quay truoc hom nay 18h30).
    - Chi can chay: python agents/telegram_run.py --schedule
      de kich hoat auto-update va predict hàng ngày.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import format_date, parse_date
from .models.weighted import (
    candidate_universe,
    ensemble_signal_score,
    fit_tuned_ranking_model,
    normalized_frequency,
    recency_score,
    train_ranking_model,
)
from .signals import clamp, resolved_model_signal_names, target_preset
from .targets import actual_targets, target_width
from .updater import DRAW_HOUR, DRAW_MINUTE, ensure_latest_data, get_latest_draw_date, get_target_date, is_before_draw, now_vietnam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("xsmb_telegram_bot")

# =========================================================================
# Config
# =========================================================================
DEFAULT_CSV_PATH = Path("xsmb_full.csv")
DEFAULT_DB_PATH = "xsmb.duckdb"
DEFAULT_TOP_K = 5
DEFAULT_TOP_N = 20

# =========================================================================
# Score explanation engine
# =========================================================================
def explain_candidate_score(
    results: Sequence,
    candidate: str,
    top_k: int = 5,
) -> Dict[str, object]:
    """Sinh giai thich diem cho 1 candidate.

    Tra ve dict gom:
    - total_score: tong diem
    - breakdown: dict the hien diem thanh phan
    - reasons: list cac ly do noi bat
    """
    rows = list(results)

    width = 2
    universe = candidate_universe(width)

    history_score = normalized_frequency(rows, candidate, "loto2")
    recency_sc = recency_score(rows, candidate, "loto2")
    signal_sc = clamp(ensemble_signal_score(rows, candidate, "loto2"))

    windows = [7, 30, 60]
    rolling_scores = {}
    for w in windows:
        window_rows = rows[-w:] if len(rows) > w else rows
        rolling_scores[f"rolling_{w}d"] = normalized_frequency(window_rows, candidate, "loto2")

    dau = candidate[0]
    duoi = candidate[1]
    dau_rows = rows[-30:] if len(rows) > 30 else rows
    duoi_rows = rows[-30:] if len(rows) > 30 else rows

    dau_count = sum(1 for r in dau_rows for item in actual_targets(r, "loto2") if item[0] == dau)
    duoi_count = sum(1 for r in duoi_rows for item in actual_targets(r, "loto2") if item[1] == duoi)
    dau_total = sum(len(actual_targets(r, "loto2")) for r in dau_rows)
    duoi_total = sum(len(actual_targets(r, "loto2")) for r in duoi_rows)

    dau_score = dau_count / max(1, dau_total)
    duoi_score = duoi_count / max(1, duoi_total)

    repeat_ratio = 1.0 if candidate[0] == candidate[1] else 0.0

    breakdown = {
        "history_freq": round(history_score, 4),
        "recency": round(recency_sc, 4),
        "signal_ensemble": round(signal_sc, 4),
        "rolling_7d": round(rolling_scores.get("rolling_7d", 0.0), 4),
        "rolling_30d": round(rolling_scores.get("rolling_30d", 0.0), 4),
        "rolling_60d": round(rolling_scores.get("rolling_60d", 0.0), 4),
        "dau_hot": round(dau_score, 4),
        "duoi_hot": round(duoi_score, 4),
        "repeat_digit": repeat_ratio,
    }

    total_score = (
        history_score * 0.14
        + recency_sc * 0.14
        + signal_sc * 0.22
        + rolling_scores.get("rolling_7d", 0.0) * 0.30
        + rolling_scores.get("rolling_30d", 0.0) * 0.22
        + dau_score * 0.05
        + duoi_score * 0.05
        + repeat_ratio * 0.04
    )

    reasons: List[str] = []
    if recency_sc > 0.1:
        reasons.append(f"Vua ra gan day (recency={recency_sc:.3f})")
    if history_score > rolling_scores.get("rolling_30d", 0.0) * 1.3:
        reasons.append("Tan suat nhieu hon trung binh dai han")
    if dau_score > 0.15:
        reasons.append(f"Đau {dau} dang nong")
    if duoi_score > 0.15:
        reasons.append(f"Duoi {duoi} dang nong")
    if signal_sc > 0.6:
        reasons.append(f"Signal manh ({signal_sc:.2f})")
    if repeat_ratio == 1.0:
        reasons.append("So lap (2 chu so giong nhau)")
    if total_score > 0.15:
        reasons.append("Tong diem cao")

    return {
        "candidate": candidate,
        "total_score": round(total_score, 4),
        "breakdown": breakdown,
        "reasons": reasons,
        "rank_relevance": "",
    }


def score_candidates_for_top_n(
    results: Sequence,
    top_n: int = 20,
    tuned: bool = False,
) -> List[Dict[str, object]]:
    """Sinh TOP N so kem diem va giai thich."""
    trainer = fit_tuned_ranking_model if tuned else train_ranking_model
    model = trainer(results, target_name="loto2", top_k=top_n, min_train_size=30) if tuned else trainer(results, target_name="loto2", top_k=top_n)

    rows: List[Dict[str, object]] = []
    for rank, (candidate, raw_score) in enumerate(model.predict(), start=1):
        explanation = explain_candidate_score(results, candidate, top_k=top_n)
        explanation["rank"] = rank
        explanation["model_score"] = round(raw_score, 4)
        explanation["reasons_text"] = "; ".join(explanation["reasons"]) if explanation["reasons"] else "Tong hop binh thuong"
        rows.append(explanation)

    return rows


# =========================================================================
# Telegram message formatting
# =========================================================================
def format_date_vietnamese(date_str: str) -> str:
    """Chuyen DD/MM/YYYY thanh ngay-thang-nam tieng Viet."""
    day, month, year = date_str.split("/")
    return f"{day}/{month}/{year}"


def format_next_draw_time() -> str:
    """Tra ve thoi gian quay so tiep theo."""
    now = now_vietnam()
    draw_time = now.replace(hour=DRAW_HOUR, minute=DRAW_MINUTE, second=0, microsecond=0)
    if now >= draw_time:
        draw_time += timedelta(days=1)
    remaining = draw_time - now
    hours = int(remaining.total_seconds() // 3600)
    minutes = int((remaining.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m"


def format_status_message(
    latest_date: Optional[str],
    total_days: int,
    next_draw_in: str,
    before_draw: bool,
) -> str:
    """Format tin nhan /status."""
    target = get_target_date()
    latest_display = format_date_vietnamese(latest_date) if latest_date else "Chua co du lieu"
    next_time = format_next_draw_time() if before_draw else "Da quay hom nay"

    status_lines = [
        "🏧 *XSMB Bot - Trang thai he thong*",
        "",
        f"📅 Ngay cuoi: *{latest_display}*",
        f"📊 Tong ngay da co: *{total_days}*",
        f"⏰ Quay tiep: *{next_time}*",
        f"🎯 Ngay du kien: *{format_date_vietnamese(target)}*",
    ]
    return "\n".join(status_lines)


def format_predict_message(
    rows: List[Dict[str, object]],
    latest_date: Optional[str],
    before_draw: bool,
    tuned: bool = False,
) -> str:
    """Format tin nhan /predict."""
    model_tag = "Tuned" if tuned else "Weighted"
    target = get_target_date()
    latest_display = format_date_vietnamese(latest_date) if latest_date else "?"

    header = [
        "🔮 *XSMB - Du doan hom nay*",
        f"📊 Model: {model_tag} Ranking",
        f"📅 Du lieu: {latest_display}",
        f"🎯 Ngay du doan: *{format_date_vietnamese(target)}*",
    ]
    if before_draw:
        header.append("⏰ Ket qua se co sau 18h30")

    header.append("")
    header.append("━" * 20)
    header.append("*TOP 20 SO CO DIEM CAO NHAT*")
    header.append("")

    lines = list(header)
    medal = ["🥇", "🥈", "🥉"]

    for row in rows:
        rank = row["rank"]
        candidate = row["candidate"]
        score = row["total_score"]
        reasons = row.get("reasons_text", "")

        if rank <= 3:
            prefix = f"{medal[rank-1]} {candidate}"
        else:
            prefix = f"  {rank:2d}. *{candidate}*"

        score_bar = "▓" * min(int(score * 20), 20)
        lines.append(f"{prefix}  [{score:.3f}] {score_bar}")

        if rank <= 5 and reasons and reasons != "Tong hop binh thuong":
            lines.append(f"      └ {reasons}")

    lines.append("")
    lines.append("━" * 20)
    lines.append(f"*Tong cong: {len(rows)} so*")

    return "\n".join(lines)


def format_top_message(
    rows: List[Dict[str, object]],
    n: int,
    latest_date: Optional[str],
) -> str:
    """Format tin nhan /top10 hoac /top20."""
    target = get_target_date()
    latest_display = format_date_vietnamese(latest_date) if latest_date else "?"
    display_rows = rows[:n]

    lines = [
            f"📊 *TOP {n} SO*",
            f"📅 Du lieu: {latest_display}",
            f"🎯 Ngay: *{format_date_vietnamese(target)}*",
            "",
            "━━━━━━━━━━━━━━━━━━",
        ]

    for row in display_rows:
        rank = row["rank"]
        candidate = row["candidate"]
        score = row["total_score"]
        lines.append(f"  {rank:2d}. *{candidate}*  ─  {score:.3f}")

    lines.append("━━━━━━━━━━━━━━━━━━")
    lines.append(f"*Diem = Weighted Frequency + Recency + Signal*")

    return "\n".join(lines)


def format_explain_message(
    candidate: str,
    explanation: Dict[str, object],
    latest_date: Optional[str],
) -> str:
    """Format tin nhan /explain [so]."""
    score = explanation["total_score"]
    breakdown = explanation["breakdown"]
    reasons = explanation["reasons"]

    lines = [
        f"🔍 *Giai thich so {candidate}*",
        f"📅 Du lieu: {format_date_vietnamese(latest_date) if latest_date else '?'}",
        "",
        "━━━━━━━━━━━━━━━━━━",
        "",
        f"🎯 *Tong diem: {score:.4f}*",
        "",
        "*Diem thanh phan:*",
        f"  📈 Tan suat dai han:   {breakdown['history_freq']:.4f}",
        f"  🕐 Gan day:            {breakdown['recency']:.4f}",
        f"  📡 Signal ensemble:    {breakdown['signal_ensemble']:.4f}",
        f"  📅 7 ngay:             {breakdown['rolling_7d']:.4f}",
        f"  📅 30 ngay:            {breakdown['rolling_30d']:.4f}",
        f"  📅 60 ngay:            {breakdown['rolling_60d']:.4f}",
        f"  🔥 Dau nong:           {breakdown['dau_hot']:.4f}",
        f"  🔥 Duoi nong:          {breakdown['duoi_hot']:.4f}",
        "",
        "*Ly do noi bat:*",
    ]

    if reasons:
        for reason in reasons:
            lines.append(f"  • {reason}")
    else:
        lines.append("  (Khong co ly do dac biet)")

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


def format_refresh_result(info: Dict[str, object]) -> str:
    """Format tin nhan /refresh."""
    is_up = info.get("is_up_to_date", False)
    latest_after = info.get("latest_after", "?")
    crawled = info.get("crawled_count", 0)
    failed = info.get("failed_count", 0)
    failed_dates = info.get("failed_dates", [])

    lines = [
        "🔄 *Refresh ket qua*",
        "",
    ]

    if is_up:
        lines.append("✅ Du lieu da cap nhat toi ngay quay gan nhat.")
    else:
        lines.append(f"⚠️  Da cap nhat, ngay cuoi: *{format_date_vietnamese(latest_after)}*")

    lines.append(f"📥 Da crawl: *{crawled}* ngay")
    if failed > 0:
        lines.append(f"❌ That bai: *{failed}* ngay")
        for fd in failed_dates[:3]:
            lines.append(f"   - {fd.get('date', '?')}: {fd.get('error', 'unknown')}")

    return "\n".join(lines)


# =========================================================================
# Bot command handler (dispatcher)
# =========================================================================
class XSMBBot:
    """Telegram Bot handler - xu ly cac lenh /predict, /top10, /top20, /explain, /status, /refresh."""

    def __init__(
        self,
        csv_path: Path = DEFAULT_CSV_PATH,
        db_path: str = DEFAULT_DB_PATH,
        top_k: int = DEFAULT_TOP_K,
        top_n: int = DEFAULT_TOP_N,
    ):
        self.csv_path = csv_path
        self.db_path = db_path
        self.top_k = top_k
        self.top_n = top_n
        self._cached_rows: Optional[List[Dict[str, object]]] = None
        self._cache_date: Optional[str] = None

    def _load_and_score(self, tuned: bool = False) -> Tuple[List[Dict[str, object]], str, int]:
        """Load data + ensure update + score candidates."""
        results, info = ensure_latest_data(self.csv_path, self.db_path)
        latest_date = get_latest_draw_date(self.csv_path, self.db_path)

        if not results:
            raise RuntimeError("Khong the load du lieu XSMB")

        rows = score_candidates_for_top_n(
            results,
            top_n=self.top_n,
            tuned=tuned,
        )
        return rows, latest_date, len(results)

    def _get_scored_rows(self, tuned: bool = False) -> List[Dict[str, object]]:
        """Get cached scored rows or recompute."""
        results, info = ensure_latest_data(self.csv_path, self.db_path)
        latest_date = get_latest_draw_date(self.csv_path, self.db_path)

        if not results:
            raise RuntimeError("Khong the load du lieu XSMB")

        return score_candidates_for_top_n(
            results,
            top_n=self.top_n,
            tuned=tuned,
        )

    def handle_start(self) -> str:
        lines = [
            "🏧 *XSMB AI Bot - Chao ban!*",
            "",
            "Day la bot soi cau XSMB bang AI, su dung",
            "XGBoost + LightGBM + Neural Network + Ensemble.",
            "",
            "Cac lenh co the su dung:",
            "",
            "  /predict  - Du doan hom nay (TOP 20)",
            "  /top10    - TOP 10 so hom nay",
            "  /top20    - TOP 20 so hom nay",
            "  /explain N - Giai thich so N (vi du /explain 27)",
            "  /status   - Trang thai he thong",
            "  /refresh  - Cap nhat du lieu moi nhat",
            "  /help     - Huong dan chi tiet",
            "",
            "⚠️  Ket qua du doan chi mang tinh tham khao.",
            "gio quay so: 18h30 hang ngay.",
        ]
        return "\n".join(lines)

    def handle_help(self) -> str:
        lines = [
            "📖 *Huong dan su dung*",
            "",
            "*1. /predict*",
            "  Du doan TOP 20 so co xac suat cao nhat hom nay.",
            "  Kem giai thich ngan gon cho TOP 5.",
            "",
            "*2. /top10 / top20*",
            "  Danh sach 10 hoac 20 so co diem cao nhat.",
            "",
            "*3. /explain [so]*",
            "  Giai thich chi tiet tai sao so do co diem cao.",
            "  Vi du: /explain 27",
            "",
            "*4. /status*",
            "  Kiem tra ngay cuoi cung, so ngay co du lieu,",
            "  va thoi gian quay so tiep theo.",
            "",
            "*5. /refresh*",
            "  Cap nhat du lieu tu internet neu thieu.",
            "",
            "⚠️  *Luu y:* Bot tu dong cap nhat du lieu",
            "truoc 18h30 hang ngay. Khong can /refresh thuong xuyen.",
        ]
        return "\n".join(lines)

    def handle_predict(self, tuned: bool = False) -> str:
        try:
            rows = self._get_scored_rows(tuned=tuned)
            latest = get_latest_draw_date(self.csv_path, self.db_path)
            return format_predict_message(rows, latest, is_before_draw(), tuned=tuned)
        except Exception as exc:
            logger.error(f"handle_predict error: {exc}")
            return f"❌ Loi: {exc}"

    def handle_top(self, n: int) -> str:
        try:
            rows = self._get_scored_rows()
            latest = get_latest_draw_date(self.csv_path, self.db_path)
            return format_top_message(rows, n, latest)
        except Exception as exc:
            logger.error(f"handle_top({n}) error: {exc}")
            return f"❌ Loi: {exc}"

    def handle_explain(self, candidate: str) -> str:
        try:
            if len(candidate) != 2 or not candidate.isdigit():
                return "❌ Vui long nhap so 2 chu so, vi du: /explain 27"

            rows = self._get_scored_rows()
            latest = get_latest_draw_date(self.csv_path, self.db_path)

            matching = [r for r in rows if r["candidate"] == candidate]
            if not matching:
                return f"❌ Khong tim thay so {candidate} trong TOP {self.top_n}"

            explanation = matching[0]
            return format_explain_message(candidate, explanation, latest)
        except Exception as exc:
            logger.error(f"handle_explain({candidate}) error: {exc}")
            return f"❌ Loi: {exc}"

    def handle_status(self) -> str:
        try:
            latest = get_latest_draw_date(self.csv_path, self.db_path)
            results, _ = ensure_latest_data(self.csv_path, self.db_path)
            total = len(results)
            return format_status_message(latest, total, format_next_draw_time(), is_before_draw())
        except Exception as exc:
            logger.error(f"handle_status error: {exc}")
            return f"❌ Loi: {exc}"

    def handle_refresh(self) -> str:
        try:
            _, info = ensure_latest_data(self.csv_path, self.db_path)
            return format_refresh_result(info)
        except Exception as exc:
            logger.error(f"handle_refresh error: {exc}")
            return f"❌ Loi: {exc}"

    def dispatch(self, command: str, args: str = "") -> str:
        """Dispatch command -> response string."""
        cmd = command.strip().lower()

        if cmd in ("/start", "start"):
            return self.handle_start()
        if cmd in ("/help", "help"):
            return self.handle_help()
        if cmd == "/predict":
            return self.handle_predict(tuned=False)
        if cmd == "/predicttuned":
            return self.handle_predict(tuned=True)
        if cmd == "/top10":
            return self.handle_top(10)
        if cmd == "/top20":
            return self.handle_top(20)
        if cmd.startswith("/explain"):
            parts = cmd.split()
            if len(parts) >= 2:
                cand = parts[1]
            elif args:
                cand = args.strip()
            else:
                return "❌ Vui long nhap so, vi du: /explain 27"
            return self.handle_explain(cand)
        if cmd in ("/status", "status"):
            return self.handle_status()
        if cmd in ("/refresh", "refresh"):
            return self.handle_refresh()

        return f"❓ Lenh '{command}' khong nhan dien. Nhap /help de xem huong dan."


# =========================================================================
# CLI launcher
# =========================================================================
def run_cli() -> None:
    parser = argparse.ArgumentParser(description="XSMB Telegram Bot")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""), help="Telegram Bot Token")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="Path to xsmb_full.csv")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to xsmb.duckdb")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K for ranking")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Top-N for display")
    parser.add_argument("--test", action="store_true", help="Test mode: run locally without Telegram")
    parser.add_argument("--port", type=int, default=8080, help="Webhook port (for production)")
    args = parser.parse_args()

    if not args.token and not args.test:
        print("ERROR: --token required (or set TELEGRAM_BOT_TOKEN env var)")
        sys.exit(1)

    bot = XSMBBot(
        csv_path=Path(args.csv),
        db_path=args.db,
        top_k=args.top_k,
        top_n=args.top_n,
    )

    if args.test:
        print("=== TEST MODE ===")
        print("")
        print(bot.handle_start())
        print("")
        print(bot.handle_status())
        print("")
        print(bot.handle_predict())
        print("")
        print(bot.handle_explain("27"))
        print("")
        print("=== TEST DONE ===")
        return

    try:
        import telegram
        from telegram.ext import CommandHandler, Dispatcher, MessageHandler, filters

        updater = telegram.Updater(token=args.token, use_context=True)
        disp = updater.dispatcher

        def cmd_start(update, context):
            update.message.reply_text(bot.handle_start(), parse_mode="Markdown")

        def cmd_help(update, context):
            update.message.reply_text(bot.handle_help(), parse_mode="Markdown")

        def cmd_predict(update, context):
            update.message.reply_text(bot.handle_predict(), parse_mode="Markdown")

        def cmd_top10(update, context):
            update.message.reply_text(bot.handle_top(10), parse_mode="Markdown")

        def cmd_top20(update, context):
            update.message.reply_text(bot.handle_top(20), parse_mode="Markdown")

        def cmd_explain(update, context):
            cand = " ".join(context.args) if context.args else ""
            update.message.reply_text(bot.handle_explain(cand), parse_mode="Markdown")

        def cmd_status(update, context):
            update.message.reply_text(bot.handle_status(), parse_mode="Markdown")

        def cmd_refresh(update, context):
            update.message.reply_text(bot.handle_refresh(), parse_mode="Markdown")

        disp.add_handler(CommandHandler("start", cmd_start))
        disp.add_handler(CommandHandler("help", cmd_help))
        disp.add_handler(CommandHandler("predict", cmd_predict))
        disp.add_handler(CommandHandler("top10", cmd_top10))
        disp.add_handler(CommandHandler("top20", cmd_top20))
        disp.add_handler(CommandHandler("explain", cmd_explain, pass_args=True))
        disp.add_handler(CommandHandler("status", cmd_status))
        disp.add_handler(CommandHandler("refresh", cmd_refresh))

        logger.info("Bot started. Press Ctrl+C to stop.")
        updater.start_polling()
        updater.idle()

    except ImportError:
        logger.error("python-telegram-bot not installed. Run: pip install python-telegram-bot")
        sys.exit(1)


if __name__ == "__main__":
    run_cli()
