from __future__ import annotations

"""Scheduler cho auto-update + predict hang ngay (18h30 XSMB).

Chay nen khi khoi dong he thong, no se:
1. Kiem tra du lieu hien tai - update neu thieu (crawl ngay missing)
2. Cho toi 18h30 (UTC+7) -> auto-update ngay hom nay
3. Sau khi co ket qua -> chay predict + luu ket qua + gui Telegram
4. Lap lai ngay hom sau

Usage:
    python -m xsmb_pipeline.scheduler --csv xsmb_full.csv --db xsmb.duckdb
    python -m xsmb_pipeline.scheduler --csv xsmb_full.csv --db xsmb.duckdb --telegram --token TOKEN
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import List, Optional

from .dataset import format_date, load_csv, write_json
from .updater import (
    DRAW_HOUR,
    DRAW_MINUTE,
    get_latest_draw_date,
    get_target_date,
    now_vietnam,
    refresh_to_latest,
)
from .models.weighted import fit_tuned_ranking_model, train_ranking_model
from .signals import clamp, ensemble_signal_score
from .targets import actual_targets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("xsmb_scheduler")


# =========================================================================
# Auto-update wrapper (luôn gọi trước mọi predict/analyze)
# =========================================================================
def ensure_data_current(
    csv_path: Path,
    db_path: str,
    force_refresh: bool = False,
) -> dict:
    """Wrapper an toàn: goi ensure_latest_data(), luôn trả dict info.

    Mọi flow predict/analyze phải gọi qua hàm này, không gọi
    trực tiếp refresh_to_latest().

    Args:
        csv_path:  Đường dẫn xsmb_full.csv
        db_path:   Đường dẫn xsmb.duckdb
        force_refresh: True = bỏ qua cache, luôn crawl

    Returns:
        Dict update info (như refresh_to_latest).
    """
    return refresh_to_latest(csv_path, db_path)


# =========================================================================
# Daily prediction (chạy sau 18h30 khi có kết quả mới)
# =========================================================================
def compute_top_n(
    results: list,
    top_n: int = 20,
    tuned: bool = False,
) -> List[dict]:
    """Sinh TOP N candidates với score + reasons.

    Dùng cho cả scheduler (lưu json) và bot (hiển thị).
    """
    trainer = fit_tuned_ranking_model if tuned else train_ranking_model
    model = trainer(results, target_name="loto2", top_k=top_n, min_train_size=30)

    rows: List[dict] = []
    for rank, (candidate, raw_score) in enumerate(model.predict(), start=1):
        width = 2
        rows.append({
            "rank": rank,
            "candidate": candidate,
            "model_score": round(float(raw_score), 4),
            "total_score": round(float(raw_score), 4),
        })

    return rows


def save_daily_prediction(
    results: list,
    output_dir: Path,
    tuned: bool = False,
    top_n: int = 20,
) -> Path:
    """Lưu kết quả predict ngày hôm nay vào file JSON.

    File được đặt tên theo ngày: predictions/2026-06-10.json

    Args:
        results:  Danh sách LotteryResult đã được ensure (đủ đến hôm qua 18h30)
        output_dir:  Thư mục lưu predictions
        tuned:    Dùng tuned model hay weighted model
        top_n:    Số candidates trả về

    Returns:
        Đường dẫn file đã lưu.
    """
    target_date = get_target_date()
    rows = compute_top_n(results, top_n=top_n, tuned=tuned)

    latest = get_latest_draw_date(Path("xsmb_full.csv"), "xsmb.duckdb")
    payload = {
        "predict_date": target_date,
        "data_latest_date": latest,
        "model": "tuned-weighted-ranking" if tuned else "weighted-ranking",
        "top_n": top_n,
        "predictions": rows,
        "generated_at": now_vietnam().isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = target_date.replace("/", "-")
    out_path = output_dir / f"{date_str}.json"

    import json as _json
    with out_path.open("w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved prediction: {out_path}")
    return out_path


# =========================================================================
# Scheduler core
# =========================================================================
def seconds_until_draw() -> float:
    """Số giây từ bây giờ đến 18h30 hôm nay (hoặc ngày mai)."""
    now = now_vietnam()
    draw_time = now.replace(hour=DRAW_HOUR, minute=DRAW_MINUTE, second=0, microsecond=0)
    if now >= draw_time:
        draw_time += timedelta(days=1)
    return (draw_time - now).total_seconds()


def wait_until_draw() -> None:
    """Block cho đến 18h30."""
    seconds = seconds_until_draw()
    if seconds > 0:
        logger.info(f"Waiting {seconds/3600:.1f}h until draw time (18h30)...")
        time.sleep(seconds)


class DailyScheduler:
    """Scheduler chạy mỗi ngày sau giờ quay số 18h30.

    Flow mỗi ngày:
        1. ensure_latest_data() - đảm bảo đủ dữ liệu
        2. wait_until_draw() - chờ đến 18h30
        3. ensure_latest_data() lại - lấy kết quả hôm nay
        4. compute_top_n() - sinh prediction
        5. save_daily_prediction() - lưu JSON
        6. Gửi Telegram (nếu token có)
        7. Chờ 25h rồi lặp lại
    """

    def __init__(
        self,
        csv_path: Path,
        db_path: str,
        predictions_dir: Path | None = None,
        telegram_token: str | None = None,
        top_n: int = 20,
        tuned: bool = False,
    ):
        self.csv_path = csv_path
        self.db_path = db_path
        self.predictions_dir = predictions_dir or Path("predictions")
        self.telegram_token = telegram_token
        self.top_n = top_n
        self.tuned = tuned
        self._running = False
        self._results_cache: Optional[list] = None
        self._bot = None

    def _ensure_data(self) -> dict:
        """Ensure data is current before any operation."""
        logger.info("Checking data currency...")
        info = refresh_to_latest(self.csv_path, self.db_path)
        latest = info.get("latest_after") or info.get("latest_before", "?")
        is_up = info.get("is_up_to_date", False)
        logger.info(
            f"Data check: latest={latest}, up_to_date={is_up}, "
            f"missing={info.get('missing_count', 0)}, "
            f"crawled={info.get('crawled_count', 0)}"
        )
        return info

    def _load_results(self) -> list:
        """Load CSV data, đảm bảo đã sorted."""
        self._results_cache = None  # invalidate cache after update
        if not self.csv_path.exists():
            return []
        try:
            results = load_csv(self.csv_path)
            from .dataset import sort_results
            self._results_cache = sort_results(results)
            return self._results_cache
        except Exception as exc:
            logger.error(f"Failed to load CSV: {exc}")
            return []

    def _run_daily_cycle(self) -> None:
        """Một chu kỳ: ensure -> wait -> ensure -> predict -> save."""
        now_ts = now_vietnam().isoformat()

        info = self._ensure_data()
        latest_before = info.get("latest_after") or info.get("latest_before")
        target = get_target_date()

        logger.info(f"[{now_ts}] Target date: {target}, latest: {latest_before}")

        if latest_before and latest_before == target:
            logger.info("Data already up to date, no need to wait for draw.")
        else:
            wait_until_draw()
            self._ensure_data()

        results = self._load_results()
        if not results:
            logger.warning("No results loaded after update cycle!")
            return

        out_path = save_daily_prediction(
            results,
            self.predictions_dir,
            tuned=self.tuned,
            top_n=self.top_n,
        )

        # Phase 10: Online Learning - process yesterday's result
        self._run_online_feedback()

        if self.telegram_token:
            self._send_telegram(results, out_path)

        self._log_prediction_summary(results)

    def _log_prediction_summary(self, results: list) -> None:
        """In ra TOP 5 prediction để log."""
        try:
            rows = compute_top_n(results, top_n=5, tuned=self.tuned)
            summary = [f"  {r['rank']}. {r['candidate']} ({r['total_score']:.4f})" for r in rows]
            logger.info("TOP 5 Prediction:\n" + "\n".join(summary))
        except Exception as exc:
            logger.error(f"Failed to log prediction: {exc}")

    def _send_telegram(self, results: list, prediction_path: Path) -> None:
        """Gửi kết quả qua Telegram bot."""
        try:
            if self._bot is None:
                import telegram
                self._bot = telegram.Bot(token=self.telegram_token)

            rows = compute_top_n(results, top_n=20, tuned=self.tuned)
            from .telegram_bot import format_predict_message

            latest = get_latest_draw_date(self.csv_path, self.db_path)
            target = get_target_date()

            now = now_vietnam()
            draw_time = now.replace(hour=DRAW_HOUR, minute=DRAW_MINUTE, second=0, microsecond=0)
            is_before_draw = now < draw_time

            message = format_predict_message(rows, latest, is_before_draw, tuned=self.tuned)

            updates = self._bot.send_message(
                chat_id=self._get_chat_id(),
                text=message,
                parse_mode="Markdown",
            )
            logger.info(f"Telegram sent: message_id={updates.message_id}")
        except ImportError:
            logger.warning("telegram package not installed, skipping Telegram send")
        except Exception as exc:
            logger.error(f"Failed to send Telegram: {exc}")

    def _get_chat_id(self) -> str:
        """Override in subclass or set via env var."""
        import os
        return os.environ.get("TELEGRAM_CHAT_ID", "")

    def _run_online_feedback(self) -> None:
        """Process online learning: compare yesterday's prediction with actual result."""
        try:
            from .online_learning import run_daily_feedback
            result = run_daily_feedback(
                predictions_dir=self.predictions_dir,
                csv_path=self.csv_path,
            )
            if result:
                status = "HIT" if result.hit else "MISS"
                logger.info(
                    f"Online feedback [{result.date}]: {status} "
                    f"(hit_count={result.hit_count}, precision={result.precision:.3f}) "
                    f"weights_updated={result.updated}"
                )
                if result.weight_delta:
                    delta_str = ", ".join(f"{k}={v:+.3f}" for k, v in result.weight_delta.items())
                    logger.info(f"  Weight changes: {delta_str}")
            else:
                logger.info("No prediction file found for yesterday to process feedback")
        except Exception as exc:
            logger.error(f"Online feedback error: {exc}")

    def run(self) -> None:
        """Chạy scheduler vĩnh viễn, mỗi ngày một chu kỳ."""
        self._running = True
        logger.info("Scheduler started. Will run daily after 18h30 draw.")
        logger.info(f"CSV: {self.csv_path}")
        logger.info(f"DB: {self.db_path}")
        logger.info(f"Predictions dir: {self.predictions_dir}")

        while self._running:
            try:
                self._run_daily_cycle()
                logger.info("Daily cycle completed. Sleeping 25h before next run...")
                time.sleep(25 * 3600)
            except KeyboardInterrupt:
                logger.info("Scheduler interrupted by user")
                self._running = False
                break
            except Exception as exc:
                logger.error(f"Daily cycle error: {exc}", exc_info=True)
                logger.info("Retrying in 1 hour...")
                time.sleep(3600)

    def stop(self) -> None:
        """Dừng scheduler."""
        self._running = False


# =========================================================================
# CLI launcher
# =========================================================================
def run_scheduler() -> None:
    parser = argparse.ArgumentParser(
        description="XSMB Daily Scheduler - auto-update + predict sau 18h30"
    )
    parser.add_argument("--csv", default="xsmb_full.csv", help="CSV dataset path")
    parser.add_argument("--db", default="xsmb.duckdb", help="DuckDB path")
    parser.add_argument("--predictions-dir", default="predictions", help="Predictions output dir")
    parser.add_argument("--token", default="", help="Telegram Bot Token")
    parser.add_argument("--top-n", type=int, default=20, help="Number of predictions")
    parser.add_argument("--tuned", action="store_true", help="Use tuned model")
    args = parser.parse_args()

    import os
    token = args.token or os.environ.get("TELEGRAM_BOT_TOKEN", "")

    scheduler = DailyScheduler(
        csv_path=Path(args.csv),
        db_path=args.db,
        predictions_dir=Path(args.predictions_dir),
        telegram_token=token or None,
        top_n=args.top_n,
        tuned=args.tuned,
    )

    try:
        scheduler.run()
    except KeyboardInterrupt:
        scheduler.stop()
        print("\nScheduler stopped.")


if __name__ == "__main__":
    run_scheduler()
