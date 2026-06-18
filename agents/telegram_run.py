"""
Telegram Bot launcher - Phase 9 (Giai doan 9 - Telegram Bot).

Su dung:
    python agents/telegram_run.py --token YOUR_TOKEN

Hoac set env var:
    export TELEGRAM_BOT_TOKEN=your_token
    python agents/telegram_run.py

Test mode (khong can token):
    python agents/telegram_run.py --test

Hoac qua CLI:
    python -m xsmb_pipeline.cli telegram-test
"""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xsmb_pipeline.telegram_bot import run_cli

if __name__ == "__main__":
    run_cli()
