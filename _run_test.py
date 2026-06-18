import sys
import json
from pathlib import Path

from xsmb_pipeline.telegram_bot import XSMBBot

bot = XSMBBot(csv_path=Path("xsmb_full.csv"), db_path="xsmb.duckdb", top_k=3, top_n=5)

outputs = {}
outputs["start"] = bot.handle_start()
outputs["status"] = bot.handle_status()
outputs["predict"] = bot.handle_predict()
outputs["top10"] = bot.handle_top(10)
outputs["explain"] = bot.handle_explain("27")

with open("_bot_test_output.json", "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)

print("Test output saved to _bot_test_output.json")
