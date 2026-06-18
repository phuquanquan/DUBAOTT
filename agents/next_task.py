from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATUS_FILE = ROOT / "agent_status.json"
PLAN_FILE = ROOT / "agent_plan.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_status() -> dict:
    return json.loads(STATUS_FILE.read_text(encoding="utf-8"))


def write_status(data: dict) -> None:
    STATUS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_task_description(task_id: str, plan_text: str) -> str:
    for line in plan_text.splitlines():
        if task_id in line:
            return line.strip().lstrip("-").replace("[ ]", "").replace("[x]", "").strip()
    return task_id


def get_phase_for_task(task_id: str) -> int:
    try:
        return int(task_id[1])
    except (IndexError, ValueError):
        return 1


def advance() -> None:
    status = read_status()
    current_task = status.get("current_task", "")
    review_status = status.get("review_status", "")

    if review_status != "approved":
        print(f"[NEXT_TASK] Task {current_task} chua duoc approve (review_status={review_status}). Khong the advance.")
        return

    plan_text = PLAN_FILE.read_text(encoding="utf-8")

    # Mark current task done trong plan
    updated_plan = plan_text.replace(
        f"- [ ] {current_task}",
        f"- [x] {current_task}"
    )
    PLAN_FILE.write_text(updated_plan, encoding="utf-8")

    # Tim task pending tiep theo
    next_task = None
    for line in updated_plan.splitlines():
        stripped = line.strip()
        if stripped.startswith("- [ ]"):
            parts = stripped.split()
            if len(parts) >= 3:
                next_task = parts[2].rstrip(":")
                break

    if not next_task:
        print("[NEXT_TASK] Tat ca task da hoan thanh! Blueprint complete.")
        status["current_task"] = None
        status["status"] = "complete"
        status["last_updated"] = now_iso()
        write_status(status)
        return

    task_description = get_task_description(next_task, updated_plan)
    new_phase = get_phase_for_task(next_task)

    new_status = {
        "current_phase": new_phase,
        "current_task": next_task,
        "task_description": task_description,
        "status": "pending",
        "assigned_to": "claude",
        "last_updated": now_iso(),
        "review_status": None,
        "review_notes": None,
        "implementation_notes": None,
        "files_changed": [],
    }
    write_status(new_status)
    print(f"[NEXT_TASK] Chuyen sang task moi: {next_task}")
    print(f"[NEXT_TASK] Mo ta: {task_description}")
    print(f"[NEXT_TASK] Claude hay doc agent_status.json va bat dau implement.")


if __name__ == "__main__":
    advance()

