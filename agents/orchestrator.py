from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATUS_FILE = ROOT / "agent_status.json"
PLAN_FILE = ROOT / "agent_plan.md"
REVIEW_FILE = ROOT / "agent_review.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_status() -> dict:
    return json.loads(STATUS_FILE.read_text(encoding="utf-8"))


def write_status(data: dict) -> None:
    STATUS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_review(message: str) -> None:
    ts = now_iso()
    line = f"\n## [{ts}]\n{message}\n"
    with open(REVIEW_FILE, "a", encoding="utf-8") as review_file:
        review_file.write(line)


def find_claude() -> str:
    claude_path = shutil.which("claude")
    if claude_path:
        return claude_path
    fallback_paths = [
        r"C:\Users\phuquan2406\AppData\Roaming\npm\claude.CMD",
        r"C:\Users\phuquan2406\AppData\Roaming\npm\claude.cmd",
    ]
    for path in fallback_paths:
        if Path(path).exists():
            return path
    raise FileNotFoundError("Khong tim thay claude CLI. Cai dat bang: npm install -g @nhà cung cấp dịch vụ AI-ai/claude-code")


def run_claude(task_description: str) -> None:
    prompt = (
        f"Doc file CLAUDE.md, agent_status.json va agent_plan.md trong project tai {ROOT}. "
        f"Task hien tai: {task_description}. "
        "Implement task nay theo dung conventions trong CLAUDE.md. "
        "Doc code hien co lien quan truoc khi viet moi. "
        "Sau khi xong, cap nhat agent_status.json: status=done, files_changed=danh sach file, implementation_notes=mo ta ngan."
    )
    print(f"\n[ORCHESTRATOR] Giao task cho Claude: {task_description}")
    try:
        claude_exe = find_claude()
        subprocess.run(
            [claude_exe, "-p", prompt, "--allowedTools", "Edit,Write,Bash"],
            cwd=str(ROOT),
            shell=False,
        )
    except FileNotFoundError as exc:
        print(f"[ORCHESTRATOR] ERROR: {exc}")
        raise


def review_task(status: dict) -> tuple[bool, str]:
    task_id = status.get("current_task", "")
    notes = status.get("implementation_notes", "")
    files = status.get("files_changed", [])
    issues: list[str] = []

    if not files:
        issues.append("Khong co file nao duoc thay doi")
    if not notes:
        issues.append("Khong co implementation_notes")

    for file_path in files:
        full_path = ROOT / file_path
        if not full_path.exists():
            issues.append(f"File khong ton tai: {file_path}")
        elif full_path.stat().st_size == 0:
            issues.append(f"File rong: {file_path}")

    for file_path in files:
        if file_path.endswith(".py"):
            full_path = ROOT / file_path
            if full_path.exists():
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(full_path)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    issues.append(f"Syntax error trong {file_path}: {result.stderr.strip()}")
                else:
                    content = full_path.read_text(encoding="utf-8")
                    if "from __future__ import annotations" not in content:
                        issues.append(f"Thieu future import trong {file_path}")

    if issues:
        review_message = f"Task {task_id} REJECTED:\n" + "\n".join(f"- {issue}" for issue in issues)
        append_review(review_message)
        print("\ncodex-5.5 REJECTED:")
        for issue in issues:
            print(f"  - {issue}")
        return False, "\n".join(issues)

    review_message = f"Task {task_id} APPROVED.\nFiles: {files}\nNotes: {notes}"
    append_review(review_message)
    print(f"\ncodex-5.5 APPROVED: {task_id}")
    return True, "ok"


def advance_task(current_task: str) -> str | None:
    plan_text = PLAN_FILE.read_text(encoding="utf-8")
    updated = plan_text.replace(f"- [ ] {current_task}", f"- [x] {current_task}")
    PLAN_FILE.write_text(updated, encoding="utf-8")
    for line in updated.splitlines():
        stripped = line.strip()
        if stripped.startswith("- [ ]"):
            parts = stripped.split()
            if len(parts) >= 3:
                return parts[2].rstrip(":")
    return None


def get_task_description(task_id: str) -> str:
    plan_text = PLAN_FILE.read_text(encoding="utf-8")
    for line in plan_text.splitlines():
        if f" {task_id}:" in line or f" {task_id} " in line:
            cleaned = line.strip()
            for prefix in ("- [ ] ", "- [x] "):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
            return cleaned
    return task_id


def get_phase_for_task(task_id: str) -> str:
    plan_text = PLAN_FILE.read_text(encoding="utf-8")
    current_phase = "Giai doan 1"
    for line in plan_text.splitlines():
        if line.startswith("## Giai doan"):
            current_phase = line.lstrip("# ").strip()
        if f" {task_id}:" in line or f" {task_id} " in line:
            return current_phase
    return current_phase


def run_loop(max_retries: int = 2) -> None:
    print("[ORCHESTRATOR] Bat dau vong lap multi-agent...")
    retry_count = 0

    while True:
        status = read_status()
        current_task = status.get("current_task")
        task_status = status.get("status", "pending")

        if not current_task:
            print("[ORCHESTRATOR] Khong con task nao. Hoan thanh!")
            break

        task_description = get_task_description(current_task)
        print(f"\n[ORCHESTRATOR] Task: {current_task} | Status: {task_status}")

        if task_status == "pending":
            status["status"] = "in_progress"
            status["last_updated"] = now_iso()
            write_status(status)
            run_claude(task_description)
            time.sleep(2)

        elif task_status == "done":
            approved, notes = review_task(status)
            if approved:
                retry_count = 0
                next_task = advance_task(current_task)
                if next_task:
                    new_status = {
                        "current_phase": get_phase_for_task(next_task),
                        "current_task": next_task,
                        "task_description": get_task_description(next_task),
                        "status": "pending",
                        "assigned_to": "claude",
                        "last_updated": now_iso(),
                        "review_status": None,
                        "review_notes": None,
                        "implementation_notes": None,
                        "files_changed": [],
                    }
                    write_status(new_status)
                    print(f"[ORCHESTRATOR] Chuyen sang task: {next_task}")
                else:
                    print("[ORCHESTRATOR] Tat ca task da hoan thanh!")
                    break
            else:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"[ORCHESTRATOR] Task {current_task} that bai {max_retries} lan. Dung lai.")
                    status["status"] = "blocked"
                    status["review_notes"] = notes
                    status["last_updated"] = now_iso()
                    write_status(status)
                    break
                status["status"] = "pending"
                status["review_notes"] = notes
                status["last_updated"] = now_iso()
                write_status(status)
                print(f"[ORCHESTRATOR] Retry {retry_count}/{max_retries} cho task {current_task}")

        elif task_status == "in_progress":
            print("[ORCHESTRATOR] Claude dang lam viec... cho 5 giay")
            time.sleep(5)

        elif task_status == "blocked":
            print(f"[ORCHESTRATOR] Task {current_task} bi block. Can nguoi xu ly.")
            break

        else:
            print(f"[ORCHESTRATOR] Trang thai khong xac dinh: {task_status}")
            break


if __name__ == "__main__":
    run_loop()
