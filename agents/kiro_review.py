from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATUS_FILE = ROOT / "agent_status.json"
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


def check_syntax(file_path: Path) -> list[str]:
    issues = []
    if file_path.suffix == ".py" and file_path.exists():
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(file_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            issues.append(f"Syntax error trong {file_path.name}: {result.stderr.strip()}")
    return issues


def check_conventions(file_path: Path) -> list[str]:
    issues = []
    if file_path.suffix != ".py" or not file_path.exists():
        return issues
    content = file_path.read_text(encoding="utf-8")
    if "from __future__ import annotations" not in content:
        issues.append(f"Thieu `from __future__ import annotations` trong {file_path.name}")
    lines = content.splitlines()
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            if "->" not in stripped and ":" in stripped:
                issues.append(f"Ham tai dong {line_num} trong {file_path.name} thieu return type hint")
    return issues


def run_tests(changed_files: list[str]) -> list[str]:
    issues = []
    test_dir = ROOT / "xsmb_pipeline" / "tests"
    if not test_dir.exists():
        return issues
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir), "-x", "-q", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        issues.append(f"Tests that bai:\n{result.stdout[-1000:]}\n{result.stderr[-500:]}")
    return issues


def review() -> None:
    status = read_status()
    task_id = status.get("current_task", "?")
    task_status = status.get("status", "")

    if task_status != "done":
        print(f"[KIRO] Task {task_id} chua done (status={task_status}). Chua can review.")
        return

    files_changed = status.get("files_changed", [])
    impl_notes = status.get("implementation_notes", "")
    issues: list[str] = []

    print(f"\n[KIRO] Bat dau review task: {task_id}")
    print(f"[KIRO] Files thay doi: {files_changed}")
    print(f"[KIRO] Implementation notes: {impl_notes}")

    if not files_changed:
        issues.append("Khong co file nao duoc thay doi")
    if not impl_notes:
        issues.append("Khong co implementation_notes")

    for rel_path in files_changed:
        full_path = ROOT / rel_path
        if not full_path.exists():
            issues.append(f"File khong ton tai: {rel_path}")
            continue
        if full_path.stat().st_size == 0:
            issues.append(f"File rong: {rel_path}")
            continue
        issues.extend(check_syntax(full_path))
        issues.extend(check_conventions(full_path))

    issues.extend(run_tests(files_changed))

    if issues:
        review_message = (
            f"Task {task_id} REJECTED by Kiro:\n"
            + "\n".join(f"- {issue}" for issue in issues)
        )
        append_review(review_message)
        status["review_status"] = "rejected"
        status["review_notes"] = "\n".join(issues)
        status["status"] = "pending"
        status["last_updated"] = now_iso()
        write_status(status)
        print(f"\n[KIRO] REJECTED — {len(issues)} van de:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        review_message = (
            f"Task {task_id} APPROVED by Kiro.\n"
            f"Files: {files_changed}\n"
            f"Notes: {impl_notes}"
        )
        append_review(review_message)
        status["review_status"] = "approved"
        status["review_notes"] = "OK"
        status["last_updated"] = now_iso()
        write_status(status)
        print(f"\n[KIRO] APPROVED task {task_id}")


if __name__ == "__main__":
    review()

