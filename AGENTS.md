# AGENTS.md - Multi-Agent Collaboration Guide

## 2 Agent trong he thong

| Agent | Role | Tool |
|-------|------|------|
| **mô hình-5.5** (Reviewer) | Phan bien, review code, phat hien loi logic, approve/reject | mô hình-5.5 CLI |
| **Claude** (Implementer) | Viet code, implement features theo thiet ke he thong | Claude Code CLI |

## Kien truc he thong

Crawler XSMB
    -> DuckDB (file xsmb.duckdb, OLAP, single-process, zero-ops)
    -> Feature Engine
    -> Feature Store
    -> Model Training (Layer1 -> Layer2 -> Layer3)
    -> nhà cung cấp dịch vụ AI (Dau/Duoi/Cham/Tong/00-99 + Ensemble)
    -> Prediction API
    -> Dashboard / Telegram Bot

## Roadmap 10 giai doan (theo thiet ke)

1. Thu thap + Lam sach du lieu
2. Xay DuckDB schema + Feature Store
3. Sinh toan bo Feature
4. Huan luyen XGBoost
5. LightGBM + CatBoost
6. nhà cung cấp dịch vụ AI (Ensemble)
7. Backtest
8. Dashboard
9. Telegram Bot
10. Online Learning

## File giao tiep giua 2 agent

- agent_plan.md     -> ke hoach tong the theo 10 giai doan (mô hình-5.5 cap nhat)
- agent_status.json -> trang thai task hien tai (Claude ghi done, mô hình-5.5 ghi approved)
- agent_review.md   -> log phan bien cua mô hình-5.5
- CLAUDE.md         -> huong dan rieng cho Claude
- AGENTS.md         -> file nay (thong tin chung cho ca 2 agent)

## Quy trinh lam viec (vong lap)

1. mô hình-5.5 doc agent_plan.md -> chon task pending -> set agent_status.json status=pending
2. Claude doc agent_status.json -> implement -> set status=done, files_changed, implementation_notes
3. mô hình-5.5 chay: python agents/kiro_review.py -> approve hoac reject
4. Neu APPROVE: python agents/next_task.py -> tu dong chuyen sang task moi
5. Neu REJECT: mô hình-5.5 ghi review_notes -> Claude sua lai -> lap lai tu buoc 2
6. Chay tu dong: python agents/orchestrator.py

## Conventions (ap dung cho moi file trong scope)

- from __future__ import annotations o dau moi file .py
- Type hints bat buoc cho tat ca function (params + return)
- Khong dung bien 1 chu cai
- Docstring tieng Viet cho cac ham domain-specific XSMB
- Test file dat trong xsmb_pipeline/tests/
- Doc code hien co truoc khi viet moi, match style hien tai
- Khong them dependency moi khi chua duoc mô hình-5.5 approve

