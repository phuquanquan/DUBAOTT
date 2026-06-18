# Agent Plan - AI Soi Cau XSMB Toan Dien (2010-2026)
# Theo dung Thiet ke He thong + Roadmap 10 Giai doan
#
# Database: DuckDB (file xsmb.duckdb) - workload OLAP/analytics single-process.

============================================================
## Giai doan 1 - Thu thap + Lam sach du lieu
============================================================
- [x] P0.1: Tao xsmb_pipeline/validator.py - validate format tung giai
- [x] P0.2: Phat hien ngay missing trong chuoi 2010-2026
- [x] P0.3: Validate lo to 00-99, normalize ngay DD/MM/YYYY
- [x] P0.4: Detect outlier (ngay co qua it hoac qua nhieu so)
- [x] P0.5: Data quality report tu dong khi load CSV
- [x] P0.6: Auto-fill hoac flag ngay thieu du lieu

============================================================
## Giai doan 2 - Xay DuckDB Schema + Feature Store
============================================================
- [x] P1.1: Setup DuckDB schema + bang draws
- [x] P1.2: Bang prizes (draw_date, prize_name, prize_index, prize_value)
- [x] P1.3: Bang loto_results (draw_date, loto_number)
- [x] P1.4: Bang digit_positions
- [x] P1.5: Bang features_daily (draw_date, feature_name, feature_value) + UPSERT
- [x] P1.6: Bang pattern_scores + UPSERT
- [x] P1.7: CLI migrate-duckdb + integrity-check

============================================================
## Giai doan 3 - Sinh toan bo Feature
============================================================
- [x] P2.1: Frequency Generator - freq 3/7/14/30/60/90/180/365 ngay
- [x] P2.2: Gan Generator - gan hien tai, trung binh, lon nhat, nho nhat
- [x] P2.3: Position Generator - DB_V1..V5, G1_V1..V5
- [x] P2.4: Cross Position Generator
- [x] P2.5: Cross Day Generator
- [x] P2.6: Pattern Generator - Pascal, Reverse, Mirror, Modulo
- [x] P2.7: Roi Generator - roi 1/2/3 ngay
- [x] P2.8: Dau/Duoi/Cham/Tong generator
- [x] P2.9-19: Feature Discovery, Genetic Programming, Symbolic Regression,
              Markov bac 1/2/3, HMM, Chu ky 7/14/30/60/90, Apriori, FP-Growth,
              Sequential Pattern, Graph AI (Degree, PageRank, Centrality)

============================================================
## Giai doan 4 - Huan luyen XGBoost
============================================================
- [x] P3.1: XGBoost ranker - train/predict/evaluate
- [x] P3.2-5: SHAP, Mutual Information, Permutation Importance, RFE

============================================================
## Giai doan 5 - LightGBM + CatBoost
============================================================
- [x] P4.1-5: LightGBM, CatBoost, Random Forest, Extra Trees, Ensemble L1

============================================================
## Giai doan 6 - Provider Models
============================================================
- [x] P5.1-4: Layer 2 Meta (Logistic, Ridge, ElasticNet, Stacking)
- [x] P5.5-9: Model A-D-E (Dau, Duoi, Cham, Tong, So 00-99)
- [x] P5.10-13: LSTM, GRU, Transformer (Layer 3)

============================================================
## Giai doan 7 - Backtest
============================================================
- [x] P6.1-2: Walk-forward + Metrics (Hit Rate, ROI, Precision@5/10)
- [x] P6.3-5: Evaluation system, Regime Detection, HMM

============================================================
## Giai doan 8 - Dashboard
============================================================
- [x] P7.1: Dashboard TOP 20 so + Score
- [x] P7.2: Giai thich ly do (Gan cao, Markov manh, Pascal xac nhan...)
- [x] P7.3: Hien thi metrics theo tung nam (2010-2026)

============================================================
## Giai doan 9 - Telegram Bot + Auto-Update
============================================================
- [x] P8.1: /predict - du doan hom nay
- [x] P8.2: /top10, /top20 - top so hom nay
- [x] P8.3: /explain [so] - giai thich tai sao so do duoc chon
- [x] P8.4: Daily Scheduler 18h30 - auto-update + predict + save JSON
- [x] P8.5: ensure_latest_data() tich hop vao TAT CA predict/analyze flows
- [x] P9.1: MLflow tracking - Model registry + Feature registry + Training runs

============================================================
## Giai doan 10 - Online Learning (HOAN THANH)
============================================================
- [x] P10.1: online_learning.py - OnlineLearningEngine, PredictionFeedback,
            DailyPrediction, compute_hit_metrics, compute_weight_adjustment
- [x] P10.2: Scheduler tich hop online learning
            (_run_online_feedback() sau khi predict)
- [x] P10.3: Don dep code trung lap
            (12 file agents/ + 6 file root + 7 file agents/)
- [x] P10.4: Fix import path (telegram_bot, mlflow_tracking,
            online_learning: from .. -> from .)
- [x] P10.5: All CLI commands tich hop ensure_latest_data() truoc predict/analyze

============================================================
## FINAL - TOAN BO 10 GIAI DOAN DA HOAN THANH
============================================================

## Chot thiet ke
- Output chinh: loto2 / 00-99
- Dau / Duoi / Cham / Tong: chi la preset / kiieu soi
- Database: DuckDB (`xsmb.duckdb`)
- All modules quy ve loto2

## Final mapping
- Phase 1: `validator.py`, `scraper.py`, `dataset.py`
- Phase 2: `database.py`
- Phase 3: `feature_generators.py`, `features.py`
- Phase 4: `models/xgboost_ranker.py`
- Phase 5: `models/sklearn_ranker.py`, `evaluate.py`
- Phase 6: `signals.py`, `models/neural_ranker.py`, `models/__init__.py`
- Phase 7: `evaluate.py`
- Phase 8: `evaluate.py` (build_full_dashboard_payload)
- Phase 9: `telegram_bot.py`, `scheduler.py`, `mlflow_tracking.py`
- Phase 10: `online_learning.py`, `scheduler.py` (tich hop)

## Architecture (cuoi cung)
Crawler -> DuckDB (xsmb.duckdb) -> Feature Engine -> Feature Store
-> Model Training (L1->L2->L3->L4) -> Provider presets -> Prediction API
-> Dashboard / Telegram Bot -> Scheduler 18h30
-> Online Learning (Predict -> Compare -> Update Weights)
-> MLflow Tracking

## Su dung cuoi cung
```bash
# Chay scheduler 18h30
python -m xsmb_pipeline.cli schedule --csv xsmb_full.csv --db xsmb.duckdb --token TOKEN

# Predict voi data cuoi cung (auto-update neu thieu)
python -m xsmb_pipeline.cli predict-ensure --csv xsmb_full.csv --top-k 5 --tuned

# Xem MLflow tracking
python -m xsmb_pipeline.cli mlflow-summary

# Test Telegram Bot
python agents/telegram_run.py --test
```
