# CLAUDE.md - Huong dan cho Claude Code Agent

## Role cua ban: IMPLEMENTER AGENT

Ban la agent VIET CODE va IMPLEMENT. Nhiem vu:
1. Doc agent_status.json -> lay current_task + task_description
2. Doc agent_plan.md -> hieu vi tri task trong 10 giai doan roadmap
3. Doc code hien co lien quan truoc khi viet moi
4. Implement dung thiet ke he thong phia duoi
5. Cap nhat agent_status.json khi xong
6. Doi mô hình-5.5 review - TUYET DOI KHONG tu y chuyen sang task moi

---

## Thiet ke he thong can nam ro

### Kien truc tong the (Database = DuckDB - file xsmb.duckdb, KHONG dung PostgreSQL)
Crawler XSMB
    -> DuckDB (xsmb.duckdb)
    -> Feature Engine
    -> Feature Store (bang features_daily)
    -> Model Training
         Layer 1: XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees
         Layer 2: Logistic Regression, Ridge, ElasticNet (meta models)
         Layer 3: LSTM, GRU, Transformer nho
    -> nhà cung cấp dịch vụ AI
         Model A: Dau (0-9)
         Model B: Duoi (0-9)
         Model C: Cham
         Model D: Tong
         Model E: So 00-99
         -> Ensemble tat ca
    -> Prediction API
    -> Dashboard (TOP 20 so + Score + Giai thich)
    -> Telegram Bot (/predict /top10 /top20 /explain)

### Database Schema (DuckDB - xsmb_pipeline/database.py)
- draws:           draw_date PK, region, special, first_prize
- prizes:          PK(draw_date, prize_name, prize_index), prize_value
- loto_results:    PK(draw_date, loto_number)
- digit_positions: PK(draw_date, prize_name, prize_index, position_index), digit
- features_daily:  PK(draw_date, feature_name), feature_value DOUBLE  -- UPSERT
- pattern_scores:  PK(pattern_id), hit_rate, roi, precision, stability  -- UPSERT

### Feature Engine can sinh
Frequency:   freq_{so}_{3|7|14|30|60|90|180|365}d
Gan:         gan_{so}_current, gan_{so}_mean, gan_{so}_max, gan_{so}_min
Position:    db_v1..v5, g1_v1..v5, g2_v1..v5, ..., g7_v1..v2
Cross Pos:   dbv1_plus_g1v3, dbv2_plus_g3v4, dbv5_minus_g4v2...
Pattern:     pascal_{so}, reverse_{so}, mirror_{so}, modulo_{so}...
Roi:         roi_1d_{so}, roi_2d_{so}, roi_3d_{so}, roi_from_db_{so}
Dau/Duoi:    dau_hot_0..9, duoi_hot_0..9
Cham:        cham_0..9
Tong:        tong_de, tong_lo
Markov:      markov1_{so}, markov2_{so}, markov3_{so}
Graph:       degree_{so}, pagerank_{so}, centrality_{so}

### Backtest Walk-Forward
2010-2016 -> Test 2017
2010-2017 -> Test 2018
...
2010-2024 -> Test 2025
Metrics: Hit Rate, ROI, Precision@5, Precision@10

### Dashboard Output
TOP 20 so | Score | Giai thich
Vi du: 27 | 92 | Gan cao, Markov manh, Pascal xac nhan, Dau 2 nong

### Telegram Bot
/predict  -> du doan hom nay
/top10    -> top 10 so
/top20    -> top 20 so
/explain [so] -> giai thich tai sao so do

---

## Cau truc thu muc hien tai

xsmb_pipeline/
  schema.py          <- LotteryResult(special, first, second, third, fourth, fifth, sixth, seventh)
  dataset.py         <- load/write CSV, flatten_numbers, derive_loto, dedupe_results
  scraper.py         <- fetch HTML, parse_result, crawl_daily
  features.py        <- feature engineering (329 lines - doc truoc khi them feature moi)
  signals.py         <- signal scoring system (494 lines)
  targets.py         <- loto2, loto3, special2, special3
  evaluate.py        <- backtesting, walk-forward
  models/
    base.py          <- RankingPrediction, RankingStrategy
    weighted.py      <- weighted ranking model hien tai
    sklearn_ranker.py
    xgboost_ranker.py
agents/
  orchestrator.py    <- vong lap tu dong mô hình-5.5 <-> Claude
  kiro_review.py     <- mô hình-5.5 chay de review sau khi Claude done
  next_task.py       <- advance sang task tiep theo sau approve

---

## Conventions BAT BUOC

- from __future__ import annotations o DONG DAU moi file .py
- Type hints day du: def ham(param: Kieu) -> KieuTraVe:
- Khong dung bien 1 chu cai (result thay r, candidate thay c, score thay s)
- Docstring tieng Viet cho cac ham XSMB domain
- Test dat trong xsmb_pipeline/tests/test_{ten_module}.py
- Doc file hien co TRUOC khi viet them, khong overwrite ham cu

---

## Sau khi implement xong

Cap nhat agent_status.json nhu sau:
{
  "status": "done",
  "files_changed": mô hình-5.5,
  "implementation_notes": "Mo ta ngan: da tao gi, ham gi, test gi"
}

Roi noi voi mô hình-5.5: "Xong task [ID], mô hình-5.5 hay chay kiro_review.py de review"
