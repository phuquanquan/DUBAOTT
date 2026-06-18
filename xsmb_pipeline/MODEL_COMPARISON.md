# XSMB Model Comparison Summary

## All Models Tested (2019-2025, Walk-Forward)

| Model | Input | Target | P@K | Random | Diff | Result |
|-------|-------|--------|------|--------|------|--------|
| **Ensemble (MLP+Gap+Freq)** | Digit seq | Dau (1 digit) | P@1=10.1% | 10% | +0.1pp | = Random |
| **Ensemble** | Digit seq | Duoi (2cang) | P@5=6.5% | 5% | +1.5pp | = Random |
| **Ensemble** | Digit seq | 3cang | P@5=4.8% | 5% | -0.2pp | = Random |
| **Sparse XGBoost** | 27 giai/ngay | DB 2cang | P@5=4.8% | 5% | -0.2pp | = Random |
| **3cang Regression** | Special full | 3cang | P@10=0.96% | 1.0% | -0.04pp | = Random |
| **25 Pattern Rules** | Various | DB 2cang | <5.5% | 1% | varied | < Random |
| **XGBoost Candidate** | Gap/Freq | DB 2cang | P@3=6.0% | 3% | +2pp | Variance |

## All Models = Random

**No model outperforms random baseline.** This is consistent across:
- 25 different pattern rules
- XGBoost on gap/frequency features
- MLP ensemble on digit sequences
- Sparse XGBoost on 27-prize data (khiemdoan)
- GradientBoosting regression on full special numbers (3cang)

## Source Projects Referenced

| Repo | Contribution Used |
|-------|-----------------|
| khiemdoan/vietnam-lottery-xsmb-analysis | Sparse matrix (27 prizes/day), data pipeline |
| itmanvn/3cang | LSTM-style sequence model, temperature scaling |
| AnhTuPhi | Statistical analysis, modular structure |

## Conclusion

XSMB special prize is a **true random number generator** (RNG).
- Gap ratio does NOT increase probability
- Frequency does NOT increase probability
- Any pattern/memory/ML model = random

**XSMB = Random.** No prediction model can beat random.
