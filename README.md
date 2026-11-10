# Recloser Outage Probability Forecasting

Daily outage-probability forecasts for 9 distribution reclosers in Puerto Princesa, built from
weather forecasts, vegetation, outage history, equipment history and calendar features.
The primary model is a tuned, calibrated XGBoost classifier; LightGBM, Random Forest and
Logistic Regression are baselines.

## Folder contents

| Path | What it is |
|---|---|
| `train.py` | Full pipeline: tuning, cross-validation, calibration, test evaluation, ablation, reports, saved model |
| `predict.py` | Scores new recloser-days with the saved model |
| `chapter5_analysis.py` | Extra analyses for the thesis chapter: SHAP, calibration, ablations (run after `train.py`) |
| `CHAPTER_5_RESULTS_AND_DISCUSSION.md` | Chapter 5 results and discussion, organized by research objective |
| `data/recloser_modeling_dataset_66f.csv` | Modeling dataset (13,473 recloser-days, 66 features); **not included in the public repository** |
| `data/feature_dictionary.csv` | Feature → feature-group mapping |
| `model/` | Saved XGBoost models (3 seeds), calibration and alert threshold (`model_card.json`) |
| `reports/` | All result tables (CSV), run log and figures |

## How to run

Requirements: Python 3.10+, `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`.

**Data availability.** The modeling dataset is derived from Palawan Electric Cooperative (PALECO)
outage records and is used for academic purposes only, so it is not published in this repository.
The trained model, the feature dictionary and all aggregated results are included. To retrain, place
`recloser_modeling_dataset_66f.csv` in `data/`.

```bash
python train.py
```

```bash
python predict.py new_days.csv predictions.csv
```

`train.py` takes about 2–5 minutes and regenerates everything in `reports/` and `model/`.
Results are deterministic (fixed seeds). `python train.py --quick` runs a fast smoke test.

`predict.py` expects a CSV with the 66 feature columns from `data/feature_dictionary.csv` and writes
`outage_probability` and `alert` per row. `--threshold` overrides the alert threshold.

## Method

**Data.** 9 reclosers, one row per recloser per day, 2022-08-31 to 2026-10-05: 13,473 rows, 418 outage
days (3.1%). 66 features in 12 groups: target-day forecast weather, lagged forecast weather, wet-weather
history, weather stress, vegetation, weather–vegetation interactions, outage history, equipment history,
equipment-category history, repair history, calendar and recloser identity.

**Split.** Chronological, with no information from the future used for training or any modeling choice.

| Period | Rows | Outages | Rate | Use |
|---|---|---|---|---|
| Development: 2022-08-31 to 2025-12-31 | 10,971 | 358 | 3.26% | Training, tuning, calibration, threshold |
| Test: 2026-01-01 to 2026-10-05 | 2,502 | 60 | 2.40% | Final evaluation only |

**Rolling-origin cross-validation.** Four half-year validation folds (2024H1, 2024H2, 2025H1, 2025H2),
each trained on all earlier data, give 213 validation outages for model selection.

**Tuning.** 25-draw random search per model family (XGBoost, LightGBM), scored by mean fold PR-AUC.
Positive-class weighting (none / √ratio / full ratio) is part of the search.

**Calibration and alert threshold.** Platt scaling is fitted on the pooled out-of-fold predictions. The
alert threshold is the one with the highest F1 among thresholds whose cross-validated recall falls within
52–68%.

**Final model.** Refit on the whole development period, averaged over 3 random seeds, and evaluated once
on 2026. Confidence intervals come from a day-block bootstrap (1,000 resamples of whole days), because
outages on the same day are correlated across reclosers.

Selected XGBoost hyperparameters: `n_estimators=250, learning_rate=0.05, max_depth=4,
min_child_weight=12, subsample=0.6, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=15, gamma=1.0`,
no positive-class weighting.

## Results

### Cross-validation (2024–2025, four folds)

| Model | PR-AUC (mean ± sd) | ROC-AUC (pooled) |
|---|---|---|
| **XGBoost** | **0.431 ± 0.110** | **0.868** |
| LightGBM | 0.433 ± 0.117 | 0.864 |
| Random Forest | 0.409 ± 0.112 | 0.855 |
| Logistic Regression | 0.092 ± 0.029 | 0.738 |

### Test period (2026)

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Lift | Precision | Recall | F1 | BSS |
|---|---|---|---|---|---|---|---|
| **XGBoost** | **0.854** (0.795–0.901) | **0.221** (0.136–0.362) | **9.2×** | **0.329** | **0.467** | **0.386** | **0.094** |
| LightGBM | 0.839 (0.784–0.886) | 0.204 (0.120–0.334) | 8.5× | 0.259 | 0.250 | 0.254 | 0.087 |
| Random Forest | 0.838 (0.782–0.885) | 0.152 (0.088–0.271) | 6.3× | 0.179 | 0.317 | 0.229 | 0.061 |
| Logistic Regression | 0.785 (0.728–0.840) | 0.076 (0.048–0.138) | 3.2× | 0.066 | 0.583 | 0.118 | 0.023 |

Lift = PR-AUC ÷ test outage rate (0.024). BSS = Brier skill score relative to the test outage rate.

### Confusion matrix: XGBoost, 2026, threshold 0.122

|  | Predicted outage | Predicted no outage | Total |
|---|---|---|---|
| **Actual outage** | **28** (TP) | 32 (FN) | 60 |
| **Actual no outage** | 57 (FP) | **2,385** (TN) | 2,442 |
| **Total** | 85 alerts | 2,417 | 2,502 |

- Precision = 28 / 85 = **32.9%**: about 1 in 3 alerts is a real outage.
- Recall = 28 / 60 = **46.7%**.
- Specificity = 2,385 / 2,442 = **97.7%**.
- About 2.1 alerts per week across all 9 reclosers.

### Alert threshold trade-off (XGBoost)

Each threshold is set on the cross-validated predictions to hit a recall target, then applied unchanged to 2026.

| Recall target | Threshold | 2026 recall | 2026 precision | TP | FP | FN | Alerts / week |
|---|---|---|---|---|---|---|---|
| 40% | 0.224 | 25.0% | 36.6% | 15 | 26 | 45 | 1.0 |
| 45% | 0.190 | 28.3% | 33.3% | 17 | 34 | 43 | 1.3 |
| 50% | 0.138 | 41.7% | 33.3% | 25 | 50 | 35 | 1.9 |
| 55% | 0.110 | 46.7% | 29.8% | 28 | 66 | 32 | 2.4 |
| 60% | 0.089 | 46.7% | 25.0% | 28 | 84 | 32 | 2.8 |
| 65% | 0.068 | 50.0% | 21.0% | 30 | 113 | 30 | 3.6 |
| 70% | 0.048 | 51.7% | 15.9% | 31 | 164 | 29 | 4.9 |

In 2026, recall levels off near 50%. Raising the recall target beyond 55% mostly adds false alarms: about
half of the 2026 outages occurred on days the model rated as low risk.

### Feature-group ablation (XGBoost)

| Configuration | Features | CV PR-AUC (mean ± sd) | Test PR-AUC (95% CI) | Test ROC-AUC |
|---|---|---|---|---|
| A1: Weather only | 31 | 0.366 ± 0.097 | 0.212 (0.129–0.348) | 0.844 |
| A2: + Vegetation & interactions | 39 | 0.376 ± 0.103 | 0.213 (0.124–0.347) | 0.841 |
| A3: + Outage & equipment history | 59 | 0.415 ± 0.107 | 0.226 (0.136–0.368) | 0.846 |
| A4: Full feature set | 66 | 0.431 ± 0.110 | 0.221 (0.136–0.362) | 0.854 |
| L1: Full − weather–vegetation interactions | 64 | 0.428 ± 0.110 | 0.221 (0.138–0.363) | 0.858 |
| L2: Full − equipment & repair history | 54 | 0.434 ± 0.111 | 0.222 (0.138–0.362) | 0.855 |
| L3: Full − outage history | 58 | 0.403 ± 0.111 | 0.208 (0.122–0.345) | 0.843 |

- **Weather carries most of the signal.** Weather alone reaches CV PR-AUC 0.366.
- **Outage history adds the most on top of weather.** Removing it lowers CV PR-AUC by 0.029.
- **Vegetation, weather–vegetation interactions and equipment/repair history** change CV PR-AUC by less
  by about 0.01 or less, which is well within fold-to-fold variation (sd ≈ 0.10).

### Feature importance (XGBoost, share of total gain)

| Group | Share |
|---|---|
| Target-day forecast weather | 48.5% |
| Lagged forecast weather | 15.0% |
| Weather stress (vs. p90) | 13.9% |
| Calendar | 7.2% |
| Outage history | 6.5% |
| Weather–vegetation interactions | 5.4% |
| Vegetation | 2.9% |
| Other groups | < 0.5% each |

The top individual features are: gust relative to its p90, mean wind speed, maximum humidity,
precipitation, minimum humidity and precipitation relative to its p90.

### Calibration

The calibrated probabilities track observed outage rates across risk deciles (`reports/calibration_table.csv`,
`reports/figures/calibration.png`). The top decile has a mean predicted probability of 14.2% against an
observed rate of 13.9%, and it contains 35 of the 60 test outages.

## Limitations

1. **Small test sample.** 60 outages in 2026 make the confidence intervals wide (PR-AUC 0.14–0.36).
2. **Lower outage rate in the test year.** 2026 had fewer outages than the development period (2.4% vs
   3.3%). PR-AUC depends on the outage rate, so it is lower on 2026 than in cross-validation, while
   ROC-AUC is stable (0.87 vs 0.85).
3. **Weather inputs.** Target-day weather comes from the Open-Meteo Historical Forecast API (ECMWF IFS
   9 km). That archive joins together the first hours of successive model runs, so its values are
   short-range forecasts and are more accurate than a forecast available the evening before. The reported
   performance is therefore an upper bound on operational day-ahead performance.
4. **Outage causes not covered by the features.** Many outages (e.g. animals, vehicles, equipment wear)
   have no weather or history signal, which caps achievable recall.
5. **Limited spatial coverage.** 9 reclosers in one city. Generalisation to other networks has not been
   tested.

## Report files (`reports/`)

| File | Content |
|---|---|
| `run_log.txt` | Full console output of the training run |
| `cv_model_comparison.csv` | Cross-validated PR-AUC per fold and pooled ROC-AUC, all models |
| `test_model_comparison.csv` | Test metrics, confusion counts and bootstrap CIs, all models |
| `confusion_matrix.csv` | XGBoost confusion matrix on 2026 |
| `threshold_tradeoff.csv` | Recall/precision/alerts at 7 recall targets |
| `ablation.csv` | Feature-group ablation (CV and test) |
| `feature_importance.csv`, `feature_group_importance.csv` | XGBoost gain importance |
| `calibration_table.csv` | Predicted vs observed outage rate by decile |
| `hyperparameter_search.csv`, `best_hyperparameters.json` | All search draws and the selected settings |
| `test_predictions.csv` | Per-row 2026 probabilities for every model and XGBoost alerts |
| `figures/` | Precision–recall curve, ROC curve, calibration plot, feature importance, threshold trade-off |
