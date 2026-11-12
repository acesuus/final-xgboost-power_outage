# Step 4. Model Development

*(Replaces the current "Step 4. Model Development" text in Section 4.7. All code comes from `train.py`
in the project folder, shortened for readability. Figures and numbers match the final model.)*

---

The model was developed in Python using Pandas, NumPy, Scikit-learn and XGBoost. Pandas and NumPy were
used to organize the records and prepare the inputs. XGBoost was used to build the outage forecasting
model. Scikit-learn was used to tune the model, adjust its probability estimates and measure its
performance. The development followed nine steps, described below.

### 4.1 Loading the Data and the List of Inputs

The prepared dataset was loaded into Python. It contained 13,473 recloser-day records from 31 August 2022
to 5 October 2026, of which 418 (3.10%) were labeled as outages. A separate feature dictionary listed the
66 inputs used by the model and the group each one belongs to (for example, weather, vegetation or outage
history).

```python
import pandas as pd

df = pd.read_csv("data/recloser_modeling_dataset_66f.csv")
feat_dict = pd.read_csv("data/feature_dictionary.csv")

all_features = feat_dict["feature"].tolist()          # the 66 model inputs
groups = feat_dict.groupby("group")["feature"].apply(list).to_dict()
```

The outage label (`outage_label`) was kept separate from the inputs. It was used only as the answer the
model learns to predict, never as an input.

### 4.2 Splitting the Data by Time

Because the model must forecast future outages, the records were split by date and not at random. All
records before 1 January 2026 formed the **development set**, used to build the model. The records from
1 January to 5 October 2026 formed the **test set**, which was set aside and used only once, at the very
end, to measure the finished model.

```python
TEST_START = "2026-01-01"

dev  = df[df["date"] <  TEST_START].reset_index(drop=True)   # 10,971 rows, 358 outages
test = df[df["date"] >= TEST_START].reset_index(drop=True)   #  2,502 rows,  60 outages

y_dev  = dev["outage_label"].to_numpy()
y_test = test["outage_label"].to_numpy()
```

Splitting by time makes sure the model is never trained on information from the period it is tested on,
as would happen in real use.

### 4.3 Preparing the Model Inputs

The dataset had no missing values and no duplicate records, so no values had to be filled in. Decision
trees, which XGBoost is built on, split each input by comparing it with a cut-off value. They are not
affected by the scale of the inputs, so the numeric inputs were used as they are, without scaling.

The recloser name is text, so it was turned into nine yes/no (1/0) columns, one for each recloser. This
is called one-hot encoding. The list of nine reclosers is fixed, so every set of records gets exactly the
same columns. The weather code was kept as a number. After this step, the 66 inputs became **74 model
input columns**: 65 numeric inputs plus 9 recloser columns.

```python
RECLOSERS = sorted(df["recloser"].unique())   # the 9 reclosers

def make_X(frame, feats):
    """Numeric inputs as they are + one 1/0 column per recloser."""
    num = [f for f in feats if f != "recloser"]
    X = frame[num].astype(float).copy()
    for r in RECLOSERS:
        X["rc_" + r.replace(" ", "_")] = (frame["recloser"] == r).astype(float)
    return X.to_numpy()

X_dev  = make_X(dev,  all_features)    # 10,971 x 74
X_test = make_X(test, all_features)    #  2,502 x 74
```

### 4.4 Setting Up Rolling Validation

To check the model during development without touching the test set, the last two years of the
development set (2024–2025) were divided into four half-year **validation folds**. For each fold, the
model was trained only on the records **before** that half-year and then asked to predict it. The
training period therefore grows with each fold, just as a real model would be retrained as new data
arrives. This is called rolling-origin cross-validation.

| Fold | Trained on | Predicts | Validation outages |
|---|---|---|---|
| 1 | 2022-08-31 to 2023-12-31 (4,392 rows) | 2024 January–June | 63 |
| 2 | 2022-08-31 to 2024-06-30 (6,030 rows) | 2024 July–December | 60 |
| 3 | 2022-08-31 to 2024-12-31 (7,686 rows) | 2025 January–June | 38 |
| 4 | 2022-08-31 to 2025-06-30 (9,315 rows) | 2025 July–December | 52 |

```python
CV_FOLDS = [("2024-01-01", "2024-06-30"),
            ("2024-07-01", "2024-12-31"),
            ("2025-01-01", "2025-06-30"),
            ("2025-07-01", "2025-12-31")]

def fold_masks():
    for val_start, val_end in CV_FOLDS:
        train_rows = (dev["date"] < val_start).to_numpy()
        val_rows   = ((dev["date"] >= val_start) & (dev["date"] <= val_end)).to_numpy()
        yield train_rows, val_rows
```

Together the four folds give **213 validation outages**, far more than a single validation period
would. Each fold's predictions are made by a model that never saw that fold, so they are called
**out-of-fold predictions**. They were collected into one validation set, which was used for tuning
(4.6), calibration (4.7) and the warning threshold (4.8).

### 4.5 Building the XGBoost Model

XGBoost builds a series of small decision trees. Each new tree tries to correct the mistakes of the trees
before it, and the model adds up the results of all trees to estimate the probability of an outage. The
model was set up for yes/no (binary) prediction. To make the result less dependent on luck, every model
was trained three times with different random seeds (42, 7 and 2024), and the three predicted
probabilities were averaged.

```python
import numpy as np
import xgboost as xgb

SEEDS = [42, 7, 2024]

def fit_models(params, X_train, y_train, seeds=SEEDS):
    models = []
    for s in seeds:
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              tree_method="hist", random_state=s, n_jobs=-1, **params)
        m.fit(X_train, y_train)
        models.append(m)
    return models

def predict_models(models, X):
    """Average predicted outage probability of the seed models."""
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

def run_cv(params, X_dev, seeds=SEEDS):
    """Rolling validation: returns out-of-fold predictions and the PR-AUC of each fold."""
    oof = np.full(len(dev), np.nan)
    fold_scores = []
    for train_rows, val_rows in fold_masks():
        models = fit_models(params, X_dev[train_rows], y_dev[train_rows], seeds)
        oof[val_rows] = predict_models(models, X_dev[val_rows])
        fold_scores.append(average_precision_score(y_dev[val_rows], oof[val_rows]))
    return oof, fold_scores
```

### 4.6 Tuning the Model Settings (Hyperparameters)

XGBoost has settings that control how the trees are built: how many trees to build, how deep each tree
can be, how fast the model learns, and how strongly it is held back from memorizing the training data
(regularization). To find good settings, a **random search** tried 25 combinations drawn from the ranges
in Table 4.7.

The search also tested whether outage days should be given more weight during training
(`scale_pos_weight`), because outages are about 30 times rarer than non-outage days. Three options were
tried: no extra weight, the square root of the class ratio, and the full class ratio.

Each combination was trained and scored on the four validation folds. Its score was the **average
precision (PR-AUC)** across the folds. PR-AUC measures how well the model ranks the rare outage days
above normal days, so it is a better guide than accuracy when outages are rare. The combination with the
highest average score was kept.

```python
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import average_precision_score

XGB_SPACE = dict(
    n_estimators=[150, 250, 400, 600], learning_rate=[0.01, 0.02, 0.03, 0.05],
    max_depth=[2, 3, 4, 5], min_child_weight=[1, 3, 6, 12, 20],
    subsample=[0.6, 0.7, 0.8, 0.9], colsample_bytree=[0.4, 0.5, 0.6, 0.75, 0.9],
    reg_alpha=[0.0, 0.5, 1.5, 3.0], reg_lambda=[1.0, 3.0, 8.0, 15.0],
    gamma=[0.0, 0.25, 1.0], spw_mode=["none", "sqrt", "full"],   # outage-day weighting
)

results = []
for params in ParameterSampler(XGB_SPACE, n_iter=25, random_state=42):
    _, fold_scores = run_cv(params, X_dev, seeds=[42])     # one seed during the search
    results.append((np.mean(fold_scores), params))

best_score, best_params = max(results, key=lambda r: r[0])
```

*(In `train.py`, `fit_models` turns `spw_mode` into the matching `scale_pos_weight` value: 1, the square
root of the class ratio, or the full ratio. This detail is left out above for clarity.)*

**Table 4.7 Hyperparameter Search Ranges and Selected Values**

| Setting | What it controls | Values tried | Selected |
|---|---|---|---|
| `n_estimators` | Number of trees | 150, 250, 400, 600 | **250** |
| `learning_rate` | How much each new tree changes the model | 0.01, 0.02, 0.03, 0.05 | **0.05** |
| `max_depth` | Maximum number of levels in a tree | 2, 3, 4, 5 | **4** |
| `min_child_weight` | Minimum amount of data needed to make a split | 1, 3, 6, 12, 20 | **12** |
| `subsample` | Share of records used to build each tree | 0.6–0.9 | **0.6** |
| `colsample_bytree` | Share of input columns used by each tree | 0.4–0.9 | **0.75** |
| `reg_alpha` | L1 regularization | 0, 0.5, 1.5, 3 | **0.5** |
| `reg_lambda` | L2 regularization | 1, 3, 8, 15 | **15** |
| `gamma` | Minimum improvement needed to make a split | 0, 0.25, 1 | **1.0** |
| `scale_pos_weight` | Extra weight given to outage days | none, √ratio, full ratio | **none (1.0)** |

The selected settings reached an average validation PR-AUC of 0.434. They describe small, shallow and
strongly regularized trees, which suits a dataset with only 358 outage days for training. Giving extra
weight to outage days did not improve the results: without weighting the average score was 0.415,
against 0.405 with √ratio weighting and 0.403 with full weighting. The class imbalance was therefore
handled through probability calibration (4.7) and the choice of warning threshold (4.8) rather than
through weighting.

### 4.7 Adjusting the Probability Estimates (Calibration)

With the selected settings, the rolling validation was run again with all three seeds to produce the
final out-of-fold predictions for 2024–2025. A model's raw probabilities do not always match how often
outages actually happen. For example, a raw value of 0.20 might correspond to outages on only 10% of such
days.

To correct this, **Platt scaling** was applied. It is a small logistic regression that learns how to
turn the model's raw probability into a calibrated one, fitted on the validation predictions and their
true labels.

```python
from sklearn.linear_model import LogisticRegression

def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

oof, fold_scores = run_cv(best_params, X_dev, seeds=SEEDS)   # 3-seed validation predictions
val = ~np.isnan(oof)                                         # the 2024-2025 validation rows

platt = LogisticRegression(C=1e4)
platt.fit(to_logit(oof[val]).reshape(-1, 1), y_dev[val])

def calibrate(raw_prob):
    return platt.predict_proba(to_logit(raw_prob).reshape(-1, 1))[:, 1]

val_prob = calibrate(oof[val])
```

The fitted calibration was

  *calibrated probability* = 1 / (1 + e<sup>−(1.146 × logit(raw probability) + 0.356)</sup>)

which only slightly adjusts the raw probabilities. This is because no extra weight was given to outage
days, so the raw probabilities were already close to the real outage rates.

### 4.8 Choosing the Warning Threshold

The model gives a probability for each recloser-day. To turn it into a yes/no **outage warning**, a
cut-off value is needed: a warning is raised when the probability is at or above it. Recall (catching
outages) was treated as the priority, but too many false alarms would make the warnings useless.

The threshold was therefore chosen on the calibrated validation predictions. Values from 0.01 to 0.60
were tested, and among the thresholds whose validation recall fell between **52% and 68%**, the one with
the highest **F1-score** was selected.

```python
def select_threshold(y, p, band=(0.52, 0.68)):
    best_th, best_f1 = None, -1.0
    for th in np.linspace(0.01, 0.60, 600):
        pred = p >= th
        tp = (pred & (y == 1)).sum()
        recall = tp / y.sum()
        f1 = 2 * tp / (pred.sum() + y.sum())
        if band[0] <= recall <= band[1] and f1 > best_f1:
            best_th, best_f1 = th, f1
    return best_th

threshold = select_threshold(y_dev[val], val_prob)   # = 0.122
```

The selected threshold is **0.122**. A recloser-day is flagged as a likely outage when its calibrated
outage probability is 12.2% or higher. On the validation set, this threshold caught 52.1% of outages,
and 30.6% of warnings were followed by an outage.

### 4.9 Training the Final Model and Saving It

Once the settings, calibration and threshold were fixed, the final model was trained on **all** records
before 2026 (10,971 records, 358 outages), again as an average of three seeds. It was then used to
predict the 2026 test set **once**, and the predictions were calibrated with the Platt scaling from
step 4.7. The test results are reported in Chapter 5.

```python
final_models = fit_models(best_params, X_dev, y_dev, seeds=SEEDS)

test_prob  = calibrate(predict_models(final_models, X_test))   # calibrated test probabilities
test_alert = (test_prob >= threshold).astype(int)              # 1 = outage warning
```

Finally, the three trained models were saved as files, together with a "model card" (`model_card.json`)
that records the list of inputs, the selected settings, the calibration values and the warning
threshold. The separate script `predict.py` uses these files to forecast outage probabilities for new
recloser-days without retraining.

```python
import json

for seed, m in zip(SEEDS, final_models):
    m.save_model(f"model/xgboost_seed{seed}.json")

card = {"features": all_features, "reclosers": RECLOSERS, "hyperparameters": best_params,
        "calibration": {"coef": 1.146, "intercept": 0.356}, "alert_threshold": threshold}
with open("model/model_card.json", "w") as f:
    json.dump(card, f, indent=2)
```

**Summary of the model development steps**

| Step | What was done | Result |
|---|---|---|
| 4.1 | Loaded the dataset and the list of inputs | 13,473 recloser-days, 66 inputs, 418 outages |
| 4.2 | Split the data by time | Development: before 2026 (10,971 rows). Test: 2026 (2,502 rows) |
| 4.3 | Prepared the inputs (one-hot encoding of the recloser) | 74 model input columns |
| 4.4 | Set up rolling validation | 4 half-year folds in 2024–2025, 213 validation outages |
| 4.5 | Built the XGBoost model | Average of 3 seed models |
| 4.6 | Tuned the hyperparameters with random search | 25 combinations; best validation PR-AUC 0.434 |
| 4.7 | Calibrated the probabilities with Platt scaling | Slope 1.146, intercept 0.356 |
| 4.8 | Chose the warning threshold | 0.122 (validation recall 52.1%) |
| 4.9 | Trained the final model on all pre-2026 data and saved it | Saved models and model card, ready for testing |

---

### Optional paragraph: baseline models

*(Add this only if you include the baseline comparison in Chapter 5.)*

To check whether XGBoost was a suitable choice, three other models were trained with the same inputs,
data split, rolling validation, calibration and threshold rule:

- **LightGBM**, another gradient-boosting model, tuned with the same 25-combination random search;
- **Random Forest**: 300 trees, maximum depth 7, balanced class weights;
- **Logistic Regression**: a linear model with L2 regularization and balanced class weights, with the
  inputs standardized first.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

rf = RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=15,
                            max_features="sqrt", class_weight="balanced_subsample")
lr = make_pipeline(StandardScaler(),
                   LogisticRegression(C=0.08, class_weight="balanced", max_iter=2000))
lgbm = lgb.LGBMClassifier(objective="binary", **best_lgb_params)   # tuned like XGBoost
```
