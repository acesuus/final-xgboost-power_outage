"""
train.py
Recloser outage probability forecasting: training, evaluation and reporting.

Pipeline
  1. Chronological split: development period (before 2026) and a held-out
     test period (2026). The test period is used only for the final evaluation.
  2. Rolling-origin cross-validation on the development period: four
     half-year validation folds (2024H1 - 2025H2), each trained on all
     earlier data.
  3. Random-search hyperparameter tuning (XGBoost, LightGBM) on those folds,
     scored by mean fold PR-AUC. Class weighting is tuned too.
  4. Probability calibration (Platt scaling) and the alert threshold
     (max F1 within a 52-68% recall band) are fitted on the pooled
     out-of-fold predictions.
  5. Final models are refit on the full development period (3-seed average)
     and evaluated once on the test period, with day-block bootstrap 95% CIs.
  6. Baselines (LightGBM, Random Forest, Logistic Regression), a feature-group
     ablation, a threshold trade-off table, calibration table, feature
     importance and figures are written to reports/.
  7. The primary model (XGBoost) is saved to model/ for predict.py.

Usage:
    python train.py            # full run, ~5 minutes
    python train.py --quick    # smoke test with fewer search draws
"""

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPORTS = HERE / "reports"
FIGS = REPORTS / "figures"
MODEL_DIR = HERE / "model"

RANDOM_STATE = 42
SEEDS = [42, 7, 2024]
TEST_START = "2026-01-01"
CV_FOLDS = [  # (val_start, val_end); each fold trains on everything before val_start
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
]
RECALL_BAND = (0.52, 0.68)
RECALL_TARGETS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
PRIMARY = "XGBoost"

XGB_SPACE = dict(
    n_estimators=[150, 250, 400, 600],
    learning_rate=[0.01, 0.02, 0.03, 0.05],
    max_depth=[2, 3, 4, 5],
    min_child_weight=[1, 3, 6, 12, 20],
    subsample=[0.6, 0.7, 0.8, 0.9],
    colsample_bytree=[0.4, 0.5, 0.6, 0.75, 0.9],
    reg_alpha=[0.0, 0.5, 1.5, 3.0],
    reg_lambda=[1.0, 3.0, 8.0, 15.0],
    gamma=[0.0, 0.25, 1.0],
    spw_mode=["none", "sqrt", "full"],
)
LGB_SPACE = dict(
    n_estimators=[150, 250, 400, 600],
    learning_rate=[0.01, 0.02, 0.03, 0.05],
    num_leaves=[4, 7, 15, 31],
    max_depth=[-1, 3, 4, 5],
    min_child_samples=[10, 20, 40, 80],
    subsample=[0.6, 0.7, 0.8, 0.9],
    subsample_freq=[1],
    colsample_bytree=[0.4, 0.5, 0.6, 0.75, 0.9],
    reg_alpha=[0.0, 0.5, 1.5, 3.0],
    reg_lambda=[1.0, 3.0, 8.0, 15.0],
    spw_mode=["none", "sqrt", "full"],
)
RF_PARAMS = dict(n_estimators=300, max_depth=7, min_samples_leaf=15, max_features="sqrt",
                 class_weight="balanced_subsample")

# =====================================================================
# DATA
# =====================================================================
df = pd.read_csv(DATA / "recloser_modeling_dataset_66f.csv")
feat_dict = pd.read_csv(DATA / "feature_dictionary.csv")
all_features = feat_dict["feature"].tolist()
groups = feat_dict.groupby("group")["feature"].apply(list).to_dict()
RECLOSERS = sorted(df["recloser"].unique())

dev = df[df["date"] < TEST_START].reset_index(drop=True)
test = df[df["date"] >= TEST_START].reset_index(drop=True)
y_dev = dev["outage_label"].to_numpy()
y_test = test["outage_label"].to_numpy()


def column_names(feats, reclosers=RECLOSERS):
    names = [f for f in feats if f != "recloser"]
    if "recloser" in feats:
        names += ["rc_" + r.replace(" ", "_") for r in reclosers]
    return names


def make_X(frame, feats, reclosers=RECLOSERS):
    """Raw numeric features + one-hot recloser (fixed category list)."""
    num = [f for f in feats if f != "recloser"]
    X = frame[num].astype(float).copy()
    if "recloser" in feats:
        for r in reclosers:
            X["rc_" + r.replace(" ", "_")] = (frame["recloser"] == r).astype(float)
    return X.to_numpy()


def fold_masks():
    for vs, ve in CV_FOLDS:
        tr = (dev["date"] < vs).to_numpy()
        va = ((dev["date"] >= vs) & (dev["date"] <= ve)).to_numpy()
        yield tr, va


# =====================================================================
# MODELS
# =====================================================================
def build_models(family, params, y_tr, seeds):
    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    p = dict(params)
    spw = {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "full": float(ratio)}[p.pop("spw_mode", "none")]
    models = []
    for s in seeds:
        if family == "xgb":
            models.append(xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                            scale_pos_weight=spw, random_state=s, n_jobs=-1, verbosity=0, **p))
        elif family == "lgb":
            models.append(lgb.LGBMClassifier(objective="binary", scale_pos_weight=spw, random_state=s,
                                             n_jobs=-1, verbose=-1, **p))
        elif family == "rf":
            models.append(RandomForestClassifier(random_state=s, n_jobs=-1, **RF_PARAMS))
        elif family == "lr":
            return [make_pipeline(StandardScaler(),
                                  LogisticRegression(C=0.08, class_weight="balanced", max_iter=2000))]
        else:
            raise ValueError(family)
    return models


def fit_models(family, params, X_tr, y_tr, seeds):
    models = build_models(family, params, y_tr, seeds)
    for m in models:
        m.fit(X_tr, y_tr)
    return models


def predict_models(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def run_cv(family, params, X_dev, seeds):
    """Rolling-origin CV. Returns pooled out-of-fold predictions and per-fold PR-AUC."""
    oof = np.full(len(dev), np.nan)
    fold_ap = []
    for tr, va in fold_masks():
        p = predict_models(fit_models(family, params, X_dev[tr], y_dev[tr], seeds), X_dev[va])
        oof[va] = p
        fold_ap.append(average_precision_score(y_dev[va], p))
    return oof, fold_ap


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Platt:
    def __init__(self, C=1e4):
        self.lr = LogisticRegression(C=C, solver="lbfgs")

    def fit(self, raw, y):
        self.lr.fit(to_logit(raw).reshape(-1, 1), y)
        return self

    def predict(self, raw):
        return self.lr.predict_proba(to_logit(raw).reshape(-1, 1))[:, 1]

    @property
    def coef(self):
        return float(self.lr.coef_[0, 0]), float(self.lr.intercept_[0])


def select_threshold(y, p, band=RECALL_BAND):
    """Max F1 among thresholds whose recall falls in the band (fallback: recall nearest the band centre)."""
    n_pos = y.sum()
    best_th, best_f1, closest_th, closest_gap = None, -1.0, None, np.inf
    for th in np.linspace(0.01, 0.60, 600):
        pred = p >= th
        tp = (pred & (y == 1)).sum()
        rec, f1 = tp / n_pos, 2 * tp / (pred.sum() + n_pos)
        if band[0] <= rec <= band[1] and f1 > best_f1:
            best_th, best_f1 = th, f1
        if abs(rec - np.mean(band)) < closest_gap:
            closest_th, closest_gap = th, abs(rec - np.mean(band))
    if best_th is None:
        print(f"  WARNING: no threshold reaches recall {band[0]:.2f}-{band[1]:.2f}; using {closest_th:.4f}")
        best_th = closest_th
    return float(best_th)


def threshold_for_recall(y, p, target):
    pos = np.sort(p[y == 1])[::-1]
    return float(pos[int(np.ceil(target * len(pos))) - 1])


# =====================================================================
# METRICS
# =====================================================================
def confusion(y, p, th):
    pred = p >= th
    return dict(TP=int((pred & (y == 1)).sum()), FP=int((pred & (y == 0)).sum()),
                FN=int((~pred & (y == 1)).sum()), TN=int((~pred & (y == 0)).sum()))


def metrics(y, p, th):
    c = confusion(y, p, th)
    alerts = c["TP"] + c["FP"]
    brier = brier_score_loss(y, p)
    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "Lift (PR-AUC / base rate)": average_precision_score(y, p) / y.mean(),
        "Precision": c["TP"] / alerts if alerts else 0.0,
        "Recall": c["TP"] / (c["TP"] + c["FN"]),
        "F1-Score": 2 * c["TP"] / (alerts + c["TP"] + c["FN"]),
        "Brier": brier,
        "BSS": 1 - brier / brier_score_loss(y, np.full(len(y), y.mean())),
        "LogLoss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "Threshold": th,
        **c,
    }


def day_bootstrap(y, p, dates, fn, n_boot, seed=0):
    """95% CI of fn(y, p), resampling whole days (outages on the same day are correlated)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(np.asarray(dates))
    by_day = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    stats = []
    for _ in range(n_boot):
        ix = np.concatenate([by_day[d] for d in rng.integers(0, len(uniq), len(uniq))])
        if y[ix].sum():
            stats.append(fn(y[ix], p[ix]))
    return tuple(np.percentile(stats, [2.5, 97.5]))


# =====================================================================
# ABLATION CONFIGURATIONS
# =====================================================================
def ablation_configs():
    weather = (groups["target_day_forecast_weather"] + groups["lagged_forecast_weather"]
               + groups["wet_weather_history"] + groups["weather_stress"])
    equip = groups["equipment_history"] + groups["repair_history"] + groups["equipment_category_history"]
    return {
        "A1: Weather only": weather,
        "A2: Weather + vegetation + interactions":
            weather + groups["vegetation"] + groups["weather_vegetation_interactions"],
        "A3: Weather + vegetation + outage & equipment history":
            [f for f in all_features if f not in groups["calendar"] and f != "recloser"],
        "A4: Full feature set": all_features,
        "L1: Full minus weather-vegetation interactions":
            [f for f in all_features if f not in groups["weather_vegetation_interactions"]],
        "L2: Full minus equipment & repair history": [f for f in all_features if f not in equip],
        "L3: Full minus outage history": [f for f in all_features if f not in groups["outage_history"]],
    }


# =====================================================================
# FIGURES
# =====================================================================
def save_figures(cal_test, primary_th, trade_df, imp_df, calib_df):
    colors = {"XGBoost": "#1f5fa8", "LightGBM": "#2a9d8f", "Random Forest": "#e9a23b", "Logistic Regression": "#9b5de5"}

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in cal_test.items():
        pr, rc, _ = precision_recall_curve(y_test, p)
        ax.plot(rc, pr, label=f"{name} (AP {average_precision_score(y_test, p):.3f})",
                color=colors.get(name), lw=2.2 if name == PRIMARY else 1.3)
    ax.axhline(y_test.mean(), color="grey", ls="--", lw=1, label=f"Base rate ({y_test.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve, test period (2026)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in cal_test.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})",
                color=colors.get(name), lw=2.2 if name == PRIMARY else 1.3)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve, test period (2026)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    lim = max(calib_df["Mean predicted"].max(), calib_df["Observed rate"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color="grey", ls="--", lw=1)
    ax.plot(calib_df["Mean predicted"], calib_df["Observed rate"], "o-", color=colors[PRIMARY])
    ax.set(xlabel="Mean predicted probability", ylabel="Observed outage rate",
           title=f"Calibration, {PRIMARY}, test period (deciles)", xlim=(0, lim), ylim=(0, lim))
    fig.tight_layout()
    fig.savefig(FIGS / "calibration.png", dpi=150)
    plt.close(fig)

    top = imp_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["Feature"], top["Gain share"], color=colors[PRIMARY])
    ax.set(xlabel="Share of total gain", title=f"{PRIMARY} feature importance (top 20)")
    fig.tight_layout()
    fig.savefig(FIGS / "feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(trade_df["Alerts per week"], trade_df["Test recall"], "o-", label="Recall", color=colors[PRIMARY])
    ax.plot(trade_df["Alerts per week"], trade_df["Test precision"], "s-", label="Precision", color="#e76f51")
    for _, r in trade_df.iterrows():
        ax.annotate(f"{r['Recall target']:.0%}", (r["Alerts per week"], r["Test recall"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set(xlabel="Alerts per week (all reclosers)", ylabel="Test period value",
           title="Alert volume vs. recall and precision (labels: recall target)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "threshold_tradeoff.png", dpi=150)
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=25, help="random-search draws per model family")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_iter, args.n_boot = 6, 200
    for d in (REPORTS, FIGS, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Development period (before {TEST_START}): {len(dev)} rows, {y_dev.sum()} outages, rate {y_dev.mean():.4f}")
    print(f"Test period ({TEST_START} onward):        {len(test)} rows, {y_test.sum()} outages, rate {y_test.mean():.4f}")
    for (vs, ve), (tr, va) in zip(CV_FOLDS, fold_masks()):
        print(f"  CV fold {vs}..{ve}: train {tr.sum():5d} rows / {y_dev[tr].sum():3d} outages, "
              f"val {va.sum():4d} rows / {y_dev[va].sum():2d} outages")

    X_dev, X_test = make_X(dev, all_features), make_X(test, all_features)

    # ---- 1. Hyperparameter search ----
    print(f"\n=== HYPERPARAMETER SEARCH ({args.n_iter} draws per family, mean CV-fold PR-AUC) ===")
    search_rows, best = [], {}
    for family, space in [("xgb", XGB_SPACE), ("lgb", LGB_SPACE)]:
        for i, params in enumerate(ParameterSampler(space, n_iter=args.n_iter, random_state=RANDOM_STATE)):
            _, fold_ap = run_cv(family, params, X_dev, seeds=[RANDOM_STATE])
            search_rows.append({"family": family, "draw": i, "cv_pr_auc": np.mean(fold_ap),
                                "cv_pr_auc_std": np.std(fold_ap), **params})
        top = max((r for r in search_rows if r["family"] == family), key=lambda r: r["cv_pr_auc"])
        best[family] = {k: top[k] for k in space}
        print(f"  {family}: best CV PR-AUC {top['cv_pr_auc']:.4f}")
    pd.DataFrame(search_rows).sort_values(["family", "cv_pr_auc"], ascending=[True, False]) \
        .to_csv(REPORTS / "hyperparameter_search.csv", index=False)

    # ---- 2. Cross-validated comparison ----
    print("\n=== CROSS-VALIDATED COMPARISON (3-seed average, 4 folds) ===")
    specs = {"XGBoost": ("xgb", best["xgb"]), "LightGBM": ("lgb", best["lgb"]),
             "Random Forest": ("rf", {}), "Logistic Regression": ("lr", {})}
    oof, fold_aps = {}, {}
    for name, (fam, params) in specs.items():
        oof[name], fold_aps[name] = run_cv(fam, params, X_dev, SEEDS)
    cv_mask = ~np.isnan(oof[PRIMARY])
    cv_df = pd.DataFrame([{"Model": n, "CV PR-AUC (mean)": np.mean(fold_aps[n]), "CV PR-AUC (std)": np.std(fold_aps[n]),
                           **{f"Fold {CV_FOLDS[i][0][:7]}": a for i, a in enumerate(fold_aps[n])},
                           "CV ROC-AUC (pooled)": roc_auc_score(y_dev[cv_mask], oof[n][cv_mask])} for n in specs])
    print(cv_df.round(4).to_string(index=False))
    cv_df.to_csv(REPORTS / "cv_model_comparison.csv", index=False)

    # ---- 3. Calibrate, threshold, refit, test ----
    print(f"\n=== TEST PERIOD EVALUATION (refit on full development period) ===")
    final_models, calibrators, thresholds, cal_test, cal_oof = {}, {}, {}, {}, {}
    test_rows = []
    for name, (fam, params) in specs.items():
        final_models[name] = fit_models(fam, params, X_dev, y_dev, SEEDS)
        calibrators[name] = Platt().fit(oof[name][cv_mask], y_dev[cv_mask])
        cal_oof[name] = calibrators[name].predict(oof[name][cv_mask])
        thresholds[name] = select_threshold(y_dev[cv_mask], cal_oof[name])
        cal_test[name] = calibrators[name].predict(predict_models(final_models[name], X_test))
        m = metrics(y_test, cal_test[name], thresholds[name])
        lo, hi = day_bootstrap(y_test, cal_test[name], test["date"], average_precision_score, args.n_boot)
        rlo, rhi = day_bootstrap(y_test, cal_test[name], test["date"], roc_auc_score, args.n_boot)
        test_rows.append({"Model": name + (" (primary)" if name == PRIMARY else ""), **m,
                          "PR-AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]", "ROC-AUC 95% CI": f"[{rlo:.3f}, {rhi:.3f}]"})
    test_df = pd.DataFrame(test_rows)
    print(test_df.drop(columns=["LogLoss"]).round(4).to_string(index=False))
    test_df.to_csv(REPORTS / "test_model_comparison.csv", index=False)

    # confusion matrix of the primary model
    c = confusion(y_test, cal_test[PRIMARY], thresholds[PRIMARY])
    cm = pd.DataFrame([[c["TP"], c["FN"]], [c["FP"], c["TN"]]],
                      index=["Actual outage", "Actual no outage"], columns=["Predicted outage", "Predicted no outage"])
    cm.to_csv(REPORTS / "confusion_matrix.csv")
    print(f"\nConfusion matrix, {PRIMARY}, threshold {thresholds[PRIMARY]:.4f}:\n{cm.to_string()}")

    # ---- 4. Threshold trade-off (primary) ----
    n_days = test["date"].nunique()
    y_cv = y_dev[cv_mask]
    trade = []
    for r in RECALL_TARGETS:
        th = threshold_for_recall(y_cv, cal_oof[PRIMARY], r)
        cv_c, te_c = confusion(y_cv, cal_oof[PRIMARY], th), confusion(y_test, cal_test[PRIMARY], th)
        alerts = te_c["TP"] + te_c["FP"]
        trade.append({"Recall target": r, "Threshold": th,
                      "CV recall": cv_c["TP"] / (cv_c["TP"] + cv_c["FN"]),
                      "CV precision": cv_c["TP"] / max(cv_c["TP"] + cv_c["FP"], 1),
                      "Test recall": te_c["TP"] / (te_c["TP"] + te_c["FN"]),
                      "Test precision": te_c["TP"] / max(alerts, 1),
                      "Test F1": 2 * te_c["TP"] / (alerts + te_c["TP"] + te_c["FN"]),
                      **{f"Test {k}": v for k, v in te_c.items()},
                      "Alerts per week": alerts / n_days * 7})
    trade_df = pd.DataFrame(trade)
    print("\n=== THRESHOLD TRADE-OFF (threshold set on CV for each recall target, applied to test) ===")
    print(trade_df.round(3).to_string(index=False))
    trade_df.to_csv(REPORTS / "threshold_tradeoff.csv", index=False)

    # ---- 5. Calibration table (primary, test deciles) ----
    bins = pd.qcut(cal_test[PRIMARY], 10, labels=False, duplicates="drop")
    calib_df = pd.DataFrame({"p": cal_test[PRIMARY], "y": y_test, "bin": bins}).groupby("bin") \
        .agg(**{"Rows": ("y", "size"), "Mean predicted": ("p", "mean"), "Observed rate": ("y", "mean"),
                "Outages": ("y", "sum")}).reset_index(drop=True)
    calib_df.index = [f"Decile {i + 1}" for i in range(len(calib_df))]
    calib_df.to_csv(REPORTS / "calibration_table.csv")

    # ---- 6. Feature importance (primary, mean gain over seed models) ----
    names = column_names(all_features)
    gains = np.zeros(len(names))
    for m in final_models[PRIMARY]:
        g = m.get_booster().get_score(importance_type="total_gain")
        gains += np.array([g.get(f"f{i}", 0.0) for i in range(len(names))])
    imp = pd.DataFrame({"Feature": ["recloser" if n.startswith("rc_") else n for n in names], "Gain": gains}) \
        .groupby("Feature", as_index=False)["Gain"].sum()
    imp = imp.merge(feat_dict.rename(columns={"feature": "Feature", "group": "Group"}), on="Feature", how="left")
    imp["Gain share"] = imp["Gain"] / imp["Gain"].sum()
    imp = imp.sort_values("Gain share", ascending=False).reset_index(drop=True)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)
    imp.groupby("Group")["Gain share"].sum().sort_values(ascending=False) \
        .to_csv(REPORTS / "feature_group_importance.csv")

    # ---- 7. Ablation (primary model) ----
    print(f"\n=== FEATURE-GROUP ABLATION ({PRIMARY}, 3-seed average) ===")
    ab_rows = []
    for ab_name, feats in ablation_configs().items():
        Xd, Xt = make_X(dev, feats), make_X(test, feats)
        o, fa = run_cv("xgb", best["xgb"], Xd, SEEDS)
        pt = predict_models(fit_models("xgb", best["xgb"], Xd, y_dev, SEEDS), Xt)
        lo, hi = day_bootstrap(y_test, pt, test["date"], average_precision_score, args.n_boot)
        ab_rows.append({"Configuration": ab_name, "Features": len(feats),
                        "CV PR-AUC (mean)": np.mean(fa), "CV PR-AUC (std)": np.std(fa),
                        "CV ROC-AUC": roc_auc_score(y_dev[cv_mask], o[cv_mask]),
                        "Test PR-AUC": average_precision_score(y_test, pt), "Test PR-AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]",
                        "Test ROC-AUC": roc_auc_score(y_test, pt)})
    ab_df = pd.DataFrame(ab_rows)
    print(ab_df.round(4).to_string(index=False))
    ab_df.to_csv(REPORTS / "ablation.csv", index=False)

    # ---- 8. Figures, predictions, saved model ----
    save_figures(cal_test, thresholds[PRIMARY], trade_df, imp, calib_df)
    pd.DataFrame({"date": test["date"], "recloser": test["recloser"], "outage_label": y_test,
                  **{f"prob_{k.replace(' ', '_')}": v for k, v in cal_test.items()},
                  "alert": (cal_test[PRIMARY] >= thresholds[PRIMARY]).astype(int)}) \
        .to_csv(REPORTS / "test_predictions.csv", index=False)

    for i, m in enumerate(final_models[PRIMARY]):
        m.save_model(MODEL_DIR / f"xgboost_seed{SEEDS[i]}.json")
    coef, intercept = calibrators[PRIMARY].coef
    card = {
        "model": "XGBoost (3-seed average) + Platt calibration",
        "trained_on": f"{dev['date'].min()} to {dev['date'].max()} ({len(dev)} rows, {int(y_dev.sum())} outages)",
        "features": all_features,
        "model_columns": names,
        "reclosers": RECLOSERS,
        "hyperparameters": best["xgb"],
        "seed_models": [f"xgboost_seed{s}.json" for s in SEEDS],
        "calibration": {"type": "platt_on_logit", "coef": coef, "intercept": intercept},
        "alert_threshold": thresholds[PRIMARY],
        "threshold_rule": f"max F1 with recall in {RECALL_BAND} on cross-validated predictions",
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    (REPORTS / "best_hyperparameters.json").write_text(json.dumps(best, indent=2))
    print(f"\nReports in {REPORTS}, model in {MODEL_DIR}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
