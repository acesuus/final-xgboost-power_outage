"""
chapter5_analysis.py
Additional analyses for Chapter 5 (Results and Discussion). Run train.py first.

Writes to reports/chapter5/:
  - dataset & split summaries (Tables 5.1-5.3)
  - validation vs test performance of the primary model (Table 5.5)
  - raw vs calibrated Brier scores and Brier skill scores (Table 5.7)
  - reliability diagram with equal-width bins (Figure 5.2)
  - SHAP analysis with XGBoost's built-in TreeSHAP (Figures 5.4-5.7, Tables 5.8-5.9)
  - Objective 4 ablation: lagged weather and cumulative outage features (Table 5.10)
  - supplementary ablations and routine vs weather-driven outage performance

"Validation" = pooled out-of-fold predictions of the four rolling CV folds
(2024H1-2025H2). Calibration and the alert threshold are fitted on these same
predictions, so validation precision/recall/F1/Brier are in-sample with
respect to those two steps; the test period (2026) is fully held out.
"""

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

import train as T

OUT = T.REPORTS / "chapter5"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY = "#1f5fa8", "#e76f51", "#8d99ae"
GROUP_LABELS = {
    "target_day_forecast_weather": "Target-day forecast weather",
    "lagged_forecast_weather": "Lagged forecast weather",
    "wet_weather_history": "Wet-weather history",
    "weather_stress": "Weather stress (vs. p90)",
    "vegetation": "Vegetation",
    "weather_vegetation_interactions": "Weather-vegetation interactions",
    "outage_history": "Outage history (cumulative/recency)",
    "equipment_history": "Equipment fault history",
    "equipment_category_history": "Equipment-category fault history",
    "repair_history": "Repair history",
    "calendar": "Temporal (calendar)",
    "identity": "Recloser identity",
}
FEATURE_GROUP = dict(zip(T.feat_dict["feature"], T.feat_dict["group"]))


def bss(y, p):
    return 1 - brier_score_loss(y, p) / brier_score_loss(y, np.full(len(y), y.mean()))


def paired_ci(y, p, p_ref, dates, fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(np.asarray(dates))
    by_day = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    d = []
    for _ in range(n_boot):
        ix = np.concatenate([by_day[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if y[ix].sum():
            d.append(fn(y[ix], p[ix]) - fn(y[ix], p_ref[ix]))
    return tuple(np.percentile(d, [2.5, 97.5]))


def evaluate_feature_set(feats, seeds=T.SEEDS):
    """Identical procedure to train.py: CV -> Platt -> threshold -> refit on dev -> test."""
    params = json.loads((T.REPORTS / "best_hyperparameters.json").read_text())["xgb"]
    Xd, Xt = T.make_X(T.dev, feats), T.make_X(T.test, feats)
    oof, fold_ap = T.run_cv("xgb", params, Xd, seeds)
    mask = ~np.isnan(oof)
    cal = T.Platt().fit(oof[mask], T.y_dev[mask])
    th = T.select_threshold(T.y_dev[mask], cal.predict(oof[mask]))
    p_test = cal.predict(T.predict_models(T.fit_models("xgb", params, Xd, T.y_dev, seeds), Xt))
    return dict(oof_raw=oof[mask], oof_cal=cal.predict(oof[mask]), fold_ap=fold_ap, th=th, p_test=p_test)


def main():
    card = json.loads((T.MODEL_DIR / "model_card.json").read_text())
    params = card["hyperparameters"]
    feats = T.all_features
    y_dev, y_test, dev, test = T.y_dev, T.y_test, T.dev, T.test
    df = T.df

    # =================================================================
    # 5.1 Dataset summary and split
    # =================================================================
    n_pos = int(df["outage_label"].sum())
    t51 = pd.DataFrame({"Characteristic": [
        "Covered reclosers", "Period covered", "Total records (recloser-days)", "Outage records (y = 1)",
        "Non-outage records (y = 0)", "Outage prevalence", "Class ratio (non-outage to outage)",
        "Input features", "Model input columns (after one-hot encoding of recloser)"],
        "Value": [df["recloser"].nunique(), f"{df['date'].min()} to {df['date'].max()}", len(df), n_pos,
                  len(df) - n_pos, f"{n_pos / len(df):.2%}", f"{(len(df) - n_pos) / n_pos:.1f} : 1",
                  len(feats), len(card["model_columns"])]})
    t51.to_csv(OUT / "table5_1_dataset_summary.csv", index=False)

    first_val = T.CV_FOLDS[0][0]
    parts = [
        ("Training (initial window)", f"{dev['date'].min()} to 2023-12-31", dev[dev["date"] < first_val]),
        ("Validation (4 rolling half-year folds)", f"{first_val} to {dev['date'].max()}", dev[dev["date"] >= first_val]),
        ("Testing (held out)", f"{test['date'].min()} to {test['date'].max()}", test),
    ]
    t52 = pd.DataFrame([{"Set": s, "Period": per, "Rows": len(f), "Proportion": f"{len(f) / len(df):.1%}",
                         "Outage rows": int(f["outage_label"].sum()),
                         "Outage prevalence": f"{f['outage_label'].mean():.2%}"} for s, per, f in parts])
    t52.to_csv(OUT / "table5_2_data_split.csv", index=False)
    folds = []
    for (vs, ve), (tr, va) in zip(T.CV_FOLDS, T.fold_masks()):
        folds.append({"Fold": f"{vs[:7]} to {ve[:7]}", "Training rows": int(tr.sum()),
                      "Training outages": int(y_dev[tr].sum()), "Validation rows": int(va.sum()),
                      "Validation outages": int(y_dev[va].sum()),
                      "Validation prevalence": f"{y_dev[va].mean():.2%}"})
    pd.DataFrame(folds).to_csv(OUT / "table5_2b_cv_folds.csv", index=False)

    # 5.2 Feature groups
    t53 = T.feat_dict.groupby("group")["feature"].agg(["count", lambda s: ", ".join(list(s)[:3])]).reset_index()
    t53.columns = ["Group", "Count", "Examples"]
    t53["Group"] = t53["Group"].map(GROUP_LABELS)
    t53.sort_values("Count", ascending=False).to_csv(OUT / "table5_3_feature_groups.csv", index=False)

    # =================================================================
    # 5.4 / 5.5 Performance and calibration of the primary model
    # =================================================================
    print("Re-running cross-validation for the primary model ...")
    X_dev, X_test = T.make_X(dev, feats), T.make_X(test, feats)
    oof, fold_ap = T.run_cv("xgb", params, X_dev, T.SEEDS)
    mask = ~np.isnan(oof)
    y_val = y_dev[mask]
    val_raw = oof[mask]
    cal = T.Platt().fit(val_raw, y_val)
    val_cal = cal.predict(val_raw)
    th = card["alert_threshold"]
    seed_models = []
    for f in card["seed_models"]:
        m = xgb.XGBClassifier()
        m.load_model(T.MODEL_DIR / f)
        seed_models.append(m)
    test_raw = np.mean([m.predict_proba(X_test)[:, 1] for m in seed_models], axis=0)
    test_cal = cal.predict(test_raw)
    assert abs(T.select_threshold(y_val, val_cal) - th) < 1e-9, "threshold mismatch with saved model"

    perf = []
    for name, y, p in [("Validation (pooled CV, 2024-2025)", y_val, val_cal), ("Test (2026)", y_test, test_cal)]:
        m = T.metrics(y, p, th)
        perf.append({"Set": name, **{k: m[k] for k in ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC",
                                                       "Brier", "BSS", "TP", "FP", "FN", "TN"]},
                     "Specificity": m["TN"] / (m["TN"] + m["FP"]), "Prevalence": y.mean()})
    perf_df = pd.DataFrame(perf)
    perf_df.to_csv(OUT / "table5_5_performance_validation_test.csv", index=False)

    # Brier table
    t57 = []
    for name, y, raw, c in [("Validation", y_val, val_raw, val_cal), ("Test", y_test, test_raw, test_cal)]:
        base = brier_score_loss(y, np.full(len(y), y.mean()))
        t57.append({"Set": name, "Raw Brier": brier_score_loss(y, raw), "Calibrated Brier": brier_score_loss(y, c),
                    "Baseline Brier (predict outage rate)": base, "BSS (raw)": bss(y, raw), "BSS (calibrated)": bss(y, c),
                    "Mean predicted (calibrated)": c.mean(), "Observed rate": y.mean()})
    pd.DataFrame(t57).to_csv(OUT / "table5_7_brier.csv", index=False)

    # Reliability diagram (equal-width bins)
    edges = np.array([0, .05, .10, .15, .20, .25, .30, .40, .50, 1.0])
    rel_rows = []
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, .75], [0, .75], color=GREY, ls="--", lw=1, label="Perfect calibration")
    for name, y, p, col, mk in [("Validation", y_val, val_cal, ORANGE, "s"), ("Test", y_test, test_cal, BLUE, "o")]:
        b = np.digitize(p, edges[1:-1])
        xs, ys, ns = [], [], []
        for k in range(len(edges) - 1):
            sel = b == k
            if sel.sum() == 0:
                continue
            rel_rows.append({"Set": name, "Bin": f"{edges[k]:.2f}-{edges[k + 1]:.2f}", "Rows": int(sel.sum()),
                             "Outages": int(y[sel].sum()), "Mean predicted": p[sel].mean(), "Observed rate": y[sel].mean()})
            xs.append(p[sel].mean()); ys.append(y[sel].mean()); ns.append(sel.sum())
        ax.plot(xs, ys, marker=mk, color=col, label=name, lw=1.5)
        for x_, y_, n_ in zip(xs, ys, ns):
            ax.annotate(str(n_), (x_, y_), textcoords="offset points", xytext=(5, -10 if name == "Test" else 6),
                        fontsize=7, color=col)
    ax.set(xlim=(0, .75), ylim=(0, .75), xlabel="Mean predicted outage probability",
           ylabel="Observed outage frequency", title="Reliability diagram (equal-width bins; labels = rows)")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "figure5_2_reliability_diagram.png", dpi=150); plt.close(fig)
    pd.DataFrame(rel_rows).to_csv(OUT / "reliability_bins.csv", index=False)

    # ROC + PR of primary model, validation and test
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name, y, p, col in [("Validation", y_val, val_cal, ORANGE), ("Test", y_test, test_cal, BLUE)]:
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC = {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, color=col, lw=2, label=f"{name} (AP = {average_precision_score(y, p):.3f})")
        axes[1].axhline(y.mean(), color=col, ls=":", lw=1)
    axes[0].plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (dotted = outage rate)")
    for a in axes:
        a.legend(loc="lower right" if a is axes[0] else "upper right")
    fig.tight_layout(); fig.savefig(OUT / "figure5_1_roc_pr_curves.png", dpi=150); plt.close(fig)

    # =================================================================
    # 5.6 SHAP (TreeSHAP, log-odds units, averaged over the 3 seed models)
    # =================================================================
    print("Computing SHAP values ...")
    cols = card["model_columns"]

    def shap_values(X):
        dm = xgb.DMatrix(X)
        return np.mean([m.get_booster().predict(dm, pred_contribs=True) for m in seed_models], axis=0)[:, :-1]

    sv_test = shap_values(X_test)
    sv_all = shap_values(np.vstack([X_dev, X_test]))
    feat_of_col = ["recloser" if c.startswith("rc_") else c for c in cols]

    def aggregate(sv):
        s = pd.DataFrame(np.abs(sv), columns=cols).T
        s["Feature"] = feat_of_col
        return s.groupby("Feature").sum().mean(axis=1)

    mean_abs = aggregate(sv_test).sort_values(ascending=False)
    mean_abs_all = aggregate(sv_all)
    rank_agree = spearmanr(mean_abs.values, mean_abs_all.reindex(mean_abs.index).values).correlation

    Xt_df = pd.DataFrame(X_test, columns=cols)
    rows = []
    for f, v in mean_abs.items():
        if f == "recloser":
            direction, rho = "Depends on recloser", np.nan
        else:
            j = cols.index(f)
            x = Xt_df[f].to_numpy()
            rho = spearmanr(x, sv_test[:, j]).correlation if np.std(x) > 0 else np.nan
            direction = ("Higher value -> higher risk" if rho > 0.2 else "Higher value -> lower risk" if rho < -0.2
                         else "Mixed / weak")
        rows.append({"Feature": f, "Group": GROUP_LABELS[FEATURE_GROUP[f]], "Mean |SHAP|": v,
                     "Share of total": v / mean_abs.sum(), "Spearman(value, SHAP)": rho, "Direction of effect": direction})
    shap_df = pd.DataFrame(rows)
    shap_df.insert(0, "Rank", range(1, len(shap_df) + 1))
    shap_df.to_csv(OUT / "shap_feature_importance_all.csv", index=False)
    shap_df.head(10).to_csv(OUT / "table5_8_top10_shap.csv", index=False)

    grp = shap_df.groupby("Group")["Mean |SHAP|"].sum().sort_values(ascending=False)
    t59 = pd.DataFrame({"Group": grp.index, "Mean |SHAP| (sum)": grp.values,
                        "Share of total (%)": 100 * grp.values / grp.sum(),
                        "Features": [int((shap_df["Group"] == g).sum()) for g in grp.index]})
    t59.to_csv(OUT / "table5_9_shap_by_group.csv", index=False)

    # Figure 5.4: bar plot
    top = shap_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["Feature"], top["Mean |SHAP|"], color=BLUE)
    ax.set(xlabel="Mean |SHAP value| (log-odds)", title="SHAP summary bar plot, top 20 features (test period)")
    fig.tight_layout(); fig.savefig(OUT / "figure5_4_shap_bar.png", dpi=150); plt.close(fig)

    # Figure 5.5: beeswarm (top 15 numeric features)
    top_feats = [f for f in shap_df["Feature"] if f != "recloser"][:15]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    for i, f in enumerate(top_feats[::-1]):
        j = cols.index(f)
        x = Xt_df[f].to_numpy()
        colr = pd.Series(x).rank(pct=True).to_numpy()
        s = sv_test[:, j]
        bins_ = np.round(s / (np.ptp(s) / 60 + 1e-12))
        counts = pd.Series(bins_).map(pd.Series(bins_).value_counts()).to_numpy()
        jitter = rng.uniform(-1, 1, len(s)) * np.minimum(0.4, 0.04 * np.sqrt(counts))
        sc = ax.scatter(s, i + jitter, c=colr, cmap="coolwarm", s=6, alpha=0.7, linewidths=0)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(range(len(top_feats))); ax.set_yticklabels(top_feats[::-1], fontsize=8)
    ax.set(xlabel="SHAP value (impact on log-odds of outage)", title="SHAP beeswarm plot, top 15 features (test period)")
    cb = fig.colorbar(sc, ax=ax, pad=0.01); cb.set_label("Feature value (percentile: low = blue, high = red)")
    fig.tight_layout(); fig.savefig(OUT / "figure5_5_shap_beeswarm.png", dpi=150); plt.close(fig)

    # Figure 5.6: group contribution
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    g = t59.iloc[::-1]
    ax.barh(g["Group"], g["Share of total (%)"], color=BLUE)
    for yv, v in enumerate(g["Share of total (%)"]):
        ax.text(v + 0.3, yv, f"{v:.1f}%", va="center", fontsize=8)
    ax.set(xlabel="Share of total mean |SHAP| (%)", title="SHAP contribution by feature group (test period)")
    fig.tight_layout(); fig.savefig(OUT / "figure5_6_shap_by_group.png", dpi=150); plt.close(fig)

    # 5.6.3 Feature-level interpretation: top feature per group
    interp, dep_feats = [], []
    for gname in t59["Group"]:
        sub = shap_df[(shap_df["Group"] == gname) & (shap_df["Feature"] != "recloser")]
        if sub.empty:
            continue
        f = sub.iloc[0]["Feature"]
        j = cols.index(f)
        x, s = Xt_df[f].to_numpy(), sv_test[:, j]
        if len(np.unique(x)) <= 3 or (x == 0).mean() > 0.5:
            lo_sel, hi_sel = x == np.min(x), x > np.min(x)
            lo_lab, hi_lab = f"= {np.min(x):g}", f"> {np.min(x):g}"
        else:
            q1, q3 = np.quantile(x, [0.25, 0.75])
            lo_sel, hi_sel = x <= q1, x >= q3
            lo_lab, hi_lab = f"<= {q1:.3g} (bottom 25%)", f">= {q3:.3g} (top 25%)"
        interp.append({"Group": gname, "Feature": f, "Low values": lo_lab, "Mean SHAP (low)": s[lo_sel].mean(),
                       "High values": hi_lab, "Mean SHAP (high)": s[hi_sel].mean(),
                       "Outage rate (low)": y_test[lo_sel].mean(), "Outage rate (high)": y_test[hi_sel].mean()})
        dep_feats.append(f)
    pd.DataFrame(interp).to_csv(OUT / "feature_level_interpretation.csv", index=False)

    dep = dep_feats[:9]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for a, f in zip(axes.ravel(), dep):
        j = cols.index(f)
        a.scatter(Xt_df[f], sv_test[:, j], s=5, alpha=0.4, color=BLUE)
        a.axhline(0, color=GREY, lw=1)
        a.set_title(f"{f}\n({GROUP_LABELS[FEATURE_GROUP[f]]})", fontsize=8)
        a.set_ylabel("SHAP", fontsize=8); a.tick_params(labelsize=7)
    for a in axes.ravel()[len(dep):]:
        a.axis("off")
    fig.suptitle("SHAP dependence: most important feature of each group (test period)")
    fig.tight_layout(); fig.savefig(OUT / "figure5_7_shap_dependence.png", dpi=150); plt.close(fig)

    # =================================================================
    # 5.7 Objective 4 ablation + supplementary ablations
    # =================================================================
    print("Running feature-set ablations ...")
    G = T.groups
    lagged_met = G["lagged_forecast_weather"] + G["wet_weather_history"]
    cum_outage = G["outage_history"]
    infra = G["equipment_history"] + G["equipment_category_history"] + G["repair_history"]
    veg = G["vegetation"] + G["weather_vegetation_interactions"]
    weather_all = G["target_day_forecast_weather"] + lagged_met + G["weather_stress"]
    sets = {
        "Full model (all features)": feats,
        "Without lagged meteorological features": [f for f in feats if f not in lagged_met],
        "Without cumulative outage-history features": [f for f in feats if f not in cum_outage],
        "Base set (without lagged meteorological and outage-history features)":
            [f for f in feats if f not in lagged_met + cum_outage],
        "Without vegetation and weather-vegetation interactions": [f for f in feats if f not in veg],
        "Without infrastructure and repair history": [f for f in feats if f not in infra],
        "Weather features only": weather_all,
        "Non-weather features only (history, vegetation, calendar, recloser)": [f for f in feats if f not in weather_all + G["weather_vegetation_interactions"]],
    }
    results = {name: evaluate_feature_set(fs) for name, fs in sets.items()}
    full = results["Full model (all features)"]
    ab = []
    for name, fs in sets.items():
        r = results[name]
        m = T.metrics(y_test, r["p_test"], r["th"])
        row = {"Feature set": name, "Features": len(fs), "Precision": m["Precision"], "Recall": m["Recall"],
               "F1": m["F1-Score"], "AUC-ROC": m["ROC-AUC"], "PR-AUC": m["PR-AUC"], "Brier": m["Brier"],
               "BSS": m["BSS"], "Threshold": r["th"], "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
               "CV PR-AUC (mean)": np.mean(r["fold_ap"]), "CV PR-AUC (std)": np.std(r["fold_ap"]),
               "CV AUC-ROC": roc_auc_score(y_val, r["oof_cal"])}
        if name != "Full model (all features)":
            lo, hi = paired_ci(y_test, r["p_test"], full["p_test"], test["date"], roc_auc_score)
            row["Test AUC-ROC diff vs full (95% CI)"] = f"{m['ROC-AUC'] - roc_auc_score(y_test, full['p_test']):+.3f} [{lo:+.3f}, {hi:+.3f}]"
            lo, hi = paired_ci(y_test, r["p_test"], full["p_test"], test["date"], average_precision_score)
            row["Test PR-AUC diff vs full (95% CI)"] = f"{m['PR-AUC'] - average_precision_score(y_test, full['p_test']):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        ab.append(row)
    ab_df = pd.DataFrame(ab)
    ab_df.to_csv(OUT / "table5_10_ablation.csv", index=False)

    # =================================================================
    # Discussion: routine vs weather-driven outages
    # =================================================================
    def subgroup(frame, y, p, label):
        out = []
        stress = ((frame["forecast_precipitation_vs_fit_p90"] >= 1) | (frame["forecast_gust_vs_fit_p90"] >= 1)).to_numpy()
        month = frame["date"].str[5:7].astype(int).to_numpy()
        wet = (month >= 6) & (month <= 11)
        for name, sel in [("Weather-stress days (rain or gust >= p90)", stress), ("Routine days", ~stress),
                          ("Wet season (Jun-Nov)", wet), ("Dry season (Dec-May)", ~wet)]:
            yy, pp = y[sel], p[sel]
            pred = pp >= th
            tp = int((pred & (yy == 1)).sum())
            out.append({"Data": label, "Subset": name, "Rows": int(sel.sum()), "Outages": int(yy.sum()),
                        "Outage rate": yy.mean(), "Alerts": int(pred.sum()), "Caught": tp,
                        "Recall": tp / max(yy.sum(), 1), "Precision": tp / max(pred.sum(), 1),
                        "AUC-ROC": roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan})
        return out

    sub = subgroup(test, y_test, test_cal, "Test (2026)") + \
        subgroup(dev[mask].reset_index(drop=True), y_val, val_cal, "Validation (2024-2025)")
    pd.DataFrame(sub).to_csv(OUT / "routine_vs_weather.csv", index=False)

    # practical: risk ranking
    order = np.argsort(-test_cal)
    rank_rows = []
    for frac in [0.01, 0.02, 0.05, 0.10, 0.20]:
        k = int(round(frac * len(order)))
        rank_rows.append({"Top share of recloser-days": f"{frac:.0%}", "Recloser-days flagged": k,
                          "Outages captured": int(y_test[order[:k]].sum()),
                          "Share of all outages": y_test[order[:k]].sum() / y_test.sum(),
                          "Outage rate among flagged": y_test[order[:k]].mean(),
                          "Min probability in group": test_cal[order[k - 1]]})
    pd.DataFrame(rank_rows).to_csv(OUT / "risk_ranking.csv", index=False)

    summary = {"shap_rank_agreement_test_vs_all_data_spearman": rank_agree,
               "threshold": th, "calibration_coef": cal.coef}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 250)
    for f in sorted(OUT.glob("*.csv")):
        print(f"\n=== {f.name} ===")
        print(pd.read_csv(f).round(4).to_string(index=False))
    print("\n", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
