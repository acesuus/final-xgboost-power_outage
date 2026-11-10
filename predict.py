"""
predict.py
Score new recloser-days with the trained model (run train.py first).

Input: a CSV with one row per recloser-day containing every feature listed in
data/feature_dictionary.csv (same column names and units as the training data).
A 'date' column is passed through if present.

Output: the input identifiers plus
    outage_probability   calibrated probability of an outage that day
    alert                1 if outage_probability >= the model's alert threshold

Usage:
    python predict.py input.csv predictions.csv
    python predict.py input.csv predictions.csv --threshold 0.11   # custom operating point
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"


def load():
    card = json.loads((MODEL_DIR / "model_card.json").read_text())
    models = []
    for f in card["seed_models"]:
        m = xgb.XGBClassifier()
        m.load_model(MODEL_DIR / f)
        models.append(m)
    return card, models


def make_X(frame, card):
    missing = [f for f in card["features"] if f not in frame.columns]
    if missing:
        raise SystemExit(f"Input is missing {len(missing)} feature column(s): {missing[:10]}")
    unknown = sorted(set(frame["recloser"]) - set(card["reclosers"]))
    if unknown:
        print(f"WARNING: reclosers not seen in training (scored with no recloser indicator): {unknown}")
    X = frame[[f for f in card["features"] if f != "recloser"]].astype(float).copy()
    for r in card["reclosers"]:
        X["rc_" + r.replace(" ", "_")] = (frame["recloser"] == r).astype(float)
    return X.to_numpy()


def predict(frame, card, models):
    raw = np.mean([m.predict_proba(make_X(frame, card))[:, 1] for m in models], axis=0)
    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    z = card["calibration"]["coef"] * np.log(raw / (1 - raw)) + card["calibration"]["intercept"]
    return 1 / (1 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--threshold", type=float, default=None, help="override the alert threshold")
    args = ap.parse_args()

    card, models = load()
    frame = pd.read_csv(args.input)
    th = card["alert_threshold"] if args.threshold is None else args.threshold
    prob = predict(frame, card, models)
    keep = [c for c in ["date", "recloser"] if c in frame.columns]
    out = frame[keep].copy()
    out["outage_probability"] = prob
    out["alert"] = (prob >= th).astype(int)
    out.to_csv(args.output, index=False)
    print(f"Scored {len(out)} rows, {int(out['alert'].sum())} alerts at threshold {th:.4f} -> {args.output}")


if __name__ == "__main__":
    main()
