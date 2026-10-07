"""Experiment 1: classical baseline.
Extract hand-crafted features -> Random Forest / XGBoost regressor -> diameter.
This establishes the floor that the CNN (Experiment 2) must beat."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import extract_features, features_to_vector  # noqa: E402
from evaluate import compute_metrics, save_metrics, plot_pred_vs_true  # noqa: E402

DATA_DIR = ROOT / "data" / "synthetic_tem"
RESULTS_DIR = ROOT / "results"


def load_split(split, meta):
    rows = [r for r in meta if r["split"] == split]
    X, y = [], []
    for r in rows:
        img = np.array(Image.open(DATA_DIR / r["filename"]))
        feats = extract_features(img, r["pixel_scale_nm_per_px"])
        X.append(features_to_vector(feats))
        y.append(r["diameter_nm"])
    return np.array(X), np.array(y)


def main():
    meta = json.load(open(DATA_DIR / "metadata.json"))
    print("Extracting features for train/val/test ...")
    X_train, y_train = load_split("train", meta)
    X_val, y_val = load_split("val", meta)
    X_test, y_test = load_split("test", meta)
    print(f"  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    from features import FEATURE_NAMES
    feature_names = FEATURE_NAMES

    results = {}

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    pred_rf = rf.predict(X_test)
    metrics_rf = compute_metrics(y_test, pred_rf)
    results["random_forest"] = metrics_rf
    plot_pred_vs_true(y_test, pred_rf, "Random Forest (classical baseline)",
                       RESULTS_DIR / "plots" / "baseline_rf_pred_vs_true.png")
    print("Random Forest:", metrics_rf)

    importances = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])
    results["random_forest_top_features"] = [[n, round(float(v), 4)] for n, v in importances[:8]]

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                                   eval_metric="mae")
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_xgb = xgb_model.predict(X_test)
    metrics_xgb = compute_metrics(y_test, pred_xgb)
    results["xgboost"] = metrics_xgb
    plot_pred_vs_true(y_test, pred_xgb, "XGBoost (classical baseline)",
                       RESULTS_DIR / "plots" / "baseline_xgb_pred_vs_true.png")
    print("XGBoost:", metrics_xgb)

    save_metrics(results, RESULTS_DIR / "baseline_classical_metrics.json")

    # save predictions for later report-writing
    pd.DataFrame({"y_true": y_test, "pred_rf": pred_rf, "pred_xgb": pred_xgb}).to_csv(
        RESULTS_DIR / "baseline_test_predictions.csv", index=False)

    print("\nSaved metrics to results/baseline_classical_metrics.json")


if __name__ == "__main__":
    main()
