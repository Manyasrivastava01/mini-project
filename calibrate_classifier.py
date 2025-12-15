# calibrate_classifier.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from xgboost import XGBClassifier
import joblib
import numpy as np


OUT = Path("outputs")
FEAT_NORM = OUT / "features_all_normalized.csv"
FEAT_RAW = OUT / "features_all.csv"
PARAMS = OUT / "xgb_tuned_params_clf.json"
MODEL_OUT = OUT / "xgb_classification_calibrated.joblib"
REPORT = OUT / "calibration_report.json"

TARGET = "mentalFatigueScore"
CLF_BINS = (32.0, 62.0)
TRAIN_RATIO = 0.7  # temporal split

def load_df():
    path = FEAT_NORM if FEAT_NORM.exists() else FEAT_RAW
    df = pd.read_csv(path, parse_dates=["window_mid"])
    df = df.sort_values(["subject_id","session_id","window_mid"]).reset_index(drop=True)
    return df

def get_X(df):
    drop = {"subject_id","session_id","window_start","window_end","window_mid","task_block",
            "mentalFatigueScore","physicalFatigueScore"}
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf,-np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return X

def y_to_classes(y):
    out = pd.cut(y, bins=[-1e18, CLF_BINS[0], CLF_BINS[1], 1e18],labels=[0,1,2], include_lowest=True).astype(int)
    return np.asarray(out)

def main():
    df = load_df()
    df = df[df[TARGET].notna()].copy()
    X = get_X(df)
    y = y_to_classes(df[TARGET].astype(float).values)

    # temporal split
    n = len(df)
    cut = int(TRAIN_RATIO * n)
    tr, te = np.arange(0, cut), np.arange(cut, n)

    # load tuned params if present
    params = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.1,
              "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2.0,
              "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
              "objective": "multi:softprob", "num_class": 3,
              "tree_method": "hist", "random_state": 42, "n_jobs": -1}
    if PARAMS.exists():
        tuned = json.load(open(PARAMS))["best_params"]
        params.update(tuned)
        params.update({"objective":"multi:softprob","num_class":3,"tree_method":"hist","random_state":42,"n_jobs":-1})

    base = XGBClassifier(**params)
    # Calibrate on validation (the temporal test split)
    cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")

    base.fit(X.iloc[tr], y[tr])
    cal.fit(X.iloc[te], y[te])  # fit calibrator on holdout

    # Evaluate on the same holdout (post-calibration)
    proba = cal.predict_proba(X.iloc[te])
    pred = proba.argmax(axis=1)
    acc = float(accuracy_score(y[te], pred))
    f1m = float(f1_score(y[te], pred, average="macro"))
    # Brier (multi-class: mean over classes)
    brier = float(np.mean(np.sum((proba - (np.eye(3)[y[te]]))**2, axis=1)))

    joblib.dump(cal, MODEL_OUT)
    with open(REPORT, "w") as f:
        json.dump({"acc_temporal": acc, "f1_macro_temporal": f1m, "brier_temporal": brier}, f, indent=2)

    print("Saved:", MODEL_OUT)
    print("Saved:", REPORT)
    print({"acc": acc, "f1_macro": f1m, "brier": brier})

if __name__ == "__main__":
    main()
