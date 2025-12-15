# retrain_with_normalized.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

OUT = Path("outputs")
FEAT_NORM = OUT / "features_all_normalized.csv"
FEAT_RAW = OUT / "features_all.csv"
TARGET = "mentalFatigueScore"
CLF_BINS = (32.0, 62.0)  # keep consistent with tuned thresholds
TRAIN_RATIO = 0.7

def load_features():
    path = FEAT_NORM if FEAT_NORM.exists() else FEAT_RAW
    if not path.exists():
        raise SystemExit("Missing features. Run make_features.py (and normalize_features.py).")
    df = pd.read_csv(path, parse_dates=["window_mid"])
    df = df.sort_values(["subject_id","session_id","window_mid"]).reset_index(drop=True)
    return df

def get_X(df, target_col=TARGET):
    drop = {"subject_id","session_id","window_start","window_end","window_mid","task_block",
            "mentalFatigueScore","physicalFatigueScore"}
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf,-np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return X

def to_classes(y):
    out = pd.cut(y, bins=[-1e18, CLF_BINS[0], CLF_BINS[1], 1e18],
                 labels=[0,1,2], include_lowest=True).astype(int)
    return np.asarray(out)

def load_tuned_params():
    reg_p = OUT / "xgb_tuned_params_reg.json"
    clf_p = OUT / "xgb_tuned_params_clf.json"

    reg = {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2.0,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
        "tree_method": "hist", "random_state": 42, "n_jobs": -1
    }

    clf = {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2.0,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
        "objective": "multi:softprob", "num_class": 3,
        "tree_method": "hist", "random_state": 42, "n_jobs": -1
    }

    if reg_p.exists():
        reg.update(json.load(open(reg_p))["best_params"])
        reg.update({"tree_method": "hist", "random_state": 42, "n_jobs": -1})

    if clf_p.exists():
        clf.update(json.load(open(clf_p))["best_params"])
        clf.update({"objective": "multi:softprob", "num_class": 3,
                    "tree_method": "hist", "random_state": 42, "n_jobs": -1})

    return reg, clf


# -------------------------
# PATCH: extract n_estimators
# -------------------------
def train_xgb(params, X, y):
    params = params.copy()
    num_round = int(params.pop("n_estimators", 500))
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(params, dtrain, num_boost_round=num_round)


def temporal_split_eval(df, reg_params, clf_params):
    df = df[df[TARGET].notna()].copy()
    X = get_X(df)
    y = df[TARGET].astype(float).values
    n = len(df); cut = int(TRAIN_RATIO * n)
    tr, te = np.arange(0, cut), np.arange(cut, n)

    # Regression
    mreg = train_xgb(reg_params, X.iloc[tr].values, y[tr])
    pred_r = mreg.predict(xgb.DMatrix(X.iloc[te].values))
    mae = float(mean_absolute_error(y[te], pred_r))

    # Classification
    y_c = to_classes(y)
    mclf = train_xgb(clf_params, X.iloc[tr].values, y_c[tr])
    proba = mclf.predict(xgb.DMatrix(X.iloc[te].values))
    yhat = proba.argmax(axis=1)
    acc = float(accuracy_score(y_c[te], yhat))
    f1m = float(f1_score(y_c[te], yhat, average="macro"))

    return mreg, mclf, {
        "reg_mae_temporal": mae,
        "clf_acc_temporal": acc,
        "clf_f1_macro_temporal": f1m
    }


def loso_eval(df, reg_params, clf_params):
    df = df[df[TARGET].notna()].copy()
    X = get_X(df)
    y = df[TARGET].astype(float).values
    y_c = to_classes(y)
    groups = df["subject_id"].astype(str).values
    gkf = GroupKFold(n_splits=len(np.unique(groups)))

    maes, accs, f1s = [], [], []

    for tr, te in gkf.split(X, y, groups):
        mreg = train_xgb(reg_params, X.iloc[tr].values, y[tr])
        pred = mreg.predict(xgb.DMatrix(X.iloc[te].values))
        maes.append(mean_absolute_error(y[te], pred))

        mclf = train_xgb(clf_params, X.iloc[tr].values, y_c[tr])
        p = mclf.predict(xgb.DMatrix(X.iloc[te].values)).argmax(axis=1)
        accs.append(accuracy_score(y_c[te], p))
        f1s.append(f1_score(y_c[te], p, average="macro"))

    return {
        "clf_LOSO_acc_mean": float(np.mean(accs)),
        "clf_LOSO_f1_macro_mean": float(np.mean(f1s)),
        "clf_LOSO_acc_std": float(np.std(accs)),
        "clf_LOSO_f1_macro_std": float(np.std(f1s)),
        "reg_LOSO_mae_mean": float(np.mean(maes)),
        "reg_LOSO_mae_std": float(np.std(maes)),
    }


def main():
    df = load_features()
    reg_params, clf_params = load_tuned_params()

    # Temporal split
    mreg, mclf, tmetrics = temporal_split_eval(df, reg_params, clf_params)

    # LOSO
    lmetrics = loso_eval(df, reg_params, clf_params)

    # Save models (normalized variants)
    reg_path = OUT / "xgb_regression_norm.json"
    clf_path = OUT / "xgb_classification_norm.json"
    mreg.save_model(str(reg_path))
    mclf.save_model(str(clf_path))

    report = {"temporal_split": TRAIN_RATIO, **tmetrics, **lmetrics}
    with open(OUT / "training_report_norm.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved models & report to:", OUT)
    print(report)


if __name__ == "__main__":
    main()
