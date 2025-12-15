# ablation_study.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from xgboost import XGBRegressor, XGBClassifier

OUT = Path("outputs")
FEAT_NORM = OUT / "features_all_normalized.csv"
FEAT_RAW = OUT / "features_all.csv"

TARGET = "mentalFatigueScore"
CLF_BINS = (32.0, 62.0)   # keep consistent with tuned thresholds
TRAIN_RATIO = 0.7

def load_df():
    path = FEAT_NORM if FEAT_NORM.exists() else FEAT_RAW
    if not path.exists():
        raise SystemExit(f"Missing features file. Expected {FEAT_NORM} or {FEAT_RAW}. Run make_features.py first.")
    df = pd.read_csv(path, parse_dates=["window_mid"], low_memory=False)
    # sort & reset correctly (drop=True avoids the multiindex error)
    df = df.sort_values(["subject_id", "session_id", "window_mid"]).reset_index(drop=True)
    # keep rows with label
    df = df[df[TARGET].notna()].copy()
    return df

def col_groups(cols):
    # EEG (headband) features: eeg_* and forehead IMU prefixed with 'fore_'
    eeg = [c for c in cols if c.startswith("eeg_") or c.startswith("fore_")]
    # Wrist features: wrist accel prefix 'wr_' and wrist-derived metrics
    wrist_core = {
        "hr_mean","hr_var",
        "hrv_sdnn","hrv_rmssd","hrv_pnn50",
        "eda_mean","eda_peak_count","eda_peak_amp_mean","eda_slope_per_min",
        "temp_mean","temp_slope_per_min",
        "wr_acc_rms_mean"
    }
    wrist = [c for c in cols if c.startswith("wr_") or c in wrist_core]

    # columns that are NOT features
    non_feat = {
        "subject_id","session_id",
        "window_start","window_end","window_mid",
        "task_block",
        "mentalFatigueScore","physicalFatigueScore"
    }
    return eeg, wrist, non_feat

def prep_X(df, cols):
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return X

def y_classes(y):
    out = pd.cut(y, bins=[-1e18, CLF_BINS[0], CLF_BINS[1], 1e18],
                 labels=[0,1,2], include_lowest=True).astype(int)
    return np.asarray(out)

def eval_temporal(df, cols):
    X = prep_X(df, cols)
    y = df[TARGET].astype(float).values
    n = len(df)
    cut = int(TRAIN_RATIO * n)
    tr, te = np.arange(0, cut), np.arange(cut, n)

    # Regression
    mreg = XGBRegressor(tree_method="hist", random_state=42)
    mreg.fit(X.iloc[tr], y[tr])
    mae = float(mean_absolute_error(y[te], mreg.predict(X.iloc[te])))

    # Classification
    y_c = y_classes(y)
    mclf = XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist", random_state=42)
    mclf.fit(X.iloc[tr], y_c[tr])
    proba = mclf.predict_proba(X.iloc[te])
    pred = proba.argmax(axis=1)
    acc = float(accuracy_score(y_c[te], pred))
    f1m = float(f1_score(y_c[te], pred, average="macro"))
    return {"reg_mae_temporal": mae, "clf_acc_temporal": acc, "clf_f1_macro_temporal": f1m}

def eval_loso(df, cols):
    X = prep_X(df, cols)
    y = df[TARGET].astype(float).values
    y_c = y_classes(y)
    groups = df["subject_id"].astype(str).values

    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    maes, accs, f1s = [], [], []
    for tr, te in gkf.split(X, y, groups):
        # Regression
        mreg = XGBRegressor(tree_method="hist", random_state=42)
        mreg.fit(X.iloc[tr], y[tr])
        maes.append(mean_absolute_error(y[te], mreg.predict(X.iloc[te])))

        # Classification
        mclf = XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist", random_state=42)
        mclf.fit(X.iloc[tr], y_c[tr])
        pred = mclf.predict_proba(X.iloc[te]).argmax(axis=1)
        accs.append(accuracy_score(y_c[te], pred))
        f1s.append(f1_score(y_c[te], pred, average="macro"))

    return {
        "reg_mae_loso_mean": float(np.mean(maes)), "reg_mae_loso_std": float(np.std(maes)),
        "clf_acc_loso_mean": float(np.mean(accs)), "clf_acc_loso_std": float(np.std(accs)),
        "clf_f1_macro_loso_mean": float(np.mean(f1s)), "clf_f1_macro_loso_std": float(np.std(f1s)),
    }

def main():
    df = load_df()
    all_cols = df.columns.tolist()
    eeg_cols, wrist_cols, non_feat = col_groups(all_cols)
    combo_cols = [c for c in all_cols if c not in non_feat]

    results = {}
    for name, cols in [
        ("EEG_only", eeg_cols),
        ("Wrist_only", wrist_cols),
        ("Combined", combo_cols),
    ]:
        if not cols:
            results[name] = {"note": "no columns found"}
            continue
        res_t = eval_temporal(df, cols)
        res_l = eval_loso(df, cols)
        results[name] = {**res_t, **res_l}

    with open(OUT / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).T.to_csv(OUT / "ablation_results.csv")

    print("Saved:", OUT / "ablation_results.json")
    print("Saved:", OUT / "ablation_results.csv")
    print(results)

if __name__ == "__main__":
    main()
