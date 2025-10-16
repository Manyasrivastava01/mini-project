import yaml
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

import xgboost as xgb


def make_class_labels(y: pd.Series, bins: list) -> pd.Series:
    # bins like [33, 66] -> 3 classes: 0,1,2
    return pd.cut(y, bins=[-1e18, bins[0], bins[1], 1e18], labels=[0, 1, 2], include_lowest=True).astype(int)


def temporal_split_idx(n: int, ratio: float):
    split = int(ratio * n)
    # guard rails
    split = min(max(split, 1), n - 1) if n > 1 else 0
    return np.arange(0, split), np.arange(split, n)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Force numeric, replace inf, drop all-NaN cols, median-impute."""
    X = df.copy()
    # force to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # replace inf/-inf -> NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    # drop columns that are entirely NaN
    X = X.loc[:, X.notna().any(axis=0)]
    if X.shape[1] == 0:
        return X
    # median impute
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return X


def get_feature_matrix(df: pd.DataFrame, target_col: str):
    drop_cols = {
        "subject_id", "session_id", "window_start", "window_end", "window_mid", target_col
    }
    # drop other label if present
    for c in ["mentalFatigueScore", "physicalFatigueScore"]:
        if c != target_col and c in df.columns:
            drop_cols.add(c)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = clean_features(X)
    return X


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    out_dir = Path(cfg["out_dir"])
    feats_path = out_dir / "features_all.csv"
    assert feats_path.exists(), f"Features file not found: {feats_path}. Run make_features.py first."

    # Parse dates if present (safe to ignore if not)
    parse_cols = ["window_start", "window_end", "window_mid"]
    use_parse = [c for c in parse_cols if c in pd.read_csv(feats_path, nrows=1).columns]
    df = pd.read_csv(feats_path, parse_dates=use_parse) if use_parse else pd.read_csv(feats_path)

    # Order by subject/session/time to make the temporal split meaningful
    sort_cols = [c for c in ["subject_id", "session_id", "window_mid"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    target_col = cfg["label_target"]
    assert target_col in df.columns, f"{target_col} not found in features_all.csv"

    # ---------------- Regression ----------------
    df_reg = df[df[target_col].notna()].copy()
    y_reg = df_reg[target_col].astype(float).values
    X_reg = get_feature_matrix(df_reg, target_col)

    if X_reg.shape[0] < 5 or X_reg.shape[1] == 0:
        raise RuntimeError("Not enough data or features for regression after cleaning.")

    tr_idx, te_idx = temporal_split_idx(len(X_reg), cfg["temporal_train_ratio"])

    # XGBoost prefers NumPy arrays (ensures no stray dtypes)
    dtrain = xgb.DMatrix(X_reg.iloc[tr_idx].values, label=y_reg[tr_idx])
    dtest  = xgb.DMatrix(X_reg.iloc[te_idx].values, label=y_reg[te_idx])
    params_reg = cfg["xgb_params_reg"].copy()
    # just to be safe, tell XGB that NaN is the missing value
    params_reg.setdefault("missing", np.nan)
    num_boost_round_reg = params_reg.pop("num_boost_round", 300)

    model_reg = xgb.train(params_reg, dtrain, num_boost_round=num_boost_round_reg)
    preds_reg = model_reg.predict(dtest)
    mae = float(mean_absolute_error(y_reg[te_idx], preds_reg))
    model_reg.save_model(str(out_dir / "xgb_regression.json"))

    # ---------------- Classification ----------------
    df_clf = df[df[target_col].notna()].copy()
    y_cont = df_clf[target_col].astype(float)
    y_clf = make_class_labels(y_cont, cfg["classification_bins"]).values
    X_clf = get_feature_matrix(df_clf, target_col)

    if X_clf.shape[0] < 5 or X_clf.shape[1] == 0:
        raise RuntimeError("Not enough data or features for classification after cleaning.")

    tr2, te2 = temporal_split_idx(len(X_clf), cfg["temporal_train_ratio"])
    dtrain_c = xgb.DMatrix(X_clf.iloc[tr2].values, label=y_clf[tr2])
    dtest_c  = xgb.DMatrix(X_clf.iloc[te2].values,  label=y_clf[te2])
    params_clf = cfg["xgb_params_clf"].copy()
    params_clf.setdefault("missing", np.nan)
    num_boost_round_clf = params_clf.pop("num_boost_round", 300)

    model_clf = xgb.train(params_clf, dtrain_c, num_boost_round=num_boost_round_clf)
    proba = model_clf.predict(dtest_c)  # shape (n,3)
    y_pred = proba.argmax(axis=1)

    acc = float(accuracy_score(y_clf[te2], y_pred))
    f1m = float(f1_score(y_clf[te2], y_pred, average="macro"))
    model_clf.save_model(str(out_dir / "xgb_classification.json"))

    # ---------------- LOSO (optional) ----------------
    loso_results = {}
    if cfg.get("do_loso", True) and "subject_id" in df_clf.columns:
        gkf = GroupKFold(n_splits=len(df_clf["subject_id"].unique()))
        groups = df_clf["subject_id"].values

        accs, f1ms = [], []
        for fold, (tr, te) in enumerate(gkf.split(X_clf, y_clf, groups)):
            dtr = xgb.DMatrix(X_clf.iloc[tr].values, label=y_clf[tr])
            dte = xgb.DMatrix(X_clf.iloc[te].values, label=y_clf[te])
            m = xgb.train(params_clf, dtr, num_boost_round=num_boost_round_clf)
            p = m.predict(dte).argmax(axis=1)
            accs.append(accuracy_score(y_clf[te], p))
            f1ms.append(f1_score(y_clf[te], p, average="macro"))

        loso_results = {
            "clf_LOSO_acc_mean": float(np.mean(accs)),
            "clf_LOSO_f1_macro_mean": float(np.mean(f1ms)),
            "clf_LOSO_acc_std": float(np.std(accs)),
            "clf_LOSO_f1_macro_std": float(np.std(f1ms)),
        }

    # ---------------- Save report ----------------
    report = {
        "temporal_split": cfg["temporal_train_ratio"],
        "regression_MAE_temporal": mae,
        "classification_acc_temporal": acc,
        "classification_f1_macro_temporal": f1m,
    }
    if loso_results:
        report["loso"] = loso_results

    import json
    with open(out_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved models & report to:", out_dir)
    print(report)


if __name__ == "__main__":
    main()
