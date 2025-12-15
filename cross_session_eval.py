# cross_session_eval.py
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

OUT_DIR = Path("outputs")

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)]
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return X

def get_X(df: pd.DataFrame, target_col: str):
    drop_cols = {"subject_id","session_id","window_start","window_end","window_mid",target_col,"task_block"}
    for c in ["mentalFatigueScore","physicalFatigueScore"]:
        if c != target_col and c in df.columns:
            drop_cols.add(c)
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

def make_class_labels(y: pd.Series, bins):
    return pd.cut(y, bins=[-1e18, bins[0], bins[1], 1e18], labels=[0,1,2], include_lowest=True).astype(int)

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    target = cfg.get("label_target", "mentalFatigueScore")
    clf_bins = cfg.get("classification_bins", [33.0, 66.0])
    params_reg = cfg["xgb_params_reg"].copy()
    n_round_reg = params_reg.pop("num_boost_round", 300)
    params_clf = cfg["xgb_params_clf"].copy()
    n_round_clf = params_clf.pop("num_boost_round", 300)

    feats = pd.read_csv(OUT_DIR / "features_all.csv", parse_dates=["window_mid"])
    feats = feats.sort_values(["subject_id","session_id","window_mid"]).reset_index(drop=True)
    subs = sorted(feats["subject_id"].astype(str).unique())

    results = []
    for sid in subs:
        dsub = feats[feats["subject_id"].astype(str) == sid].copy()
        if not set(dsub["session_id"].astype(str).unique()) >= {"01","02","03"}:
            # try without leading zeros:
            ses = set(dsub["session_id"].astype(str).unique())
            if not {"1","2","3"}.issubset(ses):
                continue

        # normalize session ids to two-digit strings
        dsub["session_id"] = dsub["session_id"].astype(str).str.zfill(2)

        dtr = dsub[dsub["session_id"].isin(["01","02"]) & dsub[target].notna()]
        dte = dsub[dsub["session_id"].isin(["03"]) & dsub[target].notna()]
        if dtr.empty or dte.empty:
            continue

        # REG
        Xtr = clean_features(get_X(dtr, target))
        Xte = clean_features(get_X(dte, target))
        ytr = dtr[target].astype(float).values
        yte = dte[target].astype(float).values

        mreg = xgb.train(params_reg, xgb.DMatrix(Xtr.values, label=ytr), num_boost_round=n_round_reg)
        pred_r = mreg.predict(xgb.DMatrix(Xte.values))
        mae = float(mean_absolute_error(yte, pred_r))

        # CLF
        ytr_c = make_class_labels(dtr[target].astype(float), clf_bins).values
        yte_c = make_class_labels(dte[target].astype(float), clf_bins).values
        mclf = xgb.train(params_clf, xgb.DMatrix(Xtr.values, label=ytr_c), num_boost_round=n_round_clf)
        proba = mclf.predict(xgb.DMatrix(Xte.values))
        yhat = proba.argmax(axis=1)
        acc = float(accuracy_score(yte_c, yhat))
        f1m = float(f1_score(yte_c, yhat, average="macro"))

        results.append({"subject_id": sid, "mae_reg": mae, "acc_clf": acc, "f1_macro_clf": f1m})

    if not results:
        print("No valid subjects with 01/02 train and 03 test.")
        return

    dfres = pd.DataFrame(results)
    dfres.to_csv(OUT_DIR / "cross_session_eval.csv", index=False)
    summary = {
        "subjects": int(dfres.shape[0]),
        "reg_mae_mean": float(dfres["mae_reg"].mean()),
        "reg_mae_std": float(dfres["mae_reg"].std()),
        "clf_acc_mean": float(dfres["acc_clf"].mean()),
        "clf_acc_std": float(dfres["acc_clf"].std()),
        "clf_f1_macro_mean": float(dfres["f1_macro_clf"].mean()),
        "clf_f1_macro_std": float(dfres["f1_macro_clf"].std()),
    }
    import json
    with open(OUT_DIR / "cross_session_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", OUT_DIR / "cross_session_eval.csv")
    print("Saved:", OUT_DIR / "cross_session_summary.json")
    print(summary)

if __name__ == "__main__":
    main()
