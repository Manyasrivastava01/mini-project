# shap_report.py
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from pathlib import Path

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
    drop_cols = {"subject_id","session_id","window_start","window_end","window_mid", target_col, "task_block"}
    # drop other label if present
    for c in ["mentalFatigueScore","physicalFatigueScore"]:
        if c != target_col and c in df.columns:
            drop_cols.add(c)
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    target_col = cfg.get("label_target", "mentalFatigueScore")

    feats = pd.read_csv(OUT_DIR / "features_all.csv", parse_dates=["window_mid","window_start","window_end"])
    feats = feats.sort_values(["subject_id","session_id","window_mid"]).reset_index(drop=True)

    X_df = clean_features(get_X(feats, target_col))
    feature_names = X_df.columns.tolist()
    dmat = xgb.DMatrix(X_df.values, feature_names=feature_names)

    model_path = OUT_DIR / "xgb_regression.json"
    if not model_path.exists():
        raise SystemExit("Train first: python train_models.py")
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # SHAP contributions from XGBoost directly
    contribs = booster.predict(dmat, pred_contribs=True)  # shape (n_samples, n_features + 1) last is bias
    shap_vals = contribs[:, :-1]
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(OUT_DIR / "shap_importance.csv", index=False)
    print("Saved:", OUT_DIR / "shap_importance.csv")

    # optional: per-task or per-subject summaries
    if "task_block" in feats.columns:
        feats_shap = feats[["subject_id","session_id","task_block"]].copy()
        for i, f in enumerate(feature_names):
            feats_shap[f] = shap_vals[:, i]
        # per-task mean |SHAP|
        per_task = feats_shap.groupby("task_block")[feature_names].apply(lambda x: np.mean(np.abs(x), axis=0))
        per_task.to_csv(OUT_DIR / "shap_per_task.csv")
        print("Saved:", OUT_DIR / "shap_per_task.csv")

if __name__ == "__main__":
    main()
