import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb


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
    drop_cols = {"subject_id","session_id","window_start","window_end","window_mid",target_col}
    for c in ["mentalFatigueScore","physicalFatigueScore"]:
        if c != target_col and c in df.columns:
            drop_cols.add(c)
    if "task_block" in df.columns:
        drop_cols.add("task_block")
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return clean_features(X)


def ema(series, alpha):
    out = []
    s = None
    for v in series:
        if s is None or np.isnan(s):
            s = v
        else:
            s = alpha * v + (1 - alpha) * s
        out.append(s)
    return np.array(out)


def persistence_flags(score_series, threshold, k):
    flags = np.zeros(len(score_series), dtype=int)
    count = 0
    for i, v in enumerate(score_series):
        if v >= threshold:
            count += 1
        else:
            count = 0
        flags[i] = 1 if count >= k else 0
    return flags


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    out_dir = Path(cfg["out_dir"])
    feats_path = out_dir / "features_all.csv"
    assert feats_path.exists(), "Run make_features.py first."

    # Load features
    parse_cols = ["window_start","window_end","window_mid"]
    cols = pd.read_csv(feats_path, nrows=1).columns.tolist()
    parse_cols = [c for c in parse_cols if c in cols]
    df = pd.read_csv(feats_path, parse_dates=parse_cols)
    df = df.sort_values(["subject_id","session_id","window_mid"]).reset_index(drop=True)

    target_col = cfg["label_target"]
    mode = cfg.get("smoothing", {}).get("mode", "regression")
    alpha = cfg.get("smoothing", {}).get("ema_alpha", 0.3)
    k = cfg.get("smoothing", {}).get("persistence_k", 3)

    # Prepare X
    X = get_X(df, target_col)

    # Load models
    reg_model_path = out_dir / "xgb_regression.json"
    clf_model_path = out_dir / "xgb_classification.json"

    outputs = []

    if mode == "regression" and reg_model_path.exists():
        booster = xgb.Booster()
        booster.load_model(str(reg_model_path))
        preds = booster.predict(xgb.DMatrix(X.values))
        df["pred_score_raw"] = preds
        df["pred_score_ema"] = ema(df["pred_score_raw"].values, alpha)

        hi = cfg.get("alerts", {}).get("high_threshold", 66.0)
        med = cfg.get("alerts", {}).get("medium_threshold", 33.0)
        df["risk_band"] = pd.cut(df["pred_score_ema"], bins=[-1e18, med, hi, 1e18],
                                 labels=["Low","Medium","High"], include_lowest=True)
        df["alert_high"] = persistence_flags(df["pred_score_ema"].values, hi, k)

        outputs.append(("regression_predictions.csv",
                        df[["subject_id","session_id","window_mid","pred_score_raw","pred_score_ema","risk_band","alert_high"] + ([ "task_block" ] if "task_block" in df.columns else []) ]))

    else:
        # classification mode
        if clf_model_path.exists():
            booster = xgb.Booster()
            booster.load_model(str(clf_model_path))
            proba = booster.predict(xgb.DMatrix(X.values))  # (n,3)
            pred_cls = proba.argmax(axis=1)
            df["pred_class_raw"] = pred_cls
            # map 0/1/2 to scores for smoothing if desired
            score_map = {0: 16.5, 1: 49.5, 2: 83.5}  # centers of bins
            df["pred_score_proxy"] = df["pred_class_raw"].map(score_map).astype(float)
            df["pred_score_ema"] = ema(df["pred_score_proxy"].values, alpha)

            hi = cfg.get("alerts", {}).get("high_threshold", 66.0)
            df["alert_high"] = persistence_flags(df["pred_score_ema"].values, hi, k)

            outputs.append(("classification_predictions.csv",
                            df[["subject_id","session_id","window_mid","pred_class_raw","pred_score_proxy","pred_score_ema","alert_high"] + ([ "task_block" ] if "task_block" in df.columns else []) ]))

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs:
        frame.to_csv(out_dir / name, index=False)
        print("Saved:", out_dir / name)


if __name__ == "__main__":
    main()
