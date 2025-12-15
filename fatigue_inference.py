# from pathlib import Path
# import json
# from typing import Tuple, Dict, Any
#
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# import yaml
#
# OUT_DIR = Path("outputs")
# TARGET = "mentalFatigueScore"
#
# # ---------- helpers shared with training ----------
#
# def get_X(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Build feature matrix from features_all(_normalized).csv-like DataFrame.
#     Must match the training-time preprocessing.
#     """
#     drop = {
#         "subject_id",
#         "session_id",
#         "window_start",
#         "window_end",
#         "window_mid",
#         "task_block",
#         "mentalFatigueScore",
#         "physicalFatigueScore",
#     }
#     X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
#     for c in X.columns:
#         X[c] = pd.to_numeric(X[c], errors="coerce")
#     X = X.replace([np.inf, -np.inf], np.nan)
#     med = X.median(numeric_only=True)
#     X = X.fillna(med)
#     return X
#
#
# def _load_thresholds(cfg: Dict[str, Any], out_dir: Path) -> Tuple[float, float]:
#     # defaults from config.yaml
#     alerts_cfg = cfg.get("alerts", {})
#     med = float(alerts_cfg.get("medium_threshold", 33.0))
#     high = float(alerts_cfg.get("high_threshold", 66.0))
#
#     # override from tuned thresholds if present
#     tt_path = out_dir / "thresholds_tuned.json"
#     if tt_path.exists():
#         try:
#             data = json.load(open(tt_path, "r"))
#             best = data.get("best_thresholds", {})
#             med = float(best.get("medium_threshold", med))
#             high = float(best.get("high_threshold", high))
#         except Exception:
#             pass
#
#     return med, high
#
#
# def _load_regression_model(out_dir: Path) -> xgb.Booster:
#     """
#     Prefer normalized model, fall back to original if needed.
#     """
#     for name in ["xgb_regression_norm.json", "xgb_regression.json"]:
#         path = out_dir / name
#         if path.exists():
#             booster = xgb.Booster()
#             booster.load_model(str(path))
#             return booster
#     raise FileNotFoundError("No regression model found in outputs/. "
#                             "Expected xgb_regression_norm.json or xgb_regression.json.")
#
#
# # ---------- engine ----------
#
# def load_engine(config_path: str = "config.yaml") -> Dict[str, Any]:
#     """
#     Load everything needed for inference: config, model, thresholds, smoothing params.
#     Returns a dict 'engine' to pass into predict_from_features.
#     """
#     cfg = yaml.safe_load(open(config_path, "r"))
#     out_dir = Path(cfg["out_dir"])
#
#     reg_model = _load_regression_model(out_dir)
#     med_th, high_th = _load_thresholds(cfg, out_dir)
#
#     alerts_cfg = cfg.get("alerts", {})
#     alpha = float(alerts_cfg.get("ema_alpha", 0.3))      # 0<alpha<=1
#     on_k = int(alerts_cfg.get("on_windows", 3))          # consecutive High to turn ON
#     off_k = int(alerts_cfg.get("off_windows", 3))        # consecutive non-High to turn OFF
#
#     return {
#         "cfg": cfg,
#         "out_dir": out_dir,
#         "reg_model": reg_model,
#         "medium_threshold": med_th,
#         "high_threshold": high_th,
#         "ema_alpha": alpha,
#         "on_windows": on_k,
#         "off_windows": off_k,
#     }
#
#
# def _band_from_score(score: float, med_th: float, high_th: float) -> str:
#     if score >= high_th:
#         return "High"
#     if score >= med_th:
#         return "Medium"
#     return "Low"
#
#
# def predict_from_features(df: pd.DataFrame, engine: Dict[str, Any]) -> pd.DataFrame:
#     """
#     Core inference function.
#     Input:  df with same schema as features_all(_normalized).csv
#     Output: new DataFrame with per-window predictions + smoothed alerts.
#     """
#
#     # sort & copy to avoid side effects
#     df = df.copy()
#     df = df.sort_values(["subject_id", "session_id", "window_mid"]).reset_index(drop=True)
#
#     X = get_X(df)
#     dmat = xgb.DMatrix(X.values)
#     y_raw = engine["reg_model"].predict(dmat)
#
#     df["pred_score_raw"] = y_raw
#
#     alpha = engine["ema_alpha"]
#     med_th = engine["medium_threshold"]
#     high_th = engine["high_threshold"]
#     on_k = engine["on_windows"]
#     off_k = engine["off_windows"]
#
#     # allocate new columns
#     df["pred_score_ema"] = np.nan
#     df["risk_band"] = "Low"
#     df["alert_high"] = 0
#
#     # apply EMA + hysteresis per subject/session
#     for (sid, sess), g_idx in df.groupby(["subject_id", "session_id"]).groups.items():
#         idx = list(g_idx)
#         ema = None
#         alert_on = 0
#         consec_high = 0
#         consec_non_high = 0
#
#         for i in idx:
#             raw = df.at[i, "pred_score_raw"]
#
#             # EMA
#             if ema is None or np.isnan(ema):
#                 ema = float(raw)
#             else:
#                 ema = float(alpha * raw + (1.0 - alpha) * ema)
#
#             df.at[i, "pred_score_ema"] = ema
#
#             band = _band_from_score(ema, med_th, high_th)
#             df.at[i, "risk_band"] = band
#
#             # hysteresis: persistence to turn alert on/off
#             if band == "High":
#                 consec_high += 1
#                 consec_non_high = 0
#             else:
#                 consec_non_high += 1
#                 consec_high = 0
#
#             if alert_on == 0 and consec_high >= on_k:
#                 alert_on = 1
#             elif alert_on == 1 and consec_non_high >= off_k:
#                 alert_on = 0
#
#             df.at[i, "alert_high"] = alert_on
#
#     return df
# fatigue_inference.py
# Single inference entrypoint:
# - reads windowed features (prefers normalized)
# - loads regression model
# - predicts fatigue score
# - applies EMA smoothing
# - assigns risk band using tuned thresholds
# - writes ONE predictions file: outputs/regression_predictions.csv

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb


OUT = Path("outputs")

# Prefer normalized features
FEAT_NORM = OUT / "features_all_normalized.csv"
FEAT_RAW  = OUT / "features_all.csv"

# Prefer normalized regression model
REG_MODEL_NORM = OUT / "xgb_regression_norm.json"
REG_MODEL_RAW  = OUT / "xgb_regression.json"

# Prefer tuned thresholds from tune_thresholds.py
THRESHOLDS_FILE = OUT / "thresholds_tuned.json"

# Output (single file)
PRED_OUT = OUT / "regression_predictions.csv"

# Columns we should NEVER use as features
DROP_COLS = {
    "subject_id", "session_id",
    "window_start", "window_end", "window_mid",
    "task_block",
    "mentalFatigueScore", "physicalFatigueScore",
}

DEFAULT_THRESHOLDS = {"medium_threshold": 32.0, "high_threshold": 62.0}


def load_features() -> pd.DataFrame:
    if FEAT_NORM.exists():
        df = pd.read_csv(FEAT_NORM, parse_dates=["window_mid", "window_start", "window_end"])
        df["_source_file"] = str(FEAT_NORM)
        return df
    if FEAT_RAW.exists():
        df = pd.read_csv(FEAT_RAW, parse_dates=["window_mid", "window_start", "window_end"])
        df["_source_file"] = str(FEAT_RAW)
        return df
    raise SystemExit("Missing features. Run make_features.py then baseline_normalization.py")


def load_regression_model() -> xgb.Booster:
    booster = xgb.Booster()
    if REG_MODEL_NORM.exists():
        booster.load_model(str(REG_MODEL_NORM))
        return booster
    if REG_MODEL_RAW.exists():
        booster.load_model(str(REG_MODEL_RAW))
        return booster
    raise SystemExit("Missing regression model. Run retrain_with_normalized.py (or train_models.py).")


def load_thresholds() -> dict:
    if THRESHOLDS_FILE.exists():
        obj = json.load(open(THRESHOLDS_FILE, "r"))
        best = obj.get("best_thresholds", obj)  # support both formats
        mt = float(best.get("medium_threshold", DEFAULT_THRESHOLDS["medium_threshold"]))
        ht = float(best.get("high_threshold", DEFAULT_THRESHOLDS["high_threshold"]))
        return {"medium_threshold": mt, "high_threshold": ht}

    # fallback
    return dict(DEFAULT_THRESHOLDS)


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").copy()

    # force numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # median impute for any remaining NaNs (safe for both normalized & raw)
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    return X


def ema_by_group(values: np.ndarray, alpha: float) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1 - alpha) * out[i - 1]
    return out


def risk_band(score: float, med_thr: float, high_thr: float) -> str:
    if score >= high_thr:
        return "High"
    if score >= med_thr:
        return "Medium"
    return "Low"


def main(alpha: float = 0.2):
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_features()
    model = load_regression_model()
    th = load_thresholds()

    # sort for stable EMA
    df = df.sort_values(["subject_id", "session_id", "window_mid"]).reset_index(drop=True)

    X = build_X(df)
    dmat = xgb.DMatrix(X.values)

    pred_raw = model.predict(dmat).astype(float)

    # per subject/session EMA smoothing
    pred_ema = np.full(len(df), np.nan, dtype=float)
    for (sid, sess), idx in df.groupby(["subject_id", "session_id"]).indices.items():
        idx_sorted = np.array(sorted(idx))
        pred_ema[idx_sorted] = ema_by_group(pred_raw[idx_sorted], alpha=alpha)

    out = pd.DataFrame({
        "subject_id": df["subject_id"].astype(str),
        "session_id": df["session_id"].astype(str),
        "window_mid": df["window_mid"],
        "task_block": df["task_block"] if "task_block" in df.columns else "",
        "mentalFatigueScore": df["mentalFatigueScore"] if "mentalFatigueScore" in df.columns else np.nan,
        "physicalFatigueScore": df["physicalFatigueScore"] if "physicalFatigueScore" in df.columns else np.nan,
        "pred_score_raw": pred_raw,
        "pred_score_ema": pred_ema,
    })

    out["risk_band"] = out["pred_score_ema"].apply(lambda s: risk_band(float(s), th["medium_threshold"], th["high_threshold"]))
    out["alert_high"] = (out["risk_band"] == "High").astype(int)

    out.to_csv(PRED_OUT, index=False)
    print(f"Features source: {df['_source_file'].iloc[0]}")
    print(f"Model used: {REG_MODEL_NORM if REG_MODEL_NORM.exists() else REG_MODEL_RAW}")
    print(f"Thresholds: medium={th['medium_threshold']} high={th['high_threshold']}")
    print(f"Saved: {PRED_OUT}")


if __name__ == "__main__":
    main(alpha=0.2)
