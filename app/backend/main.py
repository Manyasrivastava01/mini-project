from pathlib import Path
from typing import Dict, Any, List

import json
import numpy as np
import pandas as pd
import xgboost as xgb

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[2]  # project root
OUT = ROOT / "outputs"
FRONTEND_DIR = ROOT / "app" / "frontend"

MODEL_REG_PATH = OUT / "xgb_regression_norm.json"
THRESH_PATH = OUT / "thresholds_tuned.json"
NORM_STATS_PATH = OUT / "norm_stats.json"
CAUSAL_SUMMARY_PATH = OUT / "causal_summary.csv"


# ---------------- App ----------------
app = FastAPI(title="Fatigue Demo API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for local demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files at /static/*
# (so it won't conflict with /health and /predict)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------- Load artifacts ----------------
if not MODEL_REG_PATH.exists():
    raise SystemExit(f"Missing model: {MODEL_REG_PATH}")

reg_model = xgb.Booster()
reg_model.load_model(str(MODEL_REG_PATH))

thresholds = {"medium_threshold": 32.0, "high_threshold": 62.0}
if THRESH_PATH.exists():
    t = json.load(open(THRESH_PATH, "r", encoding="utf-8"))
    thresholds.update(t.get("best_thresholds", t))

norm_stats = None
if NORM_STATS_PATH.exists():
    norm_stats = json.load(open(NORM_STATS_PATH, "r", encoding="utf-8"))

causal_df = pd.DataFrame()
if CAUSAL_SUMMARY_PATH.exists():
    causal_df = pd.read_csv(CAUSAL_SUMMARY_PATH)


# ---------------- Helpers ----------------
DROP_COLS = {
    "subject_id", "session_id", "window_start", "window_end", "window_mid", "task_block",
    "mentalFatigueScore", "physicalFatigueScore"
}

def risk_band(score: float) -> str:
    if score >= float(thresholds["high_threshold"]):
        return "HIGH"
    if score >= float(thresholds["medium_threshold"]):
        return "MEDIUM"
    return "LOW"

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    med = out.median(numeric_only=True)
    out = out.fillna(med)
    return out

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply z-normalization using outputs/norm_stats.json if present.
    Supports shapes:
      {"mean":{...}, "std":{...}} OR {"mu":{...}, "sigma":{...}}
    """
    if norm_stats is None:
        return df

    mean_map = norm_stats.get("mean") or norm_stats.get("mu") or {}
    std_map = norm_stats.get("std") or norm_stats.get("sigma") or {}

    out = df.copy()
    for c in out.columns:
        if c in mean_map and c in std_map:
            mu = float(mean_map[c])
            sd_raw = float(std_map[c])
            sd = sd_raw if sd_raw != 0 else 1.0
            out[c] = (out[c] - mu) / sd
    return out

def top_causal_reasons(feature_values: Dict[str, float], k: int = 3) -> List[Dict[str, Any]]:
    """
    Works with your outputs/causal_summary.csv format:
      source,lag,sign,subjects,mean_val,mean_pval,subject_frac
    """
    if causal_df.empty:
        return []

    need = {"source", "lag", "sign"}
    if not need.issubset(set(causal_df.columns)):
        return []

    df = causal_df.copy()
    df = df[df["source"].astype(str).isin(feature_values.keys())].copy()
    if df.empty:
        return []

    # pick strongest: prefer subject_frac then abs(mean_val) then pval
    if "subject_frac" in df.columns:
        df["subject_frac"] = pd.to_numeric(df["subject_frac"], errors="coerce")
    if "mean_val" in df.columns:
        df["mean_val"] = pd.to_numeric(df["mean_val"], errors="coerce")
    if "mean_pval" in df.columns:
        df["mean_pval"] = pd.to_numeric(df["mean_pval"], errors="coerce")

    sort_cols = []
    if "subject_frac" in df.columns: sort_cols.append(("subject_frac", False))
    if "mean_val" in df.columns:
        df["_abs_val"] = df["mean_val"].abs()
        sort_cols.append(("_abs_val", False))
    if "mean_pval" in df.columns: sort_cols.append(("mean_pval", True))

    if sort_cols:
        by = [c for c, _ in sort_cols]
        asc = [a for _, a in sort_cols]
        df = df.sort_values(by=by, ascending=asc)

    out = []
    for _, r in df.head(k).iterrows():
        f = str(r["source"])
        v = float(feature_values.get(f, np.nan))
        sign = str(r.get("sign", "")).lower()
        lag = int(r.get("lag", 0)) if pd.notna(r.get("lag", 0)) else None

        if sign == "positive":
            msg = f"{f} was higher than usual ~{lag} windows ago"
        elif sign == "negative":
            msg = f"{f} was lower than usual ~{lag} windows ago"
        else:
            msg = f"{f} changed ~{lag} windows ago"

        out.append({"feature": f, "value": v, "lag": lag, "sign": sign, "message": msg})
    return out


# ---------------- Routes ----------------
@app.get("/")
def home():
    # serve UI
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"ok": True, "msg": "Frontend not found. Put index.html in app/frontend/"}

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": True,
        "thresholds": thresholds,
        "norm_stats_loaded": norm_stats is not None,
        "causal_summary_loaded": not causal_df.empty,
        "frontend_found": (FRONTEND_DIR / "index.html").exists(),
    }

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """
    payload:
      {
        "features": { "hr_mean": 80.2, "eda_mean": 0.21, ... },
        "apply_normalization": true
      }
    """
    feats = payload.get("features", {}) or {}
    apply_norm = bool(payload.get("apply_normalization", True))

    if not isinstance(feats, dict) or len(feats) == 0:
        return {"error": "features must be a non-empty object/dict"}

    df = pd.DataFrame([feats])
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    df = clean_numeric(df)
    if apply_norm:
        df = normalize_features(df)

    dmat = xgb.DMatrix(df.values, feature_names=list(df.columns))
    score = float(reg_model.predict(dmat)[0])
    band = risk_band(score)

    reasons = top_causal_reasons({c: float(df.iloc[0][c]) for c in df.columns}, k=3)

    return {
        "score": score,
        "band": band,
        "thresholds": thresholds,
        "top_reasons": reasons,
        "used_normalization": apply_norm,
        "n_features": int(df.shape[1]),
        "feature_columns": list(df.columns),
    }
