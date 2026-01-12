# app/backend/main.py
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import random

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[2]  # project root
OUT = ROOT / "outputs"
FRONTEND_DIR = ROOT / "app" / "frontend"

# Artifacts (normalized pipeline)
MODEL_REG_PATH = OUT / "xgb_regression_norm.json"
THRESH_PATH = OUT / "thresholds_tuned.json"
NORM_STATS_PATH = OUT / "norm_stats.json"
CAUSAL_SUMMARY_PATH = OUT / "causal_summary.csv"

# Feature schema sources (for /sample + expected columns + training means)
FEAT_NORM = OUT / "features_all_normalized.csv"
FEAT_RAW = OUT / "features_all.csv"

# ---------------- App ----------------
app = FastAPI(title="Fatigue Demo API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for local demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load model ----------------
if not MODEL_REG_PATH.exists():
    raise SystemExit(f"Missing model: {MODEL_REG_PATH}")

reg_model = xgb.Booster()
reg_model.load_model(str(MODEL_REG_PATH))

# ---------------- thresholds ----------------
thresholds = {"medium_threshold": 32.0, "high_threshold": 62.0}
if THRESH_PATH.exists():
    t = json.load(open(THRESH_PATH, "r", encoding="utf-8"))
    thresholds.update(t.get("best_thresholds", t))

# ---------------- norm stats ----------------
norm_stats = None
if NORM_STATS_PATH.exists():
    norm_stats = json.load(open(NORM_STATS_PATH, "r", encoding="utf-8"))

# ---------------- causal summary ----------------
causal_df = pd.DataFrame()
if CAUSAL_SUMMARY_PATH.exists():
    causal_df = pd.read_csv(CAUSAL_SUMMARY_PATH)

# ---------------- schema + training means ----------------
DROP_COLS = {
    "subject_id", "session_id", "window_start", "window_end", "window_mid", "task_block",
    "mentalFatigueScore", "physicalFatigueScore"
}

EXPECTED_FEATURES: List[str] = []
TRAIN_MEANS: Dict[str, float] = {}
TRAIN_FILE_USED: Optional[str] = None


def _load_training_schema_and_means() -> None:
    """Load expected feature columns + per-feature training mean from outputs/features_all(_normalized).csv"""
    global EXPECTED_FEATURES, TRAIN_MEANS, TRAIN_FILE_USED

    path = FEAT_NORM if FEAT_NORM.exists() else FEAT_RAW
    if not path.exists():
        EXPECTED_FEATURES = []
        TRAIN_MEANS = {}
        TRAIN_FILE_USED = None
        return

    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # keep numeric-ish columns only (but don't drop columns just because one row is bad)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # training mean for autofill
    means = df.mean(numeric_only=True)

    EXPECTED_FEATURES = list(df.columns)
    TRAIN_MEANS = {k: (float(v) if pd.notna(v) else 0.0) for k, v in means.to_dict().items()}
    TRAIN_FILE_USED = path.name


_load_training_schema_and_means()

# ---------------- helpers ----------------
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

    # median-impute within provided columns (this is NOT the training-mean autofill)
    med = out.median(numeric_only=True)
    out = out.fillna(med)
    return out


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply z-normalization using outputs/norm_stats.json if present."""
    if norm_stats is None:
        return df

    mean_map = norm_stats.get("mean") or norm_stats.get("mu") or {}
    std_map = norm_stats.get("std") or norm_stats.get("sigma") or {}

    out = df.copy()
    for c in out.columns:
        if c in mean_map and c in std_map:
            mu = float(mean_map[c])
            sd = float(std_map[c]) if float(std_map[c]) != 0 else 1.0
            out[c] = (out[c] - mu) / sd
    return out


def align_to_training_schema(user_df: pd.DataFrame) -> (pd.DataFrame, Dict[str, Any]):
    """
    Align incoming features to EXPECTED_FEATURES:
      - drop metadata columns
      - ignore unknown columns
      - autofill missing columns with TRAIN_MEANS
    Returns (aligned_df, info)
    """
    if not EXPECTED_FEATURES:
        raise ValueError(
            "Training feature schema not available. "
            "Run make_features.py and baseline_normalization.py to create outputs/features_all(_normalized).csv"
        )

    df = user_df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    provided = list(df.columns)
    unknown = [c for c in provided if c not in EXPECTED_FEATURES]
    df = df.drop(columns=unknown, errors="ignore")

    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    for c in missing:
        df[c] = TRAIN_MEANS.get(c, 0.0)

    # reorder
    df = df[EXPECTED_FEATURES]

    info = {
        "expected_feature_count": len(EXPECTED_FEATURES),
        "provided_feature_count": len(provided),
        "used_feature_count": len(EXPECTED_FEATURES),
        "missing_feature_count": len(missing),
        "unknown_feature_count": len(unknown),
        "missing_features": missing[:200],  # cap for UI
        "unknown_features": unknown[:200],
        "training_schema_source": TRAIN_FILE_USED,
        "autofilled_missing": True,
    }
    return df, info


def _json_safe_value(v):
    # Convert NaN/inf -> None, numpy scalars -> python
    if v is None:
        return None
    try:
        if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
            return None
    except Exception:
        pass
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def to_json_safe_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _json_safe_value(v) for k, v in d.items()}


def top_causal_reasons(feature_values: Dict[str, float], k: int = 3) -> List[Dict[str, Any]]:
    """
    Uses outputs/causal_summary.csv if available.
    Expected columns (your file): source, lag, sign, mean_val, mean_pval, subject_frac
    We rank by subject_frac then abs(mean_val).
    """
    if causal_df.empty:
        return []

    df = causal_df.copy()
    if "source" not in df.columns:
        return []

    df = df[df["source"].astype(str).isin(feature_values.keys())].copy()
    if df.empty:
        return []

    # rank “more stable across people” first
    if "subject_frac" in df.columns:
        df["subject_frac"] = pd.to_numeric(df["subject_frac"], errors="coerce")
    if "mean_val" in df.columns:
        df["mean_val"] = pd.to_numeric(df["mean_val"], errors="coerce")

    df["_rank"] = 0.0
    if "subject_frac" in df.columns:
        df["_rank"] += df["subject_frac"].fillna(0) * 10.0
    if "mean_val" in df.columns:
        df["_rank"] += df["mean_val"].abs().fillna(0)

    df = df.sort_values("_rank", ascending=False).head(k)

    out = []
    for _, r in df.iterrows():
        f = str(r["source"])
        v = float(feature_values.get(f, np.nan))
        lag = int(r["lag"]) if "lag" in r and pd.notna(r["lag"]) else None
        sign = str(r["sign"]) if "sign" in r and pd.notna(r["sign"]) else None

        trend = "higher" if sign == "positive" else "lower" if sign == "negative" else "changed"
        msg = f"{f} was {trend} than usual ~{lag} windows ago" if lag is not None else f"{f} was {trend} than usual"

        out.append({"feature": f, "value": v, "lag": lag, "sign": sign, "message": msg})
    return out


# ---------------- Static frontend ----------------
@app.get("/")
def home():
    p = FRONTEND_DIR / "index.html"
    if not p.exists():
        return JSONResponse({"detail": "Frontend not found"}, status_code=404)
    return FileResponse(p)


@app.get("/app.js")
def frontend_js():
    p = FRONTEND_DIR / "app.js"
    if not p.exists():
        return JSONResponse({"detail": "app.js not found"}, status_code=404)
    return FileResponse(p)


@app.get("/styles.css")
def frontend_css():
    p = FRONTEND_DIR / "styles.css"
    if not p.exists():
        return JSONResponse({"detail": "styles.css not found"}, status_code=404)
    return FileResponse(p)


# ---------------- API endpoints ----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": True,
        "thresholds": thresholds,
        "norm_stats_loaded": norm_stats is not None,
        "causal_summary_loaded": not causal_df.empty,
        "training_schema_loaded": bool(EXPECTED_FEATURES),
        "training_schema_source": TRAIN_FILE_USED,
        "expected_feature_count": len(EXPECTED_FEATURES),
        "frontend_found": (FRONTEND_DIR / "index.html").exists(),
    }


@app.get("/sample")
def sample():
    """
    Return a random sample row for UI.
    IMPORTANT: must be JSON-safe (no NaN/inf), otherwise intermittent 500.
    """
    path = FEAT_NORM if FEAT_NORM.exists() else FEAT_RAW
    if not path.exists():
        return JSONResponse({"error": f"Missing {path.name}. Run make_features.py first."}, status_code=400)

    df = pd.read_csv(path)
    if df.empty:
        return JSONResponse({"error": f"{path.name} is empty."}, status_code=400)

    # Choose a random row
    idx = random.randrange(len(df))
    row = df.iloc[idx].to_dict()

    # Keep only schema features (best for UI + avoids timestamps/labels)
    row = {k: row.get(k) for k in EXPECTED_FEATURES} if EXPECTED_FEATURES else row
    row = to_json_safe_dict(row)

    return {
        "ok": True,
        "message": "Loaded sample row into textbox",
        "source_file": path.name,
        "row_index": int(idx),
        "n_features": len(row),
        "features": row,
    }


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """
    payload:
      {
        "features": { ... },
        "apply_normalization": true
      }
    """
    feats: Dict[str, Any] = payload.get("features", {}) or {}
    apply_norm: bool = bool(payload.get("apply_normalization", True))

    if not isinstance(feats, dict) or len(feats) == 0:
        return JSONResponse({"error": "features must be a non-empty object/dict"}, status_code=400)

    user_df = pd.DataFrame([feats])

    # Align to schema (autofill missing with training means, drop unknown)
    try:
        df, schema_info = align_to_training_schema(user_df)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Numeric cleaning
    df = clean_numeric(df)

    # Normalize if requested
    if apply_norm:
        df = normalize_features(df)

    # Predict
    dmat = xgb.DMatrix(df.values, feature_names=list(df.columns))
    score = float(reg_model.predict(dmat)[0])
    band = risk_band(score)

    # Reasons from causal summary
    reasons = top_causal_reasons({c: float(df.iloc[0][c]) for c in df.columns}, k=3)

    note = (
        f"You provided {schema_info['provided_feature_count']} fields. "
        f"Model expects {schema_info['expected_feature_count']} features. "
        f"Missing={schema_info['missing_feature_count']} were auto-filled using training means. "
        f"Unknown={schema_info['unknown_feature_count']} were ignored."
    )

    return {
        "score": score,
        "band": band,
        "thresholds": thresholds,
        "warning": note,
        "schema_info": schema_info,
        "top_reasons": reasons,
        "used_normalization": apply_norm,
        "note": note,
    }
