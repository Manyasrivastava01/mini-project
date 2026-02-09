# app/backend/main.py
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import random
import re

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
# Thresholds
# - thresholds_scale_based.json: "Option B" (scale-based) defaults (recommended for paper narrative)
# - thresholds_tuned.json: older auto-tuned thresholds (may have zero "High" support depending on data)
THRESH_SCALE_PATH = OUT / "thresholds_scale_based.json"
THRESH_TUNED_PATH = OUT / "thresholds_tuned.json"
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
# Option B (scale-based): Low < 33 | Medium >= 33 and < 67 | High >= 67
# These are interpretable on a 0-100 fatigue scale and are valid even when the dataset has few/no
# extreme "High" labels (e.g., >=78). If a tuned file is present, we still prefer scale-based when
# available to avoid "High" having zero support.
thresholds = {"medium_threshold": 33.0, "high_threshold": 67.0, "mode": "scale_based"}

if THRESH_SCALE_PATH.exists():
    t = json.load(open(THRESH_SCALE_PATH, "r", encoding="utf-8"))
    thresholds.update(t.get("thresholds", t))
    thresholds["mode"] = t.get("mode", thresholds.get("mode", "scale_based"))
elif THRESH_TUNED_PATH.exists():
    t = json.load(open(THRESH_TUNED_PATH, "r", encoding="utf-8"))
    thresholds.update(t.get("best_thresholds", t))
    thresholds["mode"] = "tuned"

# ---------------- norm stats ----------------
norm_stats = None
if NORM_STATS_PATH.exists():
    norm_stats = json.load(open(NORM_STATS_PATH, "r", encoding="utf-8"))

# ---------------- causal summary ----------------
causal_df = pd.DataFrame()
if CAUSAL_SUMMARY_PATH.exists():
    causal_df = pd.read_csv(CAUSAL_SUMMARY_PATH)

# Build a quick lookup for causal strength/confidence per (source, lag)
CAUSAL_METRICS = {}  # (source, lag) -> dict
if not causal_df.empty and set(['source','lag']).issubset(set(causal_df.columns)):
    for _, row in causal_df.iterrows():
        key = (str(row.get('source')), int(row.get('lag')))
        CAUSAL_METRICS[key] = {
            'mean_val': float(row.get('mean_val')) if pd.notna(row.get('mean_val')) else None,
            'mean_pval': float(row.get('mean_pval')) if pd.notna(row.get('mean_pval')) else None,
            'subject_frac': float(row.get('subject_frac')) if pd.notna(row.get('subject_frac')) else None,
            'subjects': int(row.get('subjects')) if pd.notna(row.get('subjects')) else None,
            'sign': str(row.get('sign')) if pd.notna(row.get('sign')) else None,
        }


def confidence_label(subject_frac: Optional[float], pval: Optional[float]) -> str:
    """Heuristic confidence label from PCMCI summary."""
    if subject_frac is None or pval is None:
        return 'UNKNOWN'
    if pval <= 0.05 and subject_frac >= 0.50:
        return 'HIGH'
    if pval <= 0.10 and subject_frac >= 0.25:
        return 'MEDIUM'
    return 'LOW'


def strength_label(mean_val: Optional[float]) -> str:
    """Heuristic strength label from absolute effect size."""
    if mean_val is None:
        return 'UNKNOWN'
    a = abs(mean_val)
    if a >= 0.20:
        return 'STRONG'
    if a >= 0.10:
        return 'MODERATE'
    return 'WEAK'


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




def fill_missing_with_training_means(df: pd.DataFrame) -> (pd.DataFrame, List[str]):
    # Coerce to numeric and fill NaN/None/inf using TRAIN_MEANS.
    # Returns (filled_df, filled_columns)
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)

    filled_cols: List[str] = []
    if len(out) > 0:
        row0 = out.iloc[0]
        for c in out.columns:
            if pd.isna(row0[c]):
                out.at[out.index[0], c] = TRAIN_MEANS.get(c, 0.0)
                filled_cols.append(c)

    # Final safety: fill any remaining NaNs (shouldn't happen for single-row, but safe)
    if out.isna().any().any():
        out = out.fillna({c: TRAIN_MEANS.get(c, 0.0) for c in out.columns})
        for c in out.columns:
            if c not in filled_cols and pd.isna(df.iloc[0].get(c, np.nan)):
                filled_cols.append(c)

    return out, filled_cols

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


# ---------------- feature labeling + causal interpretation ----------------
# NOTE: these mappings are intentionally conservative (non-medical) and phrased as "may indicate".

EEG_BANDS = {"delta":"Delta","theta":"Theta","alpha":"Alpha","beta":"Beta","gamma":"Gamma"}

def feature_group(feature: str) -> str:
    f = (feature or "").lower()
    if f.startswith("eeg_"):
        return "EEG"
    if f.startswith("hrv_") or f.startswith("hr_"):
        return "Cardio"
    if f.startswith("eda_"):
        return "EDA"
    if f.startswith("temp_"):
        return "Temperature"
    if f.startswith("wr_") or f.startswith("fore_") or f.endswith("_acc_rms_mean"):
        return "Motion"
    return "Other"

def feature_label(feature: str) -> str:
    """Convert raw feature keys like 'eeg_alpha_mean_slope_per_min' to human-readable labels."""
    if not feature:
        return ""
    f = feature.strip()
    parts = f.split("_")

    # EEG power band features: eeg_<band>_mean_(mean|var|slope_per_min)
    if len(parts) >= 4 and parts[0] == "eeg" and parts[1] in EEG_BANDS and parts[2] == "mean":
        band = EEG_BANDS[parts[1]]
        stat = parts[3] if len(parts) > 3 else ""
        stat_map = {"mean":"level — mean", "var":"variability", "slope":"trend"}
        if stat == "slope":
            return f"EEG {band} level — trend (per min)"
        if stat in ("mean","var"):
            return f"EEG {band} level — {stat_map[stat]}"
        # slope_per_min comes as ['slope','per','min']
        if len(parts) >= 6 and parts[3] == "slope" and parts[4] == "per" and parts[5] == "min":
            return f"EEG {band} level — trend (per min)"
        return f"EEG {band} level"

    # EEG ratios: eeg_theta_alpha_ratio_mean / var / slope_per_min
    if f.startswith("eeg_") and "_ratio_" in f:
        # example: eeg_theta_alpha_ratio_mean
        m = re.match(r"eeg_([a-z]+)_([a-z]+)_ratio_(mean|var|slope_per_min)$", f)
        if m:
            a, b, stat = m.group(1), m.group(2), m.group(3)
            ratio = f"{EEG_BANDS.get(a,a.title())}/{EEG_BANDS.get(b,b.title())} ratio"
            if stat == "mean":
                return f"EEG {ratio} — mean"
            if stat == "var":
                return f"EEG {ratio} — variability"
            if stat == "slope_per_min":
                return f"EEG {ratio} — trend (per min)"

    # Drowsiness index
    if f.startswith("eeg_drowsiness_index_"):
        stat = f.replace("eeg_drowsiness_index_", "")
        if stat == "mean":
            return "EEG Drowsiness index — mean"
        if stat == "var":
            return "EEG Drowsiness index — variability"
        if stat == "slope_per_min":
            return "EEG Drowsiness index — trend (per min)"
        return "EEG Drowsiness index"

    # Cardio/HRV
    if f == "hr_mean":
        return "Heart rate — mean"
    if f == "hr_var":
        return "Heart rate — variability"
    if f == "hrv_sdnn":
        return "HRV SDNN"
    if f == "hrv_rmssd":
        return "HRV RMSSD"
    if f == "hrv_pnn50":
        return "HRV pNN50"

    # EDA
    if f == "eda_mean":
        return "Electrodermal activity — mean"
    if f == "eda_peak_count":
        return "EDA peaks — count"
    if f == "eda_peak_amp_mean":
        return "EDA peak amplitude — mean"
    if f == "eda_slope_per_min":
        return "EDA level — trend (per min)"

    # Temperature
    if f == "temp_mean":
        return "Skin temperature — mean"
    if f == "temp_slope_per_min":
        return "Skin temperature — trend (per min)"

    # Motion
    if f == "wr_acc_rms_mean":
        return "Wrist acceleration — RMS mean"
    if f == "fore_acc_rms_mean":
        return "Forearm acceleration — RMS mean"

    # fallback
    return f.replace("_", " ")

# Pattern-level interpretation (used to create a human-readable causal state + mitigation suggestions)
PATTERN_DEFS = {
    "reduced_vigilance": {
        "title": "Reduced cortical engagement / vigilance",
        "details": "Patterns in EEG that may align with reduced alertness or sustained attention.",
        "mitigations": [
            "Short physical activity break (2–5 minutes)",
            "Increase visual/cognitive stimulation (e.g., stand up, change posture, change task)",
            "Task rotation or brief task switching",
        ],
    },
    "drowsiness": {
        "title": "Increased drowsiness tendency",
        "details": "Markers that may align with drowsiness building over recent windows.",
        "mitigations": [
            "Take a short break and get light exposure if possible",
            "Avoid prolonged passive tasks; add interaction or checkpoints",
            "If safe to do so, hydrate and do gentle movement",
        ],
    },
    "autonomic_stress": {
        "title": "Autonomic stress / recovery imbalance",
        "details": "Heart-rate/HRV patterns that may align with stress load or reduced recovery.",
        "mitigations": [
            "Slow breathing for 1–2 minutes (comfortable pace)",
            "Micro-break + hydration",
            "Reduce intensity/pace for the next block if possible",
        ],
    },
    "sympathetic_arousal": {
        "title": "Increased sympathetic arousal",
        "details": "EDA patterns that may align with stress or higher arousal.",
        "mitigations": [
            "Brief pause and reset (30–60 seconds)",
            "Breathing or relaxation cue",
            "Reduce multitasking for the next few minutes",
        ],
    },
    "thermal_circadian": {
        "title": "Thermal / circadian-related shift",
        "details": "Temperature trends that may align with circadian dip or thermal comfort changes.",
        "mitigations": [
            "Adjust ambient temperature or airflow if possible",
            "Light movement to improve alertness",
            "Consider a brief break and reassess",
        ],
    },
    "low_movement": {
        "title": "Low movement / prolonged stillness",
        "details": "Motion features that may suggest prolonged stillness, which can correlate with fatigue buildup.",
        "mitigations": [
            "Stand up and move for 1–2 minutes",
            "Posture change + shoulder/neck mobility",
            "Add a short walking break if feasible",
        ],
    },
}

def feature_to_pattern_key(feature: str) -> Optional[str]:
    f = (feature or "").lower()
    # EEG vigilance & ratios
    if f.startswith("eeg_alpha_") or f.startswith("eeg_beta_alpha_ratio") or f.startswith("eeg_beta_"):
        return "reduced_vigilance"
    if f.startswith("eeg_theta_alpha_ratio") or f.startswith("eeg_drowsiness_index") or f.startswith("eeg_theta_") or f.startswith("eeg_delta_"):
        return "drowsiness"

    # Cardio / HRV
    if f.startswith("hrv_") or f.startswith("hr_"):
        return "autonomic_stress"

    # EDA
    if f.startswith("eda_"):
        return "sympathetic_arousal"

    # Temperature
    if f.startswith("temp_"):
        return "thermal_circadian"

    # Motion
    if f.startswith("wr_") or f.startswith("fore_") or f.endswith("_acc_rms_mean"):
        return "low_movement"

    return None


def summarize_causal_state(reasons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a deduplicated list of pattern-level summaries derived from *active, fatigue-increasing* reasons."""
    if not reasons:
        return []

    # Only build causal-state warnings from reasons that push fatigue UP.
    inc = [r for r in reasons if str(r.get("direction")) == "increase"]
    if not inc:
        return []

    seen = set()
    state: List[Dict[str, Any]] = []
    for r in inc:
        feat = str(r.get("feature") or "")
        pkey = feature_to_pattern_key(feat)
        if not pkey or pkey not in PATTERN_DEFS:
            continue
        if pkey in seen:
            continue
        seen.add(pkey)

        p = PATTERN_DEFS[pkey]
        evidence = str(r.get("message") or "")
        state.append({
            "pattern_key": pkey,
            "title": p["title"],
            "details": p["details"],
            "evidence": evidence,
        })
    return state


def mitigation_from_state(state: List[Dict[str, Any]]) -> List[str]:
    """Union of mitigation suggestions for detected pattern keys."""
    seen = set()
    out: List[str] = []
    for s in state or []:
        pkey = s.get("pattern_key")
        if not pkey or pkey not in PATTERN_DEFS:
            continue
        for tip in PATTERN_DEFS[pkey]["mitigations"]:
            if tip not in seen:
                seen.add(tip)
                out.append(tip)
    return out



def top_causal_reasons(
    feature_values: Dict[str, float],
    k: int = 3,
    activation_z: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Sample-conditional causal activation using outputs/causal_summary.csv (PCMCI learned offline).

    The causal graph is *global* (static), but explanations must be *conditional* on the current sample.

    We:
      1) Join current (normalized) feature values with PCMCI edges (source, lag, mean_val, pval, subject_frac)
      2) Mark an edge as "active" if |z| >= activation_z (baseline is ~0 after normalization)
      3) Compute directional contribution: contrib = mean_val * z
         - contrib > 0  => pushes fatigue UP
         - contrib < 0  => pushes fatigue DOWN (protective / counteracting)
      4) Rank by |contrib| (and lightly by stability), then return top-k active edges.

    Returns JSON-safe dicts (no NaN/inf).
    """
    if causal_df.empty:
        return []

    df = causal_df.copy()
    if "source" not in df.columns:
        return []

    # keep only edges whose source is in the current feature vector
    df = df[df["source"].astype(str).isin(feature_values.keys())].copy()
    if df.empty:
        return []

    # numeric coercion
    df["lag"] = pd.to_numeric(df.get("lag", 1), errors="coerce").fillna(1).astype(int)
    df["mean_val"] = pd.to_numeric(df.get("mean_val", np.nan), errors="coerce")
    df["mean_pval"] = pd.to_numeric(df.get("mean_pval", np.nan), errors="coerce")
    df["subject_frac"] = pd.to_numeric(df.get("subject_frac", np.nan), errors="coerce")

    # bring in current z-values (already normalized if apply_normalization=True)
    df["_z"] = df["source"].map(lambda s: float(feature_values.get(str(s), np.nan)))

    # treat missing z as inactive (shouldn't happen after schema alignment, but be safe)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["_z", "lag"])

    if df.empty:
        return []

    # If mean_val is missing, fall back to sign column (positive => +1, negative => -1)
    if "sign" in df.columns:
        sign_map = {"positive": 1.0, "negative": -1.0}
        df["_mean_val_filled"] = df["mean_val"]
        missing_mv = df["_mean_val_filled"].isna()
        df.loc[missing_mv, "_mean_val_filled"] = df.loc[missing_mv, "sign"].astype(str).str.lower().map(sign_map)
    else:
        df["_mean_val_filled"] = df["mean_val"]

    df["_mean_val_filled"] = pd.to_numeric(df["_mean_val_filled"], errors="coerce").fillna(0.0)

    # activation + contribution
    df["_active"] = df["_z"].abs() >= float(activation_z)
    df["_contrib"] = df["_mean_val_filled"] * df["_z"]

    active = df[df["_active"]].copy()
    if active.empty:
        return []

    # rank: |contrib| primarily, then stability lightly
    active["_stability"] = active["subject_frac"].fillna(0.0)
    active["_rank"] = active["_contrib"].abs() * (1.0 + 0.5 * active["_stability"])

    active = active.sort_values("_rank", ascending=False).head(int(k))

    out: List[Dict[str, Any]] = []
    for _, r in active.iterrows():
        src = str(r["source"])
        lag = int(r["lag"])
        z = float(r["_z"])
        mv = float(r["_mean_val_filled"])
        contrib = float(r["_contrib"])

        # labels
        direction = "increase" if contrib > 0 else "decrease"
        higher_lower = "higher" if z > 0 else "lower"
        feature_lbl = feature_label(src)

        sfrac = None if pd.isna(r.get("subject_frac", np.nan)) else float(r.get("subject_frac"))
        mpv = None if pd.isna(r.get("mean_pval", np.nan)) else float(r.get("mean_pval"))
        mval = None if pd.isna(r.get("mean_val", np.nan)) else float(r.get("mean_val"))

        conf = confidence_label(sfrac, mpv)
        strength = strength_label(mval if mval is not None else mv)

        # message that is direction-aware
        if direction == "increase":
            msg = f"{feature_lbl} was {higher_lower} than usual (~{lag} window(s) earlier), consistent with higher fatigue."
        else:
            msg = f"{feature_lbl} was {higher_lower} than usual (~{lag} window(s) earlier), consistent with lower fatigue (counteracting)."

        out.append({
            "feature": src,
            "feature_label": feature_lbl,
            "group": feature_group(src),
            "value": z,
            "lag": lag,
            "effect_mean_val": mv,
            "contribution": contrib,
            "direction": direction,  # increase/decrease fatigue
            "message": msg,
            "mean_val": mval,
            "mean_pval": mpv,
            "subject_frac": sfrac,
            "confidence": conf,
            "strength": strength,
            "active_z_threshold": float(activation_z),
        })

    return out




# ---------------- Causal graph artifacts ----------------
@app.get('/causal_graph.png')
def causal_graph_png():
    p = OUTPUTS_DIR / 'causal_graph.png'
    if not p.exists():
        return JSONResponse({'detail': 'causal_graph.png not found. Run generate_causal_graph_artifacts.py'}, status_code=404)
    return FileResponse(p)

@app.get('/causal_graph.json')
def causal_graph_json():
    p = OUTPUTS_DIR / 'causal_graph.json'
    if not p.exists():
        return JSONResponse({'detail': 'causal_graph.json not found. Run generate_causal_graph_artifacts.py'}, status_code=404)
    return FileResponse(p, media_type='application/json')

@app.get('/causal_graph_edges.csv')
def causal_graph_edges_csv():
    p = OUTPUTS_DIR / 'causal_graph_edges.csv'
    if not p.exists():
        return JSONResponse({'detail': 'causal_graph_edges.csv not found. Run generate_causal_graph_artifacts.py'}, status_code=404)
    return FileResponse(p, media_type='text/csv')
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

    # Coerce + fill any null/NaN/inf values using training means (robust to missing cells)
    df, null_filled = fill_missing_with_training_means(df)
    schema_info['null_filled_count'] = len(null_filled)
    schema_info['null_filled_features'] = null_filled[:200]

    # Normalize if requested
    if apply_norm:
        df = normalize_features(df)

    # Predict
    dmat = xgb.DMatrix(df.values, feature_names=list(df.columns))
    score = float(reg_model.predict(dmat)[0])
    band = risk_band(score)

    # Reasons from causal summary (lagged causal parents learned offline via PCMCI)
    reasons = top_causal_reasons({c: float(df.iloc[0][c]) for c in df.columns}, k=5)
    fatigue_increasing = [r for r in reasons if r.get('direction') == 'increase']
    fatigue_decreasing = [r for r in reasons if r.get('direction') == 'decrease']


    # Pattern-level causal state + suggested mitigations (conservative, non-medical)
    causal_state = summarize_causal_state(fatigue_increasing)
    mitigation_suggestions = mitigation_from_state(causal_state)
    causal_reasoning = (
        "These patterns are derived from lagged causal relationships (learned offline). "
        "The listed features are time-lagged parents of fatigue in the discovered graph, "
        "and the direction/lag indicates how they changed relative to baseline in prior window(s). "
        "This is intended for interpretability and triage, not as medical advice."
    )

    note = (
        f"You provided {schema_info['provided_feature_count']} fields. "
        f"Model expects {schema_info['expected_feature_count']} features. "
        f"Missing={schema_info['missing_feature_count']} were auto-filled using training means. Null/NaN={schema_info.get('null_filled_count',0)} were filled using training means. "
        f"Unknown={schema_info['unknown_feature_count']} were ignored."
    )

    return {
        "score": score,
        "band": band,
        "thresholds": thresholds,
        "warning": note,
        "schema_info": schema_info,
        "top_reasons": fatigue_increasing[:3],
        "protective_signals": fatigue_decreasing[:3],
        "causal_state": causal_state,
        "causal_reasoning": causal_reasoning,
        "mitigation_suggestions": mitigation_suggestions,
        "used_normalization": apply_norm,
        "note": note,
    }