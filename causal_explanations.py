# causal_explanations.py
# Build "why" text for each window using causal_summary edges + normalized features.
#
# Inputs:
#   outputs/regression_predictions.csv
#   outputs/features_all_normalized.csv
#   outputs/causal_summary.csv
#
# Output:
#   outputs/causal_explanations.csv
#
# Explanation logic (simple & robust):
# - Use stable causal edges: source_feature at lag L -> fatigue at time t
# - For each window t, fetch feature value at time t-L (previous window index shift)
# - Convert that lagged value into a within-session z-score (so magnitude is comparable)
# - Pick top K strongest drivers by |z|
# - Create a short natural-language explanation

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


OUT = Path("outputs")

PRED_PATH = OUT / "regression_predictions.csv"
FEAT_PATH = OUT / "features_all_normalized.csv"
CAUSAL_SUMMARY_PATH = OUT / "causal_summary.csv"

OUT_PATH = OUT / "causal_explanations.csv"

# ---- Tunables ----
TOP_K = 4                      # number of drivers to show in text
MIN_SUBJECT_FRAC = 0.15        # keep edges that appear in at least this fraction of subjects
MAX_MEAN_PVAL = 0.05           # keep edges that are statistically "strong enough"
MAX_LAG = 5                    # safety cap (your pipeline typically uses 1..5)

# Columns we need (predictions)
PRED_REQUIRED = {"subject_id", "session_id", "window_mid", "pred_score_ema", "risk_band"}
# Columns we need (features)
FEAT_REQUIRED = {"subject_id", "session_id", "window_mid"}
# Columns we need (causal summary)
CAUSAL_REQUIRED = {"source", "lag", "sign", "subject_frac", "mean_pval"}


def _as_str(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def load_predictions() -> pd.DataFrame:
    if not PRED_PATH.exists():
        raise SystemExit(f"Missing {PRED_PATH}. Run fatigue_inference.py (or predict_alerts.py) first.")
    df = pd.read_csv(PRED_PATH, parse_dates=["window_mid"])
    missing = PRED_REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"{PRED_PATH} missing columns: {sorted(missing)}")
    df = _as_str(df, ["subject_id", "session_id"])
    return df.sort_values(["subject_id", "session_id", "window_mid"]).reset_index(drop=True)


def load_features() -> pd.DataFrame:
    if not FEAT_PATH.exists():
        raise SystemExit(f"Missing {FEAT_PATH}. Run make_features.py then baseline_normalization.py.")
    df = pd.read_csv(FEAT_PATH, parse_dates=["window_mid"])
    missing = FEAT_REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"{FEAT_PATH} missing columns: {sorted(missing)}")
    df = _as_str(df, ["subject_id", "session_id"])
    return df.sort_values(["subject_id", "session_id", "window_mid"]).reset_index(drop=True)


def load_causal_edges() -> pd.DataFrame:
    if not CAUSAL_SUMMARY_PATH.exists():
        raise SystemExit(f"Missing {CAUSAL_SUMMARY_PATH}. Run causal_pcmci.py first.")
    edges = pd.read_csv(CAUSAL_SUMMARY_PATH)

    missing = CAUSAL_REQUIRED - set(edges.columns)
    if missing:
        raise SystemExit(f"{CAUSAL_SUMMARY_PATH} missing columns: {sorted(missing)}")

    # coerce types
    edges["lag"] = pd.to_numeric(edges["lag"], errors="coerce").fillna(0).astype(int)
    edges["subject_frac"] = pd.to_numeric(edges["subject_frac"], errors="coerce")
    edges["mean_pval"] = pd.to_numeric(edges["mean_pval"], errors="coerce")

    # filter stable edges
    edges = edges[
        (edges["lag"] >= 1) &
        (edges["lag"] <= MAX_LAG) &
        (edges["subject_frac"] >= MIN_SUBJECT_FRAC) &
        (edges["mean_pval"] <= MAX_MEAN_PVAL)
    ].copy()

    # keep only what we need
    edges = edges[["source", "lag", "sign", "subject_frac", "mean_pval"]].dropna()
    # prefer more stable + more significant
    edges = edges.sort_values(["subject_frac", "mean_pval"], ascending=[False, True]).reset_index(drop=True)
    return edges


def session_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return (s * 0)  # all zeros if constant/missing
    return (s - mu) / sd


def pretty_name(feat: str) -> str:
    # make it readable
    return (
        feat.replace("eeg_", "EEG ")
            .replace("_mean_mean", " mean")
            .replace("_slope_per_min", " slope/min")
            .replace("_var", " variance")
            .replace("_", " ")
            .strip()
    )


def direction_text(sign: str) -> str:
    s = str(sign).lower()
    if "pos" in s:
        return "higher"
    if "neg" in s:
        return "lower"
    return "changed"


def build_explanations(pred: pd.DataFrame, feats: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    # Merge predictions with features (on window_mid)
    merged = pred.merge(
        feats,
        on=["subject_id", "session_id", "window_mid"],
        how="left",
        suffixes=("", "_feat")
    )

    # Ensure every edge source exists in features; drop missing ones
    feature_cols = set(merged.columns)
    edges_ok = edges[edges["source"].isin(feature_cols)].copy()

    if edges_ok.empty:
        raise SystemExit(
            "No causal edges match the feature columns. "
            "Check that causal_summary.csv uses feature names exactly as in features_all_normalized.csv"
        )

    # We need per session indexing to shift by lag
    merged["_idx_in_session"] = merged.groupby(["subject_id", "session_id"]).cumcount()

    # Precompute z-scores per session for each source feature for comparability
    # (do only for the sources used)
    sources = edges_ok["source"].unique().tolist()
    for s in sources:
        merged[f"z_{s}"] = merged.groupby(["subject_id", "session_id"])[s].transform(session_zscore)

    # For each edge (source, lag), compute z at time (t - lag)
    for _, e in edges_ok.iterrows():
        src = e["source"]
        lag = int(e["lag"])
        merged[f"zlag_{src}_{lag}"] = merged.groupby(["subject_id", "session_id"])[f"z_{src}"].shift(lag)

    # Now build text explanations row by row
    explanation_texts = []
    top_drivers_json = []

    for i, r in merged.iterrows():
        # If no features merged, keep blank explanation
        if pd.isna(r.get(sources[0], np.nan)):
            explanation_texts.append("")
            top_drivers_json.append("[]")
            continue

        drivers = []
        for _, e in edges_ok.iterrows():
            src = e["source"]
            lag = int(e["lag"])
            zlag = r.get(f"zlag_{src}_{lag}", np.nan)
            if pd.isna(zlag):
                continue

            # score = |z| weighted slightly by stability
            stability = float(e["subject_frac"])
            score = abs(float(zlag)) * (0.7 + 0.3 * stability)

            drivers.append({
                "source": src,
                "lag": lag,
                "sign": str(e["sign"]),
                "z_lag": float(zlag),
                "subject_frac": stability,
                "mean_pval": float(e["mean_pval"]),
                "score": float(score),
            })

        if not drivers:
            explanation_texts.append("")
            top_drivers_json.append("[]")
            continue

        # pick top K
        drivers.sort(key=lambda d: d["score"], reverse=True)
        top = drivers[:TOP_K]

        # Create short human text
        parts = []
        for d in top:
            src_name = pretty_name(d["source"])
            lag = d["lag"]
            # interpret sign: if sign is negative, "lower HRV..." should correspond to fatigue increase
            # We avoid overclaiming; just say what was observed in the lagged window.
            observed = "high" if d["z_lag"] > 0.5 else ("low" if d["z_lag"] < -0.5 else "near baseline")
            parts.append(f"{src_name} was {observed} (~{lag} window(s) earlier)")

        # more direct for Medium/High; neutral for Low
        rb = str(r.get("risk_band", ""))
        if rb in ("Medium", "High"):
            text = "Likely drivers: " + "; ".join(parts) + "."
        else:
            text = "Main signals: " + "; ".join(parts) + "."

        explanation_texts.append(text)
        top_drivers_json.append(json.dumps(top))

    merged["causal_explanation"] = explanation_texts
    merged["top_causal_drivers_json"] = top_drivers_json

    # Keep output compact: predictions + explanation
    keep_cols = [
        "subject_id", "session_id", "window_mid", "task_block",
        "pred_score_ema", "risk_band", "alert_high",
        "mentalFatigueScore", "physicalFatigueScore",
        "causal_explanation", "top_causal_drivers_json"
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    return merged[keep_cols].copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    pred = load_predictions()
    feats = load_features()
    edges = load_causal_edges()

    out = build_explanations(pred, feats, edges)
    out.to_csv(OUT_PATH, index=False)

    print(f"Loaded: {PRED_PATH.name} rows={len(pred)}")
    print(f"Loaded: {FEAT_PATH.name} rows={len(feats)}")
    print(f"Using causal edges: {len(edges)} (after filtering)")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
