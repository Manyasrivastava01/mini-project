# baseline_normalization.py
"""
Per-subject baseline normalization.

For each subject:
- Identify baseline windows (task_block in {'baseline','rest'})
  or, if none, use the first N minutes per session as fallback.
- Compute per-feature mean & std over that baseline period.
- Z-score all feature columns using those stats.

Inputs:
    outputs/features_all.csv

Outputs:
    outputs/features_all_normalized.csv
    outputs/norm_stats.json
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path("outputs")
SRC = OUT / "features_all.csv"
DST = OUT / "features_all_normalized.csv"
STATS = OUT / "norm_stats.json"

BASELINE_KEYS = {"baseline", "rest"}
FALLBACK_MINUTES = 3


def is_feature_col(c: str) -> bool:
    """Return True if column is a feature (not id/label/metadata)."""
    blacklist = {
        "subject_id",
        "session_id",
        "window_start",
        "window_end",
        "window_mid",
        "mentalFatigueScore",
        "physicalFatigueScore",
        "task_block",
    }
    return c not in blacklist


def main():
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}. Run make_features.py first.")

    df = pd.read_csv(SRC, parse_dates=["window_start", "window_end", "window_mid"])
    df = df.sort_values(["subject_id", "session_id", "window_mid"]).reset_index(drop=True)

    feat_cols = [c for c in df.columns if is_feature_col(c)]

    # Ensure all feature columns are numeric float (prevents dtype warnings)
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df[feat_cols] = df[feat_cols].astype(float)

    out = df.copy()
    stats = {}

    for sid, dsub in df.groupby("subject_id"):
        dsub = dsub.copy()

        # 1) Try explicit baseline via task_block markers
        if "task_block" in dsub.columns:
            mask = dsub["task_block"].astype(str).str.lower().isin(BASELINE_KEYS)
            base_df = dsub[mask]
        else:
            base_df = pd.DataFrame()

        # 2) Fallback: first FALLBACK_MINUTES minutes of each session
        if base_df.empty:
            parts = []
            mins = pd.Timedelta(minutes=FALLBACK_MINUTES)
            for sess_id, ds in dsub.groupby("session_id"):
                if ds.empty:
                    continue
                t0 = ds["window_mid"].min()
                parts.append(
                    ds[(ds["window_mid"] >= t0) & (ds["window_mid"] <= t0 + mins)]
                )
            base_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        # If still empty, skip normalization for this subject
        if base_df.empty:
            print(f"[WARN] No baseline found for subject {sid}. Skipping normalization for this subject.")
            stats[str(sid)] = {
                "mean": {c: float("nan") for c in feat_cols},
                "std": {c: float("nan") for c in feat_cols},
                "n_baseline_rows": 0,
            }
            continue

        # ---- Compute baseline mean/std using NumPy to avoid nanops warnings ----
        base_vals = base_df[feat_cols].to_numpy(dtype=float)

        # nanmean / nanstd ignore NaNs and won't emit the nanops warning
        mu_arr = np.nanmean(base_vals, axis=0)
        # ddof=1 matches pandas std default; handle case where all NaN or single value
        with np.errstate(invalid="ignore"):
            sd_arr = np.nanstd(base_vals, axis=0, ddof=1)

        # Replace zeros with NaN to avoid divide-by-zero; these will later produce NaN
        sd_arr[sd_arr == 0] = np.nan

        mu = pd.Series(mu_arr, index=feat_cols)
        sd = pd.Series(sd_arr, index=feat_cols)

        stats[str(sid)] = {
            "mean": {c: (None if np.isnan(mu[c]) else float(mu[c])) for c in feat_cols},
            "std": {c: (None if np.isnan(sd[c]) else float(sd[c])) for c in feat_cols},
            "n_baseline_rows": int(base_df.shape[0]),
        }

        # ---- Apply z-score normalization to this subject's rows ----
        idx = out["subject_id"] == sid
        X_sub = out.loc[idx, feat_cols].to_numpy(dtype=float)

        # (X - mu) / sd, broadcasting over columns
        # where sd is NaN, result will become NaN → fine, will be handled later
        with np.errstate(invalid="ignore", divide="ignore"):
            X_norm = (X_sub - mu.values) / sd.values

        out.loc[idx, feat_cols] = X_norm

    OUT.mkdir(exist_ok=True)
    out.to_csv(DST, index=False)

    with open(STATS, "w") as f:
        json.dump(stats, f, indent=2)

    print("Saved:", DST)
    print("Saved:", STATS)


if __name__ == "__main__":
    main()
