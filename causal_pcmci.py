# causal_pcmci.py
"""
Causal discovery for mental fatigue using PCMCI (tigramite).

Target: mentalFatigueScore  (ground truth).
Input:  outputs/features_all_normalized.csv
Output:
  - outputs/causal_edges.csv      (per-subject edges)
  - outputs/causal_summary.csv    (aggregated edges across subjects)

You must install tigramite first:
    pip install tigramite
"""

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from tigramite.data_processing import DataFrame as TgDataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

OUT_DIR = Path("outputs")
FEATS_NORM = OUT_DIR / "features_all_normalized.csv"

TARGET_COL = "mentalFatigueScore"
MAX_LAG = 5                # in windows (with 15s stride -> ~1.25 minutes for lag=5)
PC_ALPHA = 0.2             # screening alpha for PC step (higher = more edges)
ALPHA_EDGE = 0.05          # significance threshold for final edges
MIN_ABS_VAL = 0.1          # minimum absolute association strength to keep edge
MAX_NAN_FRAC = 0.3         # drop feature if more than this fraction is NaN


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

ID_COLS = {
    "subject_id",
    "session_id",
    "window_start",
    "window_end",
    "window_mid",
    "task_block",
    "physicalFatigueScore",
}

def choose_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Choose numeric feature columns (excluding IDs, timestamps, labels).
    """
    cols = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        if c == TARGET_COL:
            continue
        if df[c].dtype.kind in "biufc":  # numeric
            cols.append(c)
    return cols


def prepare_subject_timeseries(df_subj: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare a numpy array (T, V) for PCMCI, where V = features + target.
    Returns dict with 'array' and 'var_names'.
    """
    # sort by time
    df_subj = df_subj.sort_values(["session_id", "window_mid"]).reset_index(drop=True)

    feat_cols = choose_feature_cols(df_subj)
    if TARGET_COL not in df_subj.columns:
        raise ValueError(f"{TARGET_COL} not in dataframe columns")

    # keep only rows where target is not NaN
    df_subj = df_subj[df_subj[TARGET_COL].notna()].reset_index(drop=True)
    if df_subj.empty:
        return {"array": None, "var_names": []}

    # numeric conversion & NaN handling
    for c in feat_cols + [TARGET_COL]:
        df_subj[c] = pd.to_numeric(df_subj[c], errors="coerce")

    # drop features with too many NaNs
    good_feats = []
    for c in feat_cols:
        nan_frac = df_subj[c].isna().mean()
        if nan_frac <= MAX_NAN_FRAC:
            good_feats.append(c)

    feat_cols = good_feats
    if not feat_cols:
        return {"array": None, "var_names": []}

    # simple imputation: forward-fill then backward-fill
    df_subj[feat_cols + [TARGET_COL]] = df_subj[feat_cols + [TARGET_COL]].ffill().bfill()

    # build array (T, V)
    var_names = feat_cols + [TARGET_COL]
    data_arr = df_subj[var_names].to_numpy(dtype=float)

    return {"array": data_arr, "var_names": var_names}


def run_pcmci_for_subject(
    subj_id: str,
    data_arr: np.ndarray,
    var_names: List[str],
) -> pd.DataFrame:
    """
    Run PCMCI on a single subject and return edges involving TARGET_COL.
    """
    if data_arr is None or data_arr.shape[0] < 50:
        # too short, skip
        return pd.DataFrame(columns=["subject_id", "source", "target", "lag", "val", "pval", "sign"])

    # last variable is target
    target_idx = len(var_names) - 1

    # tigramite expects shape (T, V)
    tg_df = TgDataFrame(data_arr, var_names=var_names)

    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=tg_df, cond_ind_test=parcorr)

    results = pcmci.run_pcmci(
        tau_max=MAX_LAG,
        pc_alpha=PC_ALPHA,
    )

    val_matrix = results["val_matrix"]   # shape (V, V, tau_max+1)
    p_matrix = results["p_matrix"]

    rows = []
    V = len(var_names)

    for src_idx in range(V):
        if src_idx == target_idx:
            continue

        src_name = var_names[src_idx]
        tgt_name = var_names[target_idx]

        for tau in range(1, MAX_LAG + 1):
            val = float(val_matrix[src_idx, target_idx, tau])
            pval = float(p_matrix[src_idx, target_idx, tau])

            if np.isnan(val) or np.isnan(pval):
                continue

            if pval <= ALPHA_EDGE and abs(val) >= MIN_ABS_VAL:
                rows.append({
                    "subject_id": subj_id,
                    "source": src_name,
                    "target": tgt_name,
                    "lag": tau,
                    "val": val,
                    "pval": pval,
                    "sign": "positive" if val > 0 else "negative",
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    assert FEATS_NORM.exists(), f"Missing {FEATS_NORM}. Run make_features.py and baseline_normalization.py first."
    df = pd.read_csv(FEATS_NORM, parse_dates=["window_mid"])

    if TARGET_COL not in df.columns:
        raise SystemExit(f"{TARGET_COL} not found in features_all_normalized.csv")

    all_edges = []

    for subj_id, df_subj in df.groupby("subject_id"):
        print(f"Running PCMCI for subject {subj_id} ...")
        prep = prepare_subject_timeseries(df_subj)
        arr, var_names = prep["array"], prep["var_names"]

        if arr is None or not var_names:
            print(f"  Skipping subject {subj_id}: not enough data or features.")
            continue

        edges_subj = run_pcmci_for_subject(str(subj_id), arr, var_names)
        if not edges_subj.empty:
            all_edges.append(edges_subj)

    if not all_edges:
        print("No causal edges found. Check configuration and data.")
        return

    edges_df = pd.concat(all_edges, ignore_index=True)
    edges_path = OUT_DIR / "causal_edges.csv"
    edges_df.to_csv(edges_path, index=False)
    print(f"Saved per-subject edges to: {edges_path}  (rows={len(edges_df)})")

    # Aggregate: stable edges across subjects
    group_cols = ["source", "lag", "sign"]
    summary = (
        edges_df
        .groupby(group_cols)
        .agg(
            subjects=("subject_id", "nunique"),
            mean_val=("val", "mean"),
            mean_pval=("pval", "mean"),
        )
        .reset_index()
    )

    # total subjects for normalization
    n_subjects = df["subject_id"].nunique()
    summary["subject_frac"] = summary["subjects"] / float(n_subjects)

    summary_path = OUT_DIR / "causal_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved aggregated summary to: {summary_path}")
    print(f"Total subjects: {n_subjects}")


if __name__ == "__main__":
    main()
