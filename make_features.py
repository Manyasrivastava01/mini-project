# make_features.py
"""
Build window-level features from the FatigueSet dataset
using only headband (EEG) + wrist sensors.

Outputs:
    outputs/features_all.csv

Relies on:
    - config.yaml  (for data_root, out_dir, window size, etc.)
    - utils.py     (load_csv, resample_1hz, window_indices, etc.)
"""

import yaml
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from utils import (
    load_csv,
    resample_1hz,
    window_indices,
    slope_per_min,
    hrv_time_domain,
    count_eda_peaks,
    find_session_dirs,
    map_labels_by_markers,
    infer_task_blocks,
    assign_blocks_to_windows,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce given columns to numeric (float), in-place-safe style."""
    if df.empty:
        return df
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------
# Core feature builder for one session
# ---------------------------------------------------------------------


def build_session_features(
    session_path: Path,
    window_seconds: int,
    stride_seconds: int,
    resample_hz: int,
    label_target: str,
    use_markers_alignment: bool,
    fallback_even_segments: bool,
) -> pd.DataFrame:
    """
    Build window-level features for a single session directory.
    Uses:
      - Headband EEG (forehead_eeg_*_abs)
      - Forehead acc/gyro
      - Wrist BVP/HR/IBI/EDA/Temp/Acc
      - Exp fatigue + markers for labels & task blocks
    """

    # --- EEG (headband) ---
    eeg_files = {
        "delta": session_path / "forehead_eeg_delta_abs.csv",
        "theta": session_path / "forehead_eeg_theta_abs.csv",
        "alpha": session_path / "forehead_eeg_alpha_abs.csv",
        "beta": session_path / "forehead_eeg_beta_abs.csv",
        "gamma": session_path / "forehead_eeg_gamma_abs.csv",
    }

    eeg_bands = {}
    for band, path in eeg_files.items():
        df_eeg = load_csv(path)
        if df_eeg.empty:
            continue
        # Muse-style channels if present
        chans = [c for c in ["TP9", "AF7", "AF8", "TP10"] if c in df_eeg.columns]
        if not chans:
            continue
        df_eeg = _coerce_numeric(df_eeg, chans)
        df_eeg[f"eeg_{band}_mean"] = df_eeg[chans].mean(axis=1)
        eeg_bands[band] = df_eeg[["timestamp", f"eeg_{band}_mean"]]

    # --- Forehead IMU ---
    fore_acc = load_csv(session_path / "forehead_acc.csv")
    fore_gyro = load_csv(session_path / "forehead_gyro.csv")

    fore_acc = _coerce_numeric(fore_acc, ["ax", "ay", "az"])
    fore_gyro = _coerce_numeric(fore_gyro, ["gx", "gy", "gz"])

    # --- Wrist signals ---
    wrist_bvp = load_csv(session_path / "wrist_bvp.csv")
    wrist_hr = load_csv(session_path / "wrist_hr.csv")
    wrist_ibi = load_csv(session_path / "wrist_ibi.csv")
    wrist_eda = load_csv(session_path / "wrist_eda.csv")
    wrist_tmp = load_csv(session_path / "wrist_skin_temperature.csv")
    wrist_acc = load_csv(session_path / "wrist_acc.csv")

    wrist_bvp = _coerce_numeric(wrist_bvp, ["bvp"])
    wrist_hr = _coerce_numeric(wrist_hr, ["hr"])
    wrist_eda = _coerce_numeric(wrist_eda, ["eda"])
    wrist_tmp = _coerce_numeric(wrist_tmp, ["temp"])
    wrist_acc = _coerce_numeric(wrist_acc, ["ax", "ay", "az"])
    wrist_ibi = _coerce_numeric(wrist_ibi, ["duration"])

    # --- Labels & markers ---
    exp_fatigue = load_csv(session_path / "exp_fatigue.csv")
    if not exp_fatigue.empty:
        for col in ["mentalFatigueScore", "physicalFatigueScore"]:
            if col in exp_fatigue.columns:
                exp_fatigue[col] = pd.to_numeric(exp_fatigue[col], errors="coerce")

    exp_markers = load_csv(session_path / "exp_markers.csv", rename={"utcTime": "timestamp"})

    # Task block inference (CRT, N-back, baseline, etc.)
    blocks = infer_task_blocks(exp_markers)
    if not blocks.empty:
        print(f"Detected {len(blocks)} task blocks in {session_path.name}")

    # -----------------------------------------------------------------
    # Build a 1 Hz base time series of all relevant channels
    # -----------------------------------------------------------------
    pieces = []

    # EEG band medians at 1 Hz
    for band, df_eeg in eeg_bands.items():
        pieces.append(resample_1hz(df_eeg, "timestamp", [f"eeg_{band}_mean"], agg="median"))

    # Forehead ACC
    if not fore_acc.empty and {"ax", "ay", "az"}.issubset(fore_acc.columns):
        a = (
            resample_1hz(fore_acc, "timestamp", ["ax", "ay", "az"], agg="mean")
            .rename(columns={"ax": "fore_ax", "ay": "fore_ay", "az": "fore_az"})
        )
        pieces.append(a)

    # Forehead Gyro
    if not fore_gyro.empty and {"gx", "gy", "gz"}.issubset(fore_gyro.columns):
        g = (
            resample_1hz(fore_gyro, "timestamp", ["gx", "gy", "gz"], agg="mean")
            .rename(columns={"gx": "fore_gx", "gy": "fore_gy", "gz": "fore_gz"})
        )
        pieces.append(g)

    # Wrist BVP / HR / EDA / Temp / ACC
    if not wrist_bvp.empty and "bvp" in wrist_bvp.columns:
        pieces.append(resample_1hz(wrist_bvp, "timestamp", ["bvp"], agg="mean"))
    if not wrist_hr.empty and "hr" in wrist_hr.columns:
        pieces.append(resample_1hz(wrist_hr, "timestamp", ["hr"], agg="mean"))
    if not wrist_eda.empty and "eda" in wrist_eda.columns:
        pieces.append(resample_1hz(wrist_eda, "timestamp", ["eda"], agg="mean"))
    if not wrist_tmp.empty and "temp" in wrist_tmp.columns:
        pieces.append(resample_1hz(wrist_tmp, "timestamp", ["temp"], agg="mean"))
    if not wrist_acc.empty and {"ax", "ay", "az"}.issubset(wrist_acc.columns):
        w = (
            resample_1hz(wrist_acc, "timestamp", ["ax", "ay", "az"], agg="mean")
            .rename(columns={"ax": "wr_ax", "ay": "wr_ay", "az": "wr_az"})
        )
        pieces.append(w)

    if not pieces:
        # no data for this session
        return pd.DataFrame()

    base = pieces[0].sort_values("timestamp")
    for p in pieces[1:]:
        base = pd.merge_asof(
            base.sort_values("timestamp"),
            p.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=1),
        )

    # Ensure numeric float types for all features
    num_cols = [c for c in base.columns if c != "timestamp"]
    base[num_cols] = base[num_cols].apply(pd.to_numeric, errors="coerce")
    base[num_cols] = base[num_cols].astype(float)

    # -----------------------------------------------------------------
    # Derived features
    # -----------------------------------------------------------------
    # EEG ratios & drowsiness index
    if "eeg_theta_mean" in base.columns and "eeg_alpha_mean" in base.columns:
        base["eeg_theta_alpha_ratio"] = base["eeg_theta_mean"] / (base["eeg_alpha_mean"] + 1e-6)
    if "eeg_beta_mean" in base.columns and "eeg_alpha_mean" in base.columns:
        base["eeg_beta_alpha_ratio"] = base["eeg_beta_mean"] / (base["eeg_alpha_mean"] + 1e-6)
    if {"eeg_theta_mean", "eeg_delta_mean", "eeg_alpha_mean", "eeg_beta_mean"}.issubset(base.columns):
        base["eeg_drowsiness_index"] = (base["eeg_theta_mean"] + base["eeg_delta_mean"]) / (
            base["eeg_alpha_mean"] + base["eeg_beta_mean"] + 1e-6
        )

    # ACC RMS
    if {"fore_ax", "fore_ay", "fore_az"}.issubset(base.columns):
        base["fore_acc_rms"] = np.sqrt(
            base["fore_ax"] ** 2 + base["fore_ay"] ** 2 + base["fore_az"] ** 2
        )
    if {"wr_ax", "wr_ay", "wr_az"}.issubset(base.columns):
        base["wr_acc_rms"] = np.sqrt(
            base["wr_ax"] ** 2 + base["wr_ay"] ** 2 + base["wr_az"] ** 2
        )

    # -----------------------------------------------------------------
    # Windowing (e.g., 60s win, 30s stride)
    # -----------------------------------------------------------------
    wins = window_indices(base["timestamp"].dropna(), win_secs=window_seconds, stride_secs=stride_seconds)

    rows = []
    for (ws, we) in wins:
        wdf = base[(base["timestamp"] >= ws) & (base["timestamp"] <= we)]
        if wdf.shape[0] < 5:
            continue

        row = {
            "window_start": ws,
            "window_end": we,
            "window_mid": ws + (we - ws) / 2,
        }

        # ------- EEG stats (bands + ratios + drowsiness index) -------
        eeg_cols = [c for c in base.columns if c.startswith("eeg_")]
        for col in eeg_cols:
            s_num = pd.to_numeric(wdf[col], errors="coerce").dropna()
            if len(s_num) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_var"] = np.nan
                row[f"{col}_slope_per_min"] = np.nan
            else:
                row[f"{col}_mean"] = float(s_num.mean())
                # Protect against var on single point
                if len(s_num) > 1:
                    with np.errstate(invalid="ignore"):
                        row[f"{col}_var"] = float(s_num.var())
                else:
                    row[f"{col}_var"] = 0.0
                row[f"{col}_slope_per_min"] = float(slope_per_min(s_num))

        # ------- HR mean / var -------
        if "hr" in base.columns:
            s_hr = pd.to_numeric(wdf["hr"], errors="coerce").dropna()
            if len(s_hr) == 0:
                row["hr_mean"] = np.nan
                row["hr_var"] = np.nan
            else:
                row["hr_mean"] = float(s_hr.mean())
                if len(s_hr) > 1:
                    with np.errstate(invalid="ignore"):
                        row["hr_var"] = float(s_hr.var())
                else:
                    row["hr_var"] = 0.0

        # ------- HRV from IBI over *this* window [ws, we] -------
        if not wrist_ibi.empty and {"timestamp", "duration"}.issubset(wrist_ibi.columns):
            # 60-second HRV window aligned with current feature window
            ibi_w = wrist_ibi[(wrist_ibi["timestamp"] >= ws) & (wrist_ibi["timestamp"] <= we)]

            MIN_IBI_COUNT = 10  # slightly relaxed for more coverage
            if ibi_w.shape[0] >= MIN_IBI_COUNT:
                durations = pd.to_numeric(ibi_w["duration"], errors="coerce").values
                sdnn, rmssd, pnn50 = hrv_time_domain(durations)
                row["hrv_sdnn"] = float(sdnn) if not np.isnan(sdnn) else np.nan
                row["hrv_rmssd"] = float(rmssd) if not np.isnan(rmssd) else np.nan
                row["hrv_pnn50"] = float(pnn50) if not np.isnan(pnn50) else np.nan
            else:
                row["hrv_sdnn"] = np.nan
                row["hrv_rmssd"] = np.nan
                row["hrv_pnn50"] = np.nan

        # ------- EDA features -------
        if "eda" in base.columns:
            s_eda = pd.to_numeric(wdf["eda"], errors="coerce").dropna()
            if len(s_eda) == 0:
                row["eda_mean"] = np.nan
                row["eda_peak_count"] = 0
                row["eda_peak_amp_mean"] = np.nan
                row["eda_slope_per_min"] = np.nan
            else:
                row["eda_mean"] = float(s_eda.mean())
                n_peaks, amp = count_eda_peaks(s_eda.values)
                row["eda_peak_count"] = int(n_peaks)
                row["eda_peak_amp_mean"] = float(amp) if not np.isnan(amp) else np.nan
                row["eda_slope_per_min"] = float(slope_per_min(s_eda))

        # ------- Temp features -------
        if "temp" in base.columns:
            s_temp = pd.to_numeric(wdf["temp"], errors="coerce").dropna()
            if len(s_temp) == 0:
                row["temp_mean"] = np.nan
                row["temp_slope_per_min"] = np.nan
            else:
                row["temp_mean"] = float(s_temp.mean())
                row["temp_slope_per_min"] = float(slope_per_min(s_temp))

        # ------- Activity features -------
        if "wr_acc_rms" in base.columns:
            s_wr = pd.to_numeric(wdf["wr_acc_rms"], errors="coerce").dropna()
            row["wr_acc_rms_mean"] = float(s_wr.mean()) if len(s_wr) > 0 else np.nan
        if "fore_acc_rms" in base.columns:
            s_fore = pd.to_numeric(wdf["fore_acc_rms"], errors="coerce").dropna()
            row["fore_acc_rms_mean"] = float(s_fore.mean()) if len(s_fore) > 0 else np.nan

        rows.append(row)

    feats = pd.DataFrame(rows)
    if feats.empty:
        return feats

    # -------- HRV gap-filling so causal discovery can use HRV --------
    hrv_cols = [c for c in ["hrv_sdnn", "hrv_rmssd", "hrv_pnn50"] if c in feats.columns]
    if hrv_cols:
        feats[hrv_cols] = feats[hrv_cols].astype(float)
        feats[hrv_cols] = (
            feats[hrv_cols]
                .interpolate(method="linear", limit=3)  # small gaps only
                .ffill()
                .bfill()
        )
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Attach task_block from markers
    # -----------------------------------------------------------------
    feats["task_block"] = assign_blocks_to_windows(feats, blocks)

    # -----------------------------------------------------------------
    # Label mapping: mentalFatigueScore (primary) + physicalFatigueScore
    # -----------------------------------------------------------------
    if use_markers_alignment:
        feats[label_target] = map_labels_by_markers(
            feats,
            exp_fatigue,
            exp_markers,
            label_col=label_target,
            fallback_even_segments=fallback_even_segments,
        )
    else:
        feats[label_target] = map_labels_by_markers(
            feats,
            exp_fatigue,
            pd.DataFrame(),  # no markers
            label_col=label_target,
            fallback_even_segments=True,
        )

    if not exp_fatigue.empty and "physicalFatigueScore" in exp_fatigue.columns:
        feats["physicalFatigueScore"] = map_labels_by_markers(
            feats,
            exp_fatigue,
            exp_markers,
            label_col="physicalFatigueScore",
            fallback_even_segments=fallback_even_segments,
        )

    return feats


# ---------------------------------------------------------------------
# Main over all subjects/sessions
# ---------------------------------------------------------------------


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    root = Path(cfg["data_root"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = find_session_dirs(root)
    all_rows = []

    for subj_id, sess_id, sess_path in sessions:
        feats = build_session_features(
            session_path=sess_path,
            window_seconds=cfg["window_seconds"],
            stride_seconds=cfg["stride_seconds"],
            resample_hz=cfg["resample_hz"],
            label_target=cfg["label_target"],
            use_markers_alignment=cfg["use_markers_alignment"],
            fallback_even_segments=cfg["fallback_even_segments"],
        )
        if feats.empty:
            continue

        feats.insert(0, "subject_id", subj_id)
        feats.insert(1, "session_id", sess_id)
        all_rows.append(feats)

    if not all_rows:
        print("No features produced. Check data_root structure and CSVs.")
        return

    features_all = pd.concat(all_rows, ignore_index=True)
    (out_dir / "features_all.csv").parent.mkdir(parents=True, exist_ok=True)
    features_all.to_csv(out_dir / "features_all.csv", index=False)

    print(
        f"Saved: {out_dir / 'features_all.csv'}   "
        f"rows={features_all.shape[0]}  cols={features_all.shape[1]}"
    )


if __name__ == "__main__":
    main()
