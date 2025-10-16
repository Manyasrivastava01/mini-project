import yaml
from pathlib import Path
import numpy as np
import pandas as pd

from utils import (
    load_csv, resample_1hz, window_indices, slope_per_min,
    hrv_time_domain, count_eda_peaks, find_session_dirs, map_labels_by_markers,
    infer_task_blocks, assign_blocks_to_windows  # 👈 add these two lines
)



def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_session_features(
    session_path: Path,
    window_seconds: int,
    stride_seconds: int,
    resample_hz: int,
    label_target: str,
    use_markers_alignment: bool,
    fallback_even_segments: bool
) -> pd.DataFrame:

    # --- Read relevant files (headband + wrist only) ---
    eeg_files = {
        "delta": session_path / "forehead_eeg_delta_abs.csv",
        "theta": session_path / "forehead_eeg_theta_abs.csv",
        "alpha": session_path / "forehead_eeg_alpha_abs.csv",
        "beta":  session_path / "forehead_eeg_beta_abs.csv",
        "gamma": session_path / "forehead_eeg_gamma_abs.csv",
    }
    eeg_bands = {}
    for band, path in eeg_files.items():
        df = load_csv(path)
        if not df.empty:
            chans = [c for c in ["TP9", "AF7", "AF8", "TP10"] if c in df.columns]
            if len(chans) == 0:
                continue
            # make sure channels are numeric (avoid string concatenation behavior in mean)
            df = _coerce_numeric(df, chans)
            df[f"eeg_{band}_mean"] = df[chans].mean(axis=1)
            eeg_bands[band] = df[["timestamp", f"eeg_{band}_mean"]]

    fore_acc = load_csv(session_path / "forehead_acc.csv")
    fore_gyro = load_csv(session_path / "forehead_gyro.csv")

    wrist_bvp = load_csv(session_path / "wrist_bvp.csv")
    wrist_hr  = load_csv(session_path / "wrist_hr.csv")
    wrist_ibi = load_csv(session_path / "wrist_ibi.csv")
    wrist_eda = load_csv(session_path / "wrist_eda.csv")
    wrist_tmp = load_csv(session_path / "wrist_skin_temperature.csv")
    wrist_acc = load_csv(session_path / "wrist_acc.csv")

    # Ensure numeric where needed BEFORE analysis
    fore_acc = _coerce_numeric(fore_acc, ["ax", "ay", "az"])
    fore_gyro = _coerce_numeric(fore_gyro, ["gx", "gy", "gz"])
    wrist_bvp = _coerce_numeric(wrist_bvp, ["bvp"])
    wrist_hr = _coerce_numeric(wrist_hr, ["hr"])
    wrist_eda = _coerce_numeric(wrist_eda, ["eda"])
    wrist_tmp = _coerce_numeric(wrist_tmp, ["temp"])
    wrist_acc = _coerce_numeric(wrist_acc, ["ax", "ay", "az"])
    wrist_ibi = _coerce_numeric(wrist_ibi, ["duration"])  # HRV uses this directly

    exp_fatigue = load_csv(session_path / "exp_fatigue.csv")
    if not exp_fatigue.empty:
        for col in ["mentalFatigueScore", "physicalFatigueScore"]:
            if col in exp_fatigue.columns:
                exp_fatigue[col] = pd.to_numeric(exp_fatigue[col], errors="coerce")

    exp_markers = load_csv(session_path / "exp_markers.csv", rename={"utcTime": "timestamp"})

    # --- Detect and attach task blocks from markers ---
    blocks = infer_task_blocks(exp_markers)
    if not blocks.empty:
        print(f"Detected {len(blocks)} task blocks in {session_path.name}")

    # --- Build 1Hz base ---
    pieces = []
    for band, df in eeg_bands.items():
        pieces.append(resample_1hz(df, "timestamp", [f"eeg_{band}_mean"], agg="median"))

    if not fore_acc.empty and {"ax", "ay", "az"}.issubset(fore_acc.columns):
        a = resample_1hz(fore_acc, "timestamp", ["ax", "ay", "az"], agg="mean").rename(
            columns={"ax": "fore_ax", "ay": "fore_ay", "az": "fore_az"}
        )
        pieces.append(a)

    if not fore_gyro.empty and {"gx", "gy", "gz"}.issubset(fore_gyro.columns):
        g = resample_1hz(fore_gyro, "timestamp", ["gx", "gy", "gz"], agg="mean").rename(
            columns={"gx": "fore_gx", "gy": "fore_gy", "gz": "fore_gz"}
        )
        pieces.append(g)

    if not wrist_bvp.empty and "bvp" in wrist_bvp.columns:
        pieces.append(resample_1hz(wrist_bvp, "timestamp", ["bvp"], agg="mean"))
    if not wrist_hr.empty and "hr" in wrist_hr.columns:
        pieces.append(resample_1hz(wrist_hr, "timestamp", ["hr"], agg="mean"))
    if not wrist_eda.empty and "eda" in wrist_eda.columns:
        pieces.append(resample_1hz(wrist_eda, "timestamp", ["eda"], agg="mean"))
    if not wrist_tmp.empty and "temp" in wrist_tmp.columns:
        pieces.append(resample_1hz(wrist_tmp, "timestamp", ["temp"], agg="mean"))
    if not wrist_acc.empty and {"ax", "ay", "az"}.issubset(wrist_acc.columns):
        w = resample_1hz(wrist_acc, "timestamp", ["ax", "ay", "az"], agg="mean").rename(
            columns={"ax": "wr_ax", "ay": "wr_ay", "az": "wr_az"}
        )
        pieces.append(w)

    if not pieces:
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

    # Derived EEG ratios
    if "eeg_theta_mean" in base.columns and "eeg_alpha_mean" in base.columns:
        base["eeg_theta_alpha_ratio"] = base["eeg_theta_mean"] / (base["eeg_alpha_mean"] + 1e-6)
    if "eeg_beta_mean" in base.columns and "eeg_alpha_mean" in base.columns:
        base["eeg_beta_alpha_ratio"] = base["eeg_beta_mean"] / (base["eeg_alpha_mean"] + 1e-6)
    if {"eeg_theta_mean", "eeg_delta_mean", "eeg_alpha_mean", "eeg_beta_mean"}.issubset(base.columns):
        base["eeg_drowsiness_index"] = (base["eeg_theta_mean"] + base["eeg_delta_mean"]) / (
            base["eeg_alpha_mean"] + base["eeg_beta_mean"] + 1e-6
        )

    if {"fore_ax", "fore_ay", "fore_az"}.issubset(base.columns):
        base["fore_acc_rms"] = np.sqrt((base["fore_ax"] ** 2 + base["fore_ay"] ** 2 + base["fore_az"] ** 2))
    if {"wr_ax", "wr_ay", "wr_az"}.issubset(base.columns):
        base["wr_acc_rms"] = np.sqrt((base["wr_ax"] ** 2 + base["wr_ay"] ** 2 + base["wr_az"] ** 2))

    # Windowing
    wins = window_indices(base["timestamp"].dropna(), win_secs=window_seconds, stride_secs=stride_seconds)
    rows = []
    for (ws, we) in wins:
        wdf = base[(base["timestamp"] >= ws) & (base["timestamp"] <= we)]
        if wdf.shape[0] < 5:
            continue
        row = {"window_start": ws, "window_end": we, "window_mid": ws + (we - ws) / 2}

        # EEG stats
        for col in [c for c in base.columns if c.startswith("eeg_")]:
            s = wdf[col]
            row[f"{col}_mean"] = float(pd.to_numeric(s, errors="coerce").mean())
            row[f"{col}_var"] = float(pd.to_numeric(s, errors="coerce").var())
            row[f"{col}_slope_per_min"] = float(slope_per_min(pd.to_numeric(s, errors="coerce")))

        # HR
        if "hr" in base.columns:
            s = pd.to_numeric(wdf["hr"], errors="coerce")
            row["hr_mean"] = float(s.mean())
            row["hr_var"] = float(s.var())

        # HRV from IBI (use original timestamps to avoid resampling bias)
        if not wrist_ibi.empty and {"timestamp", "duration"}.issubset(wrist_ibi.columns):
            ibi_w = wrist_ibi[(wrist_ibi["timestamp"] >= ws) & (wrist_ibi["timestamp"] <= we)]
            sdnn, rmssd, pnn50 = hrv_time_domain(pd.to_numeric(ibi_w["duration"], errors="coerce").values if ibi_w.shape[0] else [])
            row["hrv_sdnn"] = float(sdnn) if not np.isnan(sdnn) else np.nan
            row["hrv_rmssd"] = float(rmssd) if not np.isnan(rmssd) else np.nan
            row["hrv_pnn50"] = float(pnn50) if not np.isnan(pnn50) else np.nan

        # EDA
        if "eda" in base.columns:
            s = pd.to_numeric(wdf["eda"], errors="coerce")
            row["eda_mean"] = float(s.mean())
            n, amp = count_eda_peaks(s.values)
            row["eda_peak_count"] = int(n)
            row["eda_peak_amp_mean"] = float(amp) if not np.isnan(amp) else np.nan
            row["eda_slope_per_min"] = float(slope_per_min(s))

        # Temp
        if "temp" in base.columns:
            s = pd.to_numeric(wdf["temp"], errors="coerce")
            row["temp_mean"] = float(s.mean())
            row["temp_slope_per_min"] = float(slope_per_min(s))

        # Activity
        if "wr_acc_rms" in base.columns:
            row["wr_acc_rms_mean"] = float(pd.to_numeric(wdf["wr_acc_rms"], errors="coerce").mean())
        if "fore_acc_rms" in base.columns:
            row["fore_acc_rms_mean"] = float(pd.to_numeric(wdf["fore_acc_rms"], errors="coerce").mean())

        rows.append(row)

    feats = pd.DataFrame(rows)
    if feats.empty:
        return feats

    feats["task_block"] = assign_blocks_to_windows(feats, blocks)

    # Label mapping
    if use_markers_alignment:
        feats[label_target] = map_labels_by_markers(
            feats, exp_fatigue, exp_markers, label_col=label_target, fallback_even_segments=fallback_even_segments
        )
    else:
        feats[label_target] = map_labels_by_markers(
            feats, exp_fatigue, pd.DataFrame(), label_col=label_target, fallback_even_segments=True
        )

    # Optional: also map physical score if present
    if not exp_fatigue.empty and "physicalFatigueScore" in exp_fatigue.columns:
        feats["physicalFatigueScore"] = map_labels_by_markers(
            feats, exp_fatigue, exp_markers, label_col="physicalFatigueScore", fallback_even_segments=fallback_even_segments
        )

    return feats


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
        print("No features produced. Check data_root structure.")
        return

    features_all = pd.concat(all_rows, ignore_index=True)
    (out_dir / "features_all.csv").parent.mkdir(parents=True, exist_ok=True)
    features_all.to_csv(out_dir / "features_all.csv", index=False)
    print(f"Saved: {out_dir / 'features_all.csv'}   rows={features_all.shape[0]}  cols={features_all.shape[1]}")


if __name__ == "__main__":
    main()
