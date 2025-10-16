import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ---------------- Time helpers ----------------
def to_datetime_ms(series: pd.Series) -> pd.Series:
    """
    Robust timestamp conversion:
    1) Try numeric ms epoch
    2) Wherever that fails, try generic parser
    Never forces int casting (avoids IntCastingNaNError)
    """
    s_num = pd.to_numeric(series, errors="coerce")
    dt1 = pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")
    dt2 = pd.to_datetime(series, utc=True, errors="coerce")
    return dt1.where(dt1.notna(), dt2)


def resample_1hz(df: pd.DataFrame, time_col: str, value_cols: List[str], agg="mean") -> pd.DataFrame:
    """
    Resample to 1 Hz using lowercase 's'.
    Coerce value columns to numeric to avoid object-dtype aggregation failures.
    Assumes time_col is already datetime64[ns, UTC].
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    # Ensure datetime index base
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = to_datetime_ms(df[time_col])

    # Force numeric for all requested value columns
    for c in value_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df.set_index(time_col)[value_cols].resample("1s").agg(agg)
    return out.reset_index().rename(columns={time_col: "timestamp"})


def window_indices(ts: pd.Series, win_secs: int, stride_secs: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if len(ts) == 0:
        return []
    tmin, tmax = ts.min(), ts.max()
    if pd.isna(tmin) or pd.isna(tmax):
        return []
    starts = pd.date_range(tmin, tmax - pd.Timedelta(seconds=win_secs - 1), freq=f"{stride_secs}s")
    return [(s, s + pd.Timedelta(seconds=win_secs - 1)) for s in starts]


# ---------------- Feature helpers ----------------
def slope_per_min(series: pd.Series) -> float:
    s = series.dropna()
    if s.shape[0] < 2:
        return np.nan
    x = np.arange(s.shape[0], dtype=float)
    y = s.values.astype(float)
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return m * 60.0


def hrv_time_domain(ibi_ms: np.ndarray) -> Tuple[float, float, float]:
    ibi = np.array(ibi_ms, dtype=float)
    ibi = ibi[0:len(ibi)][~np.isnan(ibi)]
    if ibi.shape[0] < 3:
        return np.nan, np.nan, np.nan
    sdnn = np.std(ibi, ddof=1)
    diff = np.diff(ibi)
    rmssd = np.sqrt(np.mean(diff ** 2))
    pnn50 = np.mean(np.abs(diff) > 50.0) * 100.0
    return float(sdnn), float(rmssd), float(pnn50)


def count_eda_peaks(signal: np.ndarray, prominence=0.01, distance=2) -> Tuple[int, float]:
    s = np.array(signal, dtype=float)
    s = s[~np.isnan(s)]
    if s.shape[0] < 5:
        return 0, np.nan
    peaks, props = find_peaks(s, prominence=prominence, distance=distance)
    n = int(peaks.shape[0])
    amp = float(np.mean(props["prominences"])) if n > 0 else np.nan
    return n, amp


# ---------------- IO helpers ----------------
def load_csv(path: Path, rename: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Read CSV and convert any 'timestamp' column to UTC datetime (robustly).
    If 'rename' is provided (e.g., {'utcTime':'timestamp'}), we apply it BEFORE conversion.
    """
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Apply renames FIRST so we can always normalize on 'timestamp'
    if rename:
        df = df.rename(columns=rename)

    # Normalize timestamp if present (robust to floats/strings/NaNs)
    if "timestamp" in df.columns:
        df["timestamp"] = to_datetime_ms(df["timestamp"])

    return df


def find_session_dirs(root: Path):
    """
    Returns list of (subject_id, session_id, session_path) for paths matching root/NN/MM
    """
    out = []
    for subj in sorted(root.glob("[0-9][0-9]")):
        for sess in sorted(subj.glob("[0-9][0-9]")):
            out.append((subj.name, sess.name, sess))
    return out


# ---------------- Label alignment ----------------
def map_labels_by_markers(
    features: pd.DataFrame,
    exp_fatigue: pd.DataFrame,
    exp_markers: pd.DataFrame,
    label_col: str,
    fallback_even_segments: bool = True
) -> pd.Series:
    """
    Assign a session-level survey score to each feature window.

    Priority:
      1) Use exp_markers rows with 'submit_survey' (or any 'survey') to split timeline
      2) Fallback: split windows evenly by survey order

    Returns pd.Series aligned to features.index.
    """
    n = features.shape[0]
    if n == 0 or exp_fatigue.empty or label_col not in exp_fatigue.columns:
        return pd.Series([np.nan] * n, index=features.index)

    # Prepare survey times from markers (robust to different schemas)
    survey_times = pd.Series(dtype="datetime64[ns, UTC]")
    if not exp_markers.empty:
        m = exp_markers.copy()

        # Ensure we have a datetime 'timestamp' in markers
        if "timestamp" not in m.columns and "utcTime" in m.columns:
            m["timestamp"] = to_datetime_ms(m["utcTime"])
        elif "timestamp" in m.columns and not pd.api.types.is_datetime64_any_dtype(m["timestamp"]):
            m["timestamp"] = to_datetime_ms(m["timestamp"])

        if "timestamp" in m.columns:
            mask_submit = m["eventMarker"].astype(str).str.contains("submit_survey", case=False, na=False)
            mask_survey = m["eventMarker"].astype(str).str.contains("survey", case=False, na=False)
            times = m.loc[mask_submit | mask_survey, "timestamp"]
            survey_times = times.dropna().sort_values().reset_index(drop=True)

    labels = exp_fatigue.copy().reset_index(drop=True)

    y = pd.Series([np.nan] * n, index=features.index)

    if not survey_times.empty:
        k = min(len(survey_times), len(labels))
        mids = features["window_mid"]

        # Build cut points midway between survey times
        cuts = []
        for i in range(k - 1):
            cuts.append(survey_times.iloc[i] + (survey_times.iloc[i + 1] - survey_times.iloc[i]) / 2)

        segments = []
        prev = pd.Timestamp.min.tz_localize("UTC")
        for c in cuts:
            segments.append((prev, c))
            prev = c
        segments.append((prev, pd.Timestamp.max.tz_localize("UTC")))

        # Assign labels in order across segments
        for i, (a, b) in enumerate(segments[:k]):
            mask = (mids >= a) & (mids < b)
            y.loc[mask] = labels.iloc[i][label_col]

        # carry last label forward if needed — use .ffill() (no deprecated method arg)
        y = y.ffill()
        return y

    # Fallback: even segmentation by index order
    if fallback_even_segments:
        m = len(labels)
        bounds = np.linspace(0, n, m + 1, dtype=int)
        for i in range(m):
            y.iloc[bounds[i]:bounds[i + 1]] = labels.iloc[i][label_col]
        return y

    # nothing to do
    return y

def infer_task_blocks(exp_markers: pd.DataFrame, task_marker_keys=None) -> pd.DataFrame:
    """
    Build coarse task blocks from exp_markers.
    Returns a DataFrame with columns: ['start','end','task']
    Very robust: looks for any marker text containing provided keys and builds segments.
    """
    if exp_markers.empty or "eventMarker" not in exp_markers.columns:
        return pd.DataFrame(columns=["start","end","task"])

    m = exp_markers.copy()
    # ensure timestamp
    if "timestamp" not in m.columns and "utcTime" in m.columns:
        m["timestamp"] = to_datetime_ms(m["utcTime"])
    elif "timestamp" in m.columns and not pd.api.types.is_datetime64_any_dtype(m["timestamp"]):
        m["timestamp"] = to_datetime_ms(m["timestamp"])

    m = m.dropna(subset=["timestamp"]).sort_values("timestamp")
    if m.empty:
        return pd.DataFrame(columns=["start","end","task"])

    if not task_marker_keys:
        task_marker_keys = ["crt","nback","task","switch","baseline","rest"]

    rows = []
    last_time = None
    last_task = None

    for _, r in m.iterrows():
        txt = str(r.get("eventMarker", "")).lower()
        t = r["timestamp"]
        # find first matching key in text
        hit = None
        for key in task_marker_keys:
            if key in txt:
                hit = key
                break

        if hit is not None:
            if last_task is not None and last_time is not None:
                rows.append({"start": last_time, "end": t, "task": last_task})
            last_time = t
            last_task = hit

    # close final block if we have a dangling segment
    if last_task is not None and last_time is not None:
        rows.append({"start": last_time, "end": last_time + pd.Timedelta(minutes=9999), "task": last_task})

    return pd.DataFrame(rows)


def assign_blocks_to_windows(features: pd.DataFrame, blocks: pd.DataFrame) -> pd.Series:
    """
    Given feature windows (window_mid) and task blocks (start,end,task),
    returns a Series 'task_block' aligned to features.index.
    """
    out = pd.Series([np.nan]*len(features), index=features.index, dtype="object")
    if features.empty or blocks.empty:
        return out

    mids = features["window_mid"]
    for _, b in blocks.iterrows():
        mask = (mids >= b["start"]) & (mids < b["end"])
        out.loc[mask] = b["task"]
    return out.astype("string")



