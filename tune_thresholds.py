# tune_thresholds.py
"""
Grid-search thresholds on pred_score_ema to define Low/Medium/High fatigue.

Inputs:
    outputs/regression_predictions.csv
    outputs/features_all_normalized.csv (preferred) or outputs/features_all.csv

Outputs:
    outputs/thresholds_tuned.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report

OUT_DIR = Path("outputs")
TARGET_COL = "mentalFatigueScore"


def load_data(out_dir: Path = OUT_DIR, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Merge predictions with ground-truth labels."""

    # --- 1) Predictions (must contain pred_score_ema) ---
    preds_path = out_dir / "regression_predictions.csv"
    if not preds_path.exists():
        raise SystemExit(f"Missing {preds_path}. Run predict_alerts.py first.")

    preds = pd.read_csv(preds_path, parse_dates=["window_mid"])

    if "pred_score_ema" not in preds.columns:
        raise SystemExit(
            "pred_score_ema not found in regression_predictions.csv. "
            "Make sure predict_alerts.py was run with EMA enabled."
        )

    # --- 2) Features for labels: prefer normalized, fallback to raw ---
    feat_candidates = [
        out_dir / "features_all_normalized.csv",
        out_dir / "features_all.csv",
    ]

    feats = None
    for p in feat_candidates:
        if p.exists():
            tmp = pd.read_csv(p, parse_dates=["window_mid"])
            if target_col in tmp.columns:
                feats = tmp
                break

    if feats is None:
        raise SystemExit(
            f"Could not find column '{target_col}' in "
            "features_all_normalized.csv or features_all.csv in outputs/."
        )

    # Keep only ID keys + target (+ task_block if available)
    keep = ["subject_id", "session_id", "window_mid", target_col]
    if "task_block" in feats.columns:
        keep.append("task_block")

    feats = feats[keep]

    # --- 3) Merge predictions + labels ---
    df = preds.merge(
        feats,
        on=["subject_id", "session_id", "window_mid"],
        how="inner",
        suffixes=("", "_feat"),
    )

    # --- 4) Drop rows without target or pred_score_ema ---
    df = df.dropna(subset=[target_col, "pred_score_ema"])

    if df.empty:
        raise SystemExit("After merging and dropping NaNs, no rows remain for threshold tuning.")

    return df


def score_with_thresholds(y_true_cls, scores, m_thr, h_thr):
    """
    Map regression scores to 3 classes using thresholds:
        [ -inf, m_thr )  -> 0 (Low)
        [ m_thr, h_thr ) -> 1 (Medium)
        [ h_thr, +inf )  -> 2 (High)
    Return macro F1 and predicted labels.
    """
    # np.digitize bins: [m_thr, h_thr] => 0,1,2
    y_pred = np.digitize(scores, bins=[m_thr, h_thr])
    f1 = f1_score(y_true_cls, y_pred, average="macro")
    return f1, y_pred


def main():
    df = load_data()

    # Ground-truth continuous score
    y_true = df[TARGET_COL].astype(float).values
    scores = df["pred_score_ema"].astype(float).values

    # Search space for thresholds (tune as needed)
    low_limit = 10
    high_limit = 90

    best = None  # (f1, m_thr, h_thr, y_true_cls, y_pred_best)

    for m_thr in range(low_limit, 51):      # medium threshold ~ [10..50]
        for h_thr in range(m_thr + 1, high_limit + 1):  # high thr > medium thr
            # Ground truth class labels using the SAME thresholds
            y_true_cls = np.digitize(y_true, bins=[m_thr, h_thr])
            f1, y_pred = score_with_thresholds(y_true_cls, scores, m_thr, h_thr)

            if best is None or f1 > best[0]:
                best = (f1, m_thr, h_thr, y_true_cls, y_pred)

    if best is None:
        raise SystemExit("Threshold grid search failed to find any valid configuration.")

    f1_best, m_best, h_best, y_true_best, y_pred_best = best

    # Final metrics & confusion matrix using best thresholds
    cm = confusion_matrix(y_true_best, y_pred_best, labels=[0, 1, 2])
    rep = classification_report(
        y_true_best,
        y_pred_best,
        labels=[0, 1, 2],
        target_names=["Low", "Medium", "High"],
        output_dict=True,
        zero_division=0,
    )

    result = {
        "best_thresholds": {
            "medium_threshold": float(m_best),
            "high_threshold": float(h_best),
            "f1_macro": float(f1_best),
        },
        "macro_f1": float(f1_best),
        "confusion_matrix_labels": ["Low", "Medium", "High"],
        "confusion_matrix": cm.tolist(),
        "classification_report": rep,
    }

    out_path = OUT_DIR / "thresholds_tuned.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved:", out_path)
    print("Best thresholds:", result["best_thresholds"])


if __name__ == "__main__":
    main()
