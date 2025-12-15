# predict_alerts.py
from pathlib import Path
import pandas as pd
import yaml

from fatigue_inference import load_engine, predict_from_features


def main():
    cfg = yaml.safe_load(open("config.yaml", "r"))
    out_dir = Path(cfg["out_dir"])

    # Prefer normalized features if present
    feats_norm = out_dir / "features_all_normalized.csv"
    feats_raw = out_dir / "features_all.csv"
    feats_path = feats_norm if feats_norm.exists() else feats_raw

    if not feats_path.exists():
        raise SystemExit("No features found. Run make_features.py (and baseline_normalization.py) first.")

    df = pd.read_csv(feats_path, parse_dates=["window_mid"])
    if "subject_id" not in df.columns or "session_id" not in df.columns:
        raise SystemExit("features file must contain subject_id and session_id columns.")

    # Load engine (models + thresholds + smoothing settings)
    engine = load_engine("config.yaml")

    # Run inference
    df_pred = predict_from_features(df, engine)

    # Select output columns (keep task_block & labels if present)
    cols = ["subject_id", "session_id", "window_mid"]
    for opt in ["task_block", "mentalFatigueScore", "physicalFatigueScore"]:
        if opt in df_pred.columns:
            cols.append(opt)

    cols += ["pred_score_raw", "pred_score_ema", "risk_band", "alert_high"]

    out_df = df_pred[cols]

    out_path = out_dir / "regression_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
