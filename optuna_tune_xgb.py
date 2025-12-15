# optuna_tune_xgb.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, f1_score
from xgboost import XGBRegressor, XGBClassifier

OUT = Path("outputs")
FEAT = OUT / "features_all_normalized.csv"  # prefer normalized; fallback to raw
FEAT_FALLBACK = OUT / "features_all.csv"

TARGET = "mentalFatigueScore"
CLF_BINS = (32.0, 62.0)
N_SPLITS = 5
N_TRIALS = 40  # adjust for more thorough search

def load_features():
    path = FEAT if FEAT.exists() else FEAT_FALLBACK
    df = pd.read_csv(path, parse_dates=["window_mid"])
    df = df.sort_values(["subject_id","session_id","window_mid"]).reset_index(drop=True)
    return df

def get_Xy(df, target=TARGET):
    drop = {"subject_id","session_id","window_start","window_end","window_mid","task_block",
            "mentalFatigueScore","physicalFatigueScore"}
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf,-np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    y = df[target].astype(float).values
    groups = df["subject_id"].astype(str).values
    return X, y, groups

def y_to_classes(y, bins=CLF_BINS):
    return pd.cut(y, bins=[-1e18, bins[0], bins[1], 1e18],labels=[0, 1, 2], include_lowest=True).astype(int)


def tune_regression():
    df = load_features()
    df = df[df[TARGET].notna()].copy()
    X, y, groups = get_Xy(df)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        gkf = GroupKFold(n_splits=N_SPLITS)
        maes = []
        for tr, te in gkf.split(X, y, groups):
            model = XGBRegressor(**params)
            model.fit(X.iloc[tr], y[tr])
            p = model.predict(X.iloc[te])
            maes.append(mean_absolute_error(y[te], p))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize", study_name="xgb_reg_tuning")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value

def tune_classification():
    df = load_features()
    df = df[df[TARGET].notna()].copy()
    X, y, groups = get_Xy(df)
    y_c = y_to_classes(y)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "objective": "multi:softprob",
            "num_class": 3,
        }
        gkf = GroupKFold(n_splits=N_SPLITS)
        f1s = []
        for tr, te in gkf.split(X, y_c, groups):
            model = XGBClassifier(**params)
            model.fit(X.iloc[tr], y_c[tr])
            proba = model.predict_proba(X.iloc[te])
            pred = proba.argmax(axis=1)
            f1s.append(f1_score(y_c[te], pred, average="macro"))
        return float(np.mean(f1s))

    study = optuna.create_study(direction="maximize", study_name="xgb_clf_tuning")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    best_reg, score_reg = tune_regression()
    best_clf, score_clf = tune_classification()

    with open(OUT / "xgb_tuned_params_reg.json", "w") as f:
        json.dump({"best_params": best_reg, "cv_mae": score_reg}, f, indent=2)
    with open(OUT / "xgb_tuned_params_clf.json", "w") as f:
        json.dump({"best_params": best_clf, "cv_f1_macro": score_clf}, f, indent=2)

    print("Saved tuned params to outputs/")

if __name__ == "__main__":
    main()
