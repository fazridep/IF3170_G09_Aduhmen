
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.experiments.knn_common import SEED, add_date_features, feature_group, make_preprocessor
from src.preprocessing import DataPreprocessor

CANDIDATE_WEIGHTS = [0, 0.25, 0.5, 1, 2, 3, 4]
MAX_ROUNDS = 3
THRESHOLDS = np.arange(0.36, 0.62, 0.02)


def prepare_folds(train_df):
    y = train_df["is_canceled"].to_numpy()
    folds = []
    for tr_idx, va_idx in StratifiedKFold(5, shuffle=True, random_state=SEED).split(train_df, y):
        fold_train = DataPreprocessor.clean_training_dataframe(train_df.iloc[tr_idx])
        prep = make_preprocessor()
        X_tr = prep.fit_transform(add_date_features(fold_train), fold_train["is_canceled"].to_numpy())
        X_va = prep.transform(add_date_features(train_df.iloc[va_idx]))
        folds.append((X_tr, fold_train["is_canceled"].to_numpy(), X_va, va_idx))
    return folds, y, prep.final_feature_names_


def cv_macro_f1(folds, y, col_weights):
    proba = np.zeros(len(y))
    for X_tr, y_tr, X_va, va_idx in folds:
        knn = KNeighborsClassifier(25, weights="distance", p=1, n_jobs=-1)
        knn.fit(X_tr * col_weights, y_tr)
        proba[va_idx] = knn.predict_proba(X_va * col_weights)[:, 1]
    return max(f1_score(y, proba >= t, average="macro") for t in THRESHOLDS)


def main():
    train_df = pd.read_csv(ROOT_DIR / "data" / "train.csv", dtype={"id": str})
    folds, y, columns = prepare_folds(train_df)

    groups = list(dict.fromkeys(feature_group(c) for c in columns))
    weights = {g: 1.0 for g in groups}

    def score(w):
        return cv_macro_f1(folds, y, np.array([w[feature_group(c)] for c in columns]))

    best = score(weights)
    print(f"awal: {best:.4f}", flush=True)

    for rnd in range(1, MAX_ROUNDS + 1):
        improved = False
        for g in groups:
            for v in CANDIDATE_WEIGHTS:
                trial = {**weights, g: v}
                s = score(trial) if v != weights[g] else best
                if s > best + 1e-4:
                    best, weights, improved = s, trial, True
                    print(f"putaran {rnd}: {g} = {v} -> {best:.4f}", flush=True)
        if not improved:
            break

    print("\nBobot akhir (selain 1):")
    print(json.dumps({g: w for g, w in weights.items() if w != 1.0}, indent=4))


if __name__ == "__main__":
    main()
