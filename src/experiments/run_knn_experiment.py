import hashlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.experiments.knn_common import SEED, add_date_features, make_preprocessor, weight_vector
from src.models.knn import KNN
from src.preprocessing import DataPreprocessor

KNN_PARAMS = dict(n_neighbors=25, metric="manhattan", weights="distance")
# Threshold tuning untuk F1 (Lipton, Elkan & Narayanaswamy, ECML PKDD 2014)
# Ambang optimal untuk F1 umumnya bukan 0.5, sehingga ambang dipilih dari data validasi.
THRESHOLD = 0.48

# Bobot fitur hasil search_knn_weights.py (Mitchell, Machine Learning, 1997, Bab 8.2)
# Fitur tak relevan mendominasi jarak, jadi tiap sumbu fitur diberi bobot yang dipilih lewat cross-validation.
FEATURE_WEIGHTS = {
    # Sangat penting
    "special_requests": 4,
    "days_before_arrival": 4,
    "arrival_ord": 4,
    "lead_to_stay_ratio": 4,
    "is_zero_rate": 4,
    # Penting
    "booking_channel": 2,
    "nightly_rate_idr": 2,
    "total_spend_idr": 2,
    "booking_ord": 2,
    "needs_parking": 2,
    # Dikurangi
    "total_nights": 0.5,
    "is_weekend_arrival": 0.5,
    "room_category": 0.5,
    "num_adults": 0.25,
    "has_children": 0.25,
    "meal_package_is_missing": 0.25,
    "room_category_is_missing": 0.25,
    "days_before_arrival_is_missing": 0.25,
    # Diabaikan
    "num_children": 0,
    "weekend_nights": 0,
    "weekday_nights": 0,
    "arrival_year": 0,
    "is_returning_guest": 0,
    "num_adults_is_missing": 0,
    "num_children_is_missing": 0,
    "needs_parking_is_missing": 0,
    "nightly_rate_idr_is_missing": 0,
    "special_requests_is_missing": 0,
    "total_missing_cols": 0,
    "month_sin": 0,
    "arrival_dayofweek": 0,
    "booking_group_size": 0,
}


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def preprocess(train_part, other_part):
    prep = make_preprocessor()
    X_train = prep.fit_transform(add_date_features(train_part), train_part["is_canceled"].to_numpy())
    X_other = prep.transform(add_date_features(other_part))
    return X_train, X_other, prep.final_feature_names_


def train_final_and_submit(train_df, test_df):
    train_clean = DataPreprocessor.clean_training_dataframe(train_df)
    X_train, X_test, columns = preprocess(train_clean, test_df)

    model = KNN(**KNN_PARAMS, feature_weights=weight_vector(columns, FEATURE_WEIGHTS))
    model.fit(X_train, train_clean["is_canceled"].to_numpy())
    model.save(str(ROOT_DIR / "output" / "models" / "knn_kaggle.pkl"))

    submission = pd.DataFrame({
        "id": test_df["id"],
        "is_canceled": model.predict(X_test, threshold=THRESHOLD).astype(int),
    })
    sub_path = ROOT_DIR / "output" / "submissions" / "submission_knn.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False)
    return sub_path


def cross_validate(train_df):
    y = train_df["is_canceled"].to_numpy()
    names = ["Scratch (tanpa bobot)", "Scratch (berbobot)", "sklearn (berbobot)"]
    pred = {n: np.zeros(len(y), dtype=int) for n in names}
    seconds = {n: 0.0 for n in names}

    for tr_idx, va_idx in StratifiedKFold(5, shuffle=True, random_state=SEED).split(train_df, y):
        fold_train = DataPreprocessor.clean_training_dataframe(train_df.iloc[tr_idx])
        X_tr, X_va, columns = preprocess(fold_train, train_df.iloc[va_idx])
        y_tr = fold_train["is_canceled"].to_numpy()
        w = weight_vector(columns, FEATURE_WEIGHTS)

        models = {
            names[0]: KNN(**KNN_PARAMS),
            names[1]: KNN(**KNN_PARAMS, feature_weights=w),
            names[2]: KNeighborsClassifier(25, p=1, weights="distance", algorithm="brute"),
        }
        for name, model in models.items():
            start = time.time()
            if name.startswith("sklearn"):
                proba = model.fit(X_tr * w, y_tr).predict_proba(X_va * w)[:, 1]
                pred[name][va_idx] = proba >= THRESHOLD
            else:
                pred[name][va_idx] = model.fit(X_tr, y_tr).predict(X_va, threshold=THRESHOLD)
            seconds[name] += time.time() - start

    return pd.DataFrame([{
        "Model": n,
        "Macro F1": f1_score(y, pred[n], average="macro"),
        "Accuracy": accuracy_score(y, pred[n]),
        "Waktu (detik)": seconds[n],
    } for n in names])


def main():
    seed_everything()
    train_df = pd.read_csv(ROOT_DIR / "data" / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(ROOT_DIR / "data" / "test.csv", dtype={"id": str})

    sub_path = train_final_and_submit(train_df, test_df)
    print(f"Submisi: {sub_path.relative_to(ROOT_DIR)}")
    print(f"md5    : {hashlib.md5(sub_path.read_bytes()).hexdigest()}")

    if "--cv" in sys.argv:
        print(cross_validate(train_df).to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
