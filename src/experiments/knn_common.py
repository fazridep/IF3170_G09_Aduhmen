import numpy as np
import pandas as pd

from src.preprocessing import DataPreprocessor

SEED = 42
ONEHOT_PREFIXES = ("meal_package_", "room_category_", "booking_channel_")


def add_date_features(df):
    out = df.copy()
    arrival = pd.to_datetime(
        dict(year=out["arrival_year"], month=out["arrival_month"], day=out["arrival_day"]),
        errors="coerce",
    )
    out["arrival_ord"] = (arrival - pd.Timestamp("2023-01-01")).dt.days
    out["booking_ord"] = out["arrival_ord"] - out["days_before_arrival"]
    return out


def make_preprocessor():
    return DataPreprocessor(
        mode="processed_numeric",
        add_missing_indicators=True,
        add_domain_features=True,
        encoding_method="onehot",
        scaling_method="standard",
        rare_category_min_count=10,
        random_state=SEED,
    )


def feature_group(column):
    for prefix in ONEHOT_PREFIXES:
        if column.startswith(prefix) and not column.endswith("_is_missing"):
            return prefix[:-1]
    return column


def weight_vector(columns, group_weights):
    return np.array([group_weights.get(feature_group(c), 1.0) for c in columns])
