import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Daftar 8 kolom yang memiliki missing values di train.csv dan test.csv
MISSING_INDICATOR_COLS: List[str] = [
    "num_adults",
    "num_children",
    "meal_package",
    "needs_parking",
    "room_category",
    "days_before_arrival",
    "nightly_rate_idr",
    "special_requests",
]

# Daftar fitur pilihan optimal yang ortogonal & kaya sinyal untuk Tree-Augmented Naive Bayes (TAN v2)
TAN_OPTIMAL_FEATURE_SET: List[str] = [
    "hub_lead_channel_req",
    "time_ym_wknd",
    "product_room_meal_rate",
    "stay_nights_wknd",
    "party_adults_child_park",
    "history_ret_cancel_comp",
    "booking_group_size",
    "arrival_day",
    "total_spend_idr",
    "lead_to_stay_ratio",
    "rate_vs_room_month",
    "days_before_arrival_is_missing",
    "nightly_rate_idr_is_missing",
    "num_adults_is_missing",
]

# Daftar fitur lengkap untuk TAN v3 (mencakup TAN_OPTIMAL_FEATURE_SET + kolom dasar untuk Multi-Resolution Cliques)
TAN_V3_FEATURE_SET: List[str] = list(dict.fromkeys(
    TAN_OPTIMAL_FEATURE_SET
    + [
        "arrival_year",
        "arrival_month",
        "arrival_day",
        "booking_channel",
        "room_category",
        "meal_package",
        "special_requests",
        "needs_parking",
        "is_returning_guest",
        "weekend_nights",
        "weekday_nights",
        "num_adults",
        "num_children",
        "prior_cancellations",
        "prior_completed_bookings",
        "days_before_arrival",
        "nightly_rate_idr",
        "num_adults_is_missing",
        "num_children_is_missing",
        "meal_package_is_missing",
        "needs_parking_is_missing",
        "room_category_is_missing",
        "days_before_arrival_is_missing",
        "nightly_rate_idr_is_missing",
        "special_requests_is_missing",
        "is_zero_guests",
        "is_zero_nights",
        "is_zero_rate",
    ]
))


class DataPreprocessor:
    """
    Pipeline Preprocessing Modular Anti-Data Leakage untuk Tugas Besar 2 IF3170.

    Seluruh parameter statistik dipelajari HANYA dari `X_train` saat `.fit(X_train, y_train)`
    dan diterapkan secara terpisah pada `X_train`, `X_val`, maupun `X_test` melalui `.transform(X)`.
    """

    def __init__(
        self,
        mode: str = "processed_numeric",
        num_impute_strategy: str = "median",
        cat_impute_strategy: str = "mode",
        add_missing_indicators: bool = True,
        add_domain_features: bool = True,
        add_composite_interactions: bool = False,
        composite_lead_bins: int = 14,
        composite_rate_bins: int = 10,
        max_children_clip: float = 4.0,
        rare_category_min_count: int = 10,
        onehot_min_frequency: int = 1,
        add_linear_nonlinear_features: bool = False,
        oof_te_smoothing: float = 15.0,
        encoding_method: str = "onehot",
        max_onehot_cardinality: int = 10,
        scaling_method: str = "standard",
        feature_selection_method: str = "none",
        n_features_to_select: Optional[Union[int, float]] = None,
        selection_threshold: float = 0.0,
        use_pca: bool = False,
        pca_n_components: Optional[Union[int, float]] = None,
        use_rff: bool = False,
        rff_n_components: int = 1600,
        rff_gamma: float = 0.030,
        imbalance_method: str = "none",
        drop_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.mode = mode.lower()
        self.num_impute_strategy = num_impute_strategy.lower()
        self.cat_impute_strategy = cat_impute_strategy.lower()
        self.add_missing_indicators = bool(add_missing_indicators)
        self.add_domain_features = bool(add_domain_features)
        self.add_composite_interactions = bool(add_composite_interactions)
        self.composite_lead_bins = int(composite_lead_bins)
        self.composite_rate_bins = int(composite_rate_bins)
        self.max_children_clip = float(max_children_clip)
        self.rare_category_min_count = int(rare_category_min_count)
        self.onehot_min_frequency = int(onehot_min_frequency)
        self.add_linear_nonlinear_features = bool(add_linear_nonlinear_features)
        self.oof_te_smoothing = float(oof_te_smoothing)
        self.encoding_method = encoding_method.lower()
        self.max_onehot_cardinality = int(max_onehot_cardinality)
        self.scaling_method = scaling_method.lower()
        self.feature_selection_method = feature_selection_method.lower()
        self.n_features_to_select = n_features_to_select
        self.selection_threshold = float(selection_threshold)
        self.use_pca = bool(use_pca)
        self.pca_n_components = pca_n_components
        self.use_rff = bool(use_rff)
        self.rff_n_components = int(rff_n_components)
        self.rff_gamma = float(rff_gamma)
        self.imbalance_method = imbalance_method.lower()
        self.drop_cols = drop_cols if drop_cols is not None else ["id"]
        self.random_state = int(random_state)

        # Atribut yang dipelajari saat fit(X_train, y_train)
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.num_impute_values_: Dict[str, float] = {}
        self.cat_impute_values_: Dict[str, str] = {}
        self.valid_categories_: Dict[str, set] = {}

        # Parameter imputasi & grup yang dipelajari dari X_train
        self.rate_group_medians_: Dict[Tuple[str, str], float] = {}
        self.rate_room_month_medians_: Dict[Tuple[str, int], float] = {}
        self.global_rate_median_: float = 1644000.0
        self.train_booking_group_counts_: Dict[Tuple[Any, ...], int] = {}
        self.lead_quantile_edges_: np.ndarray = np.array([])
        self.rate_quantile_edges_: np.ndarray = np.array([])

        # Parameter fitur non-linear, Multi-Resolution Cliques WoE, Parzen KDE, Spline/Poly2, & RFF
        self.linear_qbin_edges_: Dict[str, np.ndarray] = {}
        self.freq_encoding_counts_: Dict[str, Dict[str, int]] = {}
        self.freq_train_n_: int = 0
        self.te_prior_: float = 0.5
        self.te_maps_: Dict[str, Dict[str, float]] = {}
        self.train_oof_te_cache_: Dict[str, np.ndarray] = {}
        self.train_n_rows_: int = 0
        self.multires_is_fitted_: bool = False
        self.multires_v6_cols_: List[str] = []
        self.multires_s_tr_: Optional[Dict[str, np.ndarray]] = None
        self.multires_y_tr_: Optional[np.ndarray] = None
        self.multires_raw_tabs_: List[Tuple[np.ndarray, np.ndarray]] = []
        self.multires_imp_tabs_: List[Tuple[np.ndarray, np.ndarray]] = []
        self.multires_v_mean_: Optional[np.ndarray] = None
        self.multires_v_std_: Optional[np.ndarray] = None
        self.train_multires_basis_cache_: Optional[np.ndarray] = None
        self.rff_source_indices_: np.ndarray = np.array([], dtype=int)
        self.rff_mean_: Optional[np.ndarray] = None
        self.rff_std_: Optional[np.ndarray] = None
        self.rff_weights_: Optional[np.ndarray] = None

        # Parameter Encoding
        self.onehot_categories_: Dict[str, List[str]] = {}
        self.onehot_infrequent_cols_: Dict[str, bool] = {}
        self.ordinal_mappings_: Dict[str, Dict[str, int]] = {}
        self.encoded_feature_names_: List[str] = []
        self.continuous_encoded_cols_: List[str] = []

        # Parameter Scaling
        self.scaler_params_: Dict[str, Tuple[float, float]] = {}

        # Parameter Feature Selection
        self.feature_scores_: Optional[np.ndarray] = None
        self.selected_mask_: Optional[np.ndarray] = None
        self.selected_feature_names_: List[str] = []

        # Parameter PCA
        self.pca_mean_: Optional[np.ndarray] = None
        self.pca_components_: Optional[np.ndarray] = None
        self.pca_explained_variance_: Optional[np.ndarray] = None
        self.pca_explained_variance_ratio_: Optional[np.ndarray] = None
        self.pca_n_components_: int = 0
        self.final_feature_names_: List[str] = []

        # Simpan ID terakhir yang diproses untuk kemudahan pembuatan submission
        self.last_ids_: Optional[pd.Series] = None
        self.is_fitted_: bool = False

    @staticmethod
    def extract_ids_and_target(
        df: pd.DataFrame,
        id_col: str = "id",
        target_col: str = "is_canceled",
    ) -> Tuple[Optional[pd.Series], pd.DataFrame, Optional[np.ndarray]]:
        """
        Memisahkan kolom identifier (`id`), matriks fitur (`X`), dan vektor target (`y`).
        """
        ids = df[id_col].astype(str).copy() if id_col in df.columns else None
        y = df[target_col].to_numpy(dtype=np.int64) if target_col in df.columns else None
        drop_c = [c for c in (id_col, target_col) if c in df.columns]
        X = df.drop(columns=drop_c).copy()
        return ids, X, y

    @staticmethod
    def clean_training_dataframe(
        df: pd.DataFrame,
        target_col: str = "is_canceled",
        drop_duplicates: bool = True,
        fix_invalid_guests: bool = True,
    ) -> pd.DataFrame:
        """
        Membersihkan DataFrame training dari baris duplikat (mengabaikan kolom `id`)
        dan memperbaiki anomali nilai tidak valid tanpa membocorkan informasi test set.
        HANYA dipanggil pada data training.
        """
        df_clean = df.copy()
        if drop_duplicates:
            subset_cols = [c for c in df_clean.columns if c != "id"]
            df_clean = df_clean.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)

        if fix_invalid_guests and "num_adults" in df_clean.columns:
            children_zero = (df_clean["num_children"].fillna(0) == 0) if "num_children" in df_clean.columns else True
            invalid_mask = (df_clean["num_adults"] == 0) & children_zero
            df_clean.loc[invalid_mask, "num_adults"] = np.nan

        if "nightly_rate_idr" in df_clean.columns:
            df_clean.loc[df_clean["nightly_rate_idr"] < 0, "nightly_rate_idr"] = np.nan

        return df_clean

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Mengonversi input X menjadi salinan DataFrame dan memisahkan kolom identifier"""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            cols = [f"feature_{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=cols)

        if "id" in df.columns:
            self.last_ids_ = df["id"].astype(str).copy()
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        if "is_canceled" in df.columns:
            df = df.drop(columns=["is_canceled"])
        return df

    def _add_missing_indicators_and_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Menambahkan indikator biner `_is_missing` pada 8 kolom yang memiliki null
           sebelum proses imputasi dilakukan.
        2. Menambahkan flag anomali (`is_zero_guests`, `is_zero_nights`, `is_zero_rate`)
           dan melakukan clipping outlier pada `num_children`.
        """
        out = df.copy()

        if self.add_missing_indicators:
            present_miss_cols = [c for c in MISSING_INDICATOR_COLS if c in out.columns]
            for col in present_miss_cols:
                out[f"{col}_is_missing"] = out[col].isna().astype(np.float64)
            if present_miss_cols:
                out["total_missing_cols"] = out[[f"{c}_is_missing" for c in present_miss_cols]].sum(axis=1)

        if self.add_domain_features:
            if ("num_adults" in out.columns) and ("num_children" in out.columns):
                na_temp = pd.to_numeric(out["num_adults"], errors="coerce").fillna(2.0)
                nc_temp = pd.to_numeric(out["num_children"], errors="coerce").fillna(0.0)
                out["is_zero_guests"] = ((na_temp + nc_temp) == 0).astype(np.float64)

            if ("weekend_nights" in out.columns) and ("weekday_nights" in out.columns):
                wknd = pd.to_numeric(out["weekend_nights"], errors="coerce").fillna(0.0)
                wkdy = pd.to_numeric(out["weekday_nights"], errors="coerce").fillna(0.0)
                out["is_zero_nights"] = ((wknd + wkdy) == 0).astype(np.float64)

            if "nightly_rate_idr" in out.columns:
                rate_temp = pd.to_numeric(out["nightly_rate_idr"], errors="coerce").fillna(1.0)
                out["is_zero_rate"] = (rate_temp <= 0).astype(np.float64)

        if "num_children" in out.columns:
            out["num_children"] = pd.to_numeric(out["num_children"], errors="coerce").clip(
                lower=0.0, upper=self.max_children_clip
            )

        if ("num_adults" in out.columns) and ("num_children" in out.columns):
            bad_guests = (pd.to_numeric(out["num_adults"], errors="coerce") == 0) & (
                pd.to_numeric(out["num_children"], errors="coerce").fillna(0.0) == 0
            )
            out.loc[bad_guests, "num_adults"] = np.nan

        if "nightly_rate_idr" in out.columns:
            rate_num = pd.to_numeric(out["nightly_rate_idr"], errors="coerce")
            if "booking_channel" in out.columns:
                bad_rate = (rate_num <= 0) & (out["booking_channel"].astype(str) != "Complimentary")
            else:
                bad_rate = rate_num <= 0
            out.loc[bad_rate, "nightly_rate_idr"] = np.nan

        return out

    def _construct_domain_features(self, df: pd.DataFrame, is_train_fit: bool = False) -> pd.DataFrame:
        """
        Membangun fitur-fitur rekayasa domain perhotelan (Hotel Booking Domain Feature Engineering)
        menggunakan parameter statistik yang dipelajari HANYA dari X_train.
        """
        out = df.copy()
        if not self.add_domain_features:
            return out

        has_nights = ("weekend_nights" in out.columns) and ("weekday_nights" in out.columns)
        has_guests = ("num_adults" in out.columns) and ("num_children" in out.columns)

        if has_nights:
            out["total_nights"] = out["weekend_nights"] + out["weekday_nights"]
            out["weekend_ratio"] = out["weekend_nights"] / (out["total_nights"] + 1.0)

        if has_guests:
            out["total_guests"] = out["num_adults"] + out["num_children"]
            out["has_children"] = (out["num_children"] > 0).astype(np.float64)

        if has_nights and ("nightly_rate_idr" in out.columns):
            out["total_spend_idr"] = out["nightly_rate_idr"] * out["total_nights"]

        if ("num_adults" in out.columns) and ("nightly_rate_idr" in out.columns):
            out["rate_per_adult"] = out["nightly_rate_idr"] / np.maximum(out["num_adults"], 1.0)

        if all(c in out.columns for c in ("room_category", "arrival_month", "nightly_rate_idr")):
            rm_keys = list(zip(out["room_category"].astype(str), out["arrival_month"].astype(int)))
            rm_med = pd.Series(
                [self.rate_room_month_medians_.get(k, self.global_rate_median_) for k in rm_keys],
                index=out.index,
                dtype=np.float64,
            )
            out["rate_vs_room_month"] = out["nightly_rate_idr"] / np.maximum(rm_med, 1.0)

        if ("prior_cancellations" in out.columns) and ("prior_completed_bookings" in out.columns):
            out["total_prior_bookings"] = out["prior_cancellations"] + out["prior_completed_bookings"]
            out["prior_cancel_ratio"] = out["prior_cancellations"] / (out["total_prior_bookings"] + 1.0)
            out["has_prior_cancel"] = (out["prior_cancellations"] > 0).astype(np.float64)

        if "special_requests" in out.columns:
            out["high_special_requests"] = (out["special_requests"] >= 3).astype(np.float64)

        if all(c in out.columns for c in ("booking_channel", "days_before_arrival", "special_requests")):
            out["is_ota_long_lead_no_req"] = (
                (out["booking_channel"].astype(str) == "Online Travel Agent")
                & (out["days_before_arrival"] > 150)
                & (out["special_requests"] == 0)
            ).astype(np.float64)

        if has_nights and ("days_before_arrival" in out.columns):
            out["lead_to_stay_ratio"] = out["days_before_arrival"] / (out["total_nights"] + 1.0)

        if ("arrival_year" in out.columns) and ("arrival_month" in out.columns):
            out["arrival_year_month"] = (out["arrival_year"] - 2023.0) * 12.0 + out["arrival_month"]
            out["month_sin"] = np.sin(2.0 * np.pi * out["arrival_month"] / 12.0)
            out["month_cos"] = np.cos(2.0 * np.pi * out["arrival_month"] / 12.0)

        if all(c in out.columns for c in ("arrival_year", "arrival_month", "arrival_day")):
            dates = pd.to_datetime(
                dict(
                    year=out["arrival_year"].round().astype(int),
                    month=out["arrival_month"].round().astype(int),
                    day=out["arrival_day"].round().astype(int),
                ),
                errors="coerce",
            )
            out["arrival_dayofweek"] = dates.dt.dayofweek.fillna(0).astype(np.float64)
            out["is_weekend_arrival"] = (out["arrival_dayofweek"] >= 5).astype(np.float64)

        # Fitur ukuran rombongan/grup pemesanan (booking_group_size)
        grp_cols = ["arrival_year", "arrival_month", "arrival_day", "booking_channel", "days_before_arrival"]
        if all(c in out.columns for c in grp_cols):
            keys = list(
                zip(
                    out["arrival_year"].round().astype(int),
                    out["arrival_month"].round().astype(int),
                    out["arrival_day"].round().astype(int),
                    out["booking_channel"].astype(str),
                    out["days_before_arrival"].round().astype(int),
                )
            )
            if is_train_fit:
                counts_map: Dict[Tuple[Any, ...], int] = {}
                for k in keys:
                    counts_map[k] = counts_map.get(k, 0) + 1
                self.train_booking_group_counts_ = counts_map
                out["booking_group_size"] = np.minimum(
                    [counts_map.get(k, 1) for k in keys], 6
                ).astype(np.float64)
            else:
                own_counts_map: Dict[Tuple[Any, ...], int] = {}
                for k in keys:
                    own_counts_map[k] = own_counts_map.get(k, 0) + 1
                own_c = np.array([own_counts_map.get(k, 1) for k in keys], dtype=np.float64)
                tr_c = np.array(
                    [self.train_booking_group_counts_.get(k, 0) + 1 for k in keys],
                    dtype=np.float64,
                )
                out["booking_group_size"] = np.minimum(np.maximum(own_c, tr_c), 6.0)

        # Faktor komposit ortogonal (Constructive Induction, Pazzani 1996)
        if self.add_composite_interactions and all(
            c in out.columns
            for c in (
                "days_before_arrival",
                "nightly_rate_idr",
                "booking_channel",
                "special_requests",
                "room_category",
                "meal_package",
            )
        ):
            if is_train_fit:
                q_lead = np.linspace(0.0, 100.0, self.composite_lead_bins + 1)[1:-1]
                q_rate = np.linspace(0.0, 100.0, self.composite_rate_bins + 1)[1:-1]
                self.lead_quantile_edges_ = np.unique(
                    np.percentile(out["days_before_arrival"].to_numpy(dtype=np.float64), q_lead)
                )
                self.rate_quantile_edges_ = np.unique(
                    np.percentile(out["nightly_rate_idr"].to_numpy(dtype=np.float64), q_rate)
                )

            lead_b = np.searchsorted(
                self.lead_quantile_edges_, out["days_before_arrival"].to_numpy(dtype=np.float64), side="right"
            ).astype(str)
            rate_b = np.searchsorted(
                self.rate_quantile_edges_, out["nightly_rate_idr"].to_numpy(dtype=np.float64), side="right"
            ).astype(str)

            req_miss = (
                out["special_requests_is_missing"].to_numpy()
                if "special_requests_is_missing" in out.columns
                else np.zeros(len(out))
            )
            park_miss = (
                out["needs_parking_is_missing"].to_numpy()
                if "needs_parking_is_missing" in out.columns
                else np.zeros(len(out))
            )
            adult_miss = (
                out["num_adults_is_missing"].to_numpy()
                if "num_adults_is_missing" in out.columns
                else np.zeros(len(out))
            )
            child_miss = (
                out["num_children_is_missing"].to_numpy()
                if "num_children_is_missing" in out.columns
                else np.zeros(len(out))
            )

            req_b = np.where(req_miss == 1, "NA", np.minimum(out["special_requests"], 3).round().astype(int).astype(str))
            park_b = np.where(park_miss == 1, "NA", out["needs_parking"].round().astype(int).astype(str))
            ret_b = out["is_returning_guest"].round().astype(int).astype(str)
            pc_b = np.minimum(out["prior_cancellations"], 2).round().astype(int).astype(str)
            pcomp_b = np.minimum(out["prior_completed_bookings"], 3).round().astype(int).astype(str)
            nights_b = np.minimum(out["total_nights"], 7).round().astype(int).astype(str)
            wknd_b = np.minimum(out["weekend_nights"], 3).round().astype(int).astype(str)
            adults_b = np.where(adult_miss == 1, "NA", np.minimum(out["num_adults"], 3).round().astype(int).astype(str))
            child_b = np.where(child_miss == 1, "NA", (out["num_children"] > 0).astype(int).astype(str))

            out["hub_lead_channel_req"] = (
                pd.Series(lead_b, index=out.index)
                + "_"
                + out["booking_channel"].astype(str)
                + "_"
                + pd.Series(req_b, index=out.index)
            )
            out["time_ym_wknd"] = (
                out["arrival_year_month"].round().astype(int).astype(str)
                + "_"
                + out["is_weekend_arrival"].round().astype(int).astype(str)
            )
            out["product_room_meal_rate"] = (
                out["room_category"].astype(str)
                + "_"
                + out["meal_package"].astype(str)
                + "_"
                + pd.Series(rate_b, index=out.index)
            )
            out["stay_nights_wknd"] = pd.Series(nights_b, index=out.index) + "_" + pd.Series(wknd_b, index=out.index)
            out["party_adults_child_park"] = (
                pd.Series(adults_b, index=out.index)
                + "_"
                + pd.Series(child_b, index=out.index)
                + "_"
                + pd.Series(park_b, index=out.index)
            )
            out["history_ret_cancel_comp"] = (
                pd.Series(ret_b, index=out.index)
                + "_"
                + pd.Series(pc_b, index=out.index)
                + "_"
                + pd.Series(pcomp_b, index=out.index)
            )

        return out

    @staticmethod
    def _stratified_kfold_indices(
        y_arr: np.ndarray,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> np.ndarray:
        """Menghasilkan array label fold (0..n_splits-1) secara terstratifikasi dengan NumPy murni."""
        n = len(y_arr)
        folds = np.zeros(n, dtype=np.int64)
        rng = np.random.RandomState(random_state)
        for cls in np.unique(y_arr):
            cls_idx = np.where(y_arr == cls)[0]
            rng.shuffle(cls_idx)
            for i, idx in enumerate(cls_idx):
                folds[idx] = i % n_splits
        return folds

    @staticmethod
    def _extract_multires_raw_struct(X_arr: np.ndarray, col_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Mengekstrak struktur atribut mentah & status missingness asli (sebelum imputasi median)
        menggunakan NumPy murni untuk kebutuhan Multi-Resolution Cliques & Parzen-Window KDE.
        """
        n = X_arr.shape[0]
        col_map = {name: idx for idx, name in enumerate(col_names)}

        def get_num(name: str, default_val: float = 0.0) -> np.ndarray:
            if name not in col_map:
                return np.full(n, default_val, dtype=np.float64)
            col = np.asarray(X_arr[:, col_map[name]], dtype=np.float64)
            return np.where(np.isnan(col), default_val, col)

        def get_miss(name: str) -> np.ndarray:
            miss_col_name = f"{name}_is_missing"
            if miss_col_name in col_map:
                return np.asarray(X_arr[:, col_map[miss_col_name]], dtype=np.float64) >= 0.5
            if name in col_map:
                c_f = np.asarray(X_arr[:, col_map[name]], dtype=np.float64)
                return np.isnan(c_f)
            return np.zeros(n, dtype=bool)

        yr = np.round(get_num("arrival_year", 2024.0)).astype(np.int64)
        mo = np.round(get_num("arrival_month", 8.0)).astype(np.int64)
        dy = np.round(get_num("arrival_day", 15.0)).astype(np.int64)
        ym = (yr - 2023) * 12 + mo
        date_id = (yr - 2023) * 400 + mo * 32 + dy

        bc = (
            np.round(get_num("booking_channel", 0.0)).astype(np.int64) + 1
            if "booking_channel" in col_map
            else np.ones(n, dtype=np.int64)
        )
        rc_miss = get_miss("room_category")
        rc_num = (
            np.round(get_num("room_category", 0.0)).astype(np.int64) + 1
            if "room_category" in col_map
            else np.zeros(n, dtype=np.int64)
        )
        rc_raw = np.where(rc_miss, 0, rc_num)

        mp_miss = get_miss("meal_package")
        mp_num = (
            np.round(get_num("meal_package", 0.0)).astype(np.int64) + 1
            if "meal_package" in col_map
            else np.zeros(n, dtype=np.int64)
        )
        mp_raw = np.where(mp_miss, 0, mp_num)

        wknd = np.clip(np.round(get_num("weekend_nights", 0.0)).astype(np.int64), 0, 7)
        wkdy = np.clip(np.round(get_num("weekday_nights", 0.0)).astype(np.int64), 0, 15)
        stay = wknd * 16 + wkdy
        tot_nights = np.clip(wknd + wkdy, 0, 12)

        dba_raw = np.where(get_miss("days_before_arrival"), np.nan, get_num("days_before_arrival", np.nan))
        rate_miss = get_miss("nightly_rate_idr") | (get_num("is_zero_rate", 0.0) >= 0.5)
        rate_raw = np.where(rate_miss, np.nan, get_num("nightly_rate_idr", np.nan))
        sr_raw = np.where(get_miss("special_requests"), np.nan, get_num("special_requests", np.nan))
        ad_miss = get_miss("num_adults") | (get_num("is_zero_guests", 0.0) >= 0.5)
        ad_raw = np.where(ad_miss, np.nan, get_num("num_adults", np.nan))
        ch_raw = np.where(get_miss("num_children"), np.nan, get_num("num_children", np.nan))
        np_raw = np.where(get_miss("needs_parking"), np.nan, get_num("needs_parking", np.nan))

        ret = np.clip(np.round(get_num("is_returning_guest", 0.0)).astype(np.int64), 0, 1)
        pc = np.clip(np.round(get_num("prior_cancellations", 0.0)).astype(np.int64), 0, 4)
        po = np.clip(np.round(get_num("prior_completed_bookings", 0.0)).astype(np.int64), 0, 8)

        return {
            "yr": yr, "mo": mo, "dy": dy, "ym": ym, "date_id": date_id,
            "bc": bc, "rc_raw": rc_raw, "mp_raw": mp_raw, "wknd": wknd,
            "wkdy": wkdy, "stay": stay, "tot_nights": tot_nights,
            "dba_raw": dba_raw, "rate_raw": rate_raw, "sr_raw": sr_raw,
            "ad_raw": ad_raw, "ch_raw": ch_raw, "np_raw": np_raw,
            "ret": ret, "pc": pc, "po": po,
        }

    @staticmethod
    def _compute_multires_parzen_kde_and_context(
        s_tr: Dict[str, np.ndarray],
        y_tr: np.ndarray,
        s_te: Dict[str, np.ndarray],
        is_self_loo: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Dual-Path Kernel-Weighted Context Imputation (hanya fitur X, tanpa label y) dan
        28 fitur Log-Likelihood Ratio (LLR) dari Strict/Near-Exact Template Kernel serta
        Relative-Scale Parzen-Window Kernel Density Estimator lintas 4 konteks kohort.
        """
        n_te = len(s_te["ym"])
        p1 = min(max(float(np.mean(y_tr)), 1e-6), 1.0 - 1e-6)
        prior_lo = math.log(p1 / (1.0 - p1))
        y_tr_f = y_tr.astype(np.float64)

        dba_imp = s_te["dba_raw"].copy()
        rate_imp = s_te["rate_raw"].copy()
        sr_imp = s_te["sr_raw"].copy()
        rc_imp = s_te["rc_raw"].copy()
        mp_imp = s_te["mp_raw"].copy()

        glob_dba = float(np.nanmedian(s_tr["dba_raw"])) if np.any(~np.isnan(s_tr["dba_raw"])) else 58.0
        glob_rate = float(np.nanmedian(s_tr["rate_raw"])) if np.any(~np.isnan(s_tr["rate_raw"])) else 1644000.0

        feats = np.zeros((n_te, 28), dtype=np.float64)
        feats[:, 20] = 25.0
        feats[:, 21] = 10.0
        feats[:, 22] = 10.0
        strict_w_arr = np.zeros(n_te, dtype=np.float64)

        def eval_pair_matrices(
            idx_te: np.ndarray,
            idx_tr: np.ndarray,
            include_bc_diff: bool = False,
            include_dy_diff: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            k_a, k_b = len(idx_te), len(idx_tr)
            mm = np.zeros((k_a, k_b), dtype=np.int64)
            mm += (s_te["wknd"][idx_te, None] != s_tr["wknd"][None, idx_tr]).astype(np.int64)
            mm += (s_te["wkdy"][idx_te, None] != s_tr["wkdy"][None, idx_tr]).astype(np.int64)

            sr_a, sr_b = s_te["sr_raw"][idx_te, None], s_tr["sr_raw"][None, idx_tr]
            obs_sr = (~np.isnan(sr_a)) & (~np.isnan(sr_b))
            sr_mm = obs_sr & (sr_a != sr_b)
            mm += sr_mm.astype(np.int64)

            for key in ("ad_raw", "ch_raw", "np_raw"):
                va, vb = s_te[key][idx_te, None], s_tr[key][None, idx_tr]
                obs = (~np.isnan(va)) & (~np.isnan(vb))
                mm += (obs & (va != vb)).astype(np.int64)
            for key in ("rc_raw", "mp_raw"):
                va, vb = s_te[key][idx_te, None], s_tr[key][None, idx_tr]
                obs = (va > 0) & (vb > 0)
                mm += (obs & (va != vb)).astype(np.int64)

            mm += (s_te["ret"][idx_te, None] != s_tr["ret"][None, idx_tr]).astype(np.int64)
            if include_bc_diff:
                mm += (s_te["bc"][idx_te, None] != s_tr["bc"][None, idx_tr]).astype(np.int64) * 2

            dba_a, dba_b = s_te["dba_raw"][idx_te, None], s_tr["dba_raw"][None, idx_tr]
            obs_dba = (~np.isnan(dba_a)) & (~np.isnan(dba_b))
            dba_scale = np.maximum(4.0, 0.045 * 0.5 * np.nan_to_num(dba_a + dba_b, nan=80.0))
            u_dba = np.where(obs_dba, np.abs(dba_a - dba_b) / dba_scale, 0.0)

            r_a, r_b = s_te["rate_raw"][idx_te, None], s_tr["rate_raw"][None, idx_tr]
            obs_r = (~np.isnan(r_a)) & (~np.isnan(r_b))
            r_scale = np.maximum(28000.0, 0.035 * 0.5 * np.nan_to_num(r_a + r_b, nan=1600000.0))
            u_rate = np.where(obs_r, np.abs(r_a - r_b) / r_scale, 0.0)

            d_cont_sq = np.minimum(u_dba ** 2, 25.0) + np.minimum(u_rate ** 2, 25.0)
            if include_dy_diff:
                dy_diff = (s_te["dy"][idx_te, None] - s_tr["dy"][None, idx_tr]) / 2.2
                d_cont_sq = d_cont_sq + np.minimum(dy_diff ** 2, 20.0)

            d_total_sq = d_cont_sq + mm.astype(np.float64) * 6.5 + np.where(sr_mm, 4.0, 0.0)
            return mm, sr_mm, u_dba, u_rate, d_cont_sq, d_total_sq

        # Konteks 1: (date_id, bc)
        b1_tr = s_tr["date_id"] * 10 + s_tr["bc"]
        b1_te = s_te["date_id"] * 10 + s_te["bc"]
        order_tr = np.argsort(b1_tr)
        u_b, start_tr, count_tr = np.unique(b1_tr[order_tr], return_index=True, return_counts=True)
        order_te = np.argsort(b1_te)
        u_b_te, start_te, count_te = np.unique(b1_te[order_te], return_index=True, return_counts=True)
        pos_c = np.clip(np.searchsorted(u_b, u_b_te), 0, len(u_b) - 1)
        valid_b = u_b[pos_c] == u_b_te

        for idx_b in np.where(valid_b)[0]:
            tr_pos = pos_c[idx_b]
            idx_tr = order_tr[start_tr[tr_pos] : start_tr[tr_pos] + count_tr[tr_pos]]
            idx_te = order_te[start_te[idx_b] : start_te[idx_b] + count_te[idx_b]]
            if is_self_loo and len(idx_tr) <= 1:
                continue
            mm, sr_mm, u_dba, u_rate, d_cont_sq, d_total_sq = eval_pair_matrices(idx_te, idx_tr)
            if is_self_loo:
                same_mask = idx_te[:, None] == idx_tr[None, :]
                d_total_sq = np.where(same_mask, 1e12, d_total_sq)
                mm = np.where(same_mask, 999, mm)
                u_dba_masked = np.where(same_mask, 999.0, u_dba)
                u_rate_masked = np.where(same_mask, 999.0, u_rate)
            else:
                u_dba_masked, u_rate_masked = u_dba, u_rate

            y_sub = y_tr_f[idx_tr]
            W_imp = np.exp(-d_total_sq / (2.0 * (1.1 ** 2)))

            na_dba = np.isnan(dba_imp[idx_te])
            if np.any(na_dba):
                tr_dba = s_tr["dba_raw"][idx_tr]
                obs_tr_dba = (~np.isnan(tr_dba)).astype(np.float64)
                if np.any(obs_tr_dba > 0):
                    w_d = W_imp[na_dba] * obs_tr_dba[None, :]
                    w_sum_d = np.sum(w_d, axis=1)
                    est_dba = (w_d @ np.nan_to_num(tr_dba, nan=0.0)) / np.maximum(w_sum_d, 1e-12)
                    dba_imp[idx_te[na_dba]] = np.where(w_sum_d > 1e-4, est_dba, float(np.nanmedian(tr_dba)))

            na_rate = np.isnan(rate_imp[idx_te])
            if np.any(na_rate):
                tr_rate = s_tr["rate_raw"][idx_tr]
                obs_tr_r = (~np.isnan(tr_rate)).astype(np.float64)
                if np.any(obs_tr_r > 0):
                    w_r = W_imp[na_rate] * obs_tr_r[None, :]
                    w_sum_r = np.sum(w_r, axis=1)
                    est_r = (w_r @ np.nan_to_num(tr_rate, nan=0.0)) / np.maximum(w_sum_r, 1e-12)
                    rate_imp[idx_te[na_rate]] = np.where(w_sum_r > 1e-4, est_r, float(np.nanmedian(tr_rate)))

            na_sr = np.isnan(sr_imp[idx_te])
            if np.any(na_sr):
                tr_sr = s_tr["sr_raw"][idx_tr]
                obs_tr_s = (~np.isnan(tr_sr)).astype(np.float64)
                if np.any(obs_tr_s > 0):
                    w_s = W_imp[na_sr] * obs_tr_s[None, :]
                    w_sum_s = np.sum(w_s, axis=1)
                    est_s = (w_s @ np.nan_to_num(tr_sr, nan=0.0)) / np.maximum(w_sum_s, 1e-12)
                    sr_imp[idx_te[na_sr]] = np.where(w_sum_s > 1e-4, np.round(est_s), float(np.nanmedian(tr_sr)))

            W_strict0 = np.where((mm == 0) & (u_dba <= 2.4) & (u_rate <= 2.4), np.exp(-0.5 * d_cont_sq), 0.0)
            ws0 = np.sum(W_strict0, axis=1)
            wy0 = W_strict0 @ y_sub
            strict_w_arr[idx_te] = ws0
            for k_m, m_prior in enumerate((0.35, 0.80, 1.60)):
                p_s0 = np.clip((wy0 + m_prior * p1) / (ws0 + m_prior), 1e-6, 1.0 - 1e-6)
                feats[idx_te, k_m] = np.log(p_s0 / (1.0 - p_s0)) - prior_lo

            W_strict1 = np.where(
                (mm <= 1) & (~sr_mm) & (u_dba <= 3.0) & (u_rate <= 3.0),
                np.exp(-0.35 * d_cont_sq) * np.where(mm == 0, 1.0, 0.25),
                0.0,
            )
            ws1 = np.sum(W_strict1, axis=1)
            wy1 = W_strict1 @ y_sub
            for k_m, m_prior in enumerate((0.60, 1.40)):
                p_s1 = np.clip((wy1 + m_prior * p1) / (ws1 + m_prior), 1e-6, 1.0 - 1e-6)
                feats[idx_te, 3 + k_m] = np.log(p_s1 / (1.0 - p_s1)) - prior_lo

            for k_h, (h, m_p) in enumerate(zip((0.55, 0.90, 1.40, 2.20, 3.50), (0.8, 1.3, 2.0, 3.0, 4.5))):
                W = np.exp(-d_total_sq / (2.0 * (h ** 2)))
                w_sum = np.sum(W, axis=1)
                wy_sum = W @ y_sub
                p_est = np.clip((wy_sum + m_p * p1) / (w_sum + m_p), 1e-6, 1.0 - 1e-6)
                feats[idx_te, 5 + k_h] = np.log(p_est / (1.0 - p_est)) - prior_lo

            W_date = np.where(same_mask, 0.0, 1.0) if is_self_loo else np.ones_like(d_total_sq)
            p_date = np.clip(((W_date @ y_sub) + 3.5 * p1) / (np.sum(W_date, axis=1) + 3.5), 1e-6, 1.0 - 1e-6)
            llr_date = np.log(p_date / (1.0 - p_date)) - prior_lo
            feats[idx_te, 10] = (ws0 / (ws0 + 0.6)) * (feats[idx_te, 0] - llr_date)
            feats[idx_te, 11] = (ws1 / (ws1 + 1.0)) * (feats[idx_te, 3] - llr_date)

            feats[idx_te, 18] = ws0
            feats[idx_te, 19] = ws1
            feats[idx_te, 20] = np.min(d_total_sq, axis=1)
            feats[idx_te, 21] = np.min(u_dba_masked, axis=1)
            feats[idx_te, 22] = np.min(u_rate_masked, axis=1)
            feats[idx_te, 23] = llr_date
            nn_idx = np.argmin(d_total_sq, axis=1)
            nn_d = d_total_sq[np.arange(len(idx_te)), nn_idx]
            nn_y = y_sub[nn_idx]
            w_nn = np.exp(-0.5 * nn_d)
            p_nn = np.clip((w_nn * nn_y + 0.8 * p1) / (w_nn + 0.8), 1e-6, 1.0 - 1e-6)
            feats[idx_te, 24] = np.log(p_nn / (1.0 - p_nn)) - prior_lo

        dba_imp = np.where(np.isnan(dba_imp), glob_dba, dba_imp)
        rate_imp = np.where(np.isnan(rate_imp), glob_rate, rate_imp)
        sr_imp_int = np.clip(np.round(np.nan_to_num(sr_imp, nan=0.0)).astype(np.int64), 0, 5)
        sr_bin_imp = np.clip(sr_imp_int, 0, 3)
        rc_imp = np.where(rc_imp == 0, 2, rc_imp)
        mp_imp = np.where(mp_imp == 0, 1, mp_imp)

        # Konteks 2: (ym, bc, stay), Konteks 3: (date_id), Konteks 4: (ym, bc, rc_raw, mp_raw)
        context_configs = [
            ((s_tr["ym"] * 10 + s_tr["bc"]) * 200 + s_tr["stay"], (s_te["ym"] * 10 + s_te["bc"]) * 200 + s_te["stay"], False, True, (0.65, 1.15, 1.90), (1.2, 2.2, 3.8), 12),
            (s_tr["date_id"], s_te["date_id"], True, False, (0.65, 1.25, 2.30), (1.0, 2.0, 3.5), 15),
            (((s_tr["ym"] * 10 + s_tr["bc"]) * 10 + s_tr["rc_raw"]) * 10 + s_tr["mp_raw"], ((s_te["ym"] * 10 + s_te["bc"]) * 10 + s_te["rc_raw"]) * 10 + s_te["mp_raw"], False, True, (0.75, 1.35, 2.20), (1.5, 2.5, 4.0), 25),
        ]
        for b_tr, b_te, inc_bc, inc_dy, hs, mps, col_off in context_configs:
            o_tr = np.argsort(b_tr)
            ub, st_tr, ct_tr = np.unique(b_tr[o_tr], return_index=True, return_counts=True)
            o_te = np.argsort(b_te)
            ub_te, st_te, ct_te = np.unique(b_te[o_te], return_index=True, return_counts=True)
            pc_idx = np.clip(np.searchsorted(ub, ub_te), 0, len(ub) - 1)
            for ib in np.where(ub[pc_idx] == ub_te)[0]:
                tp = pc_idx[ib]
                i_tr = o_tr[st_tr[tp] : st_tr[tp] + ct_tr[tp]]
                i_te = o_te[st_te[ib] : st_te[ib] + ct_te[ib]]
                if is_self_loo and len(i_tr) <= 1:
                    continue
                _, _, _, _, _, d_tot_sq = eval_pair_matrices(i_te, i_tr, include_bc_diff=inc_bc, include_dy_diff=inc_dy)
                if is_self_loo:
                    d_tot_sq = np.where(i_te[:, None] == i_tr[None, :], 1e12, d_tot_sq)
                y_sub = y_tr_f[i_tr]
                for kh, (h_v, mp_v) in enumerate(zip(hs, mps)):
                    W = np.exp(-d_tot_sq / (2.0 * (h_v ** 2)))
                    p_e = np.clip(((W @ y_sub) + mp_v * p1) / (np.sum(W, axis=1) + mp_v), 1e-6, 1.0 - 1e-6)
                    feats[i_te, col_off + kh] = np.log(p_e / (1.0 - p_e)) - prior_lo

        ctx_out = {
            "ym": s_te["ym"], "date_id": s_te["date_id"], "bc": s_te["bc"],
            "rc_imp": rc_imp, "mp_imp": mp_imp, "sr_imp": sr_imp_int,
            "sr_bin_imp": sr_bin_imp, "stay": s_te["stay"], "tot_nights": s_te["tot_nights"],
            "dba_imp": dba_imp, "rate_imp": rate_imp,
        }
        return feats, ctx_out, strict_w_arr

    @staticmethod
    def _extract_multires_raw_cliques(s: Dict[str, np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        """Mengekstrak 91 clique interaksi berorde tinggi dengan Multi-Resolution & Half-Step Shifted Discretization."""
        n = len(s["ym"])
        ym, date_id, bc = s["ym"], s["date_id"], s["bc"]
        rc, mp, stay, tot_nights = s["rc_raw"], s["mp_raw"], s["stay"], s["tot_nights"]
        ret, pc, po = s["ret"], s["pc"], s["po"]

        sr_miss = np.isnan(s["sr_raw"])
        sr_clean = np.nan_to_num(s["sr_raw"], nan=0.0).round().astype(np.int64)
        sr = np.where(sr_miss, 6, np.clip(sr_clean, 0, 5))
        sr_bin = np.where(sr_miss, 4, np.clip(sr_clean, 0, 3))
        sr_zero = np.where(sr_miss, 2, (sr_clean > 0).astype(np.int64))

        np_col = np.where(np.isnan(s["np_raw"]), 2, np.clip(np.nan_to_num(s["np_raw"], nan=0.0).round().astype(np.int64), 0, 1))
        ad = np.where(np.isnan(s["ad_raw"]), 5, np.clip(np.nan_to_num(s["ad_raw"], nan=2.0).round().astype(np.int64), 0, 4))
        ch = np.where(np.isnan(s["ch_raw"]), 4, np.clip(np.nan_to_num(s["ch_raw"], nan=0.0).round().astype(np.int64), 0, 3))
        guests = ad * 5 + ch

        dba_val = np.where(np.isnan(s["dba_raw"]), -999.0, s["dba_raw"])
        rate_val = np.where(np.isnan(s["rate_raw"]), -999.0, s["rate_raw"])

        def bin_dba(step: float, shift: float = 0.0) -> np.ndarray:
            return np.where(dba_val < 0, 0, np.floor((dba_val + shift) / step).astype(np.int64) + 1)

        def bin_rate(step: float, shift: float = 0.0) -> np.ndarray:
            return np.where(rate_val <= 0, 0, np.floor((rate_val + shift) / step).astype(np.int64) + 1)

        dba_1, dba_2, dba_2s = bin_dba(1.0), bin_dba(2.0), bin_dba(2.0, 1.0)
        dba_4, dba_4s, dba_6, dba_6s = bin_dba(4.0), bin_dba(4.0, 2.0), bin_dba(6.0), bin_dba(6.0, 3.0)
        dba_8, dba_8s, dba_12, dba_12s = bin_dba(8.0), bin_dba(8.0, 4.0), bin_dba(12.0), bin_dba(12.0, 6.0)
        dba_15, dba_15s, dba_24, dba_24s, dba_30 = bin_dba(15.0), bin_dba(15.0, 7.5), bin_dba(24.0), bin_dba(24.0, 12.0), bin_dba(30.0)

        r_25k, r_30k = bin_rate(25000.0), bin_rate(30000.0)
        r_50k, r_50ks = bin_rate(50000.0), bin_rate(50000.0, 25000.0)
        r_60k, r_60ks = bin_rate(60000.0), bin_rate(60000.0, 30000.0)
        r_80k, r_80ks = bin_rate(80000.0), bin_rate(80000.0, 40000.0)
        r_100k, r_100ks = bin_rate(100000.0), bin_rate(100000.0, 50000.0)
        r_160k, r_160ks = bin_rate(160000.0), bin_rate(160000.0, 80000.0)
        r_250k, r_250ks = bin_rate(250000.0), bin_rate(250000.0, 125000.0)

        def pack(*arrays: np.ndarray) -> np.ndarray:
            res = np.zeros(n, dtype=np.int64)
            for a in arrays:
                res = res * 1009 + a
            return res

        return [
            (pack(bc, sr, dba_8), 5.0), (pack(bc, sr, dba_8s), 5.0),
            (pack(bc, sr, dba_4), 5.0), (pack(bc, sr, dba_4s), 5.0),
            (pack(bc, sr, dba_6), 5.0), (pack(bc, sr, dba_6s), 5.0),
            (pack(bc, sr, r_100k), 5.0), (pack(bc, sr, r_100ks), 5.0),
            (pack(bc, sr, r_60k), 5.0), (pack(bc, sr, r_60ks), 5.0),
            (pack(ym, bc, sr), 5.0),
            (pack(ym, bc, dba_8), 5.0), (pack(ym, bc, dba_8s), 5.0),
            (pack(ym, bc, dba_12), 5.0), (pack(ym, bc, dba_12s), 5.0),
            (pack(ym, bc, r_100k), 5.0), (pack(ym, bc, r_100ks), 5.0),
            (pack(ym, bc, dba_15, r_160k), 4.0), (pack(ym, bc, dba_15s, r_160ks), 4.0),
            (pack(ym, bc, dba_15, r_160k, sr_bin), 4.0), (pack(ym, bc, dba_15s, r_160ks, sr_bin), 4.0),
            (pack(date_id, bc), 5.0), (pack(date_id, bc, sr_bin), 4.0), (pack(date_id, bc, sr_zero), 4.0),
            (pack(date_id, bc, dba_6), 3.0), (pack(date_id, bc, dba_6s), 3.0),
            (pack(date_id, bc, dba_8), 3.5), (pack(date_id, bc, dba_8s), 3.5),
            (pack(date_id, bc, dba_12), 3.5), (pack(date_id, bc, dba_12s), 3.5),
            (pack(date_id, bc, dba_15), 4.0), (pack(date_id, bc, dba_15s), 4.0),
            (pack(date_id, bc, r_60k), 3.0), (pack(date_id, bc, r_60ks), 3.0),
            (pack(date_id, bc, r_100k), 3.5), (pack(date_id, bc, r_100ks), 3.5),
            (pack(date_id, bc, r_160k), 4.0), (pack(date_id, bc, r_160ks), 4.0),
            (pack(date_id, bc, dba_8, r_160k), 2.0), (pack(date_id, bc, dba_8s, r_160ks), 2.0),
            (pack(date_id, bc, dba_12, r_160k, sr_zero), 2.0), (pack(date_id, bc, dba_12s, r_160ks, sr_zero), 2.0),
            (pack(date_id, bc, dba_15, r_250k), 2.5), (pack(date_id, bc, dba_15s, r_250ks), 2.5),
            (pack(date_id, bc, stay), 3.5), (pack(date_id, bc, tot_nights, sr_zero), 3.0),
            (pack(date_id, bc, rc, mp), 3.0), (pack(date_id, bc, guests), 3.0),
            (pack(date_id, dba_4, bc), 2.5), (pack(date_id, dba_4s, bc), 2.5),
            (pack(date_id, bc, stay, mp, sr_bin), 2.0), (pack(date_id, bc, stay, rc, sr_bin), 2.0),
            (pack(date_id, bc, stay, ad, sr_bin), 2.0), (pack(date_id, bc, stay, mp, rc, ad), 2.0),
            (pack(date_id, bc, dba_12, stay, sr_bin), 2.0), (pack(date_id, bc, dba_12s, stay, sr_bin), 2.0),
            (pack(date_id, bc, dba_12, mp, rc), 2.0), (pack(date_id, bc, dba_12s, mp, rc), 2.0),
            (pack(date_id, bc, r_80k, stay, sr_bin), 2.0), (pack(date_id, bc, r_80ks, stay, sr_bin), 2.0),
            (pack(date_id, bc, r_80k, mp, rc), 2.0), (pack(date_id, bc, r_80ks, mp, rc), 2.0),
            (pack(date_id, bc, dba_15, r_160k, stay), 1.8), (pack(date_id, bc, dba_15s, r_160ks, stay), 1.8),
            (pack(date_id, bc, dba_15, r_160k, ad, mp), 1.8), (pack(date_id, bc, dba_15s, r_160ks, ad, mp), 1.8),
            (pack(dba_4, r_60k, bc), 3.5), (pack(dba_4s, r_60ks, bc), 3.5),
            (pack(dba_6, r_50k, bc, sr_zero), 3.0), (pack(dba_6s, r_50ks, bc, sr_zero), 3.0),
            (pack(dba_8, r_80k, bc, stay), 3.0), (pack(dba_8s, r_80ks, bc, stay), 3.0),
            (pack(dba_8, r_100k, bc, sr_bin), 3.5), (pack(dba_8s, r_100ks, bc, sr_bin), 3.5),
            (pack(dba_12, r_100k, bc, rc, sr_zero), 3.0), (pack(dba_12s, r_100ks, bc, rc, sr_zero), 3.0),
            (pack(dba_15, r_160k, rc, mp, bc), 3.5), (pack(dba_15s, r_160ks, rc, mp, bc), 3.5),
            (pack(dba_24, r_160k, bc, stay, sr_zero), 3.5), (pack(dba_24s, r_160ks, bc, stay, sr_zero), 3.5),
            (pack(sr, np_col, ret), 8.0), (pack(ret, pc, po, bc), 5.0),
            (pack(guests, rc, mp, bc), 5.0), (pack(stay, bc, sr_bin), 5.0),
            (pack(tot_nights, rc, mp, sr_bin), 5.0),
            (pack(dba_2, bc, sr_bin), 4.0), (pack(dba_2s, bc, sr_bin), 4.0),
            (pack(r_30k, bc, rc), 4.0), (pack(r_25k, bc, sr_zero), 4.0),
            (pack(dba_1, bc), 4.0), (pack(dba_30, bc, rc, mp, stay), 4.0),
        ]

    @staticmethod
    def _extract_multires_imputed_cliques(
        s: Dict[str, np.ndarray], ctx: Dict[str, np.ndarray]
    ) -> List[Tuple[np.ndarray, float]]:
        """Membangun 30 Context-Completed & Booking-Day Shifted Shadow Cliques."""
        n = len(ctx["ym"])
        ym, date_id, bc = ctx["ym"], ctx["date_id"], ctx["bc"]
        rc, mp, sr, sr_bin, stay = ctx["rc_imp"], ctx["mp_imp"], ctx["sr_imp"], ctx["sr_bin_imp"], ctx["stay"]
        dba_imp, rate_imp = ctx["dba_imp"], ctx["rate_imp"]

        def bin_dba_i(step: float, shift: float = 0.0) -> np.ndarray:
            return np.floor((dba_imp + shift) / step).astype(np.int64) + 1

        def bin_rate_i(step: float, shift: float = 0.0) -> np.ndarray:
            return np.floor((rate_imp + shift) / step).astype(np.int64) + 1

        d4, d4s, d8, d8s = bin_dba_i(4.0), bin_dba_i(4.0, 2.0), bin_dba_i(8.0), bin_dba_i(8.0, 4.0)
        d12, d12s, d15, d15s = bin_dba_i(12.0), bin_dba_i(12.0, 6.0), bin_dba_i(15.0), bin_dba_i(15.0, 7.5)
        r60, r60s = bin_rate_i(60000.0), bin_rate_i(60000.0, 30000.0)
        r100, r100s = bin_rate_i(100000.0), bin_rate_i(100000.0, 50000.0)
        r160, r160s = bin_rate_i(160000.0), bin_rate_i(160000.0, 80000.0)

        ret = s["ret"]
        pc = np.clip(s["pc"], 0, 3)
        np_clean = np.where(np.isnan(s["np_raw"]), 2, np.clip(np.nan_to_num(s["np_raw"], nan=0.0).round().astype(np.int64), 0, 1))
        ad_clean = np.where(np.isnan(s["ad_raw"]), 5, np.clip(np.nan_to_num(s["ad_raw"], nan=2.0).round().astype(np.int64), 0, 4))
        ch_clean = np.where(np.isnan(s["ch_raw"]), 4, np.clip(np.nan_to_num(s["ch_raw"], nan=0.0).round().astype(np.int64), 0, 3))
        guests = ad_clean * 5 + ch_clean

        approx_arr = np.round((s["yr"] - 2023) * 365.25 + (s["mo"] - 1) * 30.4375 + s["dy"]).astype(np.float64)
        bday_imp = approx_arr - dba_imp

        def bin_bday(step: float, shift: float = 0.0) -> np.ndarray:
            return np.floor((bday_imp + shift) / step).astype(np.int64) + 500

        bd2, bd2s = bin_bday(2.0), bin_bday(2.0, 1.0)
        bd4, bd4s = bin_bday(4.0), bin_bday(4.0, 2.0)
        bd8, bd8s = bin_bday(8.0), bin_bday(8.0, 4.0)

        def pack(*arrays: np.ndarray) -> np.ndarray:
            res = np.zeros(n, dtype=np.int64)
            for a in arrays:
                res = res * 1009 + a
            return res

        return [
            (pack(date_id, bc, d8, r160), 2.5), (pack(date_id, bc, d8s, r160s), 2.5),
            (pack(date_id, bc, d12, r100, sr_bin), 2.5), (pack(date_id, bc, d12s, r100s, sr_bin), 2.5),
            (pack(date_id, bc, d8, sr_bin), 3.0), (pack(date_id, bc, d8s, sr_bin), 3.0),
            (pack(date_id, bc, r100, sr_bin), 3.0), (pack(date_id, bc, r100s, sr_bin), 3.0),
            (pack(date_id, bc, d15, r160, stay), 2.0), (pack(date_id, bc, d15s, r160s, stay), 2.0),
            (pack(ym, bc, d8, r100, sr_bin), 3.5), (pack(ym, bc, d8s, r100s, sr_bin), 3.5),
            (pack(d4, r60, bc, sr_bin), 3.5), (pack(d4s, r60s, bc, sr_bin), 3.5),
            (pack(d8, r100, bc, rc, mp), 3.5), (pack(d8s, r100s, bc, rc, mp), 3.5),
            (pack(bc, sr, d8), 5.0), (pack(bc, sr, d8s), 5.0),
            (pack(bc, sr, r100), 5.0), (pack(bc, sr, r100s), 5.0),
            (pack(bd2, bc, sr_bin), 3.0), (pack(bd2s, bc, sr_bin), 3.0),
            (pack(bd4, bc, sr_bin, rc), 3.0), (pack(bd4s, bc, sr_bin, rc), 3.0),
            (pack(bd8, bc, rc, mp), 3.5), (pack(bd8s, bc, rc, mp), 3.5),
            (pack(date_id, bc, np_clean, ret), 3.5),
            (pack(date_id, bc, guests, sr_bin), 3.0),
            (pack(date_id, bc, ret, pc, sr_bin), 3.0),
            (pack(ym, bc, np_clean, ret, sr_bin), 4.0),
        ]

    @staticmethod
    def _fit_multires_llr_tables(
        cliques_tr: List[Tuple[np.ndarray, float]], y_tr: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Mempelajari tabel Empirical Bayes Log-Likelihood Ratio (LLR) untuk setiap clique."""
        p1 = min(max(float(np.mean(y_tr)), 1e-6), 1.0 - 1e-6)
        prior_lo = math.log(p1 / (1.0 - p1))
        tables: List[Tuple[np.ndarray, np.ndarray]] = []
        y_float = y_tr.astype(np.float64)
        for c_tr, m in cliques_tr:
            u_vals, inv = np.unique(c_tr, return_inverse=True)
            n_tot = np.bincount(inv).astype(np.float64)
            n_pos = np.bincount(inv, weights=y_float)
            p_est = np.clip((n_pos + m * p1) / (n_tot + m), 1e-6, 1.0 - 1e-6)
            llr = np.log(p_est / (1.0 - p_est)) - prior_lo
            tables.append((u_vals, llr))
        return tables

    @staticmethod
    def _transform_multires_llr_matrix(
        cliques_te: List[Tuple[np.ndarray, float]],
        tables: List[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Mengevaluasi matriks Log-Likelihood Ratio (LLR) clique pada data validasi/uji."""
        n_te = len(cliques_te[0][0])
        n_c = len(tables)
        M = np.zeros((n_te, n_c), dtype=np.float64)
        for j, ((c_te, _), (u_vals, llr)) in enumerate(zip(cliques_te, tables)):
            idx = np.searchsorted(u_vals, c_te)
            idx_c = np.clip(idx, 0, len(u_vals) - 1)
            matched = u_vals[idx_c] == c_te
            M[:, j] = np.where(matched, llr[idx_c], 0.0)
        return M

    @staticmethod
    def _build_multires_cohort_matrix(
        X_arr: np.ndarray,
        s: Dict[str, np.ndarray],
        ctx: Dict[str, np.ndarray],
        kde28: np.ndarray,
        strict_w: np.ndarray,
        s_tr_ref: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Membangun matriks geometri kohort & Z-score relatif terhadap referensi X_train (bebas leakage)."""
        X_base = np.nan_to_num(np.asarray(X_arr, dtype=np.float64), nan=-999.0)
        dba_imp, rate_imp = ctx["dba_imp"], ctx["rate_imp"]
        sr_imp, rc_imp, mp_imp = ctx["sr_imp"].astype(np.float64), ctx["rc_imp"].astype(np.float64), ctx["mp_imp"].astype(np.float64)
        tot_nights = np.maximum(s["tot_nights"].astype(np.float64), 0.0)
        wknd, wkdy = s["wknd"].astype(np.float64), s["wkdy"].astype(np.float64)
        bc, ym, dy, date_id = s["bc"].astype(np.float64), s["ym"].astype(np.float64), s["dy"].astype(np.float64), s["date_id"].astype(np.float64)
        ad_clean = np.nan_to_num(s["ad_raw"], nan=2.0)
        ch_clean = np.nan_to_num(s["ch_raw"], nan=0.0)
        np_clean = np.nan_to_num(s["np_raw"], nan=0.0)
        tr_dba_clean = np.nan_to_num(s_tr_ref["dba_raw"], nan=58.0)
        tr_rate_clean = np.nan_to_num(s_tr_ref["rate_raw"], nan=1644000.0)

        u_d, cnt_d = np.unique(s_tr_ref["date_id"], return_counts=True)
        pos_d = np.clip(np.searchsorted(u_d, s["date_id"]), 0, len(u_d) - 1)
        date_all_cnt = np.where(u_d[pos_d] == s["date_id"], cnt_d[pos_d], 0).astype(np.float64)

        b1_tr = s_tr_ref["date_id"] * 10 + s_tr_ref["bc"]
        b1_te = s["date_id"] * 10 + s["bc"]
        u_b1, inv_b1, cnt_b1 = np.unique(b1_tr, return_inverse=True, return_counts=True)
        pos_b1 = np.clip(np.searchsorted(u_b1, b1_te), 0, len(u_b1) - 1)
        m_b1 = u_b1[pos_b1] == b1_te
        date_bc_cnt = np.where(m_b1, cnt_b1[pos_b1], 0).astype(np.float64)
        ratio_bc_on_date = date_bc_cnt / np.maximum(date_all_cnt, 1.0)

        sum_dba1 = np.bincount(inv_b1, weights=tr_dba_clean)
        sum_dba2 = np.bincount(inv_b1, weights=tr_dba_clean ** 2)
        mean_dba1 = np.where(m_b1, sum_dba1[pos_b1] / np.maximum(cnt_b1[pos_b1], 1), dba_imp)
        var_dba1 = np.where(m_b1, np.maximum(0.0, sum_dba2[pos_b1] / np.maximum(cnt_b1[pos_b1], 1) - mean_dba1 ** 2), 100.0)
        std_dba1 = np.sqrt(var_dba1)
        dba_diff_date_bc = np.abs(dba_imp - mean_dba1)
        dba_z_date_bc = (dba_imp - mean_dba1) / (std_dba1 + 5.0)

        sum_r1 = np.bincount(inv_b1, weights=tr_rate_clean)
        sum_r2 = np.bincount(inv_b1, weights=tr_rate_clean ** 2)
        mean_r1 = np.where(m_b1, sum_r1[pos_b1] / np.maximum(cnt_b1[pos_b1], 1), rate_imp)
        var_r1 = np.where(m_b1, np.maximum(0.0, sum_r2[pos_b1] / np.maximum(cnt_b1[pos_b1], 1) - mean_r1 ** 2), 1e10)
        std_r1 = np.sqrt(var_r1)
        rate_diff_date_bc = np.abs(rate_imp - mean_r1)
        rate_ratio_date_bc = rate_imp / np.maximum(mean_r1, 10000.0)
        rate_z_date_bc = (rate_imp - mean_r1) / (std_r1 + 50000.0)

        bs_tr = (s_tr_ref["date_id"] * 10 + s_tr_ref["bc"]) * 200 + s_tr_ref["stay"]
        bs_te = (s["date_id"] * 10 + s["bc"]) * 200 + s["stay"]
        u_bs, cnt_bs = np.unique(bs_tr, return_counts=True)
        pos_bs = np.clip(np.searchsorted(u_bs, bs_te), 0, len(u_bs) - 1)
        date_bc_stay_cnt = np.where(u_bs[pos_bs] == bs_te, cnt_bs[pos_bs], 0).astype(np.float64)

        b2_tr = (s_tr_ref["ym"] * 10 + s_tr_ref["bc"]) * 10 + s_tr_ref["rc_raw"]
        b2_te = (s["ym"] * 10 + s["bc"]) * 10 + s["rc_raw"]
        u_b2, inv_b2, cnt_b2 = np.unique(b2_tr, return_inverse=True, return_counts=True)
        pos_b2 = np.clip(np.searchsorted(u_b2, b2_te), 0, len(u_b2) - 1)
        m_b2 = u_b2[pos_b2] == b2_te
        sum_rb2 = np.bincount(inv_b2, weights=tr_rate_clean)
        sum_rb2_sq = np.bincount(inv_b2, weights=tr_rate_clean ** 2)
        mean_rb2 = np.where(m_b2, sum_rb2[pos_b2] / np.maximum(cnt_b2[pos_b2], 1), rate_imp)
        var_rb2 = np.where(m_b2, np.maximum(0.0, sum_rb2_sq[pos_b2] / np.maximum(cnt_b2[pos_b2], 1) - mean_rb2 ** 2), 1e10)
        rate_ratio_ym_bc_rc = rate_imp / np.maximum(mean_rb2, 10000.0)
        rate_z_ym_bc_rc = (rate_imp - mean_rb2) / (np.sqrt(var_rb2) + 50000.0)

        approx_arrival_days = (s["yr"] - 2023) * 365.25 + (s["mo"] - 1) * 30.4375 + s["dy"]
        booking_day_abs = approx_arrival_days - dba_imp
        tan_z = np.zeros(len(dba_imp), dtype=np.float64)

        return np.column_stack([
            X_base, dba_imp, rate_imp, sr_imp, rc_imp, mp_imp, tot_nights, wknd, wkdy,
            bc, ym, dy, date_id, ad_clean, ch_clean, np_clean,
            s["ret"].astype(np.float64), s["pc"].astype(np.float64), s["po"].astype(np.float64),
            rate_imp * np.maximum(tot_nights, 1.0), dba_imp / (tot_nights + 1.0),
            wknd / np.maximum(tot_nights, 1.0), rate_imp / np.maximum(ad_clean + 0.5 * ch_clean, 1.0),
            date_bc_cnt, dba_diff_date_bc, rate_diff_date_bc, rate_ratio_date_bc, rate_ratio_ym_bc_rc,
            booking_day_abs, date_all_cnt, ratio_bc_on_date, std_dba1, dba_z_date_bc,
            std_r1, rate_z_date_bc, date_bc_stay_cnt, rate_z_ym_bc_rc, strict_w, tan_z, kde28,
        ])

    @staticmethod
    def _compose_multires_spline_poly_design(
        M_r: np.ndarray, M_i: np.ndarray, kde_f: np.ndarray, sw: np.ndarray, V_sc: np.ndarray
    ) -> np.ndarray:
        """
        Menyusun matriks desain 982 fitur untuk Logistic Regression:
        Multi-Resolution Cliques WoE + Context-Imputed Shadow Cliques + Gated Parzen KDE28
        + Piecewise ReLU Hinge Splines + Degree-2 Polynomial Log-Odds Interactions.
        """
        damp1 = 1.0 / (1.0 + 0.50 * sw[:, None])
        damp2 = 1.0 / (1.0 + 1.50 * sw[:, None])
        gate_s = (sw / (sw + 0.50))[:, None]
        gate_ns = 1.0 - gate_s

        raw_mean = np.mean(M_r, axis=1, keepdims=True)
        imp_mean = np.mean(M_i, axis=1, keepdims=True)
        raw_date_mean = np.mean(M_r[:, 21:66], axis=1, keepdims=True)
        raw_ym_mean = np.mean(M_r[:, 10:21], axis=1, keepdims=True)
        raw_lr_mean = np.mean(M_r[:, 66:], axis=1, keepdims=True)
        raw_std = np.std(M_r, axis=1, keepdims=True)
        raw_max = np.max(M_r, axis=1, keepdims=True)
        raw_min = np.min(M_r, axis=1, keepdims=True)

        kde_llrs = np.hstack([kde_f[:, :18], kde_f[:, 23:28]])
        kde_geom = np.column_stack([
            np.log1p(np.maximum(kde_f[:, 18], 0.0)),
            np.log1p(np.maximum(kde_f[:, 19], 0.0)),
            np.clip(kde_f[:, 20], 0.0, 25.0) / 25.0,
            np.clip(kde_f[:, 21], 0.0, 10.0) / 10.0,
            np.clip(kde_f[:, 22], 0.0, 10.0) / 10.0,
        ])

        core_signals = np.hstack([
            np.clip(kde_f[:, [0, 1, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 23, 24, 25, 26]], -4.5, 4.5),
            np.clip(raw_mean, -4.5, 4.5),
            np.clip(imp_mean, -4.5, 4.5),
            np.clip(raw_date_mean, -4.5, 4.5),
            np.clip(raw_ym_mean, -4.5, 4.5),
            np.clip(raw_lr_mean, -4.5, 4.5),
            gate_s,
            kde_geom[:, :3],
        ])

        hinge_parts = []
        for kt in (-1.5, -0.5, 0.0, 0.5, 1.5):
            hinge_parts.append(np.maximum(0.0, core_signals - kt) * 0.35)
            hinge_parts.append(np.maximum(0.0, kt - core_signals) * 0.35)

        top20 = core_signals[:, :20]
        tri_i, tri_j = np.triu_indices(top20.shape[1], k=1)
        poly2 = (top20[:, tri_i] * top20[:, tri_j]) * 0.15

        return np.hstack([
            M_r * damp1, M_i * damp1, M_r * damp2, M_i * damp2,
            kde_llrs, kde_llrs * gate_s, kde_llrs * gate_ns,
            raw_mean * gate_ns, imp_mean * gate_ns,
            raw_date_mean * gate_ns, raw_ym_mean * gate_ns, raw_lr_mean * gate_ns,
            raw_std * gate_ns, raw_max * gate_ns, raw_min * gate_ns,
            kde_geom, V_sc * 0.22, (V_sc * 0.22) * gate_ns,
            *hinge_parts, poly2,
        ])

    def _fit_multires_kde_spline_basis(self, df_v6: pd.DataFrame, y_train: np.ndarray) -> np.ndarray:
        """Mempelajari seluruh tabel Multi-Resolution Cliques, Parzen KDE, & Spline Basis dari X_train."""
        X_tr_arr = df_v6.to_numpy(dtype=np.float64)
        col_names = list(df_v6.columns)
        n_tr = len(X_tr_arr)
        y_bin = (np.asarray(y_train).ravel() > 0).astype(np.int64)

        s_tr = self._extract_multires_raw_struct(X_tr_arr, col_names)
        kde_tr_full, ctx_tr_loo, strict_w_tr = self._compute_multires_parzen_kde_and_context(
            s_tr, y_bin, s_tr, is_self_loo=True
        )
        cliques_tr = self._extract_multires_raw_cliques(s_tr)
        imp_cliques_tr = self._extract_multires_imputed_cliques(s_tr, ctx_tr_loo)

        n_raw, n_imp = len(cliques_tr), len(imp_cliques_tr)
        pos_idx = np.where(y_bin == 1)[0]
        neg_idx = np.where(y_bin == 0)[0]
        rng = np.random.RandomState(self.random_state)
        pos_folds = np.array_split(rng.permutation(pos_idx), 5)
        neg_folds = np.array_split(rng.permutation(neg_idx), 5)

        M_raw_tr = np.zeros((n_tr, n_raw), dtype=np.float64)
        M_imp_tr = np.zeros((n_tr, n_imp), dtype=np.float64)

        for k in range(5):
            in_va = np.concatenate([pos_folds[k], neg_folds[k]])
            in_tr_mask = np.ones(n_tr, dtype=bool)
            in_tr_mask[in_va] = False
            in_tr = np.where(in_tr_mask)[0]
            y_in_tr = y_bin[in_tr]

            sub_c_tr = [(c[in_tr], m) for c, m in cliques_tr]
            sub_c_va = [(c[in_va], m) for c, m in cliques_tr]
            tabs_in = self._fit_multires_llr_tables(sub_c_tr, y_in_tr)
            M_raw_tr[in_va] = self._transform_multires_llr_matrix(sub_c_va, tabs_in)

            sub_imp_tr = [(c[in_tr], m) for c, m in imp_cliques_tr]
            sub_imp_va = [(c[in_va], m) for c, m in imp_cliques_tr]
            tabs_imp_in = self._fit_multires_llr_tables(sub_imp_tr, y_in_tr)
            M_imp_tr[in_va] = self._transform_multires_llr_matrix(sub_imp_va, tabs_imp_in)

        self.multires_s_tr_ = s_tr
        self.multires_y_tr_ = y_bin
        self.multires_raw_tabs_ = self._fit_multires_llr_tables(cliques_tr, y_bin)
        self.multires_imp_tabs_ = self._fit_multires_llr_tables(imp_cliques_tr, y_bin)

        V_tr = self._build_multires_cohort_matrix(X_tr_arr, s_tr, ctx_tr_loo, kde_tr_full, strict_w_tr, s_tr)
        V_tr_clean = np.nan_to_num(V_tr, nan=0.0, posinf=0.0, neginf=0.0)
        self.multires_v_mean_ = np.mean(V_tr_clean, axis=0)
        v_std = np.std(V_tr_clean, axis=0)
        self.multires_v_std_ = np.where(v_std < 1e-12, 1.0, v_std)
        V_tr_sc = np.clip((V_tr_clean - self.multires_v_mean_) / self.multires_v_std_, -5.0, 5.0)

        M_design_tr = self._compose_multires_spline_poly_design(M_raw_tr, M_imp_tr, kde_tr_full, strict_w_tr, V_tr_sc)
        self.train_multires_basis_cache_ = np.nan_to_num(M_design_tr, nan=0.0, posinf=0.0, neginf=0.0)
        self.train_n_rows_ = n_tr
        self.multires_is_fitted_ = True
        return self.train_multires_basis_cache_

    def _transform_multires_kde_spline_basis(self, df_v6: pd.DataFrame, is_train_fit: bool = False) -> np.ndarray:
        """Mentransformasikan data baru ke basis 982 fitur Multi-Resolution Cliques + Parzen KDE + Spline."""
        if is_train_fit and self.train_multires_basis_cache_ is not None and len(df_v6) == self.train_n_rows_:
            return self.train_multires_basis_cache_

        X_te_arr = df_v6.to_numpy(dtype=np.float64)
        col_names = list(df_v6.columns)
        s_te = self._extract_multires_raw_struct(X_te_arr, col_names)
        kde_te_full, ctx_te, strict_w_te = self._compute_multires_parzen_kde_and_context(
            self.multires_s_tr_, self.multires_y_tr_, s_te, is_self_loo=False
        )
        cliques_te = self._extract_multires_raw_cliques(s_te)
        imp_cliques_te = self._extract_multires_imputed_cliques(s_te, ctx_te)

        M_raw_te = self._transform_multires_llr_matrix(cliques_te, self.multires_raw_tabs_)
        M_imp_te = self._transform_multires_llr_matrix(imp_cliques_te, self.multires_imp_tabs_)

        V_te = self._build_multires_cohort_matrix(X_te_arr, s_te, ctx_te, kde_te_full, strict_w_te, self.multires_s_tr_)
        V_te_clean = np.nan_to_num(V_te, nan=0.0, posinf=0.0, neginf=0.0)
        V_te_sc = np.clip((V_te_clean - self.multires_v_mean_) / self.multires_v_std_, -5.0, 5.0)

        M_design_te = self._compose_multires_spline_poly_design(M_raw_te, M_imp_te, kde_te_full, strict_w_te, V_te_sc)
        return np.nan_to_num(M_design_te, nan=0.0, posinf=0.0, neginf=0.0)

    def _fit_rff(self, X_selected: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
        """
        Mempelajari Random Fourier Features (Rahimi & Recht, 2007; pasangan cos/sin
        Sutherland & Schneider, 2015) khusus pada fitur kontinu terpilih di X_train.
        """
        self.rff_source_indices_ = np.asarray(source_indices, dtype=int)
        if len(self.rff_source_indices_) == 0 or self.rff_n_components <= 0:
            self.rff_mean_ = None
            self.rff_std_ = None
            self.rff_weights_ = None
            return X_selected

        X_src = X_selected[:, self.rff_source_indices_]
        self.rff_mean_ = np.mean(X_src, axis=0)
        stds = np.std(X_src, axis=0)
        self.rff_std_ = np.where(stds < 1e-12, 1.0, stds)

        half_dim = max(1, self.rff_n_components // 2)
        rng = np.random.RandomState(self.random_state)
        self.rff_weights_ = rng.normal(
            0.0, np.sqrt(2.0 * max(self.rff_gamma, 1e-9)), size=(X_src.shape[1], half_dim)
        )
        return self._transform_rff(X_selected)

    def _transform_rff(self, X_selected: np.ndarray) -> np.ndarray:
        """Menerapkan proyeksi Random Fourier Features berbasis parameter dari X_train."""
        if (
            not self.use_rff
            or self.rff_weights_ is None
            or self.rff_mean_ is None
            or self.rff_std_ is None
            or len(self.rff_source_indices_) == 0
        ):
            return X_selected

        X_src = (X_selected[:, self.rff_source_indices_] - self.rff_mean_) / self.rff_std_
        Z = X_src @ self.rff_weights_
        out_dim = 2 * self.rff_weights_.shape[1]
        scale = np.sqrt(2.0 / float(out_dim))
        return np.hstack([X_selected, scale * np.cos(Z), scale * np.sin(Z)])

    def _compute_feature_scores(self, X_mat: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        """
        Menghitung skor kepentingan fitur terhadap target y_train di data training:
        - 'correlation': |Pearson r(X_j, y)|
        - 'anova': ANOVA F-statistic antar kelas
        - 'mutual_info' / 'mrmr': Mutual Information I(X_j; y) berbasis diskritisasi kuantil
        """
        n_samples, n_features = X_mat.shape
        scores = np.zeros(n_features, dtype=np.float64)

        if self.feature_selection_method == "correlation":
            y_float = y_arr.astype(np.float64)
            y_centered = y_float - np.mean(y_float)
            y_norm = np.sqrt(np.sum(y_centered ** 2))
            for j in range(n_features):
                x_j = X_mat[:, j]
                x_centered = x_j - np.mean(x_j)
                x_norm = np.sqrt(np.sum(x_centered ** 2))
                if x_norm > 1e-12 and y_norm > 1e-12:
                    scores[j] = abs(float(np.sum(x_centered * y_centered) / (x_norm * y_norm)))

        elif self.feature_selection_method == "anova":
            classes = np.unique(y_arr)
            k = len(classes)
            grand_mean = np.mean(X_mat, axis=0)
            ss_between = np.zeros(n_features, dtype=np.float64)
            ss_within = np.zeros(n_features, dtype=np.float64)

            for c in classes:
                X_c = X_mat[y_arr == c]
                n_c = X_c.shape[0]
                if n_c == 0:
                    continue
                mean_c = np.mean(X_c, axis=0)
                ss_between += n_c * ((mean_c - grand_mean) ** 2)
                ss_within += np.sum((X_c - mean_c) ** 2, axis=0)

            df_between = max(k - 1, 1)
            df_within = max(n_samples - k, 1)
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            scores = np.divide(ms_between, ms_within, out=np.zeros_like(ms_between), where=ms_within > 1e-12)

        elif self.feature_selection_method in ("mutual_info", "mi", "mrmr"):
            classes, y_inv = np.unique(y_arr, return_inverse=True)
            n_classes = len(classes)
            p_y = np.bincount(y_inv, minlength=n_classes).astype(np.float64) / n_samples

            for j in range(n_features):
                col = X_mat[:, j]
                unique_vals = np.unique(col)
                if len(unique_vals) <= 24:
                    _, x_bins = np.unique(col, return_inverse=True)
                    n_bins = len(unique_vals)
                else:
                    edges = np.unique(np.percentile(col, np.linspace(0, 100, 17)[1:-1]))
                    x_bins = np.searchsorted(edges, col, side="right")
                    n_bins = len(edges) + 1

                p_x = np.bincount(x_bins, minlength=n_bins).astype(np.float64) / n_samples
                joint_idx = x_bins * n_classes + y_inv
                p_xy = np.bincount(joint_idx, minlength=n_bins * n_classes).reshape(n_bins, n_classes) / n_samples

                denom = np.outer(p_x, p_y)
                valid = (p_xy > 0) & (denom > 0)
                if np.any(valid):
                    scores[j] = float(np.sum(p_xy[valid] * np.log(p_xy[valid] / denom[valid])))

        return scores

    def _select_mrmr_features(self, X_mat: np.ndarray, mi_scores: np.ndarray, k_select: int) -> np.ndarray:
        """
        Memilih k_select fitur menggunakan kriteria Minimum Redundancy Maximum Relevance (mRMR, Peng et al. 2005):
        Pada setiap langkah, pilih fitur j yang memaksimalkan:
            MI(X_j; y) - (1 / |S|) * sum_{s in S} |Corr(X_j, X_s)| * max(MI)
        """
        n_samples, n_features = X_mat.shape
        k_select = min(max(1, k_select), n_features)

        # Standarisasi matriks untuk perhitungan korelasi cepat
        means = np.mean(X_mat, axis=0, keepdims=True)
        stds = np.std(X_mat, axis=0, keepdims=True)
        stds = np.where(stds < 1e-12, 1.0, stds)
        Z = (X_mat - means) / stds
        corr_mat = np.abs(np.dot(Z.T, Z) / max(n_samples, 1))

        selected: List[int] = []
        remaining = set(range(n_features))

        first_idx = int(np.argmax(mi_scores))
        selected.append(first_idx)
        remaining.remove(first_idx)
        max_mi = max(float(np.max(mi_scores)), 1e-8)

        while len(selected) < k_select and remaining:
            best_j = -1
            best_criterion = -np.inf
            for j in remaining:
                redundancy = float(np.mean(corr_mat[j, selected])) * (0.35 * max_mi)
                crit = float(mi_scores[j]) - redundancy
                if crit > best_criterion:
                    best_criterion = crit
                    best_j = j
            selected.append(best_j)
            remaining.remove(best_j)

        mask = np.zeros(n_features, dtype=bool)
        mask[selected] = True
        return mask

    def _fit_pca(self, X_mat: np.ndarray) -> None:
        """
        Mempelajari Principal Component Analysis (PCA) dari matriks X_train menggunakan
        Singular Value Decomposition (SVD) dengan koreksi tanda deterministik (sign flip).
        """
        n_samples, n_features = X_mat.shape
        self.pca_mean_ = np.mean(X_mat, axis=0)
        X_centered = X_mat - self.pca_mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        max_abs_cols = np.argmax(np.abs(Vt), axis=1)
        signs = np.sign(Vt[np.arange(Vt.shape[0]), max_abs_cols])
        signs[signs == 0] = 1.0
        Vt = Vt * signs[:, np.newaxis]

        explained_variance = (S ** 2) / max(n_samples - 1, 1)
        total_var = float(np.sum(explained_variance))
        if total_var <= 0.0:
            explained_variance_ratio = np.zeros_like(explained_variance)
        else:
            explained_variance_ratio = explained_variance / total_var

        if self.pca_n_components is None:
            k = n_features
        elif isinstance(self.pca_n_components, float) and 0.0 < self.pca_n_components < 1.0:
            cum_var = np.cumsum(explained_variance_ratio)
            k = int(np.searchsorted(cum_var, self.pca_n_components, side="left")) + 1
            k = min(max(k, 1), n_features)
        else:
            k = min(max(int(self.pca_n_components), 1), n_features)

        self.pca_n_components_ = k
        self.pca_components_ = Vt[:k]
        self.pca_explained_variance_ = explained_variance[:k]
        self.pca_explained_variance_ratio_ = explained_variance_ratio[:k]

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "DataPreprocessor":
        """
        Mempelajari seluruh parameter preprocessing HANYA dari X_train (dan y_train).
        """
        np.random.seed(self.random_state)
        df = self._to_dataframe(X_train)

        # Identifikasi kolom numerik vs kategorikal asli
        base_cat_cols = [
            c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) in ("string", "category", "str")
        ]
        base_num_cols = [c for c in df.columns if c not in base_cat_cols]
        self.numeric_cols_ = list(base_num_cols)

        # Mode khusus DTL: biarkan numerik, kategorikal, dan nilai NaN tetap utuh
        if self.mode == "dtl_ready":
            self.categorical_cols_ = list(base_cat_cols)
            self.final_feature_names_ = list(df.columns)
            self.is_fitted_ = True
            return self

        # 1. Buat Missing Indicators & Tangani Anomali sebelum Imputasi
        df = self._add_missing_indicators_and_anomalies(df)

        # 2. Pelajari modus & kategori valid untuk fitur kategorikal asli dari X_train
        self.cat_impute_values_ = {}
        self.valid_categories_ = {}
        for col in base_cat_cols:
            s = df[col].dropna().astype(str)
            mode_val = str(s.mode().iloc[0]) if not s.empty else "Missing"
            self.cat_impute_values_[col] = mode_val
            df[col] = df[col].fillna(mode_val).astype(str)

            # Gabungkan kategori sangat langka ke mode_val jika rare_category_min_count > 0 agar tidak memicu unseen/overfit
            vc = df[col].value_counts()
            valid_cats = set(vc[vc >= self.rare_category_min_count].index.tolist())
            valid_cats.add(mode_val)
            self.valid_categories_[col] = valid_cats
            df[col] = df[col].where(df[col].isin(valid_cats), other=mode_val)

        # 3. Pelajari median grup untuk nightly_rate_idr dari X_train
        if "nightly_rate_idr" in df.columns:
            rate_s = pd.to_numeric(df["nightly_rate_idr"], errors="coerce")
            self.global_rate_median_ = float(rate_s.median()) if not rate_s.dropna().empty else 1644000.0
            if all(c in df.columns for c in ("room_category", "booking_channel")):
                temp_grp = (
                    df.assign(_rate=rate_s)
                    .groupby(["room_category", "booking_channel"])["_rate"]
                    .median()
                    .dropna()
                    .to_dict()
                )
                self.rate_group_medians_ = {
                    (str(k[0]), str(k[1])): float(v) for k, v in temp_grp.items()
                }
                keys = list(zip(df["room_category"].astype(str), df["booking_channel"].astype(str)))
                mapped_med = pd.Series(
                    [self.rate_group_medians_.get(k, self.global_rate_median_) for k in keys],
                    index=df.index,
                    dtype=np.float64,
                )
                df["nightly_rate_idr"] = rate_s.fillna(mapped_med).fillna(self.global_rate_median_)

            if all(c in df.columns for c in ("room_category", "arrival_month")):
                rm_grp = (
                    df.groupby(["room_category", df["arrival_month"].fillna(1).round().astype(int)])["nightly_rate_idr"]
                    .median()
                    .dropna()
                    .to_dict()
                )
                self.rate_room_month_medians_ = {
                    (str(k[0]), int(k[1])): float(v) for k, v in rm_grp.items()
                }

        # 4. Pelajari nilai imputasi median/mean untuk seluruh kolom numerik asli dari X_train
        self.num_impute_values_ = {}
        for col in base_num_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if self.num_impute_strategy == "mean":
                val = float(s.mean()) if not s.dropna().empty else 0.0
            else:
                val = float(s.median()) if not s.dropna().empty else 0.0
            self.num_impute_values_[col] = val
            df[col] = s.fillna(val)

        # 5. Konstruksi Fitur Domain & Faktor Komposit dari X_train
        df = self._construct_domain_features(df, is_train_fit=True)

        # Identifikasi ulang seluruh kolom kategorikal (termasuk kolom komposit baru jika ada)
        self.categorical_cols_ = [
            c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) in ("string", "category", "str")
        ]
        all_num_cols = [c for c in df.columns if c not in self.categorical_cols_]
        for col in all_num_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if col not in self.num_impute_values_:
                val = float(s.median()) if not s.dropna().empty else 0.0
                self.num_impute_values_[col] = val
            df[col] = s.fillna(self.num_impute_values_[col])

        # 6. Pelajari skema Encoding untuk Fitur Kategorikal dari X_train
        self.onehot_categories_ = {}
        self.onehot_infrequent_cols_ = {}
        self.ordinal_mappings_ = {}

        for col in self.categorical_cols_:
            s_col = df[col].astype(str)
            unique_cats = sorted(s_col.unique().tolist())
            use_onehot = (not self.add_linear_nonlinear_features) and (
                self.encoding_method == "onehot"
                or (self.encoding_method == "hybrid" and len(unique_cats) <= self.max_onehot_cardinality)
            )
            if use_onehot:
                if self.onehot_min_frequency > 1:
                    vc = s_col.value_counts()
                    freq_cats = sorted(vc[vc >= self.onehot_min_frequency].index.tolist())
                    has_infreq = bool((vc < self.onehot_min_frequency).any())
                    self.onehot_categories_[col] = freq_cats
                    self.onehot_infrequent_cols_[col] = has_infreq
                else:
                    self.onehot_categories_[col] = unique_cats
                    self.onehot_infrequent_cols_[col] = False
            else:
                # Untuk ordinal encoding, kategori unseen dipetakan ke len(unique_cats) (bucket tersendiri)
                self.ordinal_mappings_[col] = {cat: idx for idx, cat in enumerate(unique_cats)}

        df_encoded = self._apply_encoding(df)

        if self.add_linear_nonlinear_features and y_train is not None:
            opt_set = set(TAN_V3_FEATURE_SET)
            self.multires_v6_cols_ = [c for c in df_encoded.columns if c in opt_set]
            if not self.multires_v6_cols_:
                self.multires_v6_cols_ = list(df_encoded.columns)
            df_v6 = df_encoded[self.multires_v6_cols_]
            X_mat = self._fit_multires_kde_spline_basis(df_v6, np.asarray(y_train).ravel())
            self.encoded_feature_names_ = [f"multires_basis_{i}" for i in range(X_mat.shape[1])]
            self.continuous_encoded_cols_ = []
            self.scaler_params_ = {}
        else:
            self.multires_is_fitted_ = False
            self.encoded_feature_names_ = list(df_encoded.columns)
            self.continuous_encoded_cols_ = [c for c in all_num_cols if c in df_encoded.columns]

            # 7. Pelajari parameter Scaling dari X_train
            self.scaler_params_ = {}
            if self.scaling_method in ("standard", "minmax"):
                for col in self.continuous_encoded_cols_:
                    vals = df_encoded[col].to_numpy(dtype=np.float64)
                    if self.scaling_method == "standard":
                        mean_v = float(np.mean(vals))
                        std_v = float(np.std(vals))
                        if std_v < 1e-12:
                            std_v = 1.0
                        self.scaler_params_[col] = (mean_v, std_v)
                    elif self.scaling_method == "minmax":
                        min_v = float(np.min(vals))
                        max_v = float(np.max(vals))
                        rng_v = max_v - min_v
                        if rng_v < 1e-12:
                            rng_v = 1.0
                        self.scaler_params_[col] = (min_v, rng_v)

            df_scaled = self._apply_scaling(df_encoded)
            X_mat = df_scaled.to_numpy(dtype=np.float64)

        # 8. Pelajari Feature Selection dari X_train dan y_train
        n_total_features = X_mat.shape[1]
        if self.feature_selection_method in ("tan_optimal", "tan_v3"):
            self.feature_scores_ = (
                self._compute_feature_scores(X_mat, np.asarray(y_train).ravel())
                if y_train is not None
                else np.ones(n_total_features, dtype=np.float64)
            )
            opt_set = set(TAN_V3_FEATURE_SET if self.feature_selection_method == "tan_v3" else TAN_OPTIMAL_FEATURE_SET)
            mask = np.array([name in opt_set for name in self.encoded_feature_names_], dtype=bool)
            if not np.any(mask):
                mask = np.ones(n_total_features, dtype=bool)
            self.selected_mask_ = mask
        elif self.feature_selection_method != "none" and y_train is not None:
            y_arr = np.asarray(y_train).ravel()
            self.feature_scores_ = self._compute_feature_scores(X_mat, y_arr)

            if self.n_features_to_select is not None:
                if isinstance(self.n_features_to_select, float) and 0.0 < self.n_features_to_select <= 1.0:
                    k_sel = max(1, int(round(self.n_features_to_select * n_total_features)))
                else:
                    k_sel = min(max(1, int(self.n_features_to_select)), n_total_features)

                if self.feature_selection_method == "mrmr":
                    self.selected_mask_ = self._select_mrmr_features(X_mat, self.feature_scores_, k_sel)
                else:
                    top_idx = np.argsort(self.feature_scores_)[::-1][:k_sel]
                    mask = np.zeros(n_total_features, dtype=bool)
                    mask[top_idx] = True
                    self.selected_mask_ = mask
            else:
                mask = self.feature_scores_ >= self.selection_threshold
                if not np.any(mask):
                    mask[np.argmax(self.feature_scores_)] = True
                self.selected_mask_ = mask
        else:
            self.feature_scores_ = np.ones(n_total_features, dtype=np.float64)
            self.selected_mask_ = np.ones(n_total_features, dtype=bool)

        self.selected_feature_names_ = [
            name for name, keep in zip(self.encoded_feature_names_, self.selected_mask_) if keep
        ]
        X_selected = X_mat[:, self.selected_mask_]

        # 8b. Proyeksi Random Fourier Features (jika diaktifkan) pada fitur kontinu non-biner
        if self.use_rff:
            non_binary_num_set = {
                c for c in all_num_cols
                if not (
                    c.endswith("_is_missing")
                    or c.startswith("is_")
                    or c.startswith("has_")
                    or c == "needs_parking"
                )
            }
            src_indices = np.array(
                [idx for idx, name in enumerate(self.selected_feature_names_) if name in non_binary_num_set],
                dtype=int,
            )
            X_selected = self._fit_rff(X_selected, src_indices)
            if self.rff_weights_ is not None and len(self.rff_source_indices_) > 0:
                half_d = self.rff_weights_.shape[1]
                rff_names = [f"rff_cos_{k}" for k in range(half_d)] + [f"rff_sin_{k}" for k in range(half_d)]
                expanded_feature_names = list(self.selected_feature_names_) + rff_names
            else:
                expanded_feature_names = list(self.selected_feature_names_)
        else:
            expanded_feature_names = list(self.selected_feature_names_)

        # 9. Pelajari Principal Component Analysis (PCA) dari X_train jika diaktifkan
        if self.use_pca:
            self._fit_pca(X_selected)
            self.final_feature_names_ = [f"PC{i + 1}" for i in range(self.pca_n_components_)]
        else:
            self.pca_n_components_ = X_selected.shape[1]
            self.final_feature_names_ = expanded_feature_names

        self.is_fitted_ = True
        return self

    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menerapkan One-Hot / Ordinal Encoding berdasarkan kategori yang dipelajari saat fit."""
        df_num = df.drop(columns=[c for c in self.categorical_cols_ if c in df.columns]).copy()
        encoded_parts = [df_num]

        # Ordinal columns (unseen category dipetakan secara aman ke len(mapping))
        for col, mapping in self.ordinal_mappings_.items():
            if col in df.columns:
                s_str = df[col].astype(str)
                unseen_code = len(mapping)
                mapped = s_str.map(lambda v: mapping.get(v, unseen_code)).astype(np.float64)
                encoded_parts.append(pd.DataFrame({col: mapped}, index=df.index))

        # One-Hot columns (tervektorisasi + mendukung onehot_infrequent_cols_)
        n_rows = len(df)
        row_idx = np.arange(n_rows)
        for col, categories in self.onehot_categories_.items():
            has_infreq = self.onehot_infrequent_cols_.get(col, False)
            col_names = [f"{col}_{cat}".replace(" ", "_") for cat in categories]
            if has_infreq:
                col_names.append(f"{col}___infrequent__".replace(" ", "_"))
            n_ohe = len(col_names)
            if n_ohe == 0:
                continue
            mat = np.zeros((n_rows, n_ohe), dtype=np.float64)
            if col in df.columns:
                cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
                infreq_idx = len(categories) if has_infreq else -1
                codes = df[col].astype(str).map(lambda v: cat_to_idx.get(v, infreq_idx)).to_numpy(dtype=np.int64)
                valid_mask = codes >= 0
                mat[row_idx[valid_mask], codes[valid_mask]] = 1.0
            encoded_parts.append(pd.DataFrame(mat, columns=col_names, index=df.index))

        return pd.concat(encoded_parts, axis=1)

    def _apply_scaling(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
        """Menerapkan StandardScaler atau MinMaxScaler menggunakan parameter dari X_train."""
        df_out = df_encoded.copy()
        if self.scaling_method == "none":
            return df_out

        for col, (p1, p2) in self.scaler_params_.items():
            if col in df_out.columns:
                vals = df_out[col].to_numpy(dtype=np.float64)
                df_out[col] = (vals - p1) / p2
        return df_out

    def _transform_internal(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        is_train_fit: bool = False,
        return_dataframe: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Alur transformasi internal untuk data train maupun validation/test."""
        if not self.is_fitted_:
            raise RuntimeError(
                "DataPreprocessor belum di-fit! Panggil preprocessor.fit(X_train, y_train) terlebih dahulu."
            )

        df = self._to_dataframe(X)

        if self.mode == "dtl_ready":
            df_dtl = df.reindex(columns=self.final_feature_names_)
            if return_dataframe:
                return df_dtl
            return df_dtl.to_numpy()

        # 1. Missing Indicators & Penanganan Anomali
        df = self._add_missing_indicators_and_anomalies(df)

        # 2. Imputasi Kategorikal & Penanganan Rare/Unseen Categories menggunakan statistik X_train
        for col, mode_val in self.cat_impute_values_.items():
            if col in df.columns:
                s = df[col].fillna(mode_val).astype(str)
                valid_set = self.valid_categories_.get(col, {mode_val})
                df[col] = s.where(s.isin(valid_set), other=mode_val)
            else:
                df[col] = mode_val

        # 3. Imputasi Grup Median untuk nightly_rate_idr menggunakan statistik X_train
        if "nightly_rate_idr" in df.columns:
            rate_s = pd.to_numeric(df["nightly_rate_idr"], errors="coerce")
            if self.rate_group_medians_ and all(c in df.columns for c in ("room_category", "booking_channel")):
                keys = list(zip(df["room_category"].astype(str), df["booking_channel"].astype(str)))
                mapped_med = pd.Series(
                    [self.rate_group_medians_.get(k, self.global_rate_median_) for k in keys],
                    index=df.index,
                    dtype=np.float64,
                )
                df["nightly_rate_idr"] = rate_s.fillna(mapped_med).fillna(self.global_rate_median_)
            else:
                df["nightly_rate_idr"] = rate_s.fillna(self.global_rate_median_)

        # 4. Imputasi Kolom Numerik Asli menggunakan median/mean X_train
        for col in self.numeric_cols_:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(self.num_impute_values_[col])
            else:
                df[col] = self.num_impute_values_[col]

        # 5. Konstruksi Fitur Domain & Komposit
        df = self._construct_domain_features(df, is_train_fit=is_train_fit)

        for col, imp_val in self.num_impute_values_.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(imp_val)

        # 6. Encoding Kategorikal (& Basis Multi-Resolution + Parzen KDE + Spline jika diaktifkan)
        df_encoded = self._apply_encoding(df)
        if self.add_linear_nonlinear_features and self.multires_is_fitted_:
            df_v6 = df_encoded.reindex(columns=self.multires_v6_cols_, fill_value=0.0)
            X_mat = self._transform_multires_kde_spline_basis(df_v6, is_train_fit=is_train_fit)
        else:
            df_encoded = df_encoded.reindex(columns=self.encoded_feature_names_, fill_value=0.0)
            df_scaled = self._apply_scaling(df_encoded)
            X_mat = df_scaled.to_numpy(dtype=np.float64)
            X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)

        # 8. Feature Selection
        X_selected = X_mat[:, self.selected_mask_]

        # 8b. Proyeksi Random Fourier Features (jika diaktifkan)
        if self.use_rff:
            X_selected = self._transform_rff(X_selected)

        # 9. Reduksi Dimensi PCA (jika diaktifkan)
        if self.use_pca:
            X_centered = X_selected - self.pca_mean_
            X_final = np.dot(X_centered, self.pca_components_.T)
        else:
            X_final = X_selected

        if return_dataframe:
            return pd.DataFrame(X_final, columns=self.final_feature_names_, index=df.index)
        return X_final

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_dataframe: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Mentransformasikan data X (validation atau test) menggunakan parameter yang dipelajari dari X_train.
        """
        return self._transform_internal(X, is_train_fit=False, return_dataframe=return_dataframe)

    def fit_transform(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Optional[Union[pd.Series, np.ndarray]] = None,
        return_dataframe: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Mempelajari parameter dari X_train (dan y_train) lalu mentransformasikan X_train.
        """
        self.fit(X_train, y_train)
        return self._transform_internal(X_train, is_train_fit=True, return_dataframe=return_dataframe)

    def resample(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        method: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menangani Imbalanced Data HANYA pada data training menggunakan metode:
        - 'random_oversample': Random Oversampling kelas minoritas.
        - 'random_undersample': Random Undersampling kelas mayoritas.
        - 'smote': Synthetic Minority Over-sampling Technique (Chawla et al., 2002).
        - 'none': Mengembalikan X_train dan y_train tanpa perubahan.
        """
        rng = np.random.RandomState(self.random_state)
        chosen_method = (method if method is not None else self.imbalance_method).lower()

        X_arr = (
            X_train.to_numpy(dtype=np.float64)
            if hasattr(X_train, "to_numpy")
            else np.asarray(X_train, dtype=np.float64)
        )
        y_arr = np.asarray(y_train).ravel()

        if chosen_method == "none":
            return X_arr, y_arr

        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) < 2:
            return X_arr, y_arr

        maj_class = classes[np.argmax(counts)]
        min_class = classes[np.argmin(counts)]
        n_maj = int(np.max(counts))
        n_min = int(np.min(counts))

        if n_maj == n_min:
            return X_arr, y_arr

        maj_indices = np.where(y_arr == maj_class)[0]
        min_indices = np.where(y_arr == min_class)[0]

        if chosen_method == "random_oversample":
            extra_min_idx = rng.choice(min_indices, size=n_maj - n_min, replace=True)
            all_idx = np.concatenate([maj_indices, min_indices, extra_min_idx])
            rng.shuffle(all_idx)
            return X_arr[all_idx], y_arr[all_idx]

        elif chosen_method == "random_undersample":
            sampled_maj_idx = rng.choice(maj_indices, size=n_min, replace=False)
            all_idx = np.concatenate([sampled_maj_idx, min_indices])
            rng.shuffle(all_idx)
            return X_arr[all_idx], y_arr[all_idx]

        elif chosen_method == "smote":
            X_min = X_arr[min_indices]
            n_synth = n_maj - n_min
            k_neighbors = min(5, max(1, n_min - 1))

            sq_norms = np.sum(X_min ** 2, axis=1, keepdims=True)
            dists_sq = np.maximum(sq_norms + sq_norms.T - 2.0 * np.dot(X_min, X_min.T), 0.0)
            np.fill_diagonal(dists_sq, np.inf)
            nn_indices = np.argsort(dists_sq, axis=1)[:, :k_neighbors]

            base_choices = rng.randint(0, n_min, size=n_synth)
            neighbor_rank = rng.randint(0, k_neighbors, size=n_synth)
            lambdas = rng.uniform(0.0, 1.0, size=(n_synth, 1))

            X_base = X_min[base_choices]
            X_neighbor = X_min[nn_indices[base_choices, neighbor_rank]]
            X_synth = X_base + lambdas * (X_neighbor - X_base)
            y_synth = np.full(n_synth, min_class, dtype=y_arr.dtype)

            X_res = np.vstack([X_arr, X_synth])
            y_res = np.concatenate([y_arr, y_synth])
            perm = rng.permutation(len(y_res))
            return X_res[perm], y_res[perm]

        raise ValueError(f"Metode imbalance tidak dikenali: {chosen_method}")

    def fit_resample(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Menjalankan `fit_transform(X_train, y_train)` kemudian menerapkan `resample()`."""
        X_trans = self.fit_transform(X_train, y_train, return_dataframe=False)
        return self.resample(X_trans, y_train)

    def transform_for_dtl(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Menghasilkan DataFrame mentah (tanpa kolom `id` dan `is_canceled`) yang membiarkan
        fitur numerik, fitur kategorikal, dan nilai null (`np.nan`) tetap utuh untuk DTL (CART).
        """
        return self._to_dataframe(X)
