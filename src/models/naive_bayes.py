import datetime
import math
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


_CAL_ORD_LUT: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
for _cal_y in range(2020, 2028):
    for _cal_m in range(1, 13):
        for _cal_d in range(1, 32):
            try:
                _cal_dt = datetime.date(_cal_y, _cal_m, _cal_d)
                _CAL_ORD_LUT[(_cal_y, _cal_m, _cal_d)] = (_cal_dt.toordinal(), _cal_dt.weekday())
            except ValueError:
                pass


# Daftar nama fitur utama untuk komponen pohon Chow-Liu TAN
DEFAULT_TAN_TREE_FEATURES: List[str] = [
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


class NaiveBayes:
    """
    Klasifikator Bayesian Network / Naive Bayes 100% from scratch menggunakan NumPy
    (Tanpa Metode Ensemble / Pohon Keputusan):
    1. Implementasi Wajib (Sesuai Slide IF3170 2024 "Bayesian Network as Knowledge Representation"
       Halaman 27-42):
       - Diskritisasi atribut numerik menjadi kategori interval kuantil (`discretize_numeric=True`).
       - Perhitungan probabilitas prior P(Y=c) dan likelihood P(X_j=v | Y=c) dari frekuensi data
         dengan Laplace Smoothing +1 (`laplace_smoothing=1.0`).
       - Klasifikasi dengan aturan Maximum A Posteriori (MAP).
    2. Implementasi Bonus Optimasi Model (Sesuai Referensi Ilmiah):
       - Tree-Augmented Naive Bayes (TAN) via Algoritma Maximum Weight Spanning Tree Chow-Liu
         [Friedman, Geiger, & Goldszmidt, Machine Learning 1997; Chow & Liu, IEEE Trans. IT 1968].
       - Semi-Naive Multi-Resolution Factored Bayesian Contingency Tables (k-Dependence Bayesian
         Classifier) dengan Hierarchical Dirichlet / Laplace m-Estimate Smoothing
         [Sahami, KDD 1996; Pazzani, Machine Learning 1996; Webb et al., Machine Learning 2005].
       - Hidden Naive Bayes (HNB), Averaged One-Dependence Estimators (A1DE), & Selective
         Averaged Two-Dependence Super-Parent Estimators (A2DE / SP-2DE) dengan Zero-Inflated
         Safe Quantile Discretization & Smooth Top-3 CMI Level-1 Stabilization
         [Jiang, Zhang, & Cai, IEEE TKDE 2009; Webb, Boughton, & Wang, Machine Learning 2005 & 2012].
       - Flexible Naive Bayes dengan Exact Continuous 1D, 2D, 3D Wave/Product Parzen-Window KDE
         & Adaptive k-NN Compact-Support Locally Weighted Naive Bayes (LWNB) lintas-bulan
         [John & Langley, UAI 1995; Frank, Hall, & Pfahringer, UAI 2003; Wand & Jones, 1994].
       - Markov Blanket Bayesian Dependency Network Pseudolikelihood Factors
         [Heckerman, Chickering, Meek, Rounthwaite, & Kadie, JMLR 2000; Besag, JRSS 1975].
       - Hierarchical Dirichlet / Pitman-Yor Smoothing untuk k-DB Probabilities
         [MacKay & Peto, 1995; Sahami, KDD 1996].
       - Accelerated Higher-Order Discriminative CLL / WANBIA Parameter Weighting (ALRn) dengan
         Structured Group-Prior Gaussian MAP, Channel/Context Gating, & Two-Loop L-BFGS
         [Zaidi et al., JMLR 2013 & Machine Learning 2016; Liu & Nocedal, Math. Program. 1989].
    """

    def __init__(
        self,
        var_smoothing: float = 1e-9,
        laplace_smoothing: float = 1.0,
        discretize_numeric: bool = False,
        discretize_bins: int = 10,
        use_tan: bool = False,
        categorical_features: Optional[Union[List[int], List[str], List[bool], np.ndarray, str]] = None,
        tan_bins: int = 20,
        tan_blending: float = 0.92,
        class_specific_tree: bool = True,
        marginal_alpha: float = 1.0,
        tan_mode: str = "dirichlet_jm",
        decision_threshold: float = 0.5,
        cll_l2_reg: float = 450.0,
        cll_l2_grid: Optional[List[float]] = None,
        cll_n_iter: int = 15,
        random_state: int = 42,
    ):
        self.var_smoothing = float(var_smoothing)
        self.laplace_smoothing = float(laplace_smoothing)
        self.discretize_numeric = bool(discretize_numeric)
        self.discretize_bins = int(discretize_bins)
        self.use_tan = bool(use_tan)
        self.categorical_features = categorical_features
        self.tan_bins = int(tan_bins)
        self.tan_blending = float(tan_blending)
        self.class_specific_tree = bool(class_specific_tree)
        self.marginal_alpha = float(marginal_alpha)
        self.tan_mode = str(tan_mode).lower()
        self.decision_threshold = float(decision_threshold)
        self.cll_l2_reg = float(cll_l2_reg)
        self.cll_l2_grid = cll_l2_grid if cll_l2_grid is not None else [450.0]
        self.cll_n_iter = int(cll_n_iter)
        self.random_state = int(random_state)

        # Atribut yang dipelajari saat fit()
        self.classes_: Optional[np.ndarray] = None
        self.class_count_: Optional[np.ndarray] = None
        self.class_prior_: Optional[np.ndarray] = None
        self.log_prior_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        self.feature_names_in_: Optional[List[str]] = None

        # Mask dan indeks tipe fitur serta edges diskritisasi wajib (Slide IF3170 Hal. 27-42)
        self.categorical_mask_: Optional[np.ndarray] = None
        self.continuous_indices_: Optional[np.ndarray] = None
        self.categorical_indices_: Optional[np.ndarray] = None
        self.numeric_disc_edges_: Dict[int, np.ndarray] = {}
        self.numeric_disc_medians_: Dict[int, float] = {}

        # Parameter Gaussian untuk fitur kontinu: (n_classes, n_continuous)
        self.epsilon_: float = 1e-9
        self.theta_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None

        # Parameter Frekuensi Data + Laplace Smoothing (+1) untuk fitur kategorikal / hasil diskritisasi
        self.cat_log_probs_: Dict[int, Dict[str, Any]] = {}

        # Parameter Tree-Augmented Naive Bayes (TAN - Chow & Liu 1968; Friedman et al. 1997)
        self.tan_feature_indices_: Optional[np.ndarray] = None
        self.tan_feature_names_: Optional[List[str]] = None
        self.cmi_matrix_: Optional[np.ndarray] = None
        self.mi_per_class_: Optional[np.ndarray] = None
        self.tan_root_: Optional[int] = None
        self.tan_parents_: Optional[np.ndarray] = None
        self.tan_class_parents_: Optional[np.ndarray] = None
        self.tan_tree_edges_: List[Tuple[int, int, float]] = []
        self.tan_bin_edges_: List[np.ndarray] = []
        self.tan_has_nan_bin_: List[bool] = []
        self.tan_n_bins_per_feat_: Optional[np.ndarray] = None
        self.tan_marginals_: List[np.ndarray] = []
        self.tan_cpt_log_probs_: Dict[Any, Any] = {}

        # Parameter Multi-Resolution Factored Cliques, Parzen KDE, k-DB + Markov Blanket & CLL
        self.clique_tables_: List[Tuple[np.ndarray, np.ndarray]] = []
        self.imputed_clique_tables_: List[Tuple[np.ndarray, np.ndarray]] = []
        self.parzen_train_struct_: Optional[Dict[str, np.ndarray]] = None
        self.parzen_ctx_tr_loo_: Optional[Dict[str, np.ndarray]] = None
        self.parzen_y_train_: Optional[np.ndarray] = None
        self.kdb_path1_tables_: Optional[Dict[str, Any]] = None
        self.kdb_path2_tables_: Optional[Dict[str, Any]] = None
        self.hnb_sp2_path1_tables_: Optional[Dict[str, Any]] = None
        self.hnb_sp2_path2_tables_: Optional[Dict[str, Any]] = None
        self.clq_v19_tables_: List[Tuple[np.ndarray, np.ndarray]] = []
        self.parzen_rvrm_tr_: Optional[np.ndarray] = None
        self.cll_theta_: Optional[np.ndarray] = None
        self.cll_col_mean_: Optional[np.ndarray] = None
        self.cll_col_std_: Optional[np.ndarray] = None
        self.cll_col_mask_: Optional[np.ndarray] = None
        self.cll_train_oof_auc_: float = 0.0
        self.cll_train_oof_ll_: float = 0.0
        self.best_cll_l2_: float = 250.0
        self.prior_log_odds_: float = 0.0
        self.is_fitted_: bool = False

    def _detect_feature_types(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Menentukan mask boolean fitur kategorikal vs kontinu berdasarkan X_raw
        dan konfigurasi self.categorical_features.
        """
        n_samples, n_features = X_raw.shape
        cat_mask = np.zeros(n_features, dtype=bool)

        non_numeric_mask = np.zeros(n_features, dtype=bool)
        for j in range(n_features):
            col = X_raw[:, j]
            try:
                _ = col.astype(np.float64)
            except (ValueError, TypeError):
                non_numeric_mask[j] = True

        if self.categorical_features is None:
            cat_mask = non_numeric_mask.copy()
        elif isinstance(self.categorical_features, str) and self.categorical_features.lower() == "auto":
            cat_mask = non_numeric_mask.copy()
            for j in range(n_features):
                if not cat_mask[j]:
                    col_float = X_raw[:, j].astype(np.float64)
                    valid = col_float[~np.isnan(col_float)]
                    if valid.size > 0:
                        unique_vals = np.unique(valid)
                        is_integer = np.all(np.isclose(unique_vals, np.round(unique_vals)))
                        max_auto_cats = 12 if self.tan_mode == "legacy_blend" else 500
                        fname = self.feature_names_in_[j] if self.feature_names_in_ else ""
                        is_known_cont = any(
                            k in fname
                            for k in (
                                "days_before_arrival",
                                "nightly_rate_idr",
                                "total_spend_idr",
                                "rate_per_",
                                "rate_vs_",
                                "lead_to_stay",
                                "weekend_ratio",
                                "cancel_ratio",
                                "month_sin",
                                "month_cos",
                            )
                        ) and not fname.endswith("_is_missing")
                        if is_integer and len(unique_vals) <= max_auto_cats and (
                            self.tan_mode == "legacy_blend" or not is_known_cont
                        ):
                            cat_mask[j] = True
        elif isinstance(self.categorical_features, (list, tuple, np.ndarray)):
            arr = np.asarray(self.categorical_features)
            if arr.dtype == bool and arr.size == n_features:
                cat_mask = arr.copy() | non_numeric_mask
            elif arr.size > 0 and isinstance(arr[0], str) and self.feature_names_in_ is not None:
                name_to_idx = {name: idx for idx, name in enumerate(self.feature_names_in_)}
                for name in arr:
                    if name in name_to_idx:
                        cat_mask[name_to_idx[name]] = True
                cat_mask |= non_numeric_mask
            elif arr.size > 0:
                for idx in arr:
                    cat_mask[int(idx)] = True
                cat_mask |= non_numeric_mask

        return cat_mask

    @staticmethod
    def _logsumexp(log_probs: np.ndarray, axis: int = 1, keepdims: bool = True) -> np.ndarray:
        """Menghitung log(sum(exp(log_probs))) secara stabil menggunakan trik Log-Sum-Exp."""
        max_val = np.max(log_probs, axis=axis, keepdims=True)
        max_val = np.where(np.isinf(max_val), 0.0, max_val)
        sum_exp = np.sum(np.exp(log_probs - max_val), axis=axis, keepdims=True)
        lse = max_val + np.log(np.clip(sum_exp, 1e-300, None))
        if not keepdims:
            lse = np.squeeze(lse, axis=axis)
        return lse

    def _discretize_subset_matrix(
        self, X_sub: np.ndarray, feat_indices: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """Mendiskritisasi matriks subset fitur X_sub untuk komponen pohon TAN."""
        n_samples, n_sub = X_sub.shape
        X_disc = np.zeros((n_samples, n_sub), dtype=np.int64)
        if fit:
            self.tan_bin_edges_ = []
            self.tan_has_nan_bin_ = []
            self.tan_n_bins_per_feat_ = np.zeros(n_sub, dtype=np.int64)

        for local_j, global_j in enumerate(feat_indices):
            global_j = int(global_j)
            if self.categorical_mask_[global_j]:
                info = self.cat_log_probs_[global_j]
                val_to_idx = info["val_to_idx"]
                col_str = np.asarray(X_sub[:, local_j], dtype=str)
                mapped = np.array([val_to_idx.get(v, info["n_categories"]) for v in col_str], dtype=np.int64)
                X_disc[:, local_j] = mapped
                if fit:
                    self.tan_bin_edges_.append(np.array([], dtype=np.float64))
                    self.tan_has_nan_bin_.append(False)
                    self.tan_n_bins_per_feat_[local_j] = info["n_categories"] + 1
            else:
                col_num = np.asarray(X_sub[:, local_j], dtype=np.float64)
                nan_mask = np.isnan(col_num)
                cont_pos = int(np.where(self.continuous_indices_ == global_j)[0][0])
                fallback_val = float(np.mean(self.theta_[:, cont_pos]))

                if self.tan_mode == "legacy_blend":
                    col_clean = np.where(nan_mask, fallback_val, col_num)
                    if fit:
                        unique_vals = np.unique(col_clean)
                        if len(unique_vals) <= self.tan_bins:
                            edges = (
                                np.array([], dtype=np.float64)
                                if len(unique_vals) <= 1
                                else (unique_vals[:-1] + unique_vals[1:]) / 2.0
                            )
                        else:
                            quantiles = np.linspace(0.0, 100.0, self.tan_bins + 1)[1:-1]
                            edges = np.unique(np.percentile(col_clean, quantiles))
                        self.tan_bin_edges_.append(edges)
                        self.tan_has_nan_bin_.append(False)
                        self.tan_n_bins_per_feat_[local_j] = len(edges) + 1
                    else:
                        edges = self.tan_bin_edges_[local_j]
                    X_disc[:, local_j] = np.searchsorted(edges, col_clean, side="right") if edges.size > 0 else 0
                else:
                    valid_vals = col_num[~nan_mask]
                    if fit:
                        if valid_vals.size == 0:
                            edges = np.array([], dtype=np.float64)
                        else:
                            unique_vals = np.unique(valid_vals)
                            if len(unique_vals) <= self.tan_bins:
                                edges = (
                                    np.array([], dtype=np.float64)
                                    if len(unique_vals) <= 1
                                    else (unique_vals[:-1] + unique_vals[1:]) / 2.0
                                )
                            else:
                                quantiles = np.linspace(0.0, 100.0, self.tan_bins + 1)[1:-1]
                                edges = np.unique(np.percentile(valid_vals, quantiles))
                        has_nan = bool(np.any(nan_mask))
                        self.tan_bin_edges_.append(edges)
                        self.tan_has_nan_bin_.append(has_nan)
                        self.tan_n_bins_per_feat_[local_j] = len(edges) + 1 + (1 if has_nan else 0)
                    else:
                        edges = self.tan_bin_edges_[local_j]
                        has_nan = self.tan_has_nan_bin_[local_j]

                    col_filled = np.where(nan_mask, fallback_val, col_num)
                    bins_j = (
                        np.searchsorted(edges, col_filled, side="right")
                        if edges.size > 0
                        else np.zeros(n_samples, dtype=np.int64)
                    )
                    if has_nan:
                        nan_bin_idx = len(edges) + 1
                        bins_j = np.where(nan_mask, nan_bin_idx, bins_j)
                    X_disc[:, local_j] = bins_j

        return X_disc

    def _compute_cmi_and_marginals(
        self, X_disc: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Menghitung distribusi marginal P_marg(X_i | Y=c), matriks Mutual Information per kelas
        I(X_i; X_j | Y=c), serta matriks Conditional Mutual Information global I(X_i; X_j | Y).
        """
        n_samples, n_features = X_disc.shape
        n_classes = len(self.classes_)
        n_bins_arr = self.tan_n_bins_per_feat_

        class_counts = np.array([np.sum(y == c) for c in self.classes_], dtype=np.float64)
        class_priors = class_counts / max(float(n_samples), 1.0)

        marginals: List[np.ndarray] = []
        for i in range(n_features):
            b_i = int(n_bins_arr[i])
            p_i_c = np.zeros((n_classes, b_i), dtype=np.float64)
            for c_idx, c_val in enumerate(self.classes_):
                cnt = np.bincount(X_disc[y == c_val, i], minlength=b_i).astype(np.float64)
                p_i_c[c_idx] = (cnt + self.marginal_alpha) / (
                    class_counts[c_idx] + self.marginal_alpha * b_i
                )
            marginals.append(p_i_c)

        mi_per_class = np.zeros((n_classes, n_features, n_features), dtype=np.float64)
        for c_idx, c_val in enumerate(self.classes_):
            mask_c = y == c_val
            n_c = int(np.sum(mask_c))
            if n_c <= 1:
                continue
            X_c = X_disc[mask_c]

            emp_marg = [
                np.bincount(X_c[:, i], minlength=int(n_bins_arr[i])).astype(np.float64) / n_c
                for i in range(n_features)
            ]

            for i in range(n_features):
                b_i = int(n_bins_arr[i])
                p_i = emp_marg[i]
                col_i = X_c[:, i]
                for j in range(i + 1, n_features):
                    b_j = int(n_bins_arr[j])
                    p_j = emp_marg[j]
                    col_j = X_c[:, j]

                    joint_idx = col_i * b_j + col_j
                    joint_counts = np.bincount(joint_idx, minlength=b_i * b_j).reshape(b_i, b_j)
                    p_ij = joint_counts.astype(np.float64) / n_c

                    denom = np.outer(p_i, p_j)
                    valid = (p_ij > 0) & (denom > 0)
                    if np.any(valid):
                        mi_val = max(0.0, float(np.sum(p_ij[valid] * np.log(p_ij[valid] / denom[valid]))))
                        mi_per_class[c_idx, i, j] = mi_val
                        mi_per_class[c_idx, j, i] = mi_val

        cmi_global = np.tensordot(class_priors, mi_per_class, axes=(0, 0))
        return cmi_global, mi_per_class, marginals

    @staticmethod
    def _build_chow_liu_mwst(
        weight_matrix: np.ndarray,
    ) -> Tuple[int, np.ndarray, List[Tuple[int, int, float]]]:
        """Membangun Maximum Weight Spanning Tree (MWST) menggunakan algoritma Prim."""
        n_features = weight_matrix.shape[0]
        if n_features == 0:
            return 0, np.array([], dtype=np.int64), []
        if n_features == 1:
            return 0, np.array([-1], dtype=np.int64), []

        total_w_per_node = np.sum(weight_matrix, axis=1)
        root = int(np.argmax(total_w_per_node))

        in_tree = np.zeros(n_features, dtype=bool)
        in_tree[root] = True

        parents = np.full(n_features, -1, dtype=np.int64)
        best_weight = weight_matrix[root].copy()
        best_from = np.full(n_features, root, dtype=np.int64)
        best_weight[root] = -np.inf

        tree_edges: List[Tuple[int, int, float]] = []

        for _ in range(n_features - 1):
            candidates = np.where(~in_tree)[0]
            best_idx = int(candidates[np.argmax(best_weight[candidates])])
            parent_idx = int(best_from[best_idx])
            weight_val = float(weight_matrix[parent_idx, best_idx])

            in_tree[best_idx] = True
            parents[best_idx] = parent_idx
            tree_edges.append((parent_idx, best_idx, weight_val))

            for v in range(n_features):
                if not in_tree[v] and weight_matrix[best_idx, v] > best_weight[v]:
                    best_weight[v] = weight_matrix[best_idx, v]
                    best_from[v] = best_idx

        return root, parents, tree_edges

    def _fit_tan_parameters(self, X_disc: np.ndarray, y: np.ndarray) -> None:
        """Mempelajari tabel log-probabilitas kondisional P(X_i | X_{parent(i)}, Y=c) untuk TAN."""
        n_samples, n_features = X_disc.shape
        n_classes = len(self.classes_)
        self.tan_cpt_log_probs_ = {}
        n_bins_arr = self.tan_n_bins_per_feat_

        self.tan_class_parents_ = np.zeros((n_classes, n_features), dtype=np.int64)
        for c_idx in range(n_classes):
            if self.class_specific_tree and self.tan_mode in ("dirichlet_jm", "cll_v3", "cll_v4", "cll_v6", "cll_kdb_kde"):
                _, p_c, _ = self._build_chow_liu_mwst(self.mi_per_class_[c_idx])
                self.tan_class_parents_[c_idx] = p_c
            else:
                self.tan_class_parents_[c_idx] = self.tan_parents_.copy()

        if self.tan_mode == "legacy_blend":
            alpha = self.laplace_smoothing
            for i in range(n_features):
                p = int(self.tan_parents_[i])
                if p < 0:
                    continue
                b_i = int(n_bins_arr[i])
                b_p = int(n_bins_arr[p])
                cpt_log = np.zeros((n_classes, b_p, b_i), dtype=np.float64)

                for c_idx, c_val in enumerate(self.classes_):
                    mask_c = y == c_val
                    col_i_c = X_disc[mask_c, i]
                    col_p_c = X_disc[mask_c, p]
                    if col_i_c.size > 0:
                        joint_idx = col_p_c * b_i + col_i_c
                        counts_2d = np.bincount(joint_idx, minlength=b_p * b_i).reshape(b_p, b_i).astype(np.float64)
                    else:
                        counts_2d = np.zeros((b_p, b_i), dtype=np.float64)

                    parent_counts = np.sum(counts_2d, axis=1, keepdims=True)
                    smoothed_probs = (counts_2d + alpha) / (parent_counts + alpha * b_i)
                    cpt_log[c_idx] = np.log(np.clip(smoothed_probs, 1e-300, None))

                self.tan_cpt_log_probs_[i] = cpt_log
        else:
            s_dir = self.laplace_smoothing
            jm_lambda = np.clip(self.tan_blending, 0.0, 1.0)

            for c_idx, c_val in enumerate(self.classes_):
                mask_c = y == c_val
                X_c = X_disc[mask_c]
                for i in range(n_features):
                    b_i = int(n_bins_arr[i])
                    p_marg = self.tan_marginals_[i][c_idx]
                    p = int(self.tan_class_parents_[c_idx, i])

                    if p < 0:
                        self.tan_cpt_log_probs_[(c_idx, i)] = np.log(np.clip(p_marg, 1e-300, None))
                    else:
                        b_p = int(n_bins_arr[p])
                        col_i = X_c[:, i]
                        col_p = X_c[:, p]
                        if col_i.size > 0:
                            counts_2d = (
                                np.bincount(col_p * b_i + col_i, minlength=b_p * b_i)
                                .reshape(b_p, b_i)
                                .astype(np.float64)
                            )
                        else:
                            counts_2d = np.zeros((b_p, b_i), dtype=np.float64)

                        parent_counts = np.sum(counts_2d, axis=1, keepdims=True)
                        p_dir = (counts_2d + s_dir * p_marg[np.newaxis, :]) / (parent_counts + s_dir)
                        p_final = jm_lambda * p_dir + (1.0 - jm_lambda) * p_marg[np.newaxis, :]
                        self.tan_cpt_log_probs_[(c_idx, i)] = np.log(np.clip(p_final, 1e-300, None))

    def _eval_tan_dirichlet_jll(self, X_sub_disc: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
        """Menghitung joint log-likelihood dari pohon TAN v2 pada matriks diskrit X_sub_disc."""
        n_samples, n_sub = X_sub_disc.shape
        n_classes = len(self.classes_)
        jll = np.tile(log_prior, (n_samples, 1))
        n_bins_arr = self.tan_n_bins_per_feat_

        for c_idx in range(n_classes):
            for i in range(n_sub):
                b_i = int(n_bins_arr[i])
                idx_i = np.clip(X_sub_disc[:, i], 0, b_i - 1)
                p = int(self.tan_class_parents_[c_idx, i])
                cpt_log = self.tan_cpt_log_probs_[(c_idx, i)]
                if p < 0:
                    jll[:, c_idx] += cpt_log[idx_i]
                else:
                    b_p = int(n_bins_arr[p])
                    idx_p = np.clip(X_sub_disc[:, p], 0, b_p - 1)
                    jll[:, c_idx] += cpt_log[idx_p, idx_i]
        return jll

    def _extract_raw_struct(self, X_arr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Mengekstrak struktur atribut mentah & status missingness asli (sebelum imputasi)
        murni menggunakan NumPy untuk kebutuhan Multi-Resolution Cliques dan Parzen-Window KDE.
        """
        n = X_arr.shape[0]
        col_map = {name: idx for idx, name in enumerate(self.feature_names_in_ or [])}

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
                try:
                    c_f = np.asarray(X_arr[:, col_map[name]], dtype=np.float64)
                    return np.isnan(c_f)
                except (ValueError, TypeError):
                    c_s = np.asarray(X_arr[:, col_map[name]], dtype=str)
                    return (c_s == "nan") | (c_s == "None") | (c_s == "")
            return np.zeros(n, dtype=bool)

        yr = np.round(get_num("arrival_year", 2024.0)).astype(np.int64)
        mo = np.round(get_num("arrival_month", 8.0)).astype(np.int64)
        dy = np.round(get_num("arrival_day", 15.0)).astype(np.int64)
        ym = (yr - 2023) * 12 + mo
        date_id = (yr - 2023) * 400 + mo * 32 + dy

        if "booking_channel" in col_map:
            raw_bc = X_arr[:, col_map["booking_channel"]]
            try:
                bc = np.round(raw_bc.astype(np.float64)).astype(np.int64) + 1
            except (ValueError, TypeError):
                bc_map = {
                    "Online Travel Agent": 1,
                    "Offline Agent": 2,
                    "Corporate": 3,
                    "Complimentary": 4,
                    "Airline": 5,
                }
                bc = np.array([bc_map.get(str(v), 1) for v in raw_bc], dtype=np.int64)
        else:
            bc = np.ones(n, dtype=np.int64)

        rc_miss = get_miss("room_category")
        if "room_category" in col_map:
            raw_rc = X_arr[:, col_map["room_category"]]
            try:
                rc_num = np.round(raw_rc.astype(np.float64)).astype(np.int64) + 1
            except (ValueError, TypeError):
                rc_map = {f"Type {c}": idx for idx, c in enumerate(["A", "B", "C", "D", "E", "F", "G"], start=1)}
                rc_num = np.array([rc_map.get(str(v), 0) for v in raw_rc], dtype=np.int64)
            rc_raw = np.where(rc_miss, 0, rc_num)
        else:
            rc_raw = np.zeros(n, dtype=np.int64)

        mp_miss = get_miss("meal_package")
        if "meal_package" in col_map:
            raw_mp = X_arr[:, col_map["meal_package"]]
            try:
                mp_num = np.round(raw_mp.astype(np.float64)).astype(np.int64) + 1
            except (ValueError, TypeError):
                mp_map = {"Breakfast": 1, "Room Only": 2, "Half Board": 3, "Full Board": 4}
                mp_num = np.array([mp_map.get(str(v), 0) for v in raw_mp], dtype=np.int64)
            mp_raw = np.where(mp_miss, 0, mp_num)
        else:
            mp_raw = np.zeros(n, dtype=np.int64)

        wknd = np.clip(np.round(get_num("weekend_nights", 0.0)).astype(np.int64), 0, 7)
        wkdy = np.clip(np.round(get_num("weekday_nights", 0.0)).astype(np.int64), 0, 15)
        stay = wknd * 16 + wkdy
        tot_nights = np.clip(wknd + wkdy, 0, 12)

        dba_miss = get_miss("days_before_arrival")
        dba_raw = np.where(dba_miss, np.nan, get_num("days_before_arrival", np.nan))

        rate_miss = get_miss("nightly_rate_idr") | (get_num("is_zero_rate", 0.0) >= 0.5)
        rate_raw = np.where(rate_miss, np.nan, get_num("nightly_rate_idr", np.nan))

        sr_miss = get_miss("special_requests")
        sr_raw = np.where(sr_miss, np.nan, get_num("special_requests", np.nan))

        ad_miss = get_miss("num_adults") | (get_num("is_zero_guests", 0.0) >= 0.5)
        ad_raw = np.where(ad_miss, np.nan, get_num("num_adults", np.nan))

        ch_miss = get_miss("num_children")
        ch_raw = np.where(ch_miss, np.nan, get_num("num_children", np.nan))

        np_miss = get_miss("needs_parking")
        np_raw = np.where(np_miss, np.nan, get_num("needs_parking", np.nan))

        ret = np.clip(np.round(get_num("is_returning_guest", 0.0)).astype(np.int64), 0, 1)
        pc = np.clip(np.round(get_num("prior_cancellations", 0.0)).astype(np.int64), 0, 4)
        po = np.clip(np.round(get_num("prior_completed_bookings", 0.0)).astype(np.int64), 0, 8)

        return {
            "yr": yr,
            "mo": mo,
            "dy": dy,
            "ym": ym,
            "date_id": date_id,
            "bc": bc,
            "rc_raw": rc_raw,
            "mp_raw": mp_raw,
            "wknd": wknd,
            "wkdy": wkdy,
            "stay": stay,
            "tot_nights": tot_nights,
            "dba_raw": dba_raw,
            "rate_raw": rate_raw,
            "sr_raw": sr_raw,
            "ad_raw": ad_raw,
            "ch_raw": ch_raw,
            "np_raw": np_raw,
            "ret": ret,
            "pc": pc,
            "po": po,
        }

    @staticmethod
    def _compute_sharp_template_distance_matrix(
        s_te: Dict[str, np.ndarray],
        idx_te: np.ndarray,
        s_tr: Dict[str, np.ndarray],
        idx_tr: np.ndarray,
    ) -> np.ndarray:
        """
        Menghitung matriks jarak ketidaksamaan template berpasangan (Parzen-Window Kernel Distance)
        berukuran (len(idx_te), len(idx_tr)) hanya pada dimensi-dimensi yang sama-sama teramati
        (non-NaN) antara sampel uji i dan sampel latih k (John & Langley, UAI 1995).
        """
        n_a = len(idx_te)
        n_b = len(idx_tr)
        d_sq = np.zeros((n_a, n_b), dtype=np.float64)
        n_obs = np.full((n_a, n_b), 2.0, dtype=np.float64)

        # 1. Durasi menginap (wknd & wkdy)
        d_sq += np.where(s_te["wknd"][idx_te, None] != s_tr["wknd"][None, idx_tr], 6.0, 0.0)
        d_sq += np.where(s_te["wkdy"][idx_te, None] != s_tr["wkdy"][None, idx_tr], 6.0, 0.0)

        # 2. days_before_arrival (skala jitter ~4.2 hari)
        dba_a = s_te["dba_raw"][idx_te, None]
        dba_b = s_tr["dba_raw"][None, idx_tr]
        m_dba = (~np.isnan(dba_a)) & (~np.isnan(dba_b))
        diff_dba = np.where(m_dba, (dba_a - dba_b) / 4.2, 0.0)
        d_sq += np.minimum(diff_dba ** 2, 25.0)
        n_obs += m_dba.astype(np.float64) * 2.0

        # 3. nightly_rate_idr (skala jitter ~Rp 30.000)
        r_a = s_te["rate_raw"][idx_te, None]
        r_b = s_tr["rate_raw"][None, idx_tr]
        m_r = (~np.isnan(r_a)) & (~np.isnan(r_b))
        diff_r = np.where(m_r, (r_a - r_b) / 30000.0, 0.0)
        d_sq += np.minimum(diff_r ** 2, 25.0)
        n_obs += m_r.astype(np.float64) * 2.0

        # 4. special_requests
        sr_a = s_te["sr_raw"][idx_te, None]
        sr_b = s_tr["sr_raw"][None, idx_tr]
        m_sr = (~np.isnan(sr_a)) & (~np.isnan(sr_b))
        d_sq += np.where(m_sr & (sr_a != sr_b), 6.5 * np.abs(sr_a - sr_b), 0.0)
        n_obs += m_sr.astype(np.float64)

        # 5. num_adults
        ad_a = s_te["ad_raw"][idx_te, None]
        ad_b = s_tr["ad_raw"][None, idx_tr]
        m_ad = (~np.isnan(ad_a)) & (~np.isnan(ad_b))
        d_sq += np.where(m_ad & (ad_a != ad_b), 5.5 * np.abs(ad_a - ad_b), 0.0)
        n_obs += m_ad.astype(np.float64)

        # 6. num_children
        ch_a = s_te["ch_raw"][idx_te, None]
        ch_b = s_tr["ch_raw"][None, idx_tr]
        m_ch = (~np.isnan(ch_a)) & (~np.isnan(ch_b))
        d_sq += np.where(m_ch & (ch_a != ch_b), 4.0, 0.0)
        n_obs += m_ch.astype(np.float64)

        # 7. room_category & meal_package
        rc_a = s_te["rc_raw"][idx_te, None]
        rc_b = s_tr["rc_raw"][None, idx_tr]
        m_rc = (rc_a > 0) & (rc_b > 0)
        d_sq += np.where(m_rc & (rc_a != rc_b), 5.5, 0.0)
        n_obs += m_rc.astype(np.float64)

        mp_a = s_te["mp_raw"][idx_te, None]
        mp_b = s_tr["mp_raw"][None, idx_tr]
        m_mp = (mp_a > 0) & (mp_b > 0)
        d_sq += np.where(m_mp & (mp_a != mp_b), 5.5, 0.0)
        n_obs += m_mp.astype(np.float64)

        # 8. needs_parking & is_returning_guest
        np_a = s_te["np_raw"][idx_te, None]
        np_b = s_tr["np_raw"][None, idx_tr]
        m_np = (~np.isnan(np_a)) & (~np.isnan(np_b))
        d_sq += np.where(m_np & (np_a != np_b), 4.5, 0.0)

        d_sq += np.where(s_te["ret"][idx_te, None] != s_tr["ret"][None, idx_tr], 5.0, 0.0)

        return d_sq * (9.0 / np.maximum(n_obs, 4.0))

    def _compute_parzen_kde_and_context_imputation(
        self,
        s_tr: Dict[str, np.ndarray],
        y_tr: np.ndarray,
        s_te: Dict[str, np.ndarray],
        is_self_loo: bool = False,
        extended_v6: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        1. Melakukan Dual-Path Hierarchical & Kernel-Weighted Context Imputation untuk nilai NaN
           pada `days_before_arrival`, `nightly_rate_idr`, `special_requests`, `room_category`,
           dan `meal_package` murni menggunakan kemiripan fitur di `s_tr` (tanpa bocoran label y).
        2. Menghitung 18 fitur Log-Likelihood Ratio (LLR) dari Strict Zero-Mismatch Template
           Kernel, Near-Exact Template Kernel, Relative-Scale Parzen-Window Kernel Density Estimator
           (John & Langley, UAI 1995), serta Local Contrast Gating terhadap coarse date_id
           (ditambah 10 fitur geometri template & Konteks 4 bila `extended_v6=True`).
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

        n_kde_cols = 64 if extended_v6 else 18
        feats = np.zeros((n_te, n_kde_cols), dtype=np.float64)
        if extended_v6:
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
            k_a = len(idx_te)
            k_b = len(idx_tr)

            mm = np.zeros((k_a, k_b), dtype=np.int64)
            mm += (s_te["wknd"][idx_te, None] != s_tr["wknd"][None, idx_tr]).astype(np.int64)
            mm += (s_te["wkdy"][idx_te, None] != s_tr["wkdy"][None, idx_tr]).astype(np.int64)

            sr_a = s_te["sr_raw"][idx_te, None]
            sr_b = s_tr["sr_raw"][None, idx_tr]
            obs_sr = (~np.isnan(sr_a)) & (~np.isnan(sr_b))
            sr_mm = obs_sr & (sr_a != sr_b)
            mm += sr_mm.astype(np.int64)

            for key in ("ad_raw", "ch_raw", "np_raw"):
                va = s_te[key][idx_te, None]
                vb = s_tr[key][None, idx_tr]
                obs = (~np.isnan(va)) & (~np.isnan(vb))
                mm += (obs & (va != vb)).astype(np.int64)

            for key in ("rc_raw", "mp_raw"):
                va = s_te[key][idx_te, None]
                vb = s_tr[key][None, idx_tr]
                obs = (va > 0) & (vb > 0)
                mm += (obs & (va != vb)).astype(np.int64)

            mm += (s_te["ret"][idx_te, None] != s_tr["ret"][None, idx_tr]).astype(np.int64)
            if include_bc_diff:
                mm += (s_te["bc"][idx_te, None] != s_tr["bc"][None, idx_tr]).astype(np.int64) * 2

            dba_a = s_te["dba_raw"][idx_te, None]
            dba_b = s_tr["dba_raw"][None, idx_tr]
            obs_dba = (~np.isnan(dba_a)) & (~np.isnan(dba_b))
            dba_scale = np.maximum(4.0, 0.045 * 0.5 * np.nan_to_num(dba_a + dba_b, nan=80.0))
            u_dba = np.where(obs_dba, np.abs(dba_a - dba_b) / dba_scale, 0.0)

            r_a = s_te["rate_raw"][idx_te, None]
            r_b = s_tr["rate_raw"][None, idx_tr]
            obs_r = (~np.isnan(r_a)) & (~np.isnan(r_b))
            r_scale = np.maximum(28000.0, 0.035 * 0.5 * np.nan_to_num(r_a + r_b, nan=1600000.0))
            u_rate = np.where(obs_r, np.abs(r_a - r_b) / r_scale, 0.0)

            d_cont_sq = np.minimum(u_dba ** 2, 25.0) + np.minimum(u_rate ** 2, 25.0)
            if include_dy_diff:
                dy_diff = (s_te["dy"][idx_te, None] - s_tr["dy"][None, idx_tr]) / 2.2
                d_cont_sq = d_cont_sq + np.minimum(dy_diff ** 2, 20.0)

            d_total_sq = d_cont_sq + mm.astype(np.float64) * 6.5 + np.where(sr_mm, 4.0, 0.0)
            return mm, sr_mm, u_dba, u_rate, d_cont_sq, d_total_sq

        # --- Konteks 1: (date_id, bc) ---
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
                u_dba_masked = u_dba
                u_rate_masked = u_rate

            y_sub = y_tr_f[idx_tr]

            # A. Imputasi Kontekstual Berbobot Kernel (Hanya Fitur X, Tanpa Label y)
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

            # B. Exact Zero-Mismatch Template Kernel (mm == 0, u_dba <= 2.4, u_rate <= 2.4)
            W_strict0 = np.where((mm == 0) & (u_dba <= 2.4) & (u_rate <= 2.4), np.exp(-0.5 * d_cont_sq), 0.0)
            ws0 = np.sum(W_strict0, axis=1)
            wy0 = W_strict0 @ y_sub
            strict_w_arr[idx_te] = ws0

            for k_m, m_prior in enumerate((0.35, 0.80, 1.60)):
                p_s0 = np.clip((wy0 + m_prior * p1) / (ws0 + m_prior), 1e-6, 1.0 - 1e-6)
                feats[idx_te, k_m] = np.log(p_s0 / (1.0 - p_s0)) - prior_lo

            # C. Near-Exact Template Kernel (mm <= 1, ~sr_mm)
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

            # D. Multi-bandwidth Parzen KDE pada d_total_sq
            for k_h, (h, m_p) in enumerate(zip((0.55, 0.90, 1.40, 2.20, 3.50), (0.8, 1.3, 2.0, 3.0, 4.5))):
                W = np.exp(-d_total_sq / (2.0 * (h ** 2)))
                w_sum = np.sum(W, axis=1)
                wy_sum = W @ y_sub
                p_est = np.clip((wy_sum + m_p * p1) / (w_sum + m_p), 1e-6, 1.0 - 1e-6)
                feats[idx_te, 5 + k_h] = np.log(p_est / (1.0 - p_est)) - prior_lo

            # E. Local Contrast terhadap Coarse Date_id LLR
            W_date = np.ones_like(d_total_sq)
            if is_self_loo:
                W_date = np.where(same_mask, 0.0, 1.0)
            p_date = np.clip(((W_date @ y_sub) + 3.5 * p1) / (np.sum(W_date, axis=1) + 3.5), 1e-6, 1.0 - 1e-6)
            llr_date = np.log(p_date / (1.0 - p_date)) - prior_lo
            gate0 = ws0 / (ws0 + 0.6)
            feats[idx_te, 10] = gate0 * (feats[idx_te, 0] - llr_date)
            feats[idx_te, 11] = (ws1 / (ws1 + 1.0)) * (feats[idx_te, 3] - llr_date)

            if extended_v6:
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

                # Cauchy heavy-tailed Parzen kernel density pada Konteks 1
                for k_i, (h_c, m_c) in enumerate(((0.45, 0.50), (0.95, 1.10), (1.80, 2.20))):
                    W_cauchy = 1.0 / ((1.0 + d_total_sq / (h_c ** 2)) ** 2)
                    if is_self_loo:
                        W_cauchy = np.where(same_mask, 0.0, W_cauchy)
                    w_s = np.sum(W_cauchy, axis=1)
                    wy_s = W_cauchy @ y_sub
                    p_c = np.clip((wy_s + m_c * p1) / (w_s + m_c), 1e-6, 1.0 - 1e-6)
                    feats[idx_te, 28 + k_i] = np.log(p_c / (1.0 - p_c)) - prior_lo

                # Peak kernel density ratio (kelas positif vs negatif)
                W_peak = np.exp(-0.45 * d_total_sq)
                if is_self_loo:
                    W_peak = np.where(same_mask, 0.0, W_peak)
                pos_mask = y_sub == 1.0
                neg_mask = y_sub == 0.0
                max_w_pos = np.max(W_peak[:, pos_mask], axis=1) if np.any(pos_mask) else np.zeros(len(idx_te))
                max_w_neg = np.max(W_peak[:, neg_mask], axis=1) if np.any(neg_mask) else np.zeros(len(idx_te))
                feats[idx_te, 31] = np.log((max_w_pos + 0.15 * p1) / (max_w_neg + 0.15 * (1.0 - p1))) - prior_lo

                # Multi-k Local Parzen Density (Top-3, Top-5, Top-8)
                for col_k, k_nn in ((32, 3), (42, 5), (43, 8)):
                    k_top = min(k_nn, d_total_sq.shape[1])
                    part_idx = np.argpartition(d_total_sq, k_top - 1, axis=1)[:, :k_top]
                    top_d = np.take_along_axis(d_total_sq, part_idx, axis=1)
                    top_y = y_sub[part_idx]
                    w_top = np.exp(-0.5 * top_d) * (top_d < 1e9)
                    p_top = np.clip(
                        (np.sum(w_top * top_y, axis=1) + 0.9 * p1) / (np.sum(w_top, axis=1) + 0.9),
                        1e-6,
                        1.0 - 1e-6,
                    )
                    feats[idx_te, col_k] = np.log(p_top / (1.0 - p_top)) - prior_lo

                # Context-Imputed Parzen KDE pada (date_id, bc)
                dba_i_te = np.where(np.isnan(dba_imp[idx_te]), glob_dba, dba_imp[idx_te])[:, None]
                dba_i_tr = np.nan_to_num(s_tr["dba_raw"][idx_tr], nan=glob_dba)[None, :]
                r_i_te = np.where(np.isnan(rate_imp[idx_te]), glob_rate, rate_imp[idx_te])[:, None]
                r_i_tr = np.nan_to_num(s_tr["rate_raw"][idx_tr], nan=glob_rate)[None, :]
                d_imp_sq = (
                    np.minimum(((dba_i_te - dba_i_tr) / 5.0) ** 2, 25.0)
                    + np.minimum(((r_i_te - r_i_tr) / 35000.0) ** 2, 25.0)
                    + mm.astype(np.float64) * 6.0
                )
                if is_self_loo:
                    d_imp_sq = np.where(same_mask, 1e12, d_imp_sq)
                for col_imp, (h_i, m_i) in ((44, (0.75, 1.0)), (45, (1.60, 2.2))):
                    W_i = np.exp(-d_imp_sq / (2.0 * (h_i ** 2)))
                    p_i = np.clip(((W_i @ y_sub) + m_i * p1) / (np.sum(W_i, axis=1) + m_i), 1e-6, 1.0 - 1e-6)
                    feats[idx_te, col_imp] = np.log(p_i / (1.0 - p_i)) - prior_lo

        dba_imp = np.where(np.isnan(dba_imp), glob_dba, dba_imp)
        rate_imp = np.where(np.isnan(rate_imp), glob_rate, rate_imp)
        sr_imp_int = np.clip(np.round(np.nan_to_num(sr_imp, nan=0.0)).astype(np.int64), 0, 5)
        sr_bin_imp = np.clip(sr_imp_int, 0, 3)
        rc_imp = np.where(rc_imp == 0, 2, rc_imp)
        mp_imp = np.where(mp_imp == 0, 1, mp_imp)

        # --- Konteks 2: (ym, bc, stay) ---
        b2_tr = (s_tr["ym"] * 10 + s_tr["bc"]) * 200 + s_tr["stay"]
        b2_te = (s_te["ym"] * 10 + s_te["bc"]) * 200 + s_te["stay"]

        order_tr2 = np.argsort(b2_tr)
        u_b2, start_tr2, count_tr2 = np.unique(b2_tr[order_tr2], return_index=True, return_counts=True)
        order_te2 = np.argsort(b2_te)
        u_b2_te, start_te2, count_te2 = np.unique(b2_te[order_te2], return_index=True, return_counts=True)

        pos_c2 = np.clip(np.searchsorted(u_b2, u_b2_te), 0, len(u_b2) - 1)
        valid_b2 = u_b2[pos_c2] == u_b2_te

        for idx_b in np.where(valid_b2)[0]:
            tr_pos = pos_c2[idx_b]
            idx_tr = order_tr2[start_tr2[tr_pos] : start_tr2[tr_pos] + count_tr2[tr_pos]]
            idx_te = order_te2[start_te2[idx_b] : start_te2[idx_b] + count_te2[idx_b]]

            if is_self_loo and len(idx_tr) <= 1:
                continue

            mm, sr_mm, u_dba, u_rate, d_cont_sq, d_total_sq = eval_pair_matrices(idx_te, idx_tr, include_dy_diff=True)
            if is_self_loo:
                same_mask = idx_te[:, None] == idx_tr[None, :]
                d_total_sq = np.where(same_mask, 1e12, d_total_sq)

            y_sub = y_tr_f[idx_tr]
            for k_h, (h, m_p) in enumerate(zip((0.65, 1.15, 1.90), (1.2, 2.2, 3.8))):
                W = np.exp(-d_total_sq / (2.0 * (h ** 2)))
                w_sum = np.sum(W, axis=1)
                wy_sum = W @ y_sub
                p_est = np.clip((wy_sum + m_p * p1) / (w_sum + m_p), 1e-6, 1.0 - 1e-6)
                feats[idx_te, 12 + k_h] = np.log(p_est / (1.0 - p_est)) - prior_lo

        # --- Konteks 3: (date_id) lintas channel ---
        b3_tr = s_tr["date_id"]
        b3_te = s_te["date_id"]
        order_tr3 = np.argsort(b3_tr)
        u_b3, start_tr3, count_tr3 = np.unique(b3_tr[order_tr3], return_index=True, return_counts=True)
        order_te3 = np.argsort(b3_te)
        u_b3_te, start_te3, count_te3 = np.unique(b3_te[order_te3], return_index=True, return_counts=True)

        pos_c3 = np.clip(np.searchsorted(u_b3, u_b3_te), 0, len(u_b3) - 1)
        valid_b3 = u_b3[pos_c3] == u_b3_te

        for idx_b in np.where(valid_b3)[0]:
            tr_pos = pos_c3[idx_b]
            idx_tr = order_tr3[start_tr3[tr_pos] : start_tr3[tr_pos] + count_tr3[tr_pos]]
            idx_te = order_te3[start_te3[idx_b] : start_te3[idx_b] + count_te3[idx_b]]

            if is_self_loo and len(idx_tr) <= 1:
                continue

            mm, sr_mm, u_dba, u_rate, d_cont_sq, d_total_sq = eval_pair_matrices(idx_te, idx_tr, include_bc_diff=True)
            if is_self_loo:
                same_mask = idx_te[:, None] == idx_tr[None, :]
                d_total_sq = np.where(same_mask, 1e12, d_total_sq)

            y_sub = y_tr_f[idx_tr]
            for k_h, (h, m_p) in enumerate(zip((0.65, 1.25, 2.30), (1.0, 2.0, 3.5))):
                W = np.exp(-d_total_sq / (2.0 * (h ** 2)))
                w_sum = np.sum(W, axis=1)
                wy_sum = W @ y_sub
                p_est = np.clip((wy_sum + m_p * p1) / (w_sum + m_p), 1e-6, 1.0 - 1e-6)
                feats[idx_te, 15 + k_h] = np.log(p_est / (1.0 - p_est)) - prior_lo

        # --- Konteks 4..7: Kohort Global, Adjacent Arrival Window, Booking-Day Window, & (ym, bc, sr_bin) ---
        if extended_v6:
            b4_tr = ((s_tr["ym"] * 10 + s_tr["bc"]) * 10 + s_tr["rc_raw"]) * 10 + s_tr["mp_raw"]
            b4_te = ((s_te["ym"] * 10 + s_te["bc"]) * 10 + s_te["rc_raw"]) * 10 + s_te["mp_raw"]
            order_tr4 = np.argsort(b4_tr)
            u_b4, start_tr4, count_tr4 = np.unique(b4_tr[order_tr4], return_index=True, return_counts=True)
            order_te4 = np.argsort(b4_te)
            u_b4_te, start_te4, count_te4 = np.unique(b4_te[order_te4], return_index=True, return_counts=True)

            pos_c4 = np.clip(np.searchsorted(u_b4, u_b4_te), 0, len(u_b4) - 1)
            valid_b4 = u_b4[pos_c4] == u_b4_te

            for idx_b in np.where(valid_b4)[0]:
                tr_pos = pos_c4[idx_b]
                idx_tr = order_tr4[start_tr4[tr_pos] : start_tr4[tr_pos] + count_tr4[tr_pos]]
                idx_te = order_te4[start_te4[idx_b] : start_te4[idx_b] + count_te4[idx_b]]
                if is_self_loo and len(idx_tr) <= 1:
                    continue
                mm, sr_mm, u_dba, u_rate, d_cont_sq, d_total_sq = eval_pair_matrices(idx_te, idx_tr, include_dy_diff=True)
                if is_self_loo:
                    same_mask = idx_te[:, None] == idx_tr[None, :]
                    d_total_sq = np.where(same_mask, 1e12, d_total_sq)
                y_sub = y_tr_f[idx_tr]
                for k_h, (h, m_p) in enumerate(zip((0.75, 1.35, 2.20), (1.5, 2.5, 4.0))):
                    W = np.exp(-d_total_sq / (2.0 * (h ** 2)))
                    w_sum = np.sum(W, axis=1)
                    wy_sum = W @ y_sub
                    p_est = np.clip((wy_sum + m_p * p1) / (w_sum + m_p), 1e-6, 1.0 - 1e-6)
                    feats[idx_te, 25 + k_h] = np.log(p_est / (1.0 - p_est)) - prior_lo

            # Konteks 5: Adjacent Arrival Window (blok 3-hari setengah tumpang tindih x bc)
            for shift_d, col_offset in ((0, 33), (1, 35)):
                b5_tr = ((s_tr["date_id"] + shift_d) // 3) * 10 + s_tr["bc"]
                b5_te = ((s_te["date_id"] + shift_d) // 3) * 10 + s_te["bc"]
                o_tr5 = np.argsort(b5_tr)
                u_b5, s_tr5, c_tr5 = np.unique(b5_tr[o_tr5], return_index=True, return_counts=True)
                o_te5 = np.argsort(b5_te)
                u_b5_te, s_te5, c_te5 = np.unique(b5_te[o_te5], return_index=True, return_counts=True)
                p_c5 = np.clip(np.searchsorted(u_b5, u_b5_te), 0, len(u_b5) - 1)
                v_b5 = u_b5[p_c5] == u_b5_te

                for idx_b in np.where(v_b5)[0]:
                    tr_pos = p_c5[idx_b]
                    idx_tr = o_tr5[s_tr5[tr_pos] : s_tr5[tr_pos] + c_tr5[tr_pos]]
                    idx_te = o_te5[s_te5[idx_b] : s_te5[idx_b] + c_te5[idx_b]]
                    if is_self_loo and len(idx_tr) <= 1:
                        continue
                    _, _, _, _, _, d_total_sq = eval_pair_matrices(idx_te, idx_tr, include_dy_diff=True)
                    if is_self_loo:
                        same_mask = idx_te[:, None] == idx_tr[None, :]
                        d_total_sq = np.where(same_mask, 1e12, d_total_sq)
                    y_sub = y_tr_f[idx_tr]
                    for k_h, (h, m_p) in enumerate(((0.70, 1.0), (1.40, 2.2))):
                        W = np.exp(-d_total_sq / (2.0 * (h ** 2)))
                        p_e = np.clip(((W @ y_sub) + m_p * p1) / (np.sum(W, axis=1) + m_p), 1e-6, 1.0 - 1e-6)
                        feats[idx_te, col_offset + k_h] = np.log(p_e / (1.0 - p_e)) - prior_lo

            # Konteks 6: Booking-Day Window (bday // 3, bc)
            tr_dba_f = np.nan_to_num(s_tr["dba_raw"], nan=glob_dba)
            bday_tr = (s_tr["yr"] - 2023) * 365.25 + (s_tr["mo"] - 1) * 30.4375 + s_tr["dy"] - tr_dba_f
            bday_te = (s_te["yr"] - 2023) * 365.25 + (s_te["mo"] - 1) * 30.4375 + s_te["dy"] - dba_imp

            for shift_bd, col_offset in ((0.0, 37), (1.5, 39)):
                b6_tr = np.floor((bday_tr + shift_bd) / 3.0).astype(np.int64) * 10 + s_tr["bc"]
                b6_te = np.floor((bday_te + shift_bd) / 3.0).astype(np.int64) * 10 + s_te["bc"]
                o_tr6 = np.argsort(b6_tr)
                u_b6, s_tr6, c_tr6 = np.unique(b6_tr[o_tr6], return_index=True, return_counts=True)
                o_te6 = np.argsort(b6_te)
                u_b6_te, s_te6, c_te6 = np.unique(b6_te[o_te6], return_index=True, return_counts=True)
                p_c6 = np.clip(np.searchsorted(u_b6, u_b6_te), 0, len(u_b6) - 1)
                v_b6 = u_b6[p_c6] == u_b6_te

                for idx_b in np.where(v_b6)[0]:
                    tr_pos = p_c6[idx_b]
                    idx_tr = o_tr6[s_tr6[tr_pos] : s_tr6[tr_pos] + c_tr6[tr_pos]]
                    idx_te = o_te6[s_te6[idx_b] : s_te6[idx_b] + c_te6[idx_b]]
                    if is_self_loo and len(idx_tr) <= 1:
                        continue
                    _, _, _, _, _, d_total_sq = eval_pair_matrices(idx_te, idx_tr, include_dy_diff=True)
                    if is_self_loo:
                        same_mask = idx_te[:, None] == idx_tr[None, :]
                        d_total_sq = np.where(same_mask, 1e12, d_total_sq)
                    y_sub = y_tr_f[idx_tr]
                    for k_h, (h, m_p) in enumerate(((0.75, 1.2), (1.50, 2.5))):
                        W = np.exp(-d_total_sq / (2.0 * (h ** 2)))
                        p_e = np.clip(((W @ y_sub) + m_p * p1) / (np.sum(W, axis=1) + m_p), 1e-6, 1.0 - 1e-6)
                        feats[idx_te, col_offset + k_h] = np.log(p_e / (1.0 - p_e)) - prior_lo

            # Konteks 7: (ym, bc, sr_bin)
            sr_tr_bin = np.clip(np.nan_to_num(s_tr["sr_raw"], nan=0.0).round().astype(np.int64), 0, 3)
            b7_tr = (s_tr["ym"] * 10 + s_tr["bc"]) * 10 + sr_tr_bin
            b7_te = (s_te["ym"] * 10 + s_te["bc"]) * 10 + sr_bin_imp
            o_tr7 = np.argsort(b7_tr)
            u_b7, s_tr7, c_tr7 = np.unique(b7_tr[o_tr7], return_index=True, return_counts=True)
            o_te7 = np.argsort(b7_te)
            u_b7_te, s_te7, c_te7 = np.unique(b7_te[o_te7], return_index=True, return_counts=True)
            p_c7 = np.clip(np.searchsorted(u_b7, u_b7_te), 0, len(u_b7) - 1)
            v_b7 = u_b7[p_c7] == u_b7_te

            for idx_b in np.where(v_b7)[0]:
                tr_pos = p_c7[idx_b]
                idx_tr = o_tr7[s_tr7[tr_pos] : s_tr7[tr_pos] + c_tr7[tr_pos]]
                idx_te = o_te7[s_te7[idx_b] : s_te7[idx_b] + c_te7[idx_b]]
                if is_self_loo and len(idx_tr) <= 1:
                    continue
                _, _, _, _, _, d_total_sq = eval_pair_matrices(idx_te, idx_tr)
                if is_self_loo:
                    same_mask = idx_te[:, None] == idx_tr[None, :]
                    d_total_sq = np.where(same_mask, 1e12, d_total_sq)
                y_sub = y_tr_f[idx_tr]
                W = np.exp(-d_total_sq / (2.0 * (1.10 ** 2)))
                p_e = np.clip(((W @ y_sub) + 1.8 * p1) / (np.sum(W, axis=1) + 1.8), 1e-6, 1.0 - 1e-6)
                feats[idx_te, 41] = np.log(p_e / (1.0 - p_e)) - prior_lo

            # Missingness-Penalized Sharp Template & Top-2 Local Parzen Kernels across 6 Contexts (cols 46..63)
            tr_dba_imp = np.where(np.isnan(s_tr["dba_raw"]), glob_dba, s_tr["dba_raw"])
            tr_rate_imp = np.where(np.isnan(s_tr["rate_raw"]), glob_rate, s_tr["rate_raw"])

            def eval_strict_pair(
                idx_te_s: np.ndarray, idx_tr_s: np.ndarray, day_scale: float = 0.0, include_bc: bool = False
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                k_a, k_b = len(idx_te_s), len(idx_tr_s)
                mm_s = np.zeros((k_a, k_b), dtype=np.int64)
                mm_s += (s_te["wknd"][idx_te_s, None] != s_tr["wknd"][None, idx_tr_s]).astype(np.int64)
                mm_s += (s_te["wkdy"][idx_te_s, None] != s_tr["wkdy"][None, idx_tr_s]).astype(np.int64)

                sr_a = s_te["sr_raw"][idx_te_s, None]
                sr_b = s_tr["sr_raw"][None, idx_tr_s]
                na_sr_a, na_sr_b = np.isnan(sr_a), np.isnan(sr_b)
                obs_sr = (~na_sr_a) & (~na_sr_b)
                sr_mm_s = (obs_sr & (sr_a != sr_b)) | (na_sr_a != na_sr_b)
                mm_s += sr_mm_s.astype(np.int64)

                for key in ("ad_raw", "ch_raw", "np_raw"):
                    va = s_te[key][idx_te_s, None]
                    vb = s_tr[key][None, idx_tr_s]
                    na_a, na_b = np.isnan(va), np.isnan(vb)
                    obs = (~na_a) & (~na_b)
                    mm_s += ((obs & (va != vb)) | (na_a != na_b)).astype(np.int64)

                for key in ("rc_raw", "mp_raw"):
                    va = s_te[key][idx_te_s, None]
                    vb = s_tr[key][None, idx_tr_s]
                    obs = (va > 0) & (vb > 0)
                    mm_s += ((obs & (va != vb)) | ((va == 0) != (vb == 0))).astype(np.int64)

                mm_s += (s_te["ret"][idx_te_s, None] != s_tr["ret"][None, idx_tr_s]).astype(np.int64)
                if include_bc:
                    mm_s += (s_te["bc"][idx_te_s, None] != s_tr["bc"][None, idx_tr_s]).astype(np.int64) * 2

                dba_a = dba_imp[idx_te_s, None]
                dba_b = tr_dba_imp[None, idx_tr_s]
                na_dba_mm = np.isnan(s_te["dba_raw"][idx_te_s, None]) != np.isnan(s_tr["dba_raw"][None, idx_tr_s])
                dba_scale = np.maximum(3.5, 0.040 * 0.5 * (dba_a + dba_b))
                u_dba_s = np.abs(dba_a - dba_b) / dba_scale + np.where(na_dba_mm, 1.2, 0.0)

                r_a = rate_imp[idx_te_s, None]
                r_b = tr_rate_imp[None, idx_tr_s]
                na_r_mm = np.isnan(s_te["rate_raw"][idx_te_s, None]) != np.isnan(s_tr["rate_raw"][None, idx_tr_s])
                r_scale = np.maximum(25000.0, 0.032 * 0.5 * (r_a + r_b))
                u_rate_s = np.abs(r_a - r_b) / r_scale + np.where(na_r_mm, 1.2, 0.0)

                d_cont_s = np.minimum(u_dba_s ** 2, 25.0) + np.minimum(u_rate_s ** 2, 25.0)
                if day_scale > 0.0:
                    dy_diff = (s_te["date_id"][idx_te_s, None] - s_tr["date_id"][None, idx_tr_s]) / day_scale
                    d_cont_s = d_cont_s + np.minimum(dy_diff ** 2, 20.0)

                d_tot_s = d_cont_s + mm_s.astype(np.float64) * 6.5 + np.where(sr_mm_s, 4.0, 0.0)
                return mm_s, sr_mm_s, u_dba_s, u_rate_s, d_cont_s, d_tot_s

            def run_block_kernels(
                b_tr_k: np.ndarray, b_te_k: np.ndarray, col_start: int, day_scale: float = 0.0, include_bc: bool = False
            ) -> None:
                o_tr_k = np.argsort(b_tr_k)
                u_b_k, s_tr_k, c_tr_k = np.unique(b_tr_k[o_tr_k], return_index=True, return_counts=True)
                o_te_k = np.argsort(b_te_k)
                u_b_te_k, s_te_k, c_te_k = np.unique(b_te_k[o_te_k], return_index=True, return_counts=True)
                p_c_k = np.clip(np.searchsorted(u_b_k, u_b_te_k), 0, len(u_b_k) - 1)
                v_b_k = u_b_k[p_c_k] == u_b_te_k

                for idx_b_k in np.where(v_b_k)[0]:
                    tr_pos_k = p_c_k[idx_b_k]
                    idx_tr_k = o_tr_k[s_tr_k[tr_pos_k] : s_tr_k[tr_pos_k] + c_tr_k[tr_pos_k]]
                    idx_te_k = o_te_k[s_te_k[idx_b_k] : s_te_k[idx_b_k] + c_te_k[idx_b_k]]
                    if is_self_loo and len(idx_tr_k) <= 1:
                        continue
                    mm_s, sr_mm_s, u_dba_s, u_rate_s, _, d_tot_s = eval_strict_pair(
                        idx_te_k, idx_tr_k, day_scale=day_scale, include_bc=include_bc
                    )
                    if is_self_loo:
                        same_m = idx_te_k[:, None] == idx_tr_k[None, :]
                        d_tot_s = np.where(same_m, 1e12, d_tot_s)
                        mm_s = np.where(same_m, 999, mm_s)
                    y_sub_k = y_tr_f[idx_tr_k]

                    W_s = np.where(
                        (mm_s <= 1) & (~sr_mm_s) & (u_dba_s <= 2.5) & (u_rate_s <= 2.5),
                        np.exp(-0.45 * d_tot_s) * np.where(mm_s == 0, 1.0, 0.30),
                        0.0,
                    )
                    ws_k = np.sum(W_s, axis=1)
                    p_s_k = np.clip(((W_s @ y_sub_k) + 0.65 * p1) / (ws_k + 0.65), 1e-6, 1.0 - 1e-6)
                    feats[idx_te_k, col_start] = np.log(p_s_k / (1.0 - p_s_k)) - prior_lo

                    k_top_k = min(2, d_tot_s.shape[1])
                    part_k = np.argpartition(d_tot_s, k_top_k - 1, axis=1)[:, :k_top_k]
                    top_d_k = np.take_along_axis(d_tot_s, part_k, axis=1)
                    top_y_k = y_sub_k[part_k]
                    w_top_k = np.exp(-0.5 * top_d_k) * (top_d_k < 1e9)
                    p_top_k = np.clip(
                        (np.sum(w_top_k * top_y_k, axis=1) + 0.8 * p1) / (np.sum(w_top_k, axis=1) + 0.8),
                        1e-6,
                        1.0 - 1e-6,
                    )
                    feats[idx_te_k, col_start + 1] = np.log(p_top_k / (1.0 - p_top_k)) - prior_lo

                    W_g = np.exp(-d_tot_s / (2.0 * (0.55 ** 2)))
                    p_g_k = np.clip(((W_g @ y_sub_k) + 0.8 * p1) / (np.sum(W_g, axis=1) + 0.8), 1e-6, 1.0 - 1e-6)
                    feats[idx_te_k, col_start + 2] = np.log(p_g_k / (1.0 - p_g_k)) - prior_lo

            run_block_kernels(s_tr["date_id"] * 10 + s_tr["bc"], s_te["date_id"] * 10 + s_te["bc"], 46, day_scale=0.0)
            run_block_kernels(
                (s_tr["ym"] * 10 + s_tr["bc"]) * 200 + s_tr["stay"],
                (s_te["ym"] * 10 + s_te["bc"]) * 200 + s_te["stay"],
                49,
                day_scale=2.0,
            )
            run_block_kernels(s_tr["date_id"], s_te["date_id"], 52, day_scale=0.0, include_bc=True)
            tr_rc_i = np.where(s_tr["rc_raw"] == 0, 2, s_tr["rc_raw"])
            tr_mp_i = np.where(s_tr["mp_raw"] == 0, 1, s_tr["mp_raw"])
            run_block_kernels(
                ((s_tr["ym"] * 10 + s_tr["bc"]) * 10 + tr_rc_i) * 10 + tr_mp_i,
                ((s_te["ym"] * 10 + s_te["bc"]) * 10 + rc_imp) * 10 + mp_imp,
                55,
                day_scale=2.5,
            )
            bday_tr_s = (s_tr["yr"] - 2023) * 365.25 + (s_tr["mo"] - 1) * 30.4375 + s_tr["dy"] - tr_dba_imp
            bday_te_s = (s_te["yr"] - 2023) * 365.25 + (s_te["mo"] - 1) * 30.4375 + s_te["dy"] - dba_imp
            run_block_kernels(
                np.floor(bday_tr_s / 2.0).astype(np.int64) * 10 + s_tr["bc"],
                np.floor(bday_te_s / 2.0).astype(np.int64) * 10 + s_te["bc"],
                58,
                day_scale=3.0,
            )
            run_block_kernels(
                np.floor((bday_tr_s + 1.0) / 2.0).astype(np.int64) * 10 + s_tr["bc"],
                np.floor((bday_te_s + 1.0) / 2.0).astype(np.int64) * 10 + s_te["bc"],
                61,
                day_scale=3.0,
            )

        ctx_imp_out = {
            "ym": s_te["ym"],
            "date_id": s_te["date_id"],
            "bc": s_te["bc"],
            "rc_imp": rc_imp,
            "mp_imp": mp_imp,
            "sr_imp": sr_imp_int,
            "sr_bin_imp": sr_bin_imp,
            "stay": s_te["stay"],
            "tot_nights": s_te["tot_nights"],
            "dba_imp": dba_imp,
            "rate_imp": rate_imp,
        }
        return feats, ctx_imp_out, strict_w_arr

    @staticmethod
    def _extract_imputed_cliques(
        ctx: Dict[str, np.ndarray],
        s_te: Optional[Dict[str, np.ndarray]] = None,
        kde_full: Optional[np.ndarray] = None,
        s_tr_ref: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Membangun 47 Context-Completed, Booking-Day, Unsupervised Cohort & Template-Geometry
        Bayesian Contingency Cliques murni dari fitur X (tanpa target y) dengan Laplace/Dirichlet
        m-estimate smoothing [Sahami, KDD 1996; Pazzani, Machine Learning 1996].
        """
        n = len(ctx["ym"])
        ym = ctx["ym"]
        date_id = ctx["date_id"]
        bc = ctx["bc"]
        rc = ctx["rc_imp"]
        mp = ctx["mp_imp"]
        sr = ctx["sr_imp"]
        sr_bin = ctx["sr_bin_imp"]
        stay = ctx["stay"]
        dba_imp = ctx["dba_imp"]
        rate_imp = ctx["rate_imp"]

        def bin_dba_i(step: float, shift: float = 0.0) -> np.ndarray:
            return np.floor((dba_imp + shift) / step).astype(np.int64) + 1

        def bin_rate_i(step: float, shift: float = 0.0) -> np.ndarray:
            return np.floor((rate_imp + shift) / step).astype(np.int64) + 1

        d4 = bin_dba_i(4.0, 0.0)
        d4s = bin_dba_i(4.0, 2.0)
        d6 = bin_dba_i(6.0, 0.0)
        d6s = bin_dba_i(6.0, 3.0)
        d8 = bin_dba_i(8.0, 0.0)
        d8s = bin_dba_i(8.0, 4.0)
        d12 = bin_dba_i(12.0, 0.0)
        d12s = bin_dba_i(12.0, 6.0)
        d15 = bin_dba_i(15.0, 0.0)
        d15s = bin_dba_i(15.0, 7.5)

        r60 = bin_rate_i(60000.0, 0.0)
        r60s = bin_rate_i(60000.0, 30000.0)
        r80 = bin_rate_i(80000.0, 0.0)
        r80s = bin_rate_i(80000.0, 40000.0)
        r100 = bin_rate_i(100000.0, 0.0)
        r100s = bin_rate_i(100000.0, 50000.0)
        r160 = bin_rate_i(160000.0, 0.0)
        r160s = bin_rate_i(160000.0, 80000.0)

        def pack(*arrays: np.ndarray) -> np.ndarray:
            res = np.zeros(n, dtype=np.int64)
            for a in arrays:
                res = res * 1009 + a
            return res

        base_cliques: List[Tuple[np.ndarray, float]] = [
            (pack(date_id, bc, d8, r160), 2.5),
            (pack(date_id, bc, d8s, r160s), 2.5),
            (pack(date_id, bc, d12, r100, sr_bin), 2.5),
            (pack(date_id, bc, d12s, r100s, sr_bin), 2.5),
            (pack(date_id, bc, d8, sr_bin), 3.0),
            (pack(date_id, bc, d8s, sr_bin), 3.0),
            (pack(date_id, bc, r100, sr_bin), 3.0),
            (pack(date_id, bc, r100s, sr_bin), 3.0),
            (pack(date_id, bc, d15, r160, stay), 2.0),
            (pack(date_id, bc, d15s, r160s, stay), 2.0),
            (pack(ym, bc, d8, r100, sr_bin), 3.5),
            (pack(ym, bc, d8s, r100s, sr_bin), 3.5),
            (pack(d4, r60, bc, sr_bin), 3.5),
            (pack(d4s, r60s, bc, sr_bin), 3.5),
            (pack(d8, r100, bc, rc, mp), 3.5),
            (pack(d8s, r100s, bc, rc, mp), 3.5),
            (pack(bc, sr, d8), 5.0),
            (pack(bc, sr, d8s), 5.0),
            (pack(bc, sr, r100), 5.0),
            (pack(bc, sr, r100s), 5.0),
        ]

        if s_te is None or kde_full is None or s_tr_ref is None or kde_full.shape[1] < 23:
            return base_cliques

        ret = s_te["ret"]
        pc = np.clip(s_te["pc"], 0, 3)
        ad = np.clip(np.nan_to_num(s_te["ad_raw"], nan=2.0).round().astype(np.int64), 0, 4)
        ch = np.clip(np.nan_to_num(s_te["ch_raw"], nan=0.0).round().astype(np.int64), 0, 3)
        np_col = np.clip(np.nan_to_num(s_te["np_raw"], nan=0.0).round().astype(np.int64), 0, 1)

        approx_arrival_days = (s_te["yr"] - 2023) * 365.25 + (s_te["mo"] - 1) * 30.4375 + s_te["dy"]
        bday = approx_arrival_days - dba_imp

        def bin_bday(step: float, shift: float = 0.0) -> np.ndarray:
            return np.floor((bday + shift) / step).astype(np.int64) + 1000

        bd2 = bin_bday(2.0, 0.0)
        bd2s = bin_bday(2.0, 1.0)
        bd3 = bin_bday(3.0, 0.0)
        bd3s = bin_bday(3.0, 1.5)
        bd6 = bin_bday(6.0, 0.0)
        bd6s = bin_bday(6.0, 3.0)
        bd12 = bin_bday(12.0, 0.0)
        bd12s = bin_bday(12.0, 6.0)

        # Unsupervised cohort statistics from s_tr_ref (murni fitur X tanpa y)
        tr_dba_clean = np.nan_to_num(s_tr_ref["dba_raw"], nan=58.0)
        tr_rate_clean = np.nan_to_num(s_tr_ref["rate_raw"], nan=1644000.0)
        b1_tr = s_tr_ref["date_id"] * 10 + s_tr_ref["bc"]
        b1_te = s_te["date_id"] * 10 + s_te["bc"]
        u_b1, inv_b1, cnt_b1 = np.unique(b1_tr, return_inverse=True, return_counts=True)
        pos_b1 = np.clip(np.searchsorted(u_b1, b1_te), 0, len(u_b1) - 1)
        m_b1 = u_b1[pos_b1] == b1_te
        date_bc_cnt = np.where(m_b1, cnt_b1[pos_b1], 0).astype(np.float64)

        sum_dba1 = np.bincount(inv_b1, weights=tr_dba_clean)
        mean_dba1 = np.where(m_b1, sum_dba1[pos_b1] / np.maximum(cnt_b1[pos_b1], 1), dba_imp)
        dba_diff = np.abs(dba_imp - mean_dba1)

        sum_r1 = np.bincount(inv_b1, weights=tr_rate_clean)
        mean_r1 = np.where(m_b1, sum_r1[pos_b1] / np.maximum(cnt_b1[pos_b1], 1), rate_imp)
        rate_diff = np.abs(rate_imp - mean_r1)

        bs_tr = (s_tr_ref["date_id"] * 10 + s_tr_ref["bc"]) * 200 + s_tr_ref["stay"]
        bs_te = (s_te["date_id"] * 10 + s_te["bc"]) * 200 + s_te["stay"]
        u_bs, cnt_bs = np.unique(bs_tr, return_counts=True)
        pos_bs = np.clip(np.searchsorted(u_bs, bs_te), 0, len(u_bs) - 1)
        date_bc_stay_cnt = np.where(u_bs[pos_bs] == bs_te, cnt_bs[pos_bs], 0).astype(np.float64)

        cnt_bin = np.digitize(date_bc_cnt, [1, 2, 3, 5, 8, 12, 18, 25, 35]).astype(np.int64)
        stay_cnt_bin = np.digitize(date_bc_stay_cnt, [1, 2, 3, 5, 8, 12]).astype(np.int64)
        dba_diff_bin = np.digitize(dba_diff, [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]).astype(np.int64)
        rate_diff_bin = np.digitize(rate_diff, [20000.0, 50000.0, 100000.0, 200000.0, 400000.0]).astype(np.int64)

        # Unsupervised geometric template density bins (murni jarak fitur X)
        ws0 = kde_full[:, 18]
        ws1 = kde_full[:, 19]
        min_d = kde_full[:, 20]
        min_udba = kde_full[:, 21]
        min_urate = kde_full[:, 22]
        ws0_bin = np.digitize(ws0, [0.01, 0.2, 0.6, 1.2, 2.5, 5.0]).astype(np.int64)
        ws1_bin = np.digitize(ws1, [0.01, 0.3, 0.8, 1.8, 3.5, 7.0]).astype(np.int64)
        mind_bin = np.digitize(min_d, [0.15, 0.4, 0.9, 1.8, 3.5, 7.0, 15.0]).astype(np.int64)
        udba_bin = np.digitize(min_udba, [0.1, 0.3, 0.7, 1.5, 3.0, 6.0]).astype(np.int64)
        urate_bin = np.digitize(min_urate, [0.1, 0.3, 0.7, 1.5, 3.0, 6.0]).astype(np.int64)

        base_cliques.extend([
            (pack(bd2, bc, sr_bin), 2.5),
            (pack(bd2s, bc, sr_bin), 2.5),
            (pack(bd3, bc, sr_bin), 3.0),
            (pack(bd3s, bc, sr_bin), 3.0),
            (pack(bd6, bc, sr_bin, stay), 2.5),
            (pack(bd6s, bc, sr_bin, stay), 2.5),
            (pack(bd6, bc, r100), 3.0),
            (pack(bd6s, bc, r100s), 3.0),
            (pack(bd12, bc, rc, mp), 3.0),
            (pack(bd12s, bc, rc, mp), 3.0),
            (pack(bd3, bc), 4.0),
            (pack(bd3s, bc), 4.0),
            (pack(date_id, bc, dba_diff_bin, rate_diff_bin), 2.5),
            (pack(date_id, bc, ws0_bin, sr_bin), 2.5),
            (pack(date_id, bc, mind_bin, sr_bin), 2.5),
            (pack(bc, cnt_bin, stay_cnt_bin, sr_bin), 4.0),
            (pack(bc, ws0_bin, ws1_bin, mind_bin), 4.0),
            (pack(bc, udba_bin, urate_bin, sr_bin), 4.0),
            (pack(bc, dba_diff_bin, rate_diff_bin, sr_bin), 4.0),
            (pack(ym, bc, cnt_bin, sr_bin), 4.0),
            (pack(d6, r80, bc, rc, mp, sr_bin), 2.5),
            (pack(d6s, r80s, bc, rc, mp, sr_bin), 2.5),
            (pack(bc, ret, pc, sr_bin, ws0_bin), 4.0),
            (pack(bc, sr, np_col, ret, mind_bin), 4.0),
            (pack(bc, ad, ch, rc, mp, sr_bin), 4.0),
            (pack(date_id, bc, stay_cnt_bin, sr_bin), 3.0),
            (pack(ym, bc, stay_cnt_bin, rc, mp), 3.5),
        ])
        return base_cliques

    def _extract_multi_resolution_cliques(self, X_arr: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Mengekstrak 91 clique interaksi berorde tinggi (Path A: Raw Missingness Cliques)
        dengan skema Multi-Resolution & Half-Step Shifted Discretization murni NumPy.
        """
        s = self._extract_raw_struct(X_arr)
        n = len(s["ym"])
        ym = s["ym"]
        date_id = s["date_id"]
        bc = s["bc"]
        rc = s["rc_raw"]
        mp = s["mp_raw"]
        wknd = s["wknd"]
        wkdy = s["wkdy"]
        stay = s["stay"]
        tot_nights = s["tot_nights"]
        ret = s["ret"]
        pc = s["pc"]
        po = s["po"]

        sr_miss = np.isnan(s["sr_raw"])
        sr_clean = np.nan_to_num(s["sr_raw"], nan=0.0).round().astype(np.int64)
        sr = np.where(sr_miss, 6, np.clip(sr_clean, 0, 5))
        sr_bin = np.where(sr_miss, 4, np.clip(sr_clean, 0, 3))
        sr_zero = np.where(sr_miss, 2, (sr_clean > 0).astype(np.int64))

        np_miss = np.isnan(s["np_raw"])
        np_clean = np.nan_to_num(s["np_raw"], nan=0.0).round().astype(np.int64)
        np_col = np.where(np_miss, 2, np.clip(np_clean, 0, 1))

        ad_miss = np.isnan(s["ad_raw"])
        ad_clean = np.nan_to_num(s["ad_raw"], nan=2.0).round().astype(np.int64)
        ad = np.where(ad_miss, 5, np.clip(ad_clean, 0, 4))

        ch_miss = np.isnan(s["ch_raw"])
        ch_clean = np.nan_to_num(s["ch_raw"], nan=0.0).round().astype(np.int64)
        ch = np.where(ch_miss, 4, np.clip(ch_clean, 0, 3))
        guests = ad * 5 + ch

        dba_val = np.where(np.isnan(s["dba_raw"]), -999.0, s["dba_raw"])

        def bin_dba(step: float, shift: float = 0.0) -> np.ndarray:
            return np.where(dba_val < 0, 0, np.floor((dba_val + shift) / step).astype(np.int64) + 1)

        dba_1 = bin_dba(1.0, 0.0)
        dba_2 = bin_dba(2.0, 0.0)
        dba_2s = bin_dba(2.0, 1.0)
        dba_4 = bin_dba(4.0, 0.0)
        dba_4s = bin_dba(4.0, 2.0)
        dba_6 = bin_dba(6.0, 0.0)
        dba_6s = bin_dba(6.0, 3.0)
        dba_8 = bin_dba(8.0, 0.0)
        dba_8s = bin_dba(8.0, 4.0)
        dba_12 = bin_dba(12.0, 0.0)
        dba_12s = bin_dba(12.0, 6.0)
        dba_15 = bin_dba(15.0, 0.0)
        dba_15s = bin_dba(15.0, 7.5)
        dba_24 = bin_dba(24.0, 0.0)
        dba_24s = bin_dba(24.0, 12.0)
        dba_30 = bin_dba(30.0, 0.0)

        rate_val = np.where(np.isnan(s["rate_raw"]), -999.0, s["rate_raw"])

        def bin_rate(step: float, shift: float = 0.0) -> np.ndarray:
            return np.where(rate_val <= 0, 0, np.floor((rate_val + shift) / step).astype(np.int64) + 1)

        r_25k = bin_rate(25000.0, 0.0)
        r_30k = bin_rate(30000.0, 0.0)
        r_50k = bin_rate(50000.0, 0.0)
        r_50ks = bin_rate(50000.0, 25000.0)
        r_60k = bin_rate(60000.0, 0.0)
        r_60ks = bin_rate(60000.0, 30000.0)
        r_80k = bin_rate(80000.0, 0.0)
        r_80ks = bin_rate(80000.0, 40000.0)
        r_100k = bin_rate(100000.0, 0.0)
        r_100ks = bin_rate(100000.0, 50000.0)
        r_160k = bin_rate(160000.0, 0.0)
        r_160ks = bin_rate(160000.0, 80000.0)
        r_250k = bin_rate(250000.0, 0.0)
        r_250ks = bin_rate(250000.0, 125000.0)

        def pack(*arrays: np.ndarray) -> np.ndarray:
            res = np.zeros(n, dtype=np.int64)
            for a in arrays:
                res = res * 1009 + a
            return res

        return [
            (pack(bc, sr, dba_8), 5.0),
            (pack(bc, sr, dba_8s), 5.0),
            (pack(bc, sr, dba_4), 5.0),
            (pack(bc, sr, dba_4s), 5.0),
            (pack(bc, sr, dba_6), 5.0),
            (pack(bc, sr, dba_6s), 5.0),
            (pack(bc, sr, r_100k), 5.0),
            (pack(bc, sr, r_100ks), 5.0),
            (pack(bc, sr, r_60k), 5.0),
            (pack(bc, sr, r_60ks), 5.0),
            (pack(ym, bc, sr), 5.0),
            (pack(ym, bc, dba_8), 5.0),
            (pack(ym, bc, dba_8s), 5.0),
            (pack(ym, bc, dba_12), 5.0),
            (pack(ym, bc, dba_12s), 5.0),
            (pack(ym, bc, r_100k), 5.0),
            (pack(ym, bc, r_100ks), 5.0),
            (pack(ym, bc, dba_15, r_160k), 4.0),
            (pack(ym, bc, dba_15s, r_160ks), 4.0),
            (pack(ym, bc, dba_15, r_160k, sr_bin), 4.0),
            (pack(ym, bc, dba_15s, r_160ks, sr_bin), 4.0),
            (pack(date_id, bc), 5.0),
            (pack(date_id, bc, sr_bin), 4.0),
            (pack(date_id, bc, sr_zero), 4.0),
            (pack(date_id, bc, dba_6), 3.0),
            (pack(date_id, bc, dba_6s), 3.0),
            (pack(date_id, bc, dba_8), 3.5),
            (pack(date_id, bc, dba_8s), 3.5),
            (pack(date_id, bc, dba_12), 3.5),
            (pack(date_id, bc, dba_12s), 3.5),
            (pack(date_id, bc, dba_15), 4.0),
            (pack(date_id, bc, dba_15s), 4.0),
            (pack(date_id, bc, r_60k), 3.0),
            (pack(date_id, bc, r_60ks), 3.0),
            (pack(date_id, bc, r_100k), 3.5),
            (pack(date_id, bc, r_100ks), 3.5),
            (pack(date_id, bc, r_160k), 4.0),
            (pack(date_id, bc, r_160ks), 4.0),
            (pack(date_id, bc, dba_8, r_160k), 2.0),
            (pack(date_id, bc, dba_8s, r_160ks), 2.0),
            (pack(date_id, bc, dba_12, r_160k, sr_zero), 2.0),
            (pack(date_id, bc, dba_12s, r_160ks, sr_zero), 2.0),
            (pack(date_id, bc, dba_15, r_250k), 2.5),
            (pack(date_id, bc, dba_15s, r_250ks), 2.5),
            (pack(date_id, bc, stay), 3.5),
            (pack(date_id, bc, tot_nights, sr_zero), 3.0),
            (pack(date_id, bc, rc, mp), 3.0),
            (pack(date_id, bc, guests), 3.0),
            (pack(date_id, dba_4, bc), 2.5),
            (pack(date_id, dba_4s, bc), 2.5),
            (pack(date_id, bc, stay, mp, sr_bin), 2.0),
            (pack(date_id, bc, stay, rc, sr_bin), 2.0),
            (pack(date_id, bc, stay, ad, sr_bin), 2.0),
            (pack(date_id, bc, stay, mp, rc, ad), 2.0),
            (pack(date_id, bc, dba_12, stay, sr_bin), 2.0),
            (pack(date_id, bc, dba_12s, stay, sr_bin), 2.0),
            (pack(date_id, bc, dba_12, mp, rc), 2.0),
            (pack(date_id, bc, dba_12s, mp, rc), 2.0),
            (pack(date_id, bc, r_80k, stay, sr_bin), 2.0),
            (pack(date_id, bc, r_80ks, stay, sr_bin), 2.0),
            (pack(date_id, bc, r_80k, mp, rc), 2.0),
            (pack(date_id, bc, r_80ks, mp, rc), 2.0),
            (pack(date_id, bc, dba_15, r_160k, stay), 1.8),
            (pack(date_id, bc, dba_15s, r_160ks, stay), 1.8),
            (pack(date_id, bc, dba_15, r_160k, ad, mp), 1.8),
            (pack(date_id, bc, dba_15s, r_160ks, ad, mp), 1.8),
            (pack(dba_4, r_60k, bc), 3.5),
            (pack(dba_4s, r_60ks, bc), 3.5),
            (pack(dba_6, r_50k, bc, sr_zero), 3.0),
            (pack(dba_6s, r_50ks, bc, sr_zero), 3.0),
            (pack(dba_8, r_80k, bc, stay), 3.0),
            (pack(dba_8s, r_80ks, bc, stay), 3.0),
            (pack(dba_8, r_100k, bc, sr_bin), 3.5),
            (pack(dba_8s, r_100ks, bc, sr_bin), 3.5),
            (pack(dba_12, r_100k, bc, rc, sr_zero), 3.0),
            (pack(dba_12s, r_100ks, bc, rc, sr_zero), 3.0),
            (pack(dba_15, r_160k, rc, mp, bc), 3.5),
            (pack(dba_15s, r_160ks, rc, mp, bc), 3.5),
            (pack(dba_24, r_160k, bc, stay, sr_zero), 3.5),
            (pack(dba_24s, r_160ks, bc, stay, sr_zero), 3.5),
            (pack(sr, np_col, ret), 8.0),
            (pack(ret, pc, po, bc), 5.0),
            (pack(guests, rc, mp, bc), 5.0),
            (pack(stay, bc, sr_bin), 5.0),
            (pack(tot_nights, rc, mp, sr_bin), 5.0),
            (pack(dba_2, bc, sr_bin), 4.0),
            (pack(dba_2s, bc, sr_bin), 4.0),
            (pack(r_30k, bc, rc), 4.0),
            (pack(r_25k, bc, sr_zero), 4.0),
            (pack(dba_1, bc), 4.0),
            (pack(dba_30, bc, rc, mp, stay), 4.0),
        ]

    @staticmethod
    def _fit_cliques_llr_tables(
        cliques_tr: List[Tuple[np.ndarray, float]], y_tr: np.ndarray
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float]:
        """Mempelajari tabel Log-Likelihood Ratio (LLR) untuk setiap clique frekuensi Bayesian."""
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
        return tables, prior_lo

    @staticmethod
    def _fit_and_10fold_oof_cliques_llr_tables(
        cliques_tr: List[Tuple[np.ndarray, float]],
        y_tr: np.ndarray,
        pos_folds: List[np.ndarray],
        neg_folds: List[np.ndarray],
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, float]:
        """
        Menghitung secara vektorial cepat O(N):
        1. Tabel penuh Log-Likelihood Ratio (LLR) pada 100% data latih untuk inferensi test.
        2. Matriks 10-Fold Stratified Out-Of-Fold LLR melalui pengurangan frekuensi fold
           (n_tot_in_tr = n_tot_full - n_tot_fold_k), mencegah sign-flip LOO pada bin N=2
           dan menjaga keselarasan distribusi terhadap test set.
        """
        n_tr = len(y_tr)
        n_c = len(cliques_tr)
        k_folds = len(pos_folds)
        p1 = min(max(float(np.mean(y_tr)), 1e-6), 1.0 - 1e-6)
        prior_lo = math.log(p1 / (1.0 - p1))
        tables: List[Tuple[np.ndarray, np.ndarray]] = []
        M_oof = np.zeros((n_tr, n_c), dtype=np.float64)
        y_float = y_tr.astype(np.float64)
        fold_indices = [np.concatenate([pos_folds[k], neg_folds[k]]) for k in range(k_folds)]

        for j, (c_tr, m) in enumerate(cliques_tr):
            u_vals, inv = np.unique(c_tr, return_inverse=True)
            n_u = len(u_vals)
            n_tot = np.bincount(inv, minlength=n_u).astype(np.float64)
            n_pos = np.bincount(inv, weights=y_float, minlength=n_u)

            p_est = np.clip((n_pos + m * p1) / (n_tot + m), 1e-6, 1.0 - 1e-6)
            llr_full = np.log(p_est / (1.0 - p_est)) - prior_lo
            tables.append((u_vals, llr_full))

            m_fold = m * ((k_folds - 1.0) / k_folds)
            for va_k in fold_indices:
                inv_k = inv[va_k]
                tot_k = np.bincount(inv_k, minlength=n_u).astype(np.float64)
                pos_k = np.bincount(inv_k, weights=y_float[va_k], minlength=n_u)
                tot_tr_k = n_tot[inv_k] - tot_k[inv_k]
                pos_tr_k = n_pos[inv_k] - pos_k[inv_k]
                p_k = np.clip((pos_tr_k + m_fold * p1) / (tot_tr_k + m_fold), 1e-6, 1.0 - 1e-6)
                M_oof[va_k, j] = np.where(tot_tr_k > 0, np.log(p_k / (1.0 - p_k)) - prior_lo, 0.0)

        return tables, M_oof, prior_lo

    @staticmethod
    def _transform_cliques_llr_matrix(
        cliques_te: List[Tuple[np.ndarray, float]],
        tables: List[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Mengevaluasi matriks Log-Likelihood Ratio (LLR) untuk seluruh clique pada data uji."""
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
    def _compute_cal_ord_and_wd(yr: np.ndarray, mo: np.ndarray, dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Menghitung hari kalender Julian eksak (memperhitungkan tahun kabisat 2024) dan weekday (0=Senin..6=Minggu)."""
        n = len(yr)
        cal = np.empty(n, dtype=np.int64)
        wd = np.empty(n, dtype=np.int64)
        for idx in range(n):
            c_v, w_v = _CAL_ORD_LUT.get((int(yr[idx]), int(mo[idx]), int(dy[idx])), (739000, 0))
            cal[idx] = c_v
            wd[idx] = w_v
        return cal, wd

    @staticmethod
    def _compute_true_loo_cohort_stats(
        s_tr_ref: Dict[str, np.ndarray],
        ctx_tr_ref: Dict[str, np.ndarray],
        s_cur: Dict[str, np.ndarray],
        ctx_cur: Dict[str, np.ndarray],
        is_self_loo: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Menghitung statistik densitas & z-score kohort bebas bias self-counting
        (True Leave-One-Out dan normalisasi ukuran sampel referensi).
        """
        n_ref = float(len(s_tr_ref["ym"]))
        scale = 27300.0 / max((n_ref - 1.0) if is_self_loo else n_ref, 1.0)

        tr_dba = ctx_tr_ref["dba_imp"]
        te_dba = ctx_cur["dba_imp"]
        tr_rate = ctx_tr_ref["rate_imp"]
        te_rate = ctx_cur["rate_imp"]

        def loo_cnt(tr_k: np.ndarray, te_k: np.ndarray) -> np.ndarray:
            u_k, c_k = np.unique(tr_k, return_counts=True)
            pos = np.clip(np.searchsorted(u_k, te_k), 0, len(u_k) - 1)
            raw = np.where(u_k[pos] == te_k, c_k[pos], 0).astype(np.float64)
            if is_self_loo:
                raw = np.maximum(0.0, raw - 1.0)
            return raw * scale

        date_all_cnt = loo_cnt(s_tr_ref["date_id"], s_cur["date_id"])

        b1_tr = s_tr_ref["date_id"] * 10 + s_tr_ref["bc"]
        b1_te = s_cur["date_id"] * 10 + s_cur["bc"]
        u_b1, inv_b1, cnt_b1 = np.unique(b1_tr, return_inverse=True, return_counts=True)
        pos_b1 = np.clip(np.searchsorted(u_b1, b1_te), 0, len(u_b1) - 1)
        m_b1 = u_b1[pos_b1] == b1_te
        c_b1_raw = np.where(m_b1, cnt_b1[pos_b1], 0).astype(np.float64)
        c_b1_eff = np.maximum(0.0, c_b1_raw - (1.0 if is_self_loo else 0.0))
        date_bc_cnt = c_b1_eff * scale

        s1_d = np.where(m_b1, np.bincount(inv_b1, weights=tr_dba)[pos_b1], 0.0)
        s2_d = np.where(m_b1, np.bincount(inv_b1, weights=tr_dba ** 2)[pos_b1], 0.0)
        s1_r = np.where(m_b1, np.bincount(inv_b1, weights=tr_rate)[pos_b1], 0.0)
        s2_r = np.where(m_b1, np.bincount(inv_b1, weights=tr_rate ** 2)[pos_b1], 0.0)
        if is_self_loo:
            s1_d = np.where(c_b1_raw >= 1.0, s1_d - te_dba, s1_d)
            s2_d = np.where(c_b1_raw >= 1.0, s2_d - te_dba ** 2, s2_d)
            s1_r = np.where(c_b1_raw >= 1.0, s1_r - te_rate, s1_r)
            s2_r = np.where(c_b1_raw >= 1.0, s2_r - te_rate ** 2, s2_r)

        ok_b1 = c_b1_eff >= 2.0
        m_d = np.where(ok_b1, s1_d / np.maximum(c_b1_eff, 1.0), te_dba)
        v_d = np.where(ok_b1, np.maximum(0.0, s2_d / np.maximum(c_b1_eff, 1.0) - m_d ** 2), 100.0)
        z_dba = np.clip((te_dba - m_d) / (np.sqrt(v_d) + 5.0), -6.0, 6.0)

        m_r = np.where(ok_b1, s1_r / np.maximum(c_b1_eff, 1.0), te_rate)
        v_r = np.where(ok_b1, np.maximum(0.0, s2_r / np.maximum(c_b1_eff, 1.0) - m_r ** 2), 1e10)
        z_rate = np.clip((te_rate - m_r) / (np.sqrt(v_r) + 50000.0), -6.0, 6.0)

        bs_tr = (s_tr_ref["date_id"] * 10 + s_tr_ref["bc"]) * 200 + s_tr_ref["stay"]
        bs_te = (s_cur["date_id"] * 10 + s_cur["bc"]) * 200 + s_cur["stay"]
        date_bc_stay_cnt = loo_cnt(bs_tr, bs_te)
        return date_all_cnt, date_bc_cnt, date_bc_stay_cnt, z_dba, z_rate

    @classmethod
    def _build_calibrated_cll_basis(
        cls,
        tan_llr: np.ndarray,
        M_raw: np.ndarray,
        M_imp: np.ndarray,
        kde_full: np.ndarray,
        strict_w: np.ndarray,
        s_cur: Optional[Dict[str, np.ndarray]] = None,
        ctx_cur: Optional[Dict[str, np.ndarray]] = None,
        s_tr_ref: Optional[Dict[str, np.ndarray]] = None,
        ctx_tr_ref: Optional[Dict[str, np.ndarray]] = None,
        is_self_loo: bool = False,
    ) -> np.ndarray:
        """
        Menyusun matriks faktor log-likelihood Bayesian terkalibrasi (TAN + Raw Cliques +
        Context-Imputed Cliques + 7-Context Parzen KDE + Unsupervised Cohort Outlier Gating +
        Piecewise Log-Odds Calibration) untuk Discriminative CLL / WANBIA Parameter Learning
        [Zaidi et al., JMLR 2013; Greiner & Zhou, AAAI 2002; John & Langley, UAI 1995].
        Murni 1 model tunggal Bayesian Network tanpa pohon keputusan dan tanpa ensemble.
        """
        damp1 = 1.0 / (1.0 + 0.50 * strict_w[:, None])
        damp2 = 1.0 / (1.0 + 1.50 * strict_w[:, None])
        gate0 = (strict_w / (strict_w + 0.5))[:, None]
        ws1_v = kde_full[:, 19]
        gate1 = (ws1_v / (ws1_v + 1.0))[:, None]
        min_d_v = kde_full[:, 20]
        min_udba_v = kde_full[:, 21]
        min_urate_v = kde_full[:, 22]
        gate_nn = np.exp(-0.35 * np.minimum(min_d_v, 25.0))[:, None]
        gate_udba = np.exp(-0.50 * np.minimum(min_udba_v, 15.0))[:, None]
        gate_urate = np.exp(-0.50 * np.minimum(min_urate_v, 15.0))[:, None]

        if s_cur is not None and ctx_cur is not None and s_tr_ref is not None and kde_full.shape[1] >= 46:
            dba_imp = ctx_cur["dba_imp"]
            rate_imp = ctx_cur["rate_imp"]
            sr_imp = ctx_cur["sr_imp"].astype(np.float64)
            tot_nights = np.maximum(s_cur["tot_nights"].astype(np.float64), 0.0)
            wknd = s_cur["wknd"].astype(np.float64)
            ad_clean = np.nan_to_num(s_cur["ad_raw"], nan=2.0)
            ch_clean = np.nan_to_num(s_cur["ch_raw"], nan=0.0)
            np_clean = np.nan_to_num(s_cur["np_raw"], nan=0.0)
            ret = s_cur["ret"].astype(np.float64)
            pc = s_cur["pc"].astype(np.float64)
            po = s_cur["po"].astype(np.float64)

            if ctx_tr_ref is not None:
                date_all_cnt, date_bc_cnt, date_bc_stay_cnt, z_dba, z_rate = cls._compute_true_loo_cohort_stats(
                    s_tr_ref, ctx_tr_ref, s_cur, ctx_cur, is_self_loo=is_self_loo
                )
                ratio_bc_on_date = date_bc_cnt / np.maximum(date_all_cnt, 1.0)
            else:
                tr_dba_clean = np.nan_to_num(s_tr_ref["dba_raw"], nan=58.0)
                tr_rate_clean = np.nan_to_num(s_tr_ref["rate_raw"], nan=1644000.0)

                d_tr = s_tr_ref["date_id"]
                d_te = s_cur["date_id"]
                u_d, cnt_d = np.unique(d_tr, return_counts=True)
                pos_d = np.clip(np.searchsorted(u_d, d_te), 0, len(u_d) - 1)
                date_all_cnt = np.where(u_d[pos_d] == d_te, cnt_d[pos_d], 0).astype(np.float64)

                b1_tr = s_tr_ref["date_id"] * 10 + s_tr_ref["bc"]
                b1_te = s_cur["date_id"] * 10 + s_cur["bc"]
                u_b1, inv_b1, cnt_b1 = np.unique(b1_tr, return_inverse=True, return_counts=True)
                pos_b1 = np.clip(np.searchsorted(u_b1, b1_te), 0, len(u_b1) - 1)
                m_b1 = u_b1[pos_b1] == b1_te
                date_bc_cnt = np.where(m_b1, cnt_b1[pos_b1], 0).astype(np.float64)
                ratio_bc_on_date = date_bc_cnt / np.maximum(date_all_cnt, 1.0)

                sum_dba1 = np.bincount(inv_b1, weights=tr_dba_clean)
                sum_dba2 = np.bincount(inv_b1, weights=tr_dba_clean ** 2)
                mean_dba1 = np.where(m_b1, sum_dba1[pos_b1] / np.maximum(cnt_b1[pos_b1], 1), dba_imp)
                var_dba1 = np.where(
                    m_b1, np.maximum(0.0, sum_dba2[pos_b1] / np.maximum(cnt_b1[pos_b1], 1) - mean_dba1 ** 2), 100.0
                )
                z_dba = np.clip((dba_imp - mean_dba1) / (np.sqrt(var_dba1) + 5.0), -6.0, 6.0)

                sum_r1 = np.bincount(inv_b1, weights=tr_rate_clean)
                sum_r2 = np.bincount(inv_b1, weights=tr_rate_clean ** 2)
                mean_r1 = np.where(m_b1, sum_r1[pos_b1] / np.maximum(cnt_b1[pos_b1], 1), rate_imp)
                var_r1 = np.where(
                    m_b1, np.maximum(0.0, sum_r2[pos_b1] / np.maximum(cnt_b1[pos_b1], 1) - mean_r1 ** 2), 1e10
                )
                z_rate = np.clip((rate_imp - mean_r1) / (np.sqrt(var_r1) + 50000.0), -6.0, 6.0)

                bs_tr = (s_tr_ref["date_id"] * 10 + s_tr_ref["bc"]) * 200 + s_tr_ref["stay"]
                bs_te = (s_cur["date_id"] * 10 + s_cur["bc"]) * 200 + s_cur["stay"]
                u_bs, cnt_bs = np.unique(bs_tr, return_counts=True)
                pos_bs = np.clip(np.searchsorted(u_bs, bs_te), 0, len(u_bs) - 1)
                date_bc_stay_cnt = np.where(u_bs[pos_bs] == bs_te, cnt_bs[pos_bs], 0).astype(np.float64)

            gate_cohort = np.exp(-0.35 * (z_dba ** 2 + z_rate ** 2))[:, None]
            gate_cnt = (date_bc_cnt / (date_bc_cnt + 4.0))[:, None]

            cont_state = np.column_stack([
                np.log1p(np.maximum(dba_imp, 0.0)),
                np.log1p(np.maximum(rate_imp, 0.0) / 100000.0),
                sr_imp,
                tot_nights,
                wknd / np.maximum(tot_nights, 1.0),
                ad_clean,
                ch_clean,
                np_clean,
                ret,
                np.log1p(pc),
                np.log1p(po),
                np.log1p(date_all_cnt),
                np.log1p(date_bc_cnt),
                ratio_bc_on_date,
                np.log1p(date_bc_stay_cnt),
                z_dba,
                np.abs(z_dba),
                z_rate,
                np.abs(z_rate),
                np.log1p(np.maximum(strict_w, 0.0)),
                np.log1p(np.maximum(ws1_v, 0.0)),
                np.minimum(min_d_v, 25.0),
            ])

            extra_18 = kde_full[:, 28:46]
            core_signals = np.column_stack([
                tan_llr,
                np.mean(M_raw[:, 0:10], axis=1),
                np.mean(M_raw[:, 10:21], axis=1),
                np.mean(M_raw[:, 21:48], axis=1),
                np.mean(M_raw[:, 48:64], axis=1),
                np.mean(M_raw[:, 64:91], axis=1),
                np.mean(M_imp[:, 0:10], axis=1),
                np.mean(M_imp[:, 10:20], axis=1),
                np.mean(M_imp[:, 20:32], axis=1),
                np.mean(M_imp[:, 32:], axis=1),
                kde_full[:, 0],
                kde_full[:, 3],
                kde_full[:, 5],
                kde_full[:, 7],
                kde_full[:, 10],
                kde_full[:, 12],
                kde_full[:, 15],
                kde_full[:, 23],
                kde_full[:, 24],
                kde_full[:, 25],
                extra_18[:, 0],
                extra_18[:, 3],
                extra_18[:, 4],
                extra_18[:, 5],
                extra_18[:, 9],
                extra_18[:, 13],
                extra_18[:, 15],
                extra_18[:, 16],
            ])

            knots = (-2.0, -1.0, -0.35, 0.0, 0.35, 1.0, 2.0)
            hinges = []
            for k in knots:
                hinges.append(np.maximum(0.0, core_signals - k))
                hinges.append(np.maximum(0.0, k - core_signals))
            H_mat = np.hstack(hinges)

            n_c = core_signals.shape[1]
            tri_i, tri_j = np.triu_indices(n_c, k=0)
            P_mat = np.clip(core_signals[:, tri_i] * core_signals[:, tri_j], -15.0, 15.0)

            top14 = core_signals[:, [0, 1, 3, 5, 6, 8, 9, 10, 11, 12, 15, 18, 21, 22]]
            C_cross = np.hstack([cont_state[:, j : j + 1] * top14 for j in range(cont_state.shape[1])])

            bc = s_cur["bc"]
            sr_bin = ctx_cur["sr_bin_imp"]
            gates = [
                gate0,
                1.0 - gate0,
                gate1,
                gate_nn,
                gate_udba,
                gate_urate,
                gate_cohort,
                1.0 - gate_cohort,
                gate_cnt,
                (bc == 1).astype(np.float64)[:, None],
                (bc == 2).astype(np.float64)[:, None],
                (bc == 3).astype(np.float64)[:, None],
                (sr_bin == 0).astype(np.float64)[:, None],
                (sr_bin == 1).astype(np.float64)[:, None],
                (sr_bin >= 2).astype(np.float64)[:, None],
            ]
            G_mat = np.hstack([g * core_signals for g in gates])

            kde_all = np.hstack([
                kde_full[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25, 26, 27]],
                extra_18,
            ])

            base_mat = np.hstack([
                tan_llr[:, None],
                M_raw * damp1,
                M_imp * damp1,
                M_raw * damp2,
                M_imp * damp2,
                kde_all,
                cont_state,
                H_mat,
                P_mat,
                C_cross,
                G_mat,
            ])

            if kde_full.shape[1] >= 64:
                v9_18 = kde_full[:, 46:64]
                knots_v9 = (-1.5, -0.5, 0.0, 0.5, 1.5)
                h_v9 = []
                for k in knots_v9:
                    h_v9.append(np.maximum(0.0, v9_18 - k))
                    h_v9.append(np.maximum(0.0, k - v9_18))
                return np.hstack([
                    base_mat,
                    v9_18,
                    gate0 * v9_18,
                    (1.0 - gate0) * v9_18,
                    gate1 * v9_18,
                    np.hstack(h_v9),
                ])

            return base_mat

        g_tan = tan_llr[:, None]
        g_raw_bc_sr = np.mean(M_raw[:, 0:10], axis=1, keepdims=True)
        g_raw_ym = np.mean(M_raw[:, 10:21], axis=1, keepdims=True)
        g_raw_date_fine = np.mean(M_raw[:, 21:48], axis=1, keepdims=True)
        g_raw_date_deep = np.mean(M_raw[:, 48:64], axis=1, keepdims=True)
        g_raw_global = np.mean(M_raw[:, 64:91], axis=1, keepdims=True)

        g_imp_date = np.mean(M_imp[:, 0:10], axis=1, keepdims=True)
        g_imp_global = np.mean(M_imp[:, 10:20], axis=1, keepdims=True)
        g_imp_bday = np.mean(M_imp[:, 20:32], axis=1, keepdims=True) if M_imp.shape[1] > 20 else g_imp_global
        g_imp_cohort = np.mean(M_imp[:, 32:], axis=1, keepdims=True) if M_imp.shape[1] > 32 else g_imp_date

        kde_s0 = kde_full[:, 0:1]
        kde_s1 = kde_full[:, 3:4]
        kde_c1 = kde_full[:, 5:6]
        kde_c1_wide = kde_full[:, 7:8]
        kde_contrast0 = kde_full[:, 10:11]
        kde_c2 = kde_full[:, 12:13]
        kde_c3 = kde_full[:, 15:16]
        kde_date = kde_full[:, 23:24]
        kde_nn = kde_full[:, 24:25]
        kde_c4 = kde_full[:, 25:26]

        core_signals = np.hstack([
            g_tan,
            g_raw_bc_sr,
            g_raw_ym,
            g_raw_date_fine,
            g_raw_date_deep,
            g_raw_global,
            g_imp_date,
            g_imp_global,
            g_imp_bday,
            g_imp_cohort,
            kde_s0,
            kde_s1,
            kde_c1,
            kde_c1_wide,
            kde_contrast0,
            kde_c2,
            kde_c3,
            kde_date,
            kde_nn,
            kde_c4,
        ])

        knots = (-2.0, -1.0, -0.35, 0.0, 0.35, 1.0, 2.0)
        hinges = []
        for k in knots:
            hinges.append(np.maximum(0.0, core_signals - k))
            hinges.append(np.maximum(0.0, k - core_signals))
        H_mat = np.hstack(hinges)

        n_c = core_signals.shape[1]
        tri_i, tri_j = np.triu_indices(n_c, k=0)
        P_mat = np.clip(core_signals[:, tri_i] * core_signals[:, tri_j], -15.0, 15.0)

        G_mat = np.hstack([
            gate0 * core_signals,
            (1.0 - gate0) * core_signals,
            gate1 * core_signals,
            gate_nn * core_signals,
        ])

        kde_all_23 = kde_full[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25, 26, 27]]

        return np.hstack([
            tan_llr[:, None],
            M_raw * damp1,
            M_imp * damp1,
            M_raw * damp2,
            M_imp * damp2,
            kde_all_23,
            H_mat,
            P_mat,
            G_mat,
        ])

    def _fit_cll_lbfgs_weights(
        self, M: np.ndarray, y: np.ndarray, l2_reg: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Mengoptimasi parameter Conditional Log-Likelihood (CLL) berdimensi tinggi menggunakan
        algoritma Two-Loop L-BFGS murni NumPy (kompleksitas O(N*D) per iterasi, bebas
        pembentukan matriks Hessian D x D) [Liu & Nocedal, Math. Program. 1989], mendukung
        penalti Gaussian scalar maupun vektor Structured Group-Prior MAP Precision.
        """
        n, d = M.shape
        theta = np.zeros(d + 1, dtype=np.float64)
        p1 = min(max(float(np.mean(y)), 1e-6), 1.0 - 1e-6)
        theta[0] = math.log(p1 / (1.0 - p1))
        y_float = y.astype(np.float64)
        is_vec_reg = isinstance(l2_reg, np.ndarray)

        def loss_and_grad(th: np.ndarray) -> Tuple[float, np.ndarray]:
            z = np.clip(th[0] + M @ th[1:], -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - y_float
            nll = -float(
                np.sum(
                    y_float * np.log(np.clip(p, 1e-12, 1.0))
                    + (1.0 - y_float) * np.log(np.clip(1.0 - p, 1e-12, 1.0))
                )
            )
            if is_vec_reg:
                reg_loss = 0.5 * float(np.sum(l2_reg * (th[1:] ** 2)))
                g_reg = l2_reg * th[1:]
            else:
                reg_loss = 0.5 * float(l2_reg) * float(np.dot(th[1:], th[1:]))
                g_reg = float(l2_reg) * th[1:]
            g = np.empty(d + 1, dtype=np.float64)
            g[0] = float(np.sum(err))
            g[1:] = M.T @ err + g_reg
            return nll + reg_loss, g

        f_val, grad = loss_and_grad(theta)
        m_hist = 16
        s_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        rho_list: List[float] = []

        for _ in range(200):
            if float(np.max(np.abs(grad))) < 1e-5:
                break
            q = grad.copy()
            k_len = len(s_list)
            alpha_hist = np.zeros(k_len, dtype=np.float64)
            for i in range(k_len - 1, -1, -1):
                alpha_hist[i] = rho_list[i] * float(np.dot(s_list[i], q))
                q -= alpha_hist[i] * y_list[i]
            if k_len > 0:
                s_last, y_last = s_list[-1], y_list[-1]
                yy = float(np.dot(y_last, y_last))
                gamma = float(np.dot(s_last, y_last)) / yy if yy > 1e-20 else 1.0
            else:
                gnorm = float(np.linalg.norm(grad))
                gamma = 1.0 / max(gnorm, 1.0)
            r = gamma * q
            for i in range(k_len):
                beta = rho_list[i] * float(np.dot(y_list[i], r))
                r += s_list[i] * (alpha_hist[i] - beta)
            direction = -r
            g_dot_d = float(np.dot(grad, direction))
            if g_dot_d >= 0.0:
                direction = -grad
                g_dot_d = -float(np.dot(grad, grad))

            step_size = 1.0
            theta_new = theta + step_size * direction
            f_new, grad_new = loss_and_grad(theta_new)
            for _ in range(15):
                if f_new <= f_val + 1e-4 * step_size * g_dot_d:
                    break
                step_size *= 0.5
                theta_new = theta + step_size * direction
                f_new, grad_new = loss_and_grad(theta_new)

            s_k = theta_new - theta
            y_k = grad_new - grad
            sy = float(np.dot(s_k, y_k))
            if sy > 1e-14:
                if len(s_list) >= m_hist:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)
                s_list.append(s_k)
                y_list.append(y_k)
                rho_list.append(1.0 / sy)

            theta = theta_new
            f_val = f_new
            grad = grad_new

        return theta

    def _build_non_duplicate_ctx_43_features(
        self,
        X_arr: np.ndarray,
        s_cur: Dict[str, np.ndarray],
        ctx_cur: Dict[str, np.ndarray],
        s_tr_ref: Dict[str, np.ndarray],
        ctx_tr_ref: Optional[Dict[str, np.ndarray]] = None,
        is_self_loo: bool = False,
    ) -> np.ndarray:
        """
        Menyusun 43 fitur bebas duplikasi berbasis X (tanpa target y) dengan mengganti kolom
        imputasi median menggunakan hasil Context-Imputation dan menambahkan 5 fitur kohort
        ortogonal (`bday`, `z_dba`, `z_rate`, `date_bc_cnt`, `date_bc_stay_cnt`).
        """
        F_ctx = X_arr.astype(np.float64).copy()
        col_map = {name: idx for idx, name in enumerate(self.feature_names_in_ or [])}

        dba_imp = ctx_cur["dba_imp"]
        rate_imp = ctx_cur["rate_imp"]
        sr_imp = ctx_cur["sr_imp"].astype(np.float64)

        tot_nights = np.maximum(s_cur["tot_nights"].astype(np.float64), 1.0)

        if ctx_tr_ref is not None:
            cal_cur, _ = self._compute_cal_ord_and_wd(s_cur["yr"], s_cur["mo"], s_cur["dy"])
            bday = cal_cur.astype(np.float64) - dba_imp
        else:
            approx_arr = (s_cur["yr"] - 2023) * 365.25 + (s_cur["mo"] - 1) * 30.4375 + s_cur["dy"]
            bday = approx_arr - dba_imp
        total_spend_imp = rate_imp * tot_nights
        lead_to_stay_imp = dba_imp / (tot_nights + 0.5)

        if "days_before_arrival" in col_map:
            F_ctx[:, col_map["days_before_arrival"]] = dba_imp
        if "nightly_rate_idr" in col_map:
            F_ctx[:, col_map["nightly_rate_idr"]] = rate_imp
        if "special_requests" in col_map:
            F_ctx[:, col_map["special_requests"]] = sr_imp
        if "total_spend_idr" in col_map:
            F_ctx[:, col_map["total_spend_idr"]] = total_spend_imp
        if "lead_to_stay_ratio" in col_map:
            F_ctx[:, col_map["lead_to_stay_ratio"]] = lead_to_stay_imp

        if ctx_tr_ref is not None:
            _, date_bc_cnt, date_bc_stay_cnt, z_dba, z_rate = self._compute_true_loo_cohort_stats(
                s_tr_ref, ctx_tr_ref, s_cur, ctx_cur, is_self_loo=is_self_loo
            )
        else:
            tr_dba_clean = np.nan_to_num(s_tr_ref["dba_raw"], nan=58.0)
            tr_rate_clean = np.nan_to_num(s_tr_ref["rate_raw"], nan=1644000.0)

            b1_tr = s_tr_ref["date_id"] * 10 + s_tr_ref["bc"]
            b1_te = s_cur["date_id"] * 10 + s_cur["bc"]
            u_b1, inv_b1, cnt_b1 = np.unique(b1_tr, return_inverse=True, return_counts=True)
            pos_b1 = np.clip(np.searchsorted(u_b1, b1_te), 0, len(u_b1) - 1)
            m_b1 = u_b1[pos_b1] == b1_te
            date_bc_cnt = np.where(m_b1, cnt_b1[pos_b1], 0).astype(np.float64)

            sum_dba1 = np.bincount(inv_b1, weights=tr_dba_clean)
            sum_dba2 = np.bincount(inv_b1, weights=tr_dba_clean ** 2)
            mean_dba1 = np.where(m_b1, sum_dba1[pos_b1] / np.maximum(cnt_b1[pos_b1], 1), dba_imp)
            var_dba1 = np.where(
                m_b1, np.maximum(0.0, sum_dba2[pos_b1] / np.maximum(cnt_b1[pos_b1], 1) - mean_dba1 ** 2), 100.0
            )
            z_dba = np.clip((dba_imp - mean_dba1) / (np.sqrt(var_dba1) + 5.0), -6.0, 6.0)

            sum_r1 = np.bincount(inv_b1, weights=tr_rate_clean)
            sum_r2 = np.bincount(inv_b1, weights=tr_rate_clean ** 2)
            mean_r1 = np.where(m_b1, sum_r1[pos_b1] / np.maximum(cnt_b1[pos_b1], 1), rate_imp)
            var_r1 = np.where(
                m_b1, np.maximum(0.0, sum_r2[pos_b1] / np.maximum(cnt_b1[pos_b1], 1) - mean_r1 ** 2), 1e10
            )
            z_rate = np.clip((rate_imp - mean_r1) / (np.sqrt(var_r1) + 50000.0), -6.0, 6.0)

            bs_tr = (s_tr_ref["date_id"] * 10 + s_tr_ref["bc"]) * 200 + s_tr_ref["stay"]
            bs_te = (s_cur["date_id"] * 10 + s_cur["bc"]) * 200 + s_cur["stay"]
            u_bs, cnt_bs = np.unique(bs_tr, return_counts=True)
            pos_bs = np.clip(np.searchsorted(u_bs, bs_te), 0, len(u_bs) - 1)
            date_bc_stay_cnt = np.where(u_bs[pos_bs] == bs_te, cnt_bs[pos_bs], 0).astype(np.float64)

        ortho_5 = np.column_stack([bday, z_dba, z_rate, date_bc_cnt, date_bc_stay_cnt])
        return np.hstack([F_ctx, ortho_5])

    def _fit_kdb_depnet_tables(
        self,
        F_tr: np.ndarray,
        y_tr: np.ndarray,
        max_bins: int = 16,
        include_markov_blanket: bool = True,
    ) -> Dict[str, Any]:
        """
        Mempelajari diskritisasi kuantil, struktur CMI/Chow-Liu MWST/k-DB (k=0..5),
        Markov Blanket Dependency Network (Heckerman et al., JMLR 2000), dan tabel
        probabilitas bersyarat Hierarchical Dirichlet terkompresi (MacKay & Peto, 1995;
        Sahami, KDD 1996) hanya dari (F_tr, y_tr).
        """
        n_tr, d = F_tr.shape
        edges_list: List[np.ndarray] = []
        meds_list = np.zeros(d, dtype=np.float64)
        has_nan_arr = np.zeros(d, dtype=bool)
        n_bins = np.zeros(d, dtype=np.int64)
        D_tr = np.zeros((n_tr, d), dtype=np.int64)

        qs = np.linspace(0.0, 100.0, max_bins + 1)[1:-1]
        for j in range(d):
            c_tr = F_tr[:, j]
            na_tr = np.isnan(c_tr)
            valid = c_tr[~na_tr]
            u_vals = np.unique(valid) if valid.size > 0 else np.array([0.0])
            if len(u_vals) <= max_bins:
                edges = (u_vals[:-1] + u_vals[1:]) / 2.0 if len(u_vals) > 1 else np.array([], dtype=np.float64)
            else:
                edges = np.unique(np.percentile(valid, qs))
            med = float(np.median(valid)) if valid.size > 0 else 0.0
            has_na = bool(np.any(na_tr))
            edges_list.append(edges)
            meds_list[j] = med
            has_nan_arr[j] = has_na
            b_tr = (
                np.searchsorted(edges, np.where(na_tr, med, c_tr), side="right")
                if edges.size > 0
                else np.zeros(n_tr, dtype=np.int64)
            )
            if has_na:
                na_bin = len(edges) + 1
                b_tr = np.where(na_tr, na_bin, b_tr)
                n_bins[j] = na_bin + 1
            else:
                n_bins[j] = len(edges) + 1
            D_tr[:, j] = b_tr

        prev_n_bins = self.tan_n_bins_per_feat_
        self.tan_n_bins_per_feat_ = n_bins
        cmi_glob, mi_cls, _ = self._compute_cmi_and_marginals(D_tr, y_tr)
        self.tan_n_bins_per_feat_ = prev_n_bins

        _, parents_glob, _ = self._build_chow_liu_mwst(cmi_glob)
        parents_cls = np.zeros((2, d), dtype=np.int64)
        for c_idx in range(2):
            _, p_c, _ = self._build_chow_liu_mwst(mi_cls[c_idx])
            parents_cls[c_idx] = p_c

        order_nodes = np.argsort(-np.sum(cmi_glob, axis=1))
        rank_of = np.zeros(d, dtype=np.int64)
        for r_idx, feat_i in enumerate(order_nodes):
            rank_of[feat_i] = r_idx

        masks_c = [y_tr == 0, y_tr == 1]
        tot_c = [float(np.sum(masks_c[0])), float(np.sum(masks_c[1]))]
        D_c_arr = [D_tr[masks_c[0]], D_tr[masks_c[1]]]

        def fit_dense_1p(
            Dc: np.ndarray, p_col: int, feat_i: int, b_i: int, prior_tr_c: np.ndarray, s_dir: float, jm_lam: float
        ) -> np.ndarray:
            b_p = int(n_bins[p_col])
            xp = Dc[:, p_col]
            xi = Dc[:, feat_i]
            np_vec = np.bincount(xp, minlength=b_p).astype(np.float64)
            nj_mat = np.bincount(xp * b_i + xi, minlength=b_p * b_i).reshape(b_p, b_i).astype(np.float64)
            p_dir = (nj_mat + s_dir * prior_tr_c[None, :]) / (np_vec[:, None] + s_dir)
            return (jm_lam * p_dir + (1.0 - jm_lam) * prior_tr_c[None, :]).astype(np.float32)

        cnt_dtype = np.uint16 if n_tr < 65535 else np.float32

        def fit_dense_2p(Dc: np.ndarray, p1: int, p2: int, feat_i: int, b_i: int) -> Tuple[np.ndarray, np.ndarray]:
            b_p1 = int(n_bins[p1])
            b_p2 = int(n_bins[p2])
            b_12 = b_p1 * b_p2
            xp12 = Dc[:, p1] * b_p2 + Dc[:, p2]
            xi = Dc[:, feat_i]
            np_vec = np.bincount(xp12, minlength=b_12).astype(cnt_dtype)
            nj_mat = np.bincount(xp12 * b_i + xi, minlength=b_12 * b_i).reshape(b_12, b_i).astype(cnt_dtype)
            return np_vec, nj_mat

        def fit_sparse_kp(parent_c: np.ndarray, xi_c: np.ndarray, b_i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            u_p, inv_p, c_p = np.unique(parent_c, return_inverse=True, return_counts=True)
            j_keys = (inv_p * b_i + xi_c).astype(np.int64)
            u_j, c_j = np.unique(j_keys, return_counts=True)
            return (
                u_p.astype(np.int64),
                c_p.astype(cnt_dtype),
                u_j.astype(np.int32),
                c_j.astype(cnt_dtype),
            )

        feat_specs: List[Dict[str, Any]] = []
        for i in range(d):
            b_i = int(n_bins[i])
            r_i = rank_of[i]
            higher_nodes = order_nodes[:r_i]
            other_nodes = np.array([j for j in range(d) if j != i], dtype=np.int64)

            cls_specs: List[Dict[str, Any]] = []
            for c_idx in range(2):
                Dc = D_c_arr[c_idx]
                xi_c = Dc[:, i]
                cnt_i = np.bincount(xi_c, minlength=b_i).astype(np.float64)
                p0_vec = np.clip((cnt_i + 1.0) / (tot_c[c_idx] + 1.0 * b_i), 1e-12, 1.0)

                p_c1 = int(parents_cls[c_idx, i])
                p_g1 = int(parents_glob[i])
                tab_pc1 = fit_dense_1p(Dc, p_c1, i, b_i, p0_vec, 5.0, 0.68) if p_c1 >= 0 else None
                tab_pg1 = fit_dense_1p(Dc, p_g1, i, b_i, p0_vec, 5.0, 0.68) if (p_g1 >= 0 and p_g1 != p_c1) else None

                spec_c: Dict[str, Any] = {
                    "p0": p0_vec.astype(np.float32),
                    "p_c1": p_c1,
                    "p_g1": p_g1,
                    "tab_pc1": tab_pc1,
                    "tab_pg1": tab_pg1,
                }

                if len(higher_nodes) >= 2:
                    cmi_score = 0.55 * mi_cls[c_idx, i, higher_nodes] + 0.45 * cmi_glob[i, higher_nodes]
                    top_k = higher_nodes[np.argsort(-cmi_score)]
                    u1, u2 = int(top_k[0]), int(top_k[1])
                    tab_u1 = fit_dense_1p(Dc, u1, i, b_i, p0_vec, 5.0, 0.72)
                    tab_u2 = fit_dense_1p(Dc, u2, i, b_i, p0_vec, 5.0, 0.72)
                    np12, nj12 = fit_dense_2p(Dc, u1, u2, i, b_i)
                    spec_c.update({
                        "u_parents": [int(x) for x in top_k[:5]],
                        "tab_u1": tab_u1,
                        "tab_u2": tab_u2,
                        "np12": np12,
                        "nj12": nj12,
                    })
                    if len(top_k) >= 3:
                        u3 = int(top_k[2])
                        np13, nj13 = fit_dense_2p(Dc, u1, u3, i, b_i)
                        np23, nj23 = fit_dense_2p(Dc, u2, u3, i, b_i)
                        p12_c = Dc[:, u1] * 1009 + Dc[:, u2]
                        p123_c = p12_c * 1009 + Dc[:, u3]
                        spec_c["np13"] = np13
                        spec_c["nj13"] = nj13
                        spec_c["np23"] = np23
                        spec_c["nj23"] = nj23
                        spec_c["sp123"] = fit_sparse_kp(p123_c, xi_c, b_i)
                        if len(top_k) >= 4:
                            u4 = int(top_k[3])
                            p1234_c = p123_c * 1009 + Dc[:, u4]
                            spec_c["sp1234"] = fit_sparse_kp(p1234_c, xi_c, b_i)
                            if len(top_k) >= 5:
                                u5 = int(top_k[4])
                                p12345_c = p1234_c * 1009 + Dc[:, u5]
                                spec_c["sp12345"] = fit_sparse_kp(p12345_c, xi_c, b_i)
                else:
                    spec_c["u_parents"] = []

                if include_markov_blanket:
                    mb_score = 0.55 * mi_cls[c_idx, i, other_nodes] + 0.45 * cmi_glob[i, other_nodes]
                    mb_top = other_nodes[np.argsort(-mb_score)]
                    m1, m2, m3 = int(mb_top[0]), int(mb_top[1]), int(mb_top[2])
                    tab_m1 = fit_dense_1p(Dc, m1, i, b_i, p0_vec, 5.0, 0.72)
                    npm12, njm12 = fit_dense_2p(Dc, m1, m2, i, b_i)
                    mb123_c = (Dc[:, m1] * 1009 + Dc[:, m2]) * 1009 + Dc[:, m3]
                    spm123 = fit_sparse_kp(mb123_c, xi_c, b_i)
                    spec_c.update({
                        "mb_parents": [m1, m2, m3],
                        "tab_m1": tab_m1,
                        "npm12": npm12,
                        "njm12": njm12,
                        "spm123": spm123,
                    })

                cls_specs.append(spec_c)
            feat_specs.append({"cls": cls_specs})

        return {
            "d": d,
            "include_mb": include_markov_blanket,
            "edges": edges_list,
            "meds": meds_list,
            "has_nan": has_nan_arr,
            "n_bins": n_bins,
            "feats": feat_specs,
        }

    @staticmethod
    def _eval_kdb_depnet_tables(table: Dict[str, Any], F_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mengevaluasi matriks node-level LLR untuk Hierarchical k-DB (8 level) dan Markov Blanket (3 level)."""
        d = table["d"]
        include_mb = table["include_mb"]
        edges_list = table["edges"]
        meds = table["meds"]
        has_nan = table["has_nan"]
        n_bins = table["n_bins"]
        feats = table["feats"]

        n_te = F_te.shape[0]
        D_te = np.zeros((n_te, d), dtype=np.int64)
        for j in range(d):
            c_te = F_te[:, j]
            na_te = np.isnan(c_te)
            edges = edges_list[j]
            b_te = (
                np.searchsorted(edges, np.where(na_te, meds[j], c_te), side="right")
                if edges.size > 0
                else np.zeros(n_te, dtype=np.int64)
            )
            if has_nan[j]:
                b_te = np.where(na_te, len(edges) + 1, b_te)
            D_te[:, j] = np.clip(b_te, 0, int(n_bins[j]) - 1)

        def eval_sparse_kp(
            sp_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            parent_va: np.ndarray,
            xi_va: np.ndarray,
            b_i: int,
            prior_va: np.ndarray,
            s_dir: float,
            jm_lam: float,
        ) -> np.ndarray:
            u_p, c_p, u_j, c_j = sp_tuple
            if len(u_p) == 0 or len(u_j) == 0:
                return prior_va
            pos_p = np.clip(np.searchsorted(u_p, parent_va), 0, len(u_p) - 1)
            matched_p = u_p[pos_p] == parent_va
            np_va = np.where(matched_p, c_p[pos_p].astype(np.float64), 0.0)

            key_j = (pos_p * b_i + xi_va).astype(np.int32)
            pos_j = np.clip(np.searchsorted(u_j, key_j), 0, len(u_j) - 1)
            matched_j = matched_p & (u_j[pos_j] == key_j)
            nj_va = np.where(matched_j, c_j[pos_j].astype(np.float64), 0.0)
            p_dir = (nj_va + s_dir * prior_va) / (np_va + s_dir)
            return np.clip(jm_lam * p_dir + (1.0 - jm_lam) * prior_va, 1e-12, 1.0)

        n_levels = 8
        n_mb = 3 if include_mb else 0
        dag_llrs = [np.zeros((n_te, d), dtype=np.float64) for _ in range(n_levels)]
        mb_llrs = [np.zeros((n_te, d), dtype=np.float64) for _ in range(n_mb)]

        for i in range(d):
            xi_va = D_te[:, i]
            b_i = int(n_bins[i])
            log_p_va = [np.zeros((n_te, 2), dtype=np.float64) for _ in range(n_levels)]
            log_mb_va = [np.zeros((n_te, 2), dtype=np.float64) for _ in range(n_mb)]

            for c_idx in range(2):
                sc = feats[i]["cls"][c_idx]
                p0_vec = sc["p0"].astype(np.float64)
                p0_va = p0_vec[xi_va]
                log_p_va[0][:, c_idx] = np.log(p0_va)

                p_c1 = sc["p_c1"]
                p_g1 = sc["p_g1"]
                p1_c = sc["tab_pc1"].astype(np.float64)[D_te[:, p_c1], xi_va] if sc["tab_pc1"] is not None else p0_va
                if sc["tab_pg1"] is not None:
                    p1_g = sc["tab_pg1"].astype(np.float64)[D_te[:, p_g1], xi_va]
                    p1_va = np.clip(0.60 * p1_c + 0.40 * p1_g, 1e-12, 1.0)
                else:
                    p1_va = np.clip(p1_c, 1e-12, 1.0)
                log_p_va[1][:, c_idx] = np.log(p1_va)

                u_par = sc["u_parents"]
                if len(u_par) < 2:
                    for lv in range(2, n_levels):
                        log_p_va[lv][:, c_idx] = log_p_va[1][:, c_idx]
                else:
                    u1, u2 = u_par[0], u_par[1]
                    pu1_va = sc["tab_u1"].astype(np.float64)[D_te[:, u1], xi_va]
                    pu2_va = sc["tab_u2"].astype(np.float64)[D_te[:, u2], xi_va]
                    idx12 = D_te[:, u1] * int(n_bins[u2]) + D_te[:, u2]
                    np12_va = sc["np12"].astype(np.float64)[idx12]
                    nj12_va = sc["nj12"].astype(np.float64)[idx12, xi_va]
                    p2_va = np.clip(0.74 * ((nj12_va + 4.5 * pu1_va) / (np12_va + 4.5)) + 0.26 * pu1_va, 1e-12, 1.0)
                    log_p_va[2][:, c_idx] = np.log(p2_va)

                    if len(u_par) < 3:
                        for lv in range(3, n_levels):
                            log_p_va[lv][:, c_idx] = log_p_va[2][:, c_idx]
                    else:
                        u3 = u_par[2]
                        idx13 = D_te[:, u1] * int(n_bins[u3]) + D_te[:, u3]
                        p2b_va = np.clip(
                            0.74
                            * (
                                (sc["nj13"].astype(np.float64)[idx13, xi_va] + 4.5 * pu1_va)
                                / (sc["np13"].astype(np.float64)[idx13] + 4.5)
                            )
                            + 0.26 * pu1_va,
                            1e-12,
                            1.0,
                        )
                        log_p_va[3][:, c_idx] = np.log(p2b_va)

                        idx23 = D_te[:, u2] * int(n_bins[u3]) + D_te[:, u3]
                        p2c_va = np.clip(
                            0.74
                            * (
                                (sc["nj23"].astype(np.float64)[idx23, xi_va] + 4.5 * pu2_va)
                                / (sc["np23"].astype(np.float64)[idx23] + 4.5)
                            )
                            + 0.26 * pu2_va,
                            1e-12,
                            1.0,
                        )
                        log_p_va[4][:, c_idx] = np.log(p2c_va)

                        p12_va = D_te[:, u1] * 1009 + D_te[:, u2]
                        p123_va = p12_va * 1009 + D_te[:, u3]
                        p3_va = eval_sparse_kp(sc["sp123"], p123_va, xi_va, b_i, p2_va, 4.0, 0.75)
                        log_p_va[5][:, c_idx] = np.log(p3_va)

                        if len(u_par) < 4:
                            for lv in range(6, n_levels):
                                log_p_va[lv][:, c_idx] = log_p_va[5][:, c_idx]
                        else:
                            u4 = u_par[3]
                            p1234_va = p123_va * 1009 + D_te[:, u4]
                            p4_va = eval_sparse_kp(sc["sp1234"], p1234_va, xi_va, b_i, p3_va, 3.5, 0.75)
                            log_p_va[6][:, c_idx] = np.log(p4_va)

                            if len(u_par) < 5:
                                log_p_va[7][:, c_idx] = log_p_va[6][:, c_idx]
                            else:
                                u5 = u_par[4]
                                p12345_va = p1234_va * 1009 + D_te[:, u5]
                                p5_va = eval_sparse_kp(sc["sp12345"], p12345_va, xi_va, b_i, p4_va, 3.0, 0.75)
                                log_p_va[7][:, c_idx] = np.log(p5_va)

                if include_mb:
                    m1, m2, m3 = sc["mb_parents"]
                    pm1_va = sc["tab_m1"].astype(np.float64)[D_te[:, m1], xi_va]
                    idxm12 = D_te[:, m1] * int(n_bins[m2]) + D_te[:, m2]
                    pm2_va = np.clip(
                        0.74
                        * (
                            (sc["njm12"].astype(np.float64)[idxm12, xi_va] + 4.5 * pm1_va)
                            / (sc["npm12"].astype(np.float64)[idxm12] + 4.5)
                        )
                        + 0.26 * pm1_va,
                        1e-12,
                        1.0,
                    )
                    mb123_va = (D_te[:, m1] * 1009 + D_te[:, m2]) * 1009 + D_te[:, m3]
                    pm3_va = eval_sparse_kp(sc["spm123"], mb123_va, xi_va, b_i, pm2_va, 4.0, 0.75)
                    log_mb_va[0][:, c_idx] = np.log(pm1_va)
                    log_mb_va[1][:, c_idx] = np.log(pm2_va)
                    log_mb_va[2][:, c_idx] = np.log(pm3_va)

            for lv in range(n_levels):
                dag_llrs[lv][:, i] = log_p_va[lv][:, 1] - log_p_va[lv][:, 0]
            for lv in range(n_mb):
                mb_llrs[lv][:, i] = log_mb_va[lv][:, 1] - log_mb_va[lv][:, 0]

        return np.hstack(dag_llrs), (np.hstack(mb_llrs) if include_mb else np.empty((n_te, 0), dtype=np.float64))

    def _fit_and_strict_oof_kdb_depnet(
        self,
        F_tr: np.ndarray,
        y_tr: np.ndarray,
        pos_folds: List[np.ndarray],
        neg_folds: List[np.ndarray],
        max_bins: int = 16,
        include_markov_blanket: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Menghasilkan 100% Out-Of-Fold LLR tanpa data leakage (diskritisasi, struktur CMI,
        parent DAG/Markov Blanket, maupun tabel CPT dipelajari terisolasi di tiap inner fold),
        lalu melatih tabel final 100% data latih untuk inferensi test set.
        """
        n_tr, d = F_tr.shape
        n_mb = 3 if include_markov_blanket else 0
        dag_oof = np.zeros((n_tr, 8 * d), dtype=np.float64)
        mb_oof = np.zeros((n_tr, n_mb * d), dtype=np.float64)

        for k in range(len(pos_folds)):
            in_va = np.concatenate([pos_folds[k], neg_folds[k]])
            in_tr_mask = np.ones(n_tr, dtype=bool)
            in_tr_mask[in_va] = False
            in_tr = np.where(in_tr_mask)[0]

            tab_k = self._fit_kdb_depnet_tables(
                F_tr[in_tr],
                y_tr[in_tr],
                max_bins=max_bins,
                include_markov_blanket=include_markov_blanket,
            )
            d_va, m_va = self._eval_kdb_depnet_tables(tab_k, F_tr[in_va])
            dag_oof[in_va] = d_va
            if include_markov_blanket:
                mb_oof[in_va] = m_va

        full_tables = self._fit_kdb_depnet_tables(
            F_tr,
            y_tr,
            max_bins=max_bins,
            include_markov_blanket=include_markov_blanket,
        )
        return dag_oof, mb_oof, full_tables

    @staticmethod
    def _compute_soft_and_subspace_parzen_kde(
        s_tr: Dict[str, np.ndarray],
        y_tr: np.ndarray,
        s_te: Dict[str, np.ndarray],
        ctx_tr_imp: Dict[str, np.ndarray],
        ctx_te_imp: Dict[str, np.ndarray],
        is_self_loo: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menghitung 24 fitur Soft-Distance Parzen-Window KDE (Konteks 8 Full `(ym, bc)`,
        Konteks 9 Shifted 5-Day `(bday, bc)`, Konteks 10 Global Product Cohort) serta
        12 fitur Factored Subspace Conditional Parzen KDE [John & Langley, UAI 1995;
        Frank, Hall, & Pfahringer, UAI 2003] bebas kebocoran target (`p1` LOO murni).
        """
        n_te = len(s_te["ym"])
        tot_pos = float(np.sum(y_tr))
        n_tr_f = float(len(y_tr))
        p1_glob = min(max(tot_pos / max(n_tr_f, 1.0), 1e-6), 1.0 - 1e-6)
        prior_lo_glob = math.log(p1_glob / (1.0 - p1_glob))
        y_tr_f = y_tr.astype(np.float64)

        if is_self_loo and n_tr_f > 1.0:
            p1_row_all = np.clip((tot_pos - y_tr_f) / (n_tr_f - 1.0), 1e-6, 1.0 - 1e-6)
            prior_lo_all = np.log(p1_row_all / (1.0 - p1_row_all))
        else:
            p1_row_all = None
            prior_lo_all = None

        v10_out = np.zeros((n_te, 24), dtype=np.float64)
        sub12_out = np.zeros((n_te, 12), dtype=np.float64)

        tr_dba, te_dba = ctx_tr_imp["dba_imp"], ctx_te_imp["dba_imp"]
        tr_rate, te_rate = ctx_tr_imp["rate_imp"], ctx_te_imp["rate_imp"]
        tr_sr, te_sr = ctx_tr_imp["sr_imp"], ctx_te_imp["sr_imp"]
        tr_rc, te_rc = ctx_tr_imp["rc_imp"], ctx_te_imp["rc_imp"]
        tr_mp, te_mp = ctx_tr_imp["mp_imp"], ctx_te_imp["mp_imp"]

        tr_ad = np.nan_to_num(s_tr["ad_raw"], nan=2.0)
        te_ad = np.nan_to_num(s_te["ad_raw"], nan=2.0)
        tr_ch = np.nan_to_num(s_tr["ch_raw"], nan=0.0)
        te_ch = np.nan_to_num(s_te["ch_raw"], nan=0.0)
        tr_np = np.nan_to_num(s_tr["np_raw"], nan=0.0)
        te_np = np.nan_to_num(s_te["np_raw"], nan=0.0)

        tr_dba_na, te_dba_na = np.isnan(s_tr["dba_raw"]), np.isnan(s_te["dba_raw"])
        tr_r_na, te_r_na = np.isnan(s_tr["rate_raw"]), np.isnan(s_te["rate_raw"])

        tr_bday = (s_tr["yr"] - 2023) * 365.25 + (s_tr["mo"] - 1) * 30.4375 + s_tr["dy"] - tr_dba
        te_bday = (s_te["yr"] - 2023) * 365.25 + (s_te["mo"] - 1) * 30.4375 + s_te["dy"] - te_dba

        def llr_from_weights(W: np.ndarray, y_sub: np.ndarray, m_p: float, idx_te: np.ndarray) -> np.ndarray:
            p1_l = p1_row_all[idx_te] if p1_row_all is not None else p1_glob
            plo_l = prior_lo_all[idx_te] if prior_lo_all is not None else prior_lo_glob
            p_e = np.clip(((W @ y_sub) + m_p * p1_l) / (np.sum(W, axis=1) + m_p), 1e-6, 1.0 - 1e-6)
            return np.log(p_e / (1.0 - p_e)) - plo_l

        def eval_soft_pair(
            idx_te: np.ndarray, idx_tr: np.ndarray, mode: str = "ym_bc"
        ) -> Tuple[np.ndarray, np.ndarray]:
            dba_a, dba_b = te_dba[idx_te, None], tr_dba[None, idx_tr]
            u_dba = np.abs(dba_a - dba_b) / np.maximum(3.5, 0.038 * 0.5 * (dba_a + dba_b))

            r_a, r_b = te_rate[idx_te, None], tr_rate[None, idx_tr]
            u_rate = np.abs(r_a - r_b) / np.maximum(24000.0, 0.030 * 0.5 * (r_a + r_b))

            d_sq = np.minimum(u_dba ** 2, 25.0) + np.minimum(u_rate ** 2, 25.0)

            wknd_diff = np.abs(s_te["wknd"][idx_te, None] - s_tr["wknd"][None, idx_tr]).astype(np.float64)
            wkdy_diff = np.abs(s_te["wkdy"][idx_te, None] - s_tr["wkdy"][None, idx_tr]).astype(np.float64)
            d_sq += np.minimum(wknd_diff * 4.5, 14.0) + np.minimum(wkdy_diff * 3.8, 14.0)

            sr_diff = np.abs(te_sr[idx_te, None] - tr_sr[None, idx_tr]).astype(np.float64)
            d_sq += np.where(sr_diff > 0, 5.5 + 3.5 * sr_diff, 0.0)

            rc_mm = te_rc[idx_te, None] != tr_rc[None, idx_tr]
            rc_na = (s_te["rc_raw"][idx_te, None] == 0) | (s_tr["rc_raw"][None, idx_tr] == 0)
            d_sq += np.where(rc_mm, np.where(rc_na, 2.2, 5.0), 0.0)

            mp_mm = te_mp[idx_te, None] != tr_mp[None, idx_tr]
            mp_na = (s_te["mp_raw"][idx_te, None] == 0) | (s_tr["mp_raw"][None, idx_tr] == 0)
            d_sq += np.where(mp_mm, np.where(mp_na, 2.2, 5.0), 0.0)

            d_sq += np.where(te_ad[idx_te, None] != tr_ad[None, idx_tr], 4.5, 0.0)
            d_sq += np.where(te_ch[idx_te, None] != tr_ch[None, idx_tr], 3.5, 0.0)
            d_sq += np.where(te_np[idx_te, None] != tr_np[None, idx_tr], 4.0, 0.0)
            d_sq += np.where(s_te["ret"][idx_te, None] != s_tr["ret"][None, idx_tr], 4.5, 0.0)
            d_sq += np.where(s_te["pc"][idx_te, None] != s_tr["pc"][None, idx_tr], 4.0, 0.0)

            if mode == "ym_bc":
                dy_diff = np.abs(s_te["dy"][idx_te, None] - s_tr["dy"][None, idx_tr]).astype(np.float64)
                bd_diff = np.abs(te_bday[idx_te, None] - tr_bday[None, idx_tr])
                d_temporal = np.minimum((dy_diff / 2.0) ** 2, (bd_diff / 2.0) ** 2)
                return d_sq, d_sq + np.minimum(d_temporal, 16.0)
            if mode == "bday_bc":
                bd_diff = np.abs(te_bday[idx_te, None] - tr_bday[None, idx_tr])
                return d_sq, d_sq + np.minimum((bd_diff / 1.5) ** 2, 16.0)
            ym_diff = np.abs(s_te["ym"][idx_te, None] - s_tr["ym"][None, idx_tr]).astype(np.float64)
            return d_sq, d_sq + np.minimum((ym_diff / 1.2) ** 2, 16.0)

        def eval_subspace_kernels(
            idx_te: np.ndarray, idx_tr: np.ndarray, include_dy: bool = False
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            dba_a, dba_b = te_dba[idx_te, None], tr_dba[None, idx_tr]
            dba_sc = np.maximum(3.5, 0.038 * 0.5 * (dba_a + dba_b))
            na_dba_any = te_dba_na[idx_te, None] | tr_dba_na[None, idx_tr]
            u_dba_sq = np.minimum((np.abs(dba_a - dba_b) / dba_sc + np.where(na_dba_any, 0.75, 0.0)) ** 2, 25.0)

            r_a, r_b = te_rate[idx_te, None], tr_rate[None, idx_tr]
            r_sc = np.maximum(24000.0, 0.030 * 0.5 * (r_a + r_b))
            na_r_any = te_r_na[idx_te, None] | tr_r_na[None, idx_tr]
            u_rate_sq = np.minimum((np.abs(r_a - r_b) / r_sc + np.where(na_r_any, 0.75, 0.0)) ** 2, 25.0)

            wknd_diff = np.abs(s_te["wknd"][idx_te, None] - s_tr["wknd"][None, idx_tr]).astype(np.float64)
            wkdy_diff = np.abs(s_te["wkdy"][idx_te, None] - s_tr["wkdy"][None, idx_tr]).astype(np.float64)
            d_stay = np.minimum(wknd_diff * 4.5, 14.0) + np.minimum(wkdy_diff * 3.8, 14.0)

            sr_diff = np.abs(te_sr[idx_te, None] - tr_sr[None, idx_tr]).astype(np.float64)
            d_sr = np.where(sr_diff > 0, 5.5 + 3.5 * sr_diff, 0.0)

            rc_mm = te_rc[idx_te, None] != tr_rc[None, idx_tr]
            mp_mm = te_mp[idx_te, None] != tr_mp[None, idx_tr]
            d_prod = np.where(rc_mm, 4.0, 0.0) + np.where(mp_mm, 4.0, 0.0)

            d_party = (
                np.where(te_ad[idx_te, None] != tr_ad[None, idx_tr], 4.0, 0.0)
                + np.where(te_ch[idx_te, None] != tr_ch[None, idx_tr], 3.0, 0.0)
                + np.where(te_np[idx_te, None] != tr_np[None, idx_tr], 3.5, 0.0)
            )
            d_hist = (
                np.where(s_te["ret"][idx_te, None] != s_tr["ret"][None, idx_tr], 4.5, 0.0)
                + np.where(s_te["pc"][idx_te, None] != s_tr["pc"][None, idx_tr], 4.0, 0.0)
            )
            d_dy = (
                np.minimum(((s_te["date_id"][idx_te, None] - s_tr["date_id"][None, idx_tr]) / 1.5) ** 2, 12.0)
                if include_dy
                else 0.0
            )

            d_sub_lead = u_dba_sq + d_stay + d_sr + d_hist + 0.15 * (u_rate_sq + d_prod + d_party) + d_dy
            d_sub_rate = u_rate_sq + d_prod + d_stay + d_sr + d_hist + 0.15 * (u_dba_sq + d_party) + d_dy
            d_sub_core = u_dba_sq + u_rate_sq + d_sr + d_hist + 0.18 * (d_stay + d_prod + d_party) + d_dy
            d_sub_full = u_dba_sq + u_rate_sq + d_stay + d_sr + d_prod + d_party + d_hist + d_dy
            return d_sub_lead, d_sub_rate, d_sub_core, d_sub_full

        # 1. Konteks 8: Full (ym, bc)
        b8_tr = s_tr["ym"] * 10 + s_tr["bc"]
        b8_te = s_te["ym"] * 10 + s_te["bc"]
        o_tr8 = np.argsort(b8_tr)
        u_b8, s_tr8, c_tr8 = np.unique(b8_tr[o_tr8], return_index=True, return_counts=True)
        o_te8 = np.argsort(b8_te)
        u_b8_te, s_te8, c_te8 = np.unique(b8_te[o_te8], return_index=True, return_counts=True)
        p_c8 = np.clip(np.searchsorted(u_b8, u_b8_te), 0, len(u_b8) - 1)
        v_b8 = u_b8[p_c8] == u_b8_te

        for idx_b in np.where(v_b8)[0]:
            tr_pos = p_c8[idx_b]
            idx_tr = o_tr8[s_tr8[tr_pos] : s_tr8[tr_pos] + c_tr8[tr_pos]]
            idx_te = o_te8[s_te8[idx_b] : s_te8[idx_b] + c_te8[idx_b]]
            if is_self_loo and len(idx_tr) <= 1:
                continue
            d_sq, d_sq_temp = eval_soft_pair(idx_te, idx_tr, mode="ym_bc")
            if is_self_loo:
                same_m = idx_te[:, None] == idx_tr[None, :]
                d_sq = np.where(same_m, 1e12, d_sq)
                d_sq_temp = np.where(same_m, 1e12, d_sq_temp)
            y_sub = y_tr_f[idx_tr]

            for k_h, (h, m_p) in enumerate(((0.50, 0.7), (0.95, 1.4), (1.65, 2.5), (2.80, 4.0))):
                v10_out[idx_te, k_h] = llr_from_weights(np.exp(-d_sq_temp / (2.0 * (h ** 2))), y_sub, m_p, idx_te)

            for k_h, (h, m_p) in enumerate(((0.45, 0.7), (0.90, 1.4), (1.60, 2.5))):
                v10_out[idx_te, 4 + k_h] = llr_from_weights(np.exp(-d_sq / (2.0 * (h ** 2))), y_sub, m_p, idx_te)

            p1_l = p1_row_all[idx_te] if p1_row_all is not None else p1_glob
            plo_l = prior_lo_all[idx_te] if prior_lo_all is not None else prior_lo_glob
            for col_k, k_nn in ((7, 1), (8, 3), (9, 7)):
                k_top = min(k_nn, d_sq_temp.shape[1])
                part_k = np.argpartition(d_sq_temp, k_top - 1, axis=1)[:, :k_top]
                top_d = np.take_along_axis(d_sq_temp, part_k, axis=1)
                top_y = y_sub[part_k]
                w_top = np.exp(-0.45 * top_d) * (top_d < 1e9)
                p_top = np.clip((np.sum(w_top * top_y, axis=1) + 0.8 * p1_l) / (np.sum(w_top, axis=1) + 0.8), 1e-6, 1.0 - 1e-6)
                v10_out[idx_te, col_k] = np.log(p_top / (1.0 - p_top)) - plo_l

        # 2. Konteks 9: Shifted 5-Day Booking-Day Window x Channel
        for shift_bd, col_off in ((0.0, 10), (2.5, 14)):
            b9_tr = np.floor((tr_bday + shift_bd) / 5.0).astype(np.int64) * 10 + s_tr["bc"]
            b9_te = np.floor((te_bday + shift_bd) / 5.0).astype(np.int64) * 10 + s_te["bc"]
            o_tr9 = np.argsort(b9_tr)
            u_b9, s_tr9, c_tr9 = np.unique(b9_tr[o_tr9], return_index=True, return_counts=True)
            o_te9 = np.argsort(b9_te)
            u_b9_te, s_te9, c_te9 = np.unique(b9_te[o_te9], return_index=True, return_counts=True)
            p_c9 = np.clip(np.searchsorted(u_b9, u_b9_te), 0, len(u_b9) - 1)
            v_b9 = u_b9[p_c9] == u_b9_te

            for idx_b in np.where(v_b9)[0]:
                tr_pos = p_c9[idx_b]
                idx_tr = o_tr9[s_tr9[tr_pos] : s_tr9[tr_pos] + c_tr9[tr_pos]]
                idx_te = o_te9[s_te9[idx_b] : s_te9[idx_b] + c_te9[idx_b]]
                if is_self_loo and len(idx_tr) <= 1:
                    continue
                _, d_sq_bd = eval_soft_pair(idx_te, idx_tr, mode="bday_bc")
                if is_self_loo:
                    same_m = idx_te[:, None] == idx_tr[None, :]
                    d_sq_bd = np.where(same_m, 1e12, d_sq_bd)
                y_sub = y_tr_f[idx_tr]

                for k_h, (h, m_p) in enumerate(((0.55, 0.8), (1.10, 1.6), (2.00, 3.0))):
                    v10_out[idx_te, col_off + k_h] = llr_from_weights(
                        np.exp(-d_sq_bd / (2.0 * (h ** 2))), y_sub, m_p, idx_te
                    )

                p1_l = p1_row_all[idx_te] if p1_row_all is not None else p1_glob
                plo_l = prior_lo_all[idx_te] if prior_lo_all is not None else prior_lo_glob
                k_top = min(3, d_sq_bd.shape[1])
                part_k = np.argpartition(d_sq_bd, k_top - 1, axis=1)[:, :k_top]
                top_d = np.take_along_axis(d_sq_bd, part_k, axis=1)
                top_y = y_sub[part_k]
                w_top = np.exp(-0.45 * top_d) * (top_d < 1e9)
                p_top = np.clip((np.sum(w_top * top_y, axis=1) + 0.8 * p1_l) / (np.sum(w_top, axis=1) + 0.8), 1e-6, 1.0 - 1e-6)
                v10_out[idx_te, col_off + 3] = np.log(p_top / (1.0 - p_top)) - plo_l

        # 3. Konteks 10: Global Product Cohort (bc, sr_bin, rc_imp, mp_imp)
        b10_tr = ((s_tr["bc"] * 10 + ctx_tr_imp["sr_bin_imp"]) * 10 + tr_rc) * 10 + tr_mp
        b10_te = ((s_te["bc"] * 10 + ctx_te_imp["sr_bin_imp"]) * 10 + te_rc) * 10 + te_mp
        o_tr10 = np.argsort(b10_tr)
        u_b10, s_tr10, c_tr10 = np.unique(b10_tr[o_tr10], return_index=True, return_counts=True)
        o_te10 = np.argsort(b10_te)
        u_b10_te, s_te10, c_te10 = np.unique(b10_te[o_te10], return_index=True, return_counts=True)
        p_c10 = np.clip(np.searchsorted(u_b10, u_b10_te), 0, len(u_b10) - 1)
        v_b10 = u_b10[p_c10] == u_b10_te

        for idx_b in np.where(v_b10)[0]:
            tr_pos = p_c10[idx_b]
            idx_tr = o_tr10[s_tr10[tr_pos] : s_tr10[tr_pos] + c_tr10[tr_pos]]
            idx_te = o_te10[s_te10[idx_b] : s_te10[idx_b] + c_te10[idx_b]]
            if is_self_loo and len(idx_tr) <= 1:
                continue
            if len(idx_tr) > 1200:
                step_s = max(1, len(idx_tr) // 1000)
                idx_tr_sub = idx_tr[::step_s]
            else:
                idx_tr_sub = idx_tr
            d_sq, d_sq_ym = eval_soft_pair(idx_te, idx_tr_sub, mode="global_prod")
            if is_self_loo:
                same_m = idx_te[:, None] == idx_tr_sub[None, :]
                d_sq = np.where(same_m, 1e12, d_sq)
                d_sq_ym = np.where(same_m, 1e12, d_sq_ym)
            y_sub = y_tr_f[idx_tr_sub]

            for k_h, (h, m_p) in enumerate(((0.55, 0.8), (1.15, 1.8), (2.10, 3.2))):
                v10_out[idx_te, 18 + k_h] = llr_from_weights(np.exp(-d_sq_ym / (2.0 * (h ** 2))), y_sub, m_p, idx_te)

            for k_h, (h, m_p) in enumerate(((0.50, 0.8), (1.10, 1.8), (2.00, 3.2))):
                v10_out[idx_te, 21 + k_h] = llr_from_weights(np.exp(-d_sq / (2.0 * (h ** 2))), y_sub, m_p, idx_te)

        # 4. Factored Subspace Conditional Parzen Kernels on (date_id, bc)
        b1_tr = s_tr["date_id"] * 10 + s_tr["bc"]
        b1_te = s_te["date_id"] * 10 + s_te["bc"]
        o_tr1 = np.argsort(b1_tr)
        u_b1, s_tr1, c_tr1 = np.unique(b1_tr[o_tr1], return_index=True, return_counts=True)
        o_te1 = np.argsort(b1_te)
        u_b1_te, s_te1, c_te1 = np.unique(b1_te[o_te1], return_index=True, return_counts=True)
        p_c1 = np.clip(np.searchsorted(u_b1, u_b1_te), 0, len(u_b1) - 1)
        v_b1 = u_b1[p_c1] == u_b1_te

        for idx_b in np.where(v_b1)[0]:
            tr_pos = p_c1[idx_b]
            idx_tr = o_tr1[s_tr1[tr_pos] : s_tr1[tr_pos] + c_tr1[tr_pos]]
            idx_te = o_te1[s_te1[idx_b] : s_te1[idx_b] + c_te1[idx_b]]
            if is_self_loo and len(idx_tr) <= 1:
                continue
            d_lead, d_rate, d_core, d_full = eval_subspace_kernels(idx_te, idx_tr, include_dy=False)
            if is_self_loo:
                same_m = idx_te[:, None] == idx_tr[None, :]
                d_lead = np.where(same_m, 1e12, d_lead)
                d_rate = np.where(same_m, 1e12, d_rate)
                d_core = np.where(same_m, 1e12, d_core)
                d_full = np.where(same_m, 1e12, d_full)
            y_sub = y_tr_f[idx_tr]

            for d_idx, d_mat in enumerate((d_lead, d_rate, d_core, d_full)):
                for k_h, (h, m_p) in enumerate(((0.55, 0.7), (1.15, 1.5))):
                    sub12_out[idx_te, d_idx * 2 + k_h] = llr_from_weights(
                        np.exp(-d_mat / (2.0 * (h ** 2))), y_sub, m_p, idx_te
                    )

        # 5. Factored Subspace Conditional Parzen Kernels on 3-Day Arrival Window (date_id // 3, bc)
        b5_tr = (s_tr["date_id"] // 3) * 10 + s_tr["bc"]
        b5_te = (s_te["date_id"] // 3) * 10 + s_te["bc"]
        o_tr5 = np.argsort(b5_tr)
        u_b5, s_tr5, c_tr5 = np.unique(b5_tr[o_tr5], return_index=True, return_counts=True)
        o_te5 = np.argsort(b5_te)
        u_b5_te, s_te5, c_te5 = np.unique(b5_te[o_te5], return_index=True, return_counts=True)
        p_c5 = np.clip(np.searchsorted(u_b5, u_b5_te), 0, len(u_b5) - 1)
        v_b5 = u_b5[p_c5] == u_b5_te

        for idx_b in np.where(v_b5)[0]:
            tr_pos = p_c5[idx_b]
            idx_tr = o_tr5[s_tr5[tr_pos] : s_tr5[tr_pos] + c_tr5[tr_pos]]
            idx_te = o_te5[s_te5[idx_b] : s_te5[idx_b] + c_te5[idx_b]]
            if is_self_loo and len(idx_tr) <= 1:
                continue
            d_lead, d_rate, d_core, d_full = eval_subspace_kernels(idx_te, idx_tr, include_dy=True)
            if is_self_loo:
                same_m = idx_te[:, None] == idx_tr[None, :]
                d_lead = np.where(same_m, 1e12, d_lead)
                d_rate = np.where(same_m, 1e12, d_rate)
                d_core = np.where(same_m, 1e12, d_core)
                d_full = np.where(same_m, 1e12, d_full)
            y_sub = y_tr_f[idx_tr]

            for d_idx, d_mat in enumerate((d_lead, d_rate, d_core, d_full)):
                sub12_out[idx_te, 8 + d_idx] = llr_from_weights(
                    np.exp(-d_mat / (2.0 * (0.85 ** 2))), y_sub, 1.2, idx_te
                )

        return v10_out, sub12_out

    @staticmethod
    def _build_kdb_kde_cll_basis_and_prior(
        base_v9: np.ndarray,
        v10_24: np.ndarray,
        sub12: np.ndarray,
        nodes_list: List[Tuple[np.ndarray, int, float]],
        strict_w: np.ndarray,
        kde_64: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menyusun matriks faktor Bayesian berdimensi tinggi untuk `cll_kdb_kde` beserta
        vektor Structured Group-Prior Precision L2 (MAP) agar setiap blok fitur (core KDE,
        cliques, node-level k-DB / Markov Blanket, gating, splines, dan interaksi) mendapat
        skala regularisasi Gaussian yang proporsional terhadap kompleksitasnya.
        """
        damp1 = 1.0 / (1.0 + 0.50 * strict_w[:, None])
        damp2 = 1.0 / (1.0 + 1.50 * strict_w[:, None])
        gate0 = (strict_w / (strict_w + 0.5))[:, None]
        ws1_v = kde_64[:, 19]
        gate1 = (ws1_v / (ws1_v + 1.0))[:, None]
        min_d_v = kde_64[:, 20]
        gate_nn = np.exp(-0.35 * np.minimum(min_d_v, 25.0))[:, None]

        blocks: List[np.ndarray] = []
        l2_blocks: List[np.ndarray] = []

        blocks.append(base_v9)
        n_b9 = base_v9.shape[1]
        l2_b9 = np.full(n_b9, 750.0, dtype=np.float64)
        l2_b9[:318] = 450.0
        l2_b9[277:318] = 220.0
        l2_blocks.append(l2_b9)

        all_net_sums = []
        for nodes_mat, d_f, reg_node in nodes_list:
            m_d1 = nodes_mat * damp1
            m_d2 = nodes_mat * damp2
            blocks.extend([m_d1, m_d2])
            l2_blocks.append(np.full(m_d1.shape[1], reg_node, dtype=np.float64))
            l2_blocks.append(np.full(m_d2.shape[1], reg_node, dtype=np.float64))

            n_lv = nodes_mat.shape[1] // d_f
            sc = 1.0 / math.sqrt(d_f)
            sums_mat = np.column_stack([
                np.sum(nodes_mat[:, lv * d_f : (lv + 1) * d_f], axis=1) * sc
                for lv in range(n_lv)
            ])
            all_net_sums.append(sums_mat)

        net_sums_full = np.hstack(all_net_sums)
        v_kde = np.hstack([v10_24, sub12, net_sums_full])

        blocks.append(v_kde)
        l2_blocks.append(np.full(v_kde.shape[1], 240.0, dtype=np.float64))

        for g_mat in (gate0 * v_kde, (1.0 - gate0) * v_kde, gate1 * v_kde, (1.0 - gate_nn) * v_kde):
            blocks.append(g_mat)
            l2_blocks.append(np.full(g_mat.shape[1], 650.0, dtype=np.float64))

        knots = (-1.5, -0.5, 0.0, 0.5, 1.5)
        h_v = []
        for k in knots:
            h_v.append(np.maximum(0.0, v_kde - k))
            h_v.append(np.maximum(0.0, k - v_kde))
        H_mat = np.hstack(h_v)
        blocks.append(H_mat)
        l2_blocks.append(np.full(H_mat.shape[1], 850.0, dtype=np.float64))

        s_key = np.column_stack([
            kde_64[:, 0],
            kde_64[:, 5],
            kde_64[:, 12],
            kde_64[:, 24],
            kde_64[:, 25],
            v10_24[:, 0],
            v10_24[:, 1],
            v10_24[:, 4],
            v10_24[:, 8],
            v10_24[:, 10],
            v10_24[:, 14],
            v10_24[:, 18],
            sub12[:, 0],
            sub12[:, 2],
            sub12[:, 4],
            sub12[:, 6],
            net_sums_full[:, 0],
            net_sums_full[:, 1],
            net_sums_full[:, 2],
            net_sums_full[:, 5],
            net_sums_full[:, 6],
            net_sums_full[:, 7],
        ])
        ti, tj = np.triu_indices(s_key.shape[1], k=0)
        P_v = np.clip(s_key[:, ti] * s_key[:, tj], -15.0, 15.0)
        blocks.append(P_v)
        l2_blocks.append(np.full(P_v.shape[1], 950.0, dtype=np.float64))

        return np.hstack(blocks), np.concatenate(l2_blocks)

    @staticmethod
    def _discretize_zi_safe(
        F_tr: np.ndarray, max_bins: int
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Diskritisasi kuantil adaptif yang aman untuk fitur zero-inflated (`prior_completed_bookings`,
        `prior_cancellations`, `num_children`) agar batas persentil tidak kolaps menjadi 1 bin konstan.
        """
        n_tr, d = F_tr.shape
        edges_list: List[np.ndarray] = []
        meds_list = np.zeros(d, dtype=np.float64)
        has_nan_arr = np.zeros(d, dtype=bool)
        n_bins = np.zeros(d, dtype=np.int64)
        D_tr = np.zeros((n_tr, d), dtype=np.int64)
        qs = np.linspace(0.0, 100.0, max_bins + 1)[1:-1]

        for j in range(d):
            c_tr = F_tr[:, j]
            na_tr = np.isnan(c_tr)
            valid = c_tr[~na_tr]
            if valid.size > 0:
                u_vals, u_cnts = np.unique(valid, return_counts=True)
            else:
                u_vals, u_cnts = np.array([0.0]), np.array([1])

            if len(u_vals) <= max_bins:
                edges = (u_vals[:-1] + u_vals[1:]) / 2.0 if len(u_vals) > 1 else np.array([], dtype=np.float64)
            else:
                edges = np.unique(np.percentile(valid, qs))
                if len(edges) <= 2:
                    dom_val = u_vals[int(np.argmax(u_cnts))]
                    e_list: List[float] = []
                    below = valid[valid < dom_val]
                    above = valid[valid > dom_val]
                    if below.size > 0:
                        e_list.append(0.5 * (float(np.max(below)) + float(dom_val)))
                        u_b = np.unique(below)
                        if len(u_b) <= max_bins // 3:
                            e_list.extend(((u_b[:-1] + u_b[1:]) / 2.0).tolist())
                        else:
                            e_list.extend(
                                np.percentile(below, np.linspace(0.0, 100.0, (max_bins // 3) + 1)[1:-1]).tolist()
                            )
                    if above.size > 0:
                        e_list.append(0.5 * (float(dom_val) + float(np.min(above))))
                        u_a = np.unique(above)
                        rem = max(3, max_bins - len(e_list) - 1)
                        if len(u_a) <= rem:
                            e_list.extend(((u_a[:-1] + u_a[1:]) / 2.0).tolist())
                        else:
                            e_list.extend(np.percentile(above, np.linspace(0.0, 100.0, rem + 1)[1:-1]).tolist())
                    edges = np.unique(np.asarray(e_list, dtype=np.float64))

            med = float(np.median(valid)) if valid.size > 0 else 0.0
            has_na = bool(np.any(na_tr))
            edges_list.append(edges)
            meds_list[j] = med
            has_nan_arr[j] = has_na
            b_tr = (
                np.searchsorted(edges, np.where(na_tr, med, c_tr), side="right")
                if edges.size > 0
                else np.zeros(n_tr, dtype=np.int64)
            )
            if has_na:
                na_bin = len(edges) + 1
                b_tr = np.where(na_tr, na_bin, b_tr)
                n_bins[j] = na_bin + 1
            else:
                n_bins[j] = len(edges) + 1
            D_tr[:, j] = b_tr

        return D_tr, edges_list, meds_list, has_nan_arr, n_bins

    def _fit_hnb_sp2de_tables(
        self,
        F_tr: np.ndarray,
        y_tr: np.ndarray,
        max_bins: int = 16,
        sp_max_pairs: int = 6,
        top_a1de: int = 0,
        smooth_k1: bool = False,
    ) -> Dict[str, Any]:
        """
        Mempelajari tabel probabilitas untuk:
        - Tree-Stabilized Level-1 Chow-Liu (`0.35 * mi_cls + 0.65 * cmi_glob`) + optional Top-3 CMI Smooth-K1
        - Hidden Naive Bayes (HNB gamma=1 & gamma=2) [Jiang, Zhang, & Cai, IEEE TKDE 2009]
        - Selective Super-Parent A2DE (SP-2DE) & A1DE [Webb, Boughton, & Wang, ML 2005; Webb et al., ML 2012]
        """
        n_tr, d = F_tr.shape
        D_tr, edges_list, meds_list, has_nan_arr, n_bins = self._discretize_zi_safe(F_tr, max_bins)
        prev_b = self.tan_n_bins_per_feat_
        self.tan_n_bins_per_feat_ = n_bins
        cmi_glob, mi_cls, _ = self._compute_cmi_and_marginals(D_tr, y_tr)
        self.tan_n_bins_per_feat_ = prev_b

        _, parents_glob, _ = self._build_chow_liu_mwst(cmi_glob)
        parents_cls = np.zeros((2, d), dtype=np.int64)
        for c_idx in range(2):
            _, p_c, _ = self._build_chow_liu_mwst(0.35 * mi_cls[c_idx] + 0.65 * cmi_glob)
            parents_cls[c_idx] = p_c

        order_nodes = np.argsort(-np.sum(cmi_glob, axis=1))
        masks_c = [y_tr == 0, y_tr == 1]
        tot_c = [float(np.sum(masks_c[0])), float(np.sum(masks_c[1]))]
        D_c_arr = [D_tr[masks_c[0]], D_tr[masks_c[1]]]
        cnt_dtype = np.uint16 if n_tr < 65535 else np.float32

        p0_all: List[List[np.ndarray]] = [[], []]
        for c_idx in range(2):
            Dc = D_c_arr[c_idx]
            for i in range(d):
                b_i = int(n_bins[i])
                cnt_i = np.bincount(Dc[:, i], minlength=b_i).astype(np.float64)
                p0_all[c_idx].append(np.clip((cnt_i + 1.0) / (tot_c[c_idx] + 1.0 * b_i), 1e-12, 1.0))

        def fit_1p(Dc: np.ndarray, p_col: int, feat_i: int, b_i: int, prior_c: np.ndarray, s_dir: float, jm: float) -> np.ndarray:
            b_p = int(n_bins[p_col])
            xp, xi = Dc[:, p_col], Dc[:, feat_i]
            np_vec = np.bincount(xp, minlength=b_p).astype(np.float64)
            nj_mat = np.bincount(xp * b_i + xi, minlength=b_p * b_i).reshape(b_p, b_i).astype(np.float64)
            p_dir = (nj_mat + s_dir * prior_c[None, :]) / (np_vec[:, None] + s_dir)
            return (jm * p_dir + (1.0 - jm) * prior_c[None, :]).astype(np.float32)

        def fit_2p_counts(Dc: np.ndarray, p1: int, p2: int, feat_i: int, b_i: int) -> Tuple[np.ndarray, np.ndarray]:
            b_p1, b_p2 = int(n_bins[p1]), int(n_bins[p2])
            b_12 = b_p1 * b_p2
            xp12, xi = Dc[:, p1] * b_p2 + Dc[:, p2], Dc[:, feat_i]
            return (
                np.bincount(xp12, minlength=b_12).astype(cnt_dtype),
                np.bincount(xp12 * b_i + xi, minlength=b_12 * b_i).reshape(b_12, b_i).astype(cnt_dtype),
            )

        onep_cache: List[List[Dict[int, np.ndarray]]] = [[{} for _ in range(d)] for _ in range(2)]
        hnb_w: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[], []]
        for c_idx in range(2):
            Dc = D_c_arr[c_idx]
            mi_mat = np.maximum(0.0, 0.50 * mi_cls[c_idx] + 0.50 * cmi_glob)
            np.fill_diagonal(mi_mat, 0.0)
            for i in range(d):
                b_i = int(n_bins[i])
                p0_v = p0_all[c_idx][i]
                top_j = np.argsort(-mi_mat[i])[: min(10, d - 1)]
                valid_j = np.array([int(j) for j in top_j if j != i and mi_mat[i, j] > 1e-6], dtype=np.int64)
                if len(valid_j) > 0:
                    w1 = mi_mat[i, valid_j]
                    w1 = w1 / np.sum(w1)
                    w2 = (w1 ** 2)
                    w2 = w2 / np.sum(w2)
                    for j in valid_j:
                        onep_cache[c_idx][i][int(j)] = fit_1p(Dc, int(j), i, b_i, p0_v, 5.0, 0.72)
                    hnb_w[c_idx].append((valid_j, w1.astype(np.float64), w2.astype(np.float64)))
                else:
                    hnb_w[c_idx].append((np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)))

        n_sp_cand = min(d, 6 if sp_max_pairs > 10 else 5)
        sp_nodes = [int(x) for x in order_nodes[:n_sp_cand]]
        sp_pairs = [(sp_nodes[a], sp_nodes[b]) for a in range(len(sp_nodes)) for b in range(a + 1, len(sp_nodes))][:sp_max_pairs]
        sp2_tables = []
        for s1, s2 in sp_pairs:
            pair_cls = []
            for c_idx in range(2):
                Dc = D_c_arr[c_idx]
                per_feat = {}
                for i in range(d):
                    if i == s1 or i == s2:
                        continue
                    b_i = int(n_bins[i])
                    p0_v = p0_all[c_idx][i]
                    if s1 not in onep_cache[c_idx][i]:
                        onep_cache[c_idx][i][s1] = fit_1p(Dc, s1, i, b_i, p0_v, 5.0, 0.72)
                    if s2 not in onep_cache[c_idx][i]:
                        onep_cache[c_idx][i][s2] = fit_1p(Dc, s2, i, b_i, p0_v, 5.0, 0.72)
                    np12, nj12 = fit_2p_counts(Dc, s1, s2, i, b_i)
                    per_feat[i] = (onep_cache[c_idx][i][s1], onep_cache[c_idx][i][s2], np12, nj12)
                pair_cls.append(per_feat)
            sp2_tables.append(((s1, s2), pair_cls))

        a1de_specs = []
        if top_a1de > 0:
            a1de_nodes = [int(x) for x in order_nodes[: min(top_a1de, d)]]
            for s_node in a1de_nodes:
                cls_tabs: List[List[Optional[np.ndarray]]] = []
                for c_idx in range(2):
                    Dc = D_c_arr[c_idx]
                    tabs_s: List[Optional[np.ndarray]] = []
                    for i in range(d):
                        if i == s_node:
                            tabs_s.append(None)
                        else:
                            if s_node not in onep_cache[c_idx][i]:
                                onep_cache[c_idx][i][s_node] = fit_1p(
                                    Dc, s_node, i, int(n_bins[i]), p0_all[c_idx][i], 5.0, 0.72
                                )
                            tabs_s.append(onep_cache[c_idx][i][s_node])
                    cls_tabs.append(tabs_s)
                a1de_specs.append({"s": s_node, "tabs": cls_tabs})

        feat_specs = []
        for i in range(d):
            b_i = int(n_bins[i])
            cls_specs = []
            for c_idx in range(2):
                Dc = D_c_arr[c_idx]
                p0_v = p0_all[c_idx][i]
                p_c1, p_g1 = int(parents_cls[c_idx, i]), int(parents_glob[i])
                vj, w1, w2 = hnb_w[c_idx][i]
                sk_items = []
                if smooth_k1 and d > 1:
                    s_vec = 0.45 * np.maximum(0.0, mi_cls[c_idx][i]) + 0.55 * np.maximum(0.0, cmi_glob[i])
                    s_vec[i] = -1.0
                    top3 = np.argsort(-s_vec)[: min(3, d - 1)]
                    ws = np.maximum(s_vec[top3], 1e-6)
                    ws = ws / np.sum(ws)
                    for p_col, w_p in zip(top3, ws):
                        p_int = int(p_col)
                        if p_int not in onep_cache[c_idx][i]:
                            onep_cache[c_idx][i][p_int] = fit_1p(Dc, p_int, i, b_i, p0_v, 5.0, 0.72)
                        sk_items.append((p_int, onep_cache[c_idx][i][p_int], float(w_p)))
                cls_specs.append({
                    "p0": p0_v.astype(np.float32),
                    "p_c1": p_c1,
                    "p_g1": p_g1,
                    "tab_pc1": fit_1p(Dc, p_c1, i, b_i, p0_v, 5.0, 0.68) if p_c1 >= 0 else None,
                    "tab_pg1": fit_1p(Dc, p_g1, i, b_i, p0_v, 5.0, 0.68) if (p_g1 >= 0 and p_g1 != p_c1) else None,
                    "hnb_items": [(int(j), onep_cache[c_idx][i][int(j)], float(a1), float(a2)) for j, a1, a2 in zip(vj, w1, w2)],
                    "sk_items": sk_items,
                })
            feat_specs.append(cls_specs)

        return {
            "d": d,
            "edges": edges_list,
            "meds": meds_list,
            "has_nan": has_nan_arr,
            "n_bins": n_bins,
            "feats": feat_specs,
            "sp2": sp2_tables,
            "a1de": a1de_specs,
            "smooth_k1": bool(smooth_k1),
        }

    @staticmethod
    def _eval_hnb_sp2de_tables(
        table: Dict[str, Any], Fb: np.ndarray, return_a1de: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Mengevaluasi node-level LLR untuk Level-1 (Stabilized/Smooth-K1), HNB (gamma=1,2), Selective SP-2DE, dan A1DE."""
        d = table["d"]
        nb_te = Fb.shape[0]
        D_te = np.zeros((nb_te, d), dtype=np.int64)
        for j in range(d):
            c_te = Fb[:, j]
            na_te = np.isnan(c_te)
            ed = table["edges"][j]
            b_te = np.searchsorted(ed, np.where(na_te, table["meds"][j], c_te), side="right") if ed.size > 0 else np.zeros(nb_te, dtype=np.int64)
            if table["has_nan"][j]:
                b_te = np.where(na_te, len(ed) + 1, b_te)
            D_te[:, j] = np.clip(b_te, 0, int(table["n_bins"][j]) - 1)

        k1_llr = np.zeros((nb_te, d), dtype=np.float64)
        sk_llr = np.zeros((nb_te, d), dtype=np.float64) if table.get("smooth_k1", False) else None
        hnb_llr = np.zeros((nb_te, 2 * d), dtype=np.float64)
        sp2_llr = np.zeros((nb_te, d + len(table["sp2"])), dtype=np.float64)

        for i in range(d):
            xi = D_te[:, i]
            lp_k1 = np.zeros((nb_te, 2), dtype=np.float64)
            lp_sk = np.zeros((nb_te, 2), dtype=np.float64) if sk_llr is not None else None
            lp_h1 = np.zeros((nb_te, 2), dtype=np.float64)
            lp_h2 = np.zeros((nb_te, 2), dtype=np.float64)
            for c_idx in range(2):
                sc = table["feats"][i][c_idx]
                p0_v = sc["p0"].astype(np.float64)[xi]
                p_c1, p_g1 = sc["p_c1"], sc["p_g1"]
                p1_c = sc["tab_pc1"].astype(np.float64)[D_te[:, p_c1], xi] if sc["tab_pc1"] is not None else p0_v
                if sc["tab_pg1"] is not None:
                    p1_g = sc["tab_pg1"].astype(np.float64)[D_te[:, p_g1], xi]
                    p1_v = np.clip(0.50 * p1_c + 0.50 * p1_g, 1e-12, 1.0)
                else:
                    p1_v = np.clip(p1_c, 1e-12, 1.0)
                lp_k1[:, c_idx] = np.log(p1_v)

                if lp_sk is not None:
                    sk_items = sc.get("sk_items", [])
                    if sk_items:
                        p_mix = np.zeros(nb_te, dtype=np.float64)
                        for p_col, tab_sk, w_p in sk_items:
                            p_mix += w_p * tab_sk.astype(np.float64)[D_te[:, p_col], xi]
                        lp_sk[:, c_idx] = np.log(np.clip(p_mix, 1e-12, 1.0))
                    else:
                        lp_sk[:, c_idx] = lp_k1[:, c_idx]

                items = sc["hnb_items"]
                if items:
                    acc1 = np.zeros(nb_te, dtype=np.float64)
                    acc2 = np.zeros(nb_te, dtype=np.float64)
                    for j, tab, a1, a2 in items:
                        pv = tab.astype(np.float64)[D_te[:, j], xi]
                        acc1 += a1 * pv
                        acc2 += a2 * pv
                    lp_h1[:, c_idx] = np.log(np.clip(acc1, 1e-12, 1.0))
                    lp_h2[:, c_idx] = np.log(np.clip(acc2, 1e-12, 1.0))
                else:
                    lp_h1[:, c_idx] = lp_k1[:, c_idx]
                    lp_h2[:, c_idx] = lp_k1[:, c_idx]
            k1_llr[:, i] = lp_k1[:, 1] - lp_k1[:, 0]
            if sk_llr is not None and lp_sk is not None:
                sk_llr[:, i] = lp_sk[:, 1] - lp_sk[:, 0]
            hnb_llr[:, i] = lp_h1[:, 1] - lp_h1[:, 0]
            hnb_llr[:, d + i] = lp_h2[:, 1] - lp_h2[:, 0]

        if sk_llr is not None:
            k1_llr = 0.50 * k1_llr + 0.50 * sk_llr

        cnt_node = np.zeros(d, dtype=np.float64)
        for p_idx, ((s1, s2), pair_cls) in enumerate(table["sp2"]):
            b_s2 = int(table["n_bins"][s2])
            idx12 = D_te[:, s1] * b_s2 + D_te[:, s2]
            lp_pair = np.zeros((nb_te, 2), dtype=np.float64)
            for i in pair_cls[0].keys():
                xi = D_te[:, i]
                cnt_node[i] += 1.0
                lp_f = np.zeros((nb_te, 2), dtype=np.float64)
                for c_idx in range(2):
                    t_s1, t_s2, np12, nj12 = pair_cls[c_idx][i]
                    prior = 0.50 * t_s1.astype(np.float64)[D_te[:, s1], xi] + 0.50 * t_s2.astype(np.float64)[D_te[:, s2], xi]
                    p2v = np.clip(
                        0.75 * ((nj12.astype(np.float64)[idx12, xi] + 4.5 * prior) / (np12.astype(np.float64)[idx12] + 4.5))
                        + 0.25 * prior,
                        1e-12,
                        1.0,
                    )
                    lp_f[:, c_idx] = np.log(p2v)
                sp2_llr[:, i] += lp_f[:, 1] - lp_f[:, 0]
                lp_pair += lp_f
            sp2_llr[:, d + p_idx] = (lp_pair[:, 1] - lp_pair[:, 0]) / math.sqrt(max(d - 2, 1))
        for i in range(d):
            if cnt_node[i] > 0:
                sp2_llr[:, i] /= cnt_node[i]

        if not return_a1de:
            return k1_llr, hnb_llr, sp2_llr

        a1de_specs = table.get("a1de", [])
        a1de_llr = np.zeros((nb_te, len(a1de_specs)), dtype=np.float64)
        sc_a1 = 1.0 / math.sqrt(max(d, 1))
        for si, sp_info in enumerate(a1de_specs):
            s_node = int(sp_info["s"])
            xs_e = D_te[:, s_node]
            lp_a = np.zeros((nb_te, 2), dtype=np.float64)
            for c_idx in range(2):
                lp_a[:, c_idx] += np.log(table["feats"][s_node][c_idx]["p0"].astype(np.float64)[xs_e])
                tabs_c = sp_info["tabs"][c_idx]
                for i, tab in enumerate(tabs_c):
                    if tab is not None:
                        lp_a[:, c_idx] += np.log(np.clip(tab.astype(np.float64)[xs_e, D_te[:, i]], 1e-12, 1.0))
            a1de_llr[:, si] = (lp_a[:, 1] - lp_a[:, 0]) * sc_a1
        return k1_llr, hnb_llr, sp2_llr, a1de_llr

    def _fit_and_strict_oof_hnb_sp2de(
        self,
        F_tr: np.ndarray,
        y_tr: np.ndarray,
        pos_folds: List[np.ndarray],
        neg_folds: List[np.ndarray],
        max_bins: int = 16,
        sp_max_pairs: int = 6,
        top_a1de: int = 0,
        smooth_k1: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
    ]:
        """Menghasilkan 100% Out-Of-Fold LLR untuk Level-1, HNB, SP-2DE, dan A1DE di 10 inner folds."""
        n_tr, d = F_tr.shape
        n_sp = sp_max_pairs
        k1_oof = np.zeros((n_tr, d), dtype=np.float64)
        hnb_oof = np.zeros((n_tr, 2 * d), dtype=np.float64)
        sp2_oof = np.zeros((n_tr, d + n_sp), dtype=np.float64)
        a1_cols = min(top_a1de, d) if top_a1de > 0 else 0
        a1de_oof = np.zeros((n_tr, a1_cols), dtype=np.float64)

        for k in range(len(pos_folds)):
            in_va = np.concatenate([pos_folds[k], neg_folds[k]])
            m_tr = np.ones(n_tr, dtype=bool)
            m_tr[in_va] = False
            in_tr = np.where(m_tr)[0]
            tb_k = self._fit_hnb_sp2de_tables(
                F_tr[in_tr],
                y_tr[in_tr],
                max_bins=max_bins,
                sp_max_pairs=sp_max_pairs,
                top_a1de=top_a1de,
                smooth_k1=smooth_k1,
            )
            if top_a1de > 0:
                k1_v, hnb_v, sp2_v, a1_v = self._eval_hnb_sp2de_tables(tb_k, F_tr[in_va], return_a1de=True)
                a1de_oof[in_va] = a1_v[:, :a1_cols]
            else:
                k1_v, hnb_v, sp2_v = self._eval_hnb_sp2de_tables(tb_k, F_tr[in_va])
            k1_oof[in_va] = k1_v
            hnb_oof[in_va] = hnb_v
            sp2_oof[in_va, : min(sp2_v.shape[1], sp2_oof.shape[1])] = sp2_v[:, : sp2_oof.shape[1]]

        tb_full = self._fit_hnb_sp2de_tables(
            F_tr,
            y_tr,
            max_bins=max_bins,
            sp_max_pairs=sp_max_pairs,
            top_a1de=top_a1de,
            smooth_k1=smooth_k1,
        )
        if top_a1de > 0:
            return k1_oof, hnb_oof, sp2_oof, a1de_oof, tb_full
        return k1_oof, hnb_oof, sp2_oof, tb_full

    @staticmethod
    def _exact_1d_kde_sorted_window(
        grp_tr: np.ndarray,
        grp_te: np.ndarray,
        y_tr: np.ndarray,
        feat_specs: List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]],
        is_self_loo: bool = False,
        prior_m: float = 3.0,
    ) -> np.ndarray:
        """
        Continuous 1D Conditional Parzen-Window KDE [John & Langley, UAI 1995] dengan exact
        sorted binary-search windowing (|x_te - x_tr| <= 5.0 * h) & exact Leave-One-Out.
        """
        n_te = len(grp_te)
        out = np.zeros((n_te, len(feat_specs) * 3), dtype=np.float64)
        tot_pos, n_tr_f = float(np.sum(y_tr)), float(len(y_tr))
        p1_glob = min(max(tot_pos / max(n_tr_f, 1.0), 1e-6), 1.0 - 1e-6)
        plo_glob = math.log(p1_glob / (1.0 - p1_glob))
        y_tr_f = y_tr.astype(np.float64)
        p1_row_all = np.clip((tot_pos - y_tr_f) / max(n_tr_f - 1.0, 1.0), 1e-6, 1.0 - 1e-6) if is_self_loo else None
        plo_all = np.log(p1_row_all / (1.0 - p1_row_all)) if p1_row_all is not None else None

        for g_val in np.unique(grp_tr):
            idx_tr = np.where(grp_tr == g_val)[0]
            idx_te = np.where(grp_te == g_val)[0]
            if len(idx_tr) < 10 or len(idx_te) == 0:
                continue
            p1_l = p1_row_all[idx_te] if p1_row_all is not None else p1_glob
            plo_l = plo_all[idx_te] if plo_all is not None else plo_glob
            for f_idx, (tr_arr, te_arr, bws) in enumerate(feat_specs):
                v_tr, v_te = tr_arr[idx_tr], te_arr[idx_te]
                o_tr = np.argsort(v_tr, kind="mergesort")
                v_tr_s, y_tr_s, orig_tr_s = v_tr[o_tr], y_tr_f[idx_tr[o_tr]].astype(np.float32), idx_tr[o_tr]
                o_te = np.argsort(v_te, kind="mergesort")
                v_te_s, orig_te_s = v_te[o_te], idx_te[o_te]

                for h_k, h_val in enumerate(bws):
                    ws_acc = np.zeros(len(idx_te), dtype=np.float64)
                    wy_acc = np.zeros(len(idx_te), dtype=np.float64)
                    max_w = 5.0 * h_val
                    inv_h = np.float32(1.0 / h_val)
                    for st in range(0, len(idx_te), 256):
                        en = min(st + 256, len(idx_te))
                        sub_v = v_te_s[st:en]
                        lo = int(np.searchsorted(v_tr_s, float(sub_v[0]) - max_w, side="left"))
                        hi = int(np.searchsorted(v_tr_s, float(sub_v[-1]) + max_w, side="right"))
                        if hi <= lo:
                            continue
                        u = (sub_v[:, None] - v_tr_s[None, lo:hi]).astype(np.float32) * inv_h
                        u_sq = u * u
                        W = np.where(u_sq <= 25.0, np.exp(np.float32(-0.5) * np.minimum(u_sq, 25.0)), np.float32(0.0))
                        if is_self_loo:
                            same = orig_te_s[st:en, None] == orig_tr_s[None, lo:hi]
                            W[same] = 0.0
                        ws_acc[o_te[st:en]] = np.sum(W, axis=1, dtype=np.float64)
                        wy_acc[o_te[st:en]] = (W @ y_tr_s[lo:hi]).astype(np.float64)
                    p_est = np.clip((wy_acc + prior_m * p1_l) / (ws_acc + prior_m), 1e-6, 1.0 - 1e-6)
                    out[idx_te, f_idx * 3 + h_k] = np.log(p_est / (1.0 - p_est)) - plo_l
        return out

    @classmethod
    def _compute_v19_continuous_kde_and_cohort(
        cls,
        s_tr: Dict[str, np.ndarray],
        rvrm_tr: np.ndarray,
        y_tr: np.ndarray,
        s_te: Dict[str, np.ndarray],
        rvrm_te: np.ndarray,
        ctx_tr: Dict[str, np.ndarray],
        ctx_te: Dict[str, np.ndarray],
        is_self_loo: bool = False,
    ) -> Tuple[List[Tuple[np.ndarray, float, float]], np.ndarray, np.ndarray]:
        """
        Menghitung 12 irisan Conditional 1D Parzen KDE (141 fitur), 6 pasangan Exact 2D Joint
        Continuous Parzen KDE (`float32` BLAS), dan 11 fitur True Leave-One-Out Cohort Density
        & Anomaly ternormalisasi ukuran sampel.
        """
        n_tr, n_te = len(y_tr), len(s_te["ym"])
        tr_doy = (s_tr["mo"] - 1) * 30.4375 + s_tr["dy"]
        te_doy = (s_te["mo"] - 1) * 30.4375 + s_te["dy"]
        tr_bday = (s_tr["yr"] - 2023) * 365.25 + tr_doy - ctx_tr["dba_imp"]
        te_bday = (s_te["yr"] - 2023) * 365.25 + te_doy - ctx_te["dba_imp"]

        tr_den = np.maximum(np.nan_to_num(s_tr["ad_raw"], nan=2.0), 1.0) + 0.5 * np.maximum(
            np.nan_to_num(s_tr["ch_raw"], nan=0.0), 0.0
        )
        te_den = np.maximum(np.nan_to_num(s_te["ad_raw"], nan=2.0), 1.0) + 0.5 * np.maximum(
            np.nan_to_num(s_te["ch_raw"], nan=0.0), 0.0
        )
        tr_rpn = ctx_tr["rate_imp"] / tr_den
        te_rpn = ctx_te["rate_imp"] / te_den
        tr_spend = ctx_tr["rate_imp"] * np.maximum(s_tr["tot_nights"].astype(np.float64), 1.0)
        te_spend = ctx_te["rate_imp"] * np.maximum(s_te["tot_nights"].astype(np.float64), 1.0)
        tr_lts = ctx_tr["dba_imp"] / (np.maximum(s_tr["tot_nights"].astype(np.float64), 1.0) + 0.5)
        te_lts = ctx_te["dba_imp"] / (np.maximum(s_te["tot_nights"].astype(np.float64), 1.0) + 0.5)

        dba_s = (ctx_tr["dba_imp"], ctx_te["dba_imp"], (1.5, 4.5, 12.0))
        rate_s = (ctx_tr["rate_imp"], ctx_te["rate_imp"], (20000.0, 60000.0, 150000.0))
        bday_s = (tr_bday, te_bday, (1.0, 3.0, 7.0))
        rvrm_s = (rvrm_tr, rvrm_te, (0.04, 0.12, 0.30))
        doy_s = (tr_doy, te_doy, (1.5, 5.0, 14.0))
        rpn_s = (tr_rpn, te_rpn, (18000.0, 50000.0, 120000.0))
        spend_s = (tr_spend, te_spend, (60000.0, 180000.0, 450000.0))
        lts_s = (tr_lts, te_lts, (0.6, 2.0, 5.0))

        grp_bc_sr_tr = s_tr["bc"] * 2 + (ctx_tr["sr_imp"] > 0).astype(np.int64)
        grp_bc_sr_te = s_te["bc"] * 2 + (ctx_te["sr_imp"] > 0).astype(np.int64)
        tr_wk_f = (s_tr["wknd"] > 0).astype(np.int64)
        te_wk_f = (s_te["wknd"] > 0).astype(np.int64)

        slice_configs = [
            (grp_bc_sr_tr, grp_bc_sr_te, [dba_s, rate_s, bday_s, rpn_s, spend_s, lts_s, rvrm_s, doy_s], 240.0, 650.0),
            (s_tr["bc"] * 10 + ctx_tr["rc_imp"], s_te["bc"] * 10 + ctx_te["rc_imp"],
             [(ctx_tr["rate_imp"], ctx_te["rate_imp"], (25000.0, 75000.0, 180000.0)), (ctx_tr["dba_imp"], ctx_te["dba_imp"], (2.0, 6.0, 16.0)), (tr_bday, te_bday, (1.5, 4.0, 9.0))], 260.0, 700.0),
            (s_tr["bc"] * 10 + ctx_tr["mp_imp"], s_te["bc"] * 10 + ctx_te["mp_imp"],
             [(ctx_tr["rate_imp"], ctx_te["rate_imp"], (25000.0, 75000.0, 180000.0)), (ctx_tr["dba_imp"], ctx_te["dba_imp"], (2.0, 6.0, 16.0)), (tr_bday, te_bday, (1.5, 4.0, 9.0))], 260.0, 700.0),
            (s_tr["ym"] * 10 + s_tr["bc"], s_te["ym"] * 10 + s_te["bc"],
             [(ctx_tr["rate_imp"], ctx_te["rate_imp"], (22000.0, 65000.0, 150000.0)), (ctx_tr["dba_imp"], ctx_te["dba_imp"], (2.0, 5.5, 14.0))], 280.0, 720.0),
            (s_tr["bc"] * 2 + s_tr["ret"], s_te["bc"] * 2 + s_te["ret"], [dba_s, rate_s, bday_s, rvrm_s], 260.0, 680.0),
            (s_tr["bc"] * 4 + np.clip(ctx_tr["sr_imp"], 0, 2), s_te["bc"] * 4 + np.clip(ctx_te["sr_imp"], 0, 2), [dba_s, rate_s, bday_s], 260.0, 680.0),
            (s_tr["bc"], s_te["bc"], [rpn_s, spend_s, rvrm_s, doy_s, lts_s], 260.0, 680.0),
            (s_tr["bc"] * 2 + tr_wk_f, s_te["bc"] * 2 + te_wk_f, [rate_s, dba_s, bday_s], 260.0, 680.0),
            (s_tr["bc"] * 4 + (s_tr["yr"] - 2022), s_te["bc"] * 4 + (s_te["yr"] - 2022), [rate_s, dba_s, bday_s, rvrm_s, doy_s], 260.0, 680.0),
            (s_tr["bc"] * 8 + np.clip(s_tr["tot_nights"], 1, 4), s_te["bc"] * 8 + np.clip(s_te["tot_nights"], 1, 4), [rate_s, dba_s, bday_s], 260.0, 680.0),
            ((s_tr["bc"] * 2 + tr_wk_f) * 2 + (ctx_tr["sr_imp"] > 0).astype(np.int64),
             (s_te["bc"] * 2 + te_wk_f) * 2 + (ctx_te["sr_imp"] > 0).astype(np.int64), [rate_s, dba_s, bday_s, rvrm_s], 260.0, 680.0),
            (ctx_tr["rc_imp"] * 10 + ctx_tr["mp_imp"], ctx_te["rc_imp"] * 10 + ctx_te["mp_imp"],
             [(ctx_tr["rate_imp"], ctx_te["rate_imp"], (25000.0, 75000.0, 180000.0)), (ctx_tr["dba_imp"], ctx_te["dba_imp"], (2.0, 6.0, 15.0)), (tr_bday, te_bday, (1.5, 4.0, 9.0)), rvrm_s], 260.0, 680.0),
        ]

        k1d_blocks = [
            (cls._exact_1d_kde_sorted_window(gtr, gte, y_tr, specs, is_self_loo=is_self_loo), r_l, r_h)
            for gtr, gte, specs, r_l, r_h in slice_configs
        ]

        # Exact 2D Joint Continuous Parzen KDE inside (bc, has_sr) in float32 BLAS
        k2d = np.zeros((n_te, 18), dtype=np.float64)
        tot_pos = float(np.sum(y_tr))
        p1_glob = min(max(tot_pos / max(float(n_tr), 1.0), 1e-6), 1.0 - 1e-6)
        plo_glob = math.log(p1_glob / (1.0 - p1_glob))
        y_tr_f = y_tr.astype(np.float64)
        y_tr_32 = y_tr.astype(np.float32)
        p1_row_all = np.clip((tot_pos - y_tr_f) / max(float(n_tr) - 1.0, 1.0), 1e-6, 1.0 - 1e-6) if is_self_loo else None
        plo_all = np.log(p1_row_all / (1.0 - p1_row_all)) if p1_row_all is not None else None
        c_bws = (np.float32(-0.5 / (0.65 ** 2)), np.float32(-0.5 / (1.25 ** 2)), np.float32(-0.5 / (2.40 ** 2)))

        for g_val in np.unique(grp_bc_sr_tr):
            idx_tr = np.where(grp_bc_sr_tr == g_val)[0]
            idx_te = np.where(grp_bc_sr_te == g_val)[0]
            if len(idx_tr) < 10 or len(idx_te) == 0:
                continue
            y_sub32 = y_tr_32[idx_tr]
            p1_l = p1_row_all[idx_te] if p1_row_all is not None else p1_glob
            plo_l = plo_all[idx_te] if plo_all is not None else plo_glob

            dba_tr_s = ctx_tr["dba_imp"][idx_tr].astype(np.float32)
            dba_te_s = ctx_te["dba_imp"][idx_te].astype(np.float32)
            r_tr_s = ctx_tr["rate_imp"][idx_tr].astype(np.float32)
            r_te_s = ctx_te["rate_imp"][idx_te].astype(np.float32)
            bd_tr_s = tr_bday[idx_tr].astype(np.float32)
            bd_te_s = te_bday[idx_te].astype(np.float32)
            lts_tr_s = tr_lts[idx_tr].astype(np.float32)
            lts_te_s = te_lts[idx_te].astype(np.float32)
            rv_tr_s = rvrm_tr[idx_tr].astype(np.float32)
            rv_te_s = rvrm_te[idx_te].astype(np.float32)
            dy_tr_s = tr_doy[idx_tr].astype(np.float32)
            dy_te_s = te_doy[idx_te].astype(np.float32)

            ws = [np.zeros(len(idx_te), dtype=np.float64) for _ in range(18)]
            wy = [np.zeros(len(idx_te), dtype=np.float64) for _ in range(18)]
            for st in range(0, len(idx_te), 512):
                en = min(st + 512, len(idx_te))
                sub_i = idx_te[st:en]
                same = (sub_i[:, None] == idx_tr[None, :]) if is_self_loo else None
                u_dba = np.minimum(((dba_te_s[st:en, None] - dba_tr_s[None, :]) / np.maximum(np.float32(3.5), np.float32(0.02) * (dba_te_s[st:en, None] + dba_tr_s[None, :]))) ** 2, np.float32(25.0))
                u_r = np.minimum(((r_te_s[st:en, None] - r_tr_s[None, :]) / np.maximum(np.float32(25000.0), np.float32(0.016) * (r_te_s[st:en, None] + r_tr_s[None, :]))) ** 2, np.float32(25.0))
                u_bd = np.minimum(((bd_te_s[st:en, None] - bd_tr_s[None, :]) / np.float32(1.5)) ** 2, np.float32(25.0))
                u_lts = np.minimum(((lts_te_s[st:en, None] - lts_tr_s[None, :]) / np.maximum(np.float32(1.2), np.float32(0.04) * (lts_te_s[st:en, None] + lts_tr_s[None, :]))) ** 2, np.float32(25.0))
                u_rv = np.minimum(((rv_te_s[st:en, None] - rv_tr_s[None, :]) / np.float32(0.06)) ** 2, np.float32(25.0))
                u_dy = np.minimum(((dy_te_s[st:en, None] - dy_tr_s[None, :]) / np.float32(3.0)) ** 2, np.float32(25.0))
                for p_k, d2 in enumerate((u_dba + u_r, u_bd + u_r, u_dba + u_bd, u_dba + u_lts, u_r + u_rv, u_dy + u_r)):
                    for h_k, c_h in enumerate(c_bws):
                        W = np.exp(c_h * d2)
                        if same is not None:
                            W[same] = 0.0
                        col = p_k * 3 + h_k
                        ws[col][st:en] = np.sum(W, axis=1, dtype=np.float64)
                        wy[col][st:en] = (W @ y_sub32).astype(np.float64)
            for col in range(18):
                p_est = np.clip((wy[col] + 2.5 * p1_l) / (ws[col] + 2.5), 1e-6, 1.0 - 1e-6)
                k2d[idx_te, col] = np.log(p_est / (1.0 - p_est)) - plo_l

        # True Leave-One-Out & Sample-Size-Normalized Cohort Densities & Z-Scores (11 cols)
        n_ref = float(len(s_tr["ym"]))
        scale = 27300.0 / max((n_ref - 1.0) if is_self_loo else n_ref, 1.0)
        bday_int = np.round(te_bday).astype(np.int64)
        tr_bday_int = np.round(tr_bday).astype(np.int64)

        def loo_cnt(tr_keys: np.ndarray, cur_keys: np.ndarray) -> np.ndarray:
            u_k, c_k = np.unique(tr_keys, return_counts=True)
            pos = np.clip(np.searchsorted(u_k, cur_keys), 0, len(u_k) - 1)
            raw = np.where(u_k[pos] == cur_keys, c_k[pos], 0).astype(np.float64)
            if is_self_loo:
                raw = np.maximum(0.0, raw - 1.0)
            return raw * scale

        def loo_rate_z(tr_keys: np.ndarray, cur_keys: np.ndarray, min_c: float, floor_s: float) -> Tuple[np.ndarray, np.ndarray]:
            u_g, inv_g, cnt_g = np.unique(tr_keys, return_inverse=True, return_counts=True)
            pos_g = np.clip(np.searchsorted(u_g, cur_keys), 0, len(u_g) - 1)
            m_g = u_g[pos_g] == cur_keys
            sum1 = np.bincount(inv_g, weights=ctx_tr["rate_imp"])
            sum2 = np.bincount(inv_g, weights=ctx_tr["rate_imp"] ** 2)
            c_raw = np.where(m_g, cnt_g[pos_g], 0).astype(np.float64)
            s1 = np.where(m_g, sum1[pos_g], 0.0)
            s2 = np.where(m_g, sum2[pos_g], 0.0)
            r_cur = ctx_te["rate_imp"]
            if is_self_loo:
                c_eff = np.maximum(0.0, c_raw - 1.0)
                s1 = np.where(c_raw >= 1.0, s1 - r_cur, s1)
                s2 = np.where(c_raw >= 1.0, s2 - r_cur ** 2, s2)
            else:
                c_eff = c_raw
            ok = c_eff >= min_c
            m_v = np.where(ok, s1 / np.maximum(c_eff, 1.0), r_cur)
            v_v = np.where(ok, np.maximum(0.0, s2 / np.maximum(c_eff, 1.0) - m_v ** 2), 1e10)
            return np.clip((r_cur - m_v) / (np.sqrt(v_v) + floor_s), -6.0, 6.0), m_v

        d_all_c = loo_cnt(s_tr["date_id"], s_te["date_id"])
        d_bc_c = loo_cnt(s_tr["date_id"] * 10 + s_tr["bc"], s_te["date_id"] * 10 + s_te["bc"])
        d_bs_c = loo_cnt((s_tr["date_id"] * 10 + s_tr["bc"]) * 200 + s_tr["stay"], (s_te["date_id"] * 10 + s_te["bc"]) * 200 + s_te["stay"])
        bd_all_c = loo_cnt(tr_bday_int, bday_int)
        bd_bc_c = loo_cnt(tr_bday_int * 10 + s_tr["bc"], bday_int * 10 + s_te["bc"])
        bd2_bc_c = loo_cnt((tr_bday_int // 2) * 10 + s_tr["bc"], (bday_int // 2) * 10 + s_te["bc"])

        z_rb, _ = loo_rate_z((tr_bday_int // 2) * 10 + s_tr["bc"], (bday_int // 2) * 10 + s_te["bc"], 2.0, 50000.0)
        z_rp, m_pr_v = loo_rate_z(
            (((s_tr["ym"] * 10 + s_tr["bc"]) * 10 + ctx_tr["rc_imp"]) * 10 + ctx_tr["mp_imp"]) * 2 + tr_wk_f,
            (((s_te["ym"] * 10 + s_te["bc"]) * 10 + ctx_te["rc_imp"]) * 10 + ctx_te["mp_imp"]) * 2 + te_wk_f,
            3.0,
            45000.0,
        )
        log_rp = np.log(np.clip(ctx_te["rate_imp"] / np.maximum(m_pr_v, 100000.0), 0.25, 4.0))
        coh_11 = np.column_stack([
            np.log1p(d_all_c), np.log1p(d_bc_c), np.log1p(d_bs_c),
            np.log1p(bd_all_c), np.log1p(bd_bc_c), np.log1p(bd2_bc_c),
            z_rb, np.abs(z_rb), z_rp, np.abs(z_rp), log_rp,
        ])
        return k1d_blocks, k2d, coh_11

    @staticmethod
    def _extract_v19_orthogonal_cliques(
        s: Dict[str, np.ndarray], ctx: Dict[str, np.ndarray]
    ) -> List[Tuple[np.ndarray, float]]:
        """Mengekstrak 18 clique ortogonal tambahan (booking-date dow, weekly cohort, rate-per-guest)."""
        n = len(s["ym"])
        bday = (s["yr"] - 2023) * 365.25 + (s["mo"] - 1) * 30.4375 + s["dy"] - ctx["dba_imp"]
        den = np.maximum(np.nan_to_num(s["ad_raw"], nan=2.0), 1.0) + 0.5 * np.maximum(np.nan_to_num(s["ch_raw"], nan=0.0), 0.0)
        rpn = ctx["rate_imp"] / den

        def pack(*arrays: np.ndarray) -> np.ndarray:
            res = np.zeros(n, dtype=np.int64)
            for a in arrays:
                res = res * 1009 + a
            return res

        bd_int = np.round(bday).astype(np.int64) + 1000
        bd2 = np.floor(bday / 2.0).astype(np.int64) + 1000
        bd2s = np.floor((bday + 1.0) / 2.0).astype(np.int64) + 1000
        bd4 = np.floor(bday / 4.0).astype(np.int64) + 1000
        bd_wk = np.floor(bday / 7.0).astype(np.int64) + 1000
        arr_wk = s["date_id"] // 7
        bc, rc, mp = s["bc"], ctx["rc_imp"], ctx["mp_imp"]
        sr_z = (ctx["sr_imp"] > 0).astype(np.int64)
        sr_b = ctx["sr_bin_imp"]
        stay, ret, did, ym = s["stay"], s["ret"], s["date_id"], s["ym"]
        r80 = np.floor(ctx["rate_imp"] / 80000.0).astype(np.int64) + 1
        r50 = np.floor(ctx["rate_imp"] / 50000.0).astype(np.int64) + 1
        d12 = np.floor(ctx["dba_imp"] / 12.0).astype(np.int64) + 1
        rpn50 = np.floor(rpn / 50000.0).astype(np.int64) + 1
        ad = np.clip(np.nan_to_num(s["ad_raw"], nan=2.0).round().astype(np.int64), 0, 4)
        ch = np.clip(np.nan_to_num(s["ch_raw"], nan=0.0).round().astype(np.int64), 0, 3)
        np_c = np.clip(np.nan_to_num(s["np_raw"], nan=0.0).round().astype(np.int64), 0, 1)
        return [
            (pack(bd_int, bc, rc, mp, r50), 2.5),
            (pack(bd_int, bc, rc, sr_z), 2.5),
            (pack(bd2, bc, rc, mp, sr_z), 2.5),
            (pack(bd2s, bc, rc, mp, sr_z), 2.5),
            (pack(bd2, bc, stay, sr_b), 2.5),
            (pack(bd4, bc, rc, r80, sr_z), 2.5),
            (pack(bd_wk, bc, rc, sr_b), 3.0),
            (pack(bd_wk, bc, r80, d12), 3.0),
            (pack(arr_wk, bc, rc, sr_z), 3.0),
            (pack(arr_wk, bc, r80, sr_b), 3.0),
            (pack(bd_int % 7, bc, sr_b), 5.0),
            (pack(did % 7, bc, stay, sr_z), 5.0),
            (pack(rpn50, bc, sr_b), 4.0),
            (pack(rpn50, ym, bc, rc), 3.5),
            (pack(bc, rc, mp, stay, sr_b), 4.0),
            (pack(bc, sr_b, ret, r80, d12), 3.5),
            (pack(bc, ad, ch, np_c, ret, sr_z), 4.0),
            (pack(did, bc, rc, mp, sr_z), 2.5),
        ]

    @staticmethod
    def _build_v19_cll_basis_and_prior(
        m_base_in: np.ndarray,
        l2_base_in: np.ndarray,
        n_base_cols: int,
        d_p1: int,
        d_p2: int,
        k1d_blocks: List[Tuple[np.ndarray, float, float]],
        k2d: np.ndarray,
        coh_11: np.ndarray,
        clq_v19: np.ndarray,
        bn38_hnb: np.ndarray,
        bn43_hnb: np.ndarray,
        bn38_sp2: np.ndarray,
        bn43_sp2: np.ndarray,
        bn38_k1: np.ndarray,
        bn43_k1: np.ndarray,
        strict_w: np.ndarray,
        l2_mult: float = 2.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Menyatukan basis k-DB, HNB, SP-2DE, Continuous 1D/2D KDE, dan LOO Cohort dengan Group Prior L2."""
        m_base = m_base_in.copy()
        d1 = 1.0 / (1.0 + 0.50 * strict_w[:, None])
        d2 = 1.0 / (1.0 + 1.50 * strict_w[:, None])

        # Stabilize Level-1 node LLRs dynamically without breaking higher levels or net_sums
        off1 = n_base_cols + d_p1
        off2 = n_base_cols + 8 * d_p1 + d_p1
        off3 = n_base_cols + 22 * d_p1 + d_p2
        off4 = n_base_cols + 22 * d_p1 + 9 * d_p2
        m_base[:, off1 : off1 + d_p1] = bn38_k1 * d1
        m_base[:, off2 : off2 + d_p1] = bn38_k1 * d2
        m_base[:, off3 : off3 + d_p2] = bn43_k1 * d1
        m_base[:, off4 : off4 + d_p2] = bn43_k1 * d2

        parts = [m_base]
        l2_parts = [l2_base_in * l2_mult]

        knots = (-1.0, 0.0, 1.0)
        for mat, r_lin, r_h in k1d_blocks + [(k2d, 240.0, 650.0), (coh_11, 320.0, 750.0)]:
            h_mat = np.hstack([mat] + [np.maximum(0.0, mat - k) for k in knots] + [np.maximum(0.0, k - mat) for k in knots])
            n0 = mat.shape[1]
            parts.append(h_mat)
            l2_parts.append(
                np.concatenate([
                    np.full(n0, r_lin * l2_mult, dtype=np.float64),
                    np.full(h_mat.shape[1] - n0, r_h * l2_mult, dtype=np.float64),
                ])
            )

        for bn_mat, reg_b in ((bn38_hnb, 1100.0), (bn43_hnb, 1200.0), (bn38_sp2, 1100.0), (bn43_sp2, 1200.0)):
            parts.append(bn_mat)
            l2_parts.append(np.full(bn_mat.shape[1], reg_b * l2_mult, dtype=np.float64))

        clq_d = np.hstack([clq_v19 * d1, clq_v19 * d2])
        parts.append(clq_d)
        l2_parts.append(np.full(clq_d.shape[1], 550.0 * l2_mult, dtype=np.float64))

        return np.hstack(parts), np.concatenate(l2_parts)

    @staticmethod
    def _eval_dirichlet_lwnb_pair(
        W_mat: np.ndarray,
        ys: np.ndarray,
        y0s: np.ndarray,
        oh_s1: np.ndarray,
        oh_s0: np.ndarray,
        p1_loc: Union[float, np.ndarray],
        plo_loc: Union[float, np.ndarray],
        offsets: List[Tuple[int, int]],
        cols_te: List[np.ndarray],
        ie_sub: np.ndarray,
        p_glob_c1: np.ndarray,
        p_glob_c0: np.ndarray,
        glob_llr_vec: np.ndarray,
        m_dir: float,
        m_pr: float,
        fd_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Menghitung pair (prior_only, LWNB_total) ter-kalibrasi Dirichlet [Frank et al., UAI 2003]."""
        w1 = (W_mat @ ys).astype(np.float64)
        w0 = (W_mat @ y0s).astype(np.float64)
        pe = np.clip((w1 + m_pr * p1_loc) / (w1 + w0 + m_pr), 1e-6, 1.0 - 1e-6)
        pr_d = np.log(pe / (1.0 - pe)) - plo_loc
        w1_c = (W_mat @ oh_s1).astype(np.float64)
        w0_c = (W_mat @ oh_s0).astype(np.float64)
        n_q = len(ie_sub)
        tot_fd = np.zeros(n_q, dtype=np.float64)
        row_ar = np.arange(n_q)
        for f_i, (lo, hi) in enumerate(offsets):
            v_q = np.clip(cols_te[f_i][ie_sub], 0, (hi - lo) - 1)
            c_idx = lo + v_q
            p_l1 = np.clip((w1_c[row_ar, c_idx] + m_dir * p_glob_c1[c_idx]) / (w1 + m_dir), 1e-6, 1.0 - 1e-6)
            p_l0 = np.clip((w0_c[row_ar, c_idx] + m_dir * p_glob_c0[c_idx]) / (w0 + m_dir), 1e-6, 1.0 - 1e-6)
            tot_fd += np.clip(np.log(p_l1 / p_l0) - glob_llr_vec[c_idx], -2.5, 2.5)
        return pr_d, np.clip(pr_d + fd_scale * tot_fd, -10.0, 10.0)

    def _compute_v20_continuous_kde_and_lwnb(
        self,
        s_tr: Dict[str, np.ndarray],
        ctx_tr: Dict[str, np.ndarray],
        y_tr: np.ndarray,
        rvrm_tr: np.ndarray,
        s_te: Dict[str, np.ndarray],
        ctx_te: Dict[str, np.ndarray],
        rvrm_te: np.ndarray,
        is_self_loo: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Menghitung 4 blok kernel KDE & LWNB across-month / date-free berdimensi ringkas untuk V20:
        - bk (24 kol): 3D Parzen batch/wave kalender sejati inside (ym, bc) & (bc, yr)
        - lw (17 kol): Across-month Wave/Seas/Full Parzen + Dirichlet LWNB (k=45, 100, h=1.15)
        - pr (14 kol): Date-free Product+Behavior Parzen & LWNB (k=40, 100) + Full multi-k (25, 50, 100)
        - k2f (12 kol): Date-free 2D Parzen inside (bc) untuk isolated rows tanpa cohort kembar
        Juga mengembalikan wd_te (weekday kalender) untuk gated summary signals.
        """
        cal_tr, wd_tr = self._compute_cal_ord_and_wd(s_tr["yr"], s_tr["mo"], s_tr["dy"])
        cal_te, wd_te = (cal_tr, wd_tr) if is_self_loo else self._compute_cal_ord_and_wd(s_te["yr"], s_te["mo"], s_te["dy"])

        n_tr, n_te = len(y_tr), len(s_te["ym"])
        y_f = y_tr.astype(np.float64)
        y_32 = y_tr.astype(np.float32)
        y0_32 = np.float32(1.0) - y_32
        tot_pos = float(np.sum(y_tr))
        p1_g = min(max(tot_pos / max(float(n_tr), 1.0), 1e-6), 1.0 - 1e-6)
        plo_g = math.log(p1_g / (1.0 - p1_g))
        p1_r = np.clip((tot_pos - y_f) / max(float(n_tr) - 1.0, 1.0), 1e-6, 1.0 - 1e-6) if is_self_loo else None
        plo_r = np.log(p1_r / (1.0 - p1_r)) if p1_r is not None else None
        n_scale = 27300.0 / max((float(n_tr) - 1.0) if is_self_loo else float(n_tr), 1.0)

        dba_tr, dba_te = ctx_tr["dba_imp"].astype(np.float32), ctx_te["dba_imp"].astype(np.float32)
        rate_tr, rate_te = ctx_tr["rate_imp"].astype(np.float32), ctx_te["rate_imp"].astype(np.float32)
        bday_tr = (cal_tr.astype(np.float64) - ctx_tr["dba_imp"]).astype(np.float32)
        bday_te = (cal_te.astype(np.float64) - ctx_te["dba_imp"]).astype(np.float32)
        cal32_tr, cal32_te = (cal_tr - 738500).astype(np.float32), (cal_te - 738500).astype(np.float32)
        rv_tr, rv_te = rvrm_tr.astype(np.float32), rvrm_te.astype(np.float32)

        # 1. 3D Continuous Batch/Wave Calendar Parzen KDE (24 kol)
        bk = np.zeros((n_te, 24), dtype=np.float64)
        for grp_tr, grp_te, col_off, bws in (
            (s_tr["ym"] * 10 + s_tr["bc"], s_te["ym"] * 10 + s_te["bc"], 0, (0.70, 1.40, 2.60)),
            (s_tr["bc"] * 4 + (s_tr["yr"] - 2022), s_te["bc"] * 4 + (s_te["yr"] - 2022), 12, (0.75, 1.50, 2.80)),
        ):
            c_bws = [np.float32(-0.5 / (h ** 2)) for h in bws]
            for gv in np.unique(grp_tr):
                it = np.where(grp_tr == gv)[0]
                ie = np.where(grp_te == gv)[0]
                if len(it) < 12 or len(ie) == 0:
                    continue
                y_s = y_32[it]
                p1_l = p1_r[ie] if p1_r is not None else p1_g
                plo_l = plo_r[ie] if plo_r is not None else plo_g
                ws = [np.zeros(len(ie), dtype=np.float64) for _ in range(9)]
                wy = [np.zeros(len(ie), dtype=np.float64) for _ in range(9)]
                for st in range(0, len(ie), 512):
                    en = min(st + 512, len(ie))
                    same = (ie[st:en, None] == it[None, :]) if is_self_loo else None
                    u_cal = np.minimum(((cal32_te[ie[st:en], None] - cal32_tr[None, it]) / np.float32(2.0)) ** 2, np.float32(25.0))
                    u_bd = np.minimum(((bday_te[ie[st:en], None] - bday_tr[None, it]) / np.float32(1.5)) ** 2, np.float32(25.0))
                    u_r = np.minimum(
                        (
                            (rate_te[ie[st:en], None] - rate_tr[None, it])
                            / np.maximum(np.float32(25000.0), np.float32(0.015) * (rate_te[ie[st:en], None] + rate_tr[None, it]))
                        ) ** 2,
                        np.float32(25.0),
                    )
                    u_dba = np.minimum(
                        (
                            (dba_te[ie[st:en], None] - dba_tr[None, it])
                            / np.maximum(np.float32(3.5), np.float32(0.02) * (dba_te[ie[st:en], None] + dba_tr[None, it]))
                        ) ** 2,
                        np.float32(25.0),
                    )
                    u_rv = np.minimum(((rv_te[ie[st:en], None] - rv_tr[None, it]) / np.float32(0.06)) ** 2, np.float32(25.0))
                    for p_k, d2 in enumerate((u_cal + u_bd + u_r, u_cal + u_bd + u_dba, u_bd + u_r + u_rv)):
                        for h_k, c_h in enumerate(c_bws):
                            W = np.exp(c_h * d2)
                            if same is not None:
                                W[same] = np.float32(0.0)
                            ci = p_k * 3 + h_k
                            ws[ci][st:en] = np.sum(W, axis=1, dtype=np.float64)
                            wy[ci][st:en] = (W @ y_s).astype(np.float64)
                for ci in range(9):
                    pe = np.clip((wy[ci] + 2.5 * p1_l) / (ws[ci] + 2.5), 1e-6, 1.0 - 1e-6)
                    bk[ie, col_off + ci] = np.log(pe / (1.0 - pe)) - plo_l
                for d_k, s_col in enumerate((1, 4, 7)):
                    bk[ie, col_off + 9 + d_k] = np.log1p(ws[s_col] * n_scale)

        # 2. Bangun representasi fitur & one-hot untuk lw (17), pr (14), dan k2f (12)
        dba_q = np.unique(np.percentile(ctx_tr["dba_imp"], np.linspace(0, 100, 9)[1:-1]))
        rate_q = np.unique(np.percentile(ctx_tr["rate_imp"], np.linspace(0, 100, 9)[1:-1]))

        def build_oh_and_vecs(s: Dict[str, np.ndarray], ctx: Dict[str, np.ndarray], cal: np.ndarray, wd: np.ndarray, rv: np.ndarray):
            n = len(cal)
            sr_c = np.clip(ctx["sr_bin_imp"], 0, 3)
            rc_c = np.clip(ctx["rc_imp"] - 1, 0, 6)
            mp_c = np.clip(ctx["mp_imp"] - 1, 0, 3)
            sty_c = np.clip(s["tot_nights"], 0, 5)
            wk_c = (s["wknd"] > 0).astype(np.int64)
            ret_c = s["ret"]
            np_c = np.where(np.isnan(s["np_raw"]), 2, np.clip(np.nan_to_num(s["np_raw"], nan=0.0).round().astype(np.int64), 0, 1))
            wd_c = np.clip(wd, 0, 6)
            dba_c = np.searchsorted(dba_q, ctx["dba_imp"], side="right")
            rt_c = np.searchsorted(rate_q, ctx["rate_imp"], side="right")
            miss_raw = np.isnan(s["dba_raw"]).astype(np.int64) + np.isnan(s["rate_raw"]).astype(np.int64) + np.isnan(s["sr_raw"]).astype(np.int64)
            miss_2 = (miss_raw > 0).astype(np.int64)
            miss_3 = np.clip(miss_raw, 0, 2)

            base_cards = [4, 7, 4, 6, 2, 2, 3, 7, len(dba_q) + 1, len(rate_q) + 1]
            base_cols = [sr_c, rc_c, mp_c, sty_c, wk_c, ret_c, np_c, wd_c, dba_c, rt_c]

            def encode_oh(miss_arr: np.ndarray, miss_card: int):
                cards = base_cards + [miss_card]
                cols_i = base_cols + [miss_arr]
                oh = np.zeros((n, sum(cards)), dtype=np.float32)
                offs, cur = [], 0
                ar = np.arange(n)
                for c_arr, kc in zip(cols_i, cards):
                    oh[ar, cur + np.clip(c_arr, 0, kc - 1)] = 1.0
                    offs.append((cur, cur + kc))
                    cur += kc
                return oh, offs, cols_i

            oh2, off2, cols2 = encode_oh(miss_2, 2)
            oh3, off3, cols3 = encode_oh(miss_3, 3)

            cal_f = (cal - 738800).astype(np.float32)
            bd_f = (cal.astype(np.float64) - ctx["dba_imp"] - 738800.0).astype(np.float32)
            dba_l = np.log1p(np.maximum(ctx["dba_imp"], 0.0)).astype(np.float32)
            rate_l = np.log1p(np.maximum(ctx["rate_imp"], 0.0) / 100000.0).astype(np.float32)
            doy = ((s["mo"] - 1) * 30.4375 + s["dy"]).astype(np.float32)
            doy_s = np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)
            doy_c = np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)
            tot_n = np.maximum(s["tot_nights"].astype(np.float32), 1.0)
            spend_l = np.log1p(np.maximum(ctx["rate_imp"], 0.0) * tot_n / 100000.0).astype(np.float32)
            lts_l = np.log1p(np.maximum(ctx["dba_imp"], 0.0) / (tot_n + 0.5)).astype(np.float32)

            V_wave = np.column_stack([cal_f / 3.0, bd_f / 2.0, dba_l / 0.25, rate_l / 0.08, rv / 0.08]).astype(np.float32)
            V_full = np.column_stack([
                cal_f / 3.5, bd_f / 2.2, dba_l / 0.22, rate_l / 0.07, rv / 0.08,
                s["wknd"].astype(np.float32) / 0.75, s["wkdy"].astype(np.float32) / 0.95,
                ctx["sr_imp"].astype(np.float32) / 0.55, s["ret"].astype(np.float32) * 2.0,
                np.nan_to_num(s["np_raw"], nan=-1.0).astype(np.float32) * 1.4,
                oh2[:, 4:15] * np.float32(1.4),
            ]).astype(np.float32)
            V_seas = np.column_stack([
                doy_s / 0.08, doy_c / 0.08, dba_l / 0.22, rate_l / 0.07, rv / 0.08,
                s["wknd"].astype(np.float32) / 0.85, s["wkdy"].astype(np.float32) / 1.05,
                ctx["sr_imp"].astype(np.float32) / 0.60, oh2[:, 4:15] * np.float32(1.4),
            ]).astype(np.float32)
            V_prod = np.column_stack([
                dba_l / 0.28, rate_l / 0.10, rv / 0.08, spend_l / 0.16, lts_l / 0.25,
                s["wknd"].astype(np.float32) / 0.70, s["wkdy"].astype(np.float32) / 0.90,
                ctx["sr_imp"].astype(np.float32) / 0.50, s["ret"].astype(np.float32) * 2.2,
                np.minimum(s["pc"].astype(np.float32), 2.0) * 2.5,
                np.minimum(s["po"].astype(np.float32), 2.0) * 2.5,
                np.nan_to_num(s["np_raw"], nan=-1.0).astype(np.float32) * 1.4,
                np.isnan(s["dba_raw"]).astype(np.float32) * 2.0,
                np.isnan(s["rate_raw"]).astype(np.float32) * 1.8,
                np.isnan(s["sr_raw"]).astype(np.float32) * 1.5,
                (wd == 6).astype(np.float32) * 0.9,
                oh3[:, 4:15] * np.float32(1.4),
            ]).astype(np.float32)
            return V_wave, V_full, V_seas, V_prod, dba_l, rate_l, oh2, off2, cols2, oh3, off3, cols3

        Vw_tr, Vf_tr, Vs_tr, Vp_tr, dba_l_tr, rt_l_tr, oh2_tr, off2, _, oh3_tr, off3, _ = build_oh_and_vecs(
            s_tr, ctx_tr, cal_tr, wd_tr, rv_tr
        )
        Vw_te, Vf_te, Vs_te, Vp_te, dba_l_te, rt_l_te, _, _, cols2_te, _, _, cols3_te = build_oh_and_vecs(
            s_te, ctx_te, cal_te, wd_te, rv_te
        )

        def glob_oh_stats(oh: np.ndarray, offs: List[Tuple[int, int]]):
            p1 = np.zeros(oh.shape[1], dtype=np.float64)
            p0 = np.zeros(oh.shape[1], dtype=np.float64)
            n1_v, n0_v = float(np.sum(y_tr == 1)), float(np.sum(y_tr == 0))
            for lo, hi in offs:
                kc = hi - lo
                p1[lo:hi] = (np.sum(oh[y_tr == 1, lo:hi], axis=0) + 1.0) / (n1_v + kc)
                p0[lo:hi] = (np.sum(oh[y_tr == 0, lo:hi], axis=0) + 1.0) / (n0_v + kc)
            return p1, p0, np.log(p1 / p0)

        p1_g2, p0_g2, llr_g2 = glob_oh_stats(oh2_tr, off2)
        p1_g3, p0_g3, llr_g3 = glob_oh_stats(oh3_tr, off3)

        sr2d_tr, sr2d_te = ctx_tr["sr_imp"].astype(np.float32), ctx_te["sr_imp"].astype(np.float32)
        sty2d_tr = np.minimum(s_tr["tot_nights"].astype(np.float32), 7.0)
        sty2d_te = np.minimum(s_te["tot_nights"].astype(np.float32), 7.0)
        c_bws_2d = (np.float32(-0.5 / 0.50 ** 2), np.float32(-0.5 / 1.00 ** 2), np.float32(-0.5 / 2.00 ** 2))

        lw = np.zeros((n_te, 17), dtype=np.float64)
        pr = np.zeros((n_te, 14), dtype=np.float64)
        k2f = np.zeros((n_te, 12), dtype=np.float64)

        for bc_v in np.unique(s_tr["bc"]):
            it = np.where(s_tr["bc"] == bc_v)[0]
            ie = np.where(s_te["bc"] == bc_v)[0]
            if len(it) < 15 or len(ie) == 0:
                continue
            ys = y_32[it]
            y0s = y0_32[it]
            p1_l = p1_r[ie] if p1_r is not None else p1_g
            plo_l = plo_r[ie] if plo_r is not None else plo_g
            oh2_s1, oh2_s0 = oh2_tr[it] * ys[:, None], oh2_tr[it] * y0s[:, None]
            oh3_s1, oh3_s0 = oh3_tr[it] * ys[:, None], oh3_tr[it] * y0s[:, None]
            k_big = min(100, len(it) - (1 if is_self_loo else 0))

            # a) V_wave (lw[:, 0..3], lw[:, 15..16])
            Vw_t, Vw_e = Vw_tr[it], Vw_te[ie]
            w2t, w2e = np.sum(Vw_t * Vw_t, axis=1), np.sum(Vw_e * Vw_e, axis=1)
            for st in range(0, len(ie), 512):
                en = min(st + 512, len(ie))
                d2 = np.maximum(w2e[st:en, None] + w2t[None, :] - np.float32(2.0) * (Vw_e[st:en] @ Vw_t.T), np.float32(0.0))
                same = (ie[st:en, None] == it[None, :]) if is_self_loo else None
                if same is not None:
                    d2[same] = np.float32(1e12)
                p1_c = p1_l[st:en] if isinstance(p1_l, np.ndarray) else p1_l
                plo_c = plo_l[st:en] if isinstance(plo_l, np.ndarray) else plo_l
                for h_i, h in enumerate((0.55, 1.15, 2.20)):
                    W = np.exp(np.maximum(np.float32(-0.5 / (h ** 2)) * d2, np.float32(-25.0)))
                    if same is not None:
                        W[same] = np.float32(0.0)
                    ws_b = np.sum(W, axis=1, dtype=np.float64)
                    pe = np.clip((W @ ys + 2.5 * p1_c) / (ws_b + 2.5), 1e-6, 1.0 - 1e-6)
                    lw[ie[st:en], h_i] = np.log(pe / (1.0 - pe)) - plo_c
                    if h_i == 1:
                        lw[ie[st:en], 3] = np.log1p(ws_b * n_scale)
                        pr_w, lw_w = self._eval_dirichlet_lwnb_pair(
                            W, ys, y0s, oh2_s1, oh2_s0, p1_c, plo_c, off2, cols2_te, ie[st:en], p1_g2, p0_g2, llr_g2, 14.0, 3.0, 0.50
                        )
                        lw[ie[st:en], 15] = pr_w
                        lw[ie[st:en], 16] = lw_w

            # b) V_seas (lw[:, 4..6])
            Vs_t, Vs_e = Vs_tr[it], Vs_te[ie]
            s2t, s2e = np.sum(Vs_t * Vs_t, axis=1), np.sum(Vs_e * Vs_e, axis=1)
            for st in range(0, len(ie), 512):
                en = min(st + 512, len(ie))
                d2 = np.maximum(s2e[st:en, None] + s2t[None, :] - np.float32(2.0) * (Vs_e[st:en] @ Vs_t.T), np.float32(0.0))
                same = (ie[st:en, None] == it[None, :]) if is_self_loo else None
                if same is not None:
                    d2[same] = np.float32(1e12)
                p1_c = p1_l[st:en] if isinstance(p1_l, np.ndarray) else p1_l
                plo_c = plo_l[st:en] if isinstance(plo_l, np.ndarray) else plo_l
                for h_i, h in enumerate((0.65, 1.35, 2.50)):
                    W = np.exp(np.maximum(np.float32(-0.5 / (h ** 2)) * d2, np.float32(-25.0)))
                    if same is not None:
                        W[same] = np.float32(0.0)
                    pe = np.clip((W @ ys + 2.5 * p1_c) / (np.sum(W, axis=1) + 2.5), 1e-6, 1.0 - 1e-6)
                    lw[ie[st:en], 4 + h_i] = np.log(pe / (1.0 - pe)) - plo_c

            # c) V_full shared (lw[:, 7..14] + pr[:, 8..13])
            Vf_t, Vf_e = Vf_tr[it], Vf_te[ie]
            f2t, f2e = np.sum(Vf_t * Vf_t, axis=1), np.sum(Vf_e * Vf_e, axis=1)
            for st in range(0, len(ie), 512):
                en = min(st + 512, len(ie))
                d2 = np.maximum(f2e[st:en, None] + f2t[None, :] - np.float32(2.0) * (Vf_e[st:en] @ Vf_t.T), np.float32(0.0))
                same = (ie[st:en, None] == it[None, :]) if is_self_loo else None
                if same is not None:
                    d2[same] = np.float32(1e12)
                p1_c = p1_l[st:en] if isinstance(p1_l, np.ndarray) else p1_l
                plo_c = plo_l[st:en] if isinstance(plo_l, np.ndarray) else plo_l
                for h_i, h in enumerate((0.65, 1.30, 2.40)):
                    W = np.exp(np.maximum(np.float32(-0.5 / (h ** 2)) * d2, np.float32(-25.0)))
                    if same is not None:
                        W[same] = np.float32(0.0)
                    ws_b = np.sum(W, axis=1, dtype=np.float64)
                    pe = np.clip((W @ ys + 2.5 * p1_c) / (ws_b + 2.5), 1e-6, 1.0 - 1e-6)
                    lw[ie[st:en], 8 + h_i] = np.log(pe / (1.0 - pe)) - plo_c
                    if h_i == 1:
                        lw[ie[st:en], 7] = np.log1p(ws_b * n_scale)
                if k_big >= 10:
                    pk = np.argpartition(d2, k_big - 1, axis=1)[:, :k_big]
                    dk_u = np.take_along_axis(d2, pk, axis=1)
                    ord_k = np.argsort(dk_u, axis=1)
                    pk_s = np.take_along_axis(pk, ord_k, axis=1)
                    dk_s = np.sqrt(np.take_along_axis(d2, pk_s, axis=1))
                    row_b = np.arange(en - st)[:, None]
                    for col_o, kv, use_pr, md, mp, fds in (
                        (11, min(45, k_big), False, 12.0, 3.0, 0.50),
                        (13, k_big, False, 12.0, 3.0, 0.50),
                        (8, min(25, k_big), True, 10.0, 2.5, 1.0),
                        (10, min(50, k_big), True, 10.0, 2.5, 1.0),
                        (12, k_big, True, 10.0, 2.5, 1.0),
                    ):
                        rk = np.maximum(dk_s[:, kv - 1 : kv], np.float32(1e-4))
                        wk = (np.float32(1.0) - np.minimum(dk_s[:, :kv] / rk, np.float32(1.0)) ** 3) ** 3
                        W_k = np.zeros((en - st, len(it)), dtype=np.float32)
                        W_k[np.repeat(row_b, kv, axis=1), pk_s[:, :kv]] = wk
                        if use_pr:
                            pr_k, lw_k = self._eval_dirichlet_lwnb_pair(
                                W_k, ys, y0s, oh3_s1, oh3_s0, p1_c, plo_c, off3, cols3_te, ie[st:en], p1_g3, p0_g3, llr_g3, md, mp, fds
                            )
                            pr[ie[st:en], col_o] = pr_k
                            pr[ie[st:en], col_o + 1] = lw_k
                        else:
                            pr_k, lw_k = self._eval_dirichlet_lwnb_pair(
                                W_k, ys, y0s, oh2_s1, oh2_s0, p1_c, plo_c, off2, cols2_te, ie[st:en], p1_g2, p0_g2, llr_g2, md, mp, fds
                            )
                            lw[ie[st:en], col_o] = pr_k
                            lw[ie[st:en], col_o + 1] = lw_k

            # d) V_prod Date-Free (pr[:, 0..7]) + Date-Free 2D Parzen (k2f[:, 0..11])
            Vp_t, Vp_e = Vp_tr[it], Vp_te[ie]
            p2t, p2e = np.sum(Vp_t * Vp_t, axis=1), np.sum(Vp_e * Vp_e, axis=1)
            d_tr_s, d_te_s = dba_l_tr[it], dba_l_te[ie]
            r_tr_s, r_te_s = rt_l_tr[it], rt_l_te[ie]
            rv_tr_s, rv_te_s = rv_tr[it], rv_te[ie]
            sr_tr_s, sr_te_s = sr2d_tr[it], sr2d_te[ie]
            st_tr_s, st_te_s = sty2d_tr[it], sty2d_te[ie]
            ws2 = [np.zeros(len(ie), dtype=np.float64) for _ in range(12)]
            wy2 = [np.zeros(len(ie), dtype=np.float64) for _ in range(12)]

            for st in range(0, len(ie), 512):
                en = min(st + 512, len(ie))
                same = (ie[st:en, None] == it[None, :]) if is_self_loo else None
                p1_c = p1_l[st:en] if isinstance(p1_l, np.ndarray) else p1_l
                plo_c = plo_l[st:en] if isinstance(plo_l, np.ndarray) else plo_l

                d2p = np.maximum(p2e[st:en, None] + p2t[None, :] - np.float32(2.0) * (Vp_e[st:en] @ Vp_t.T), np.float32(0.0))
                if same is not None:
                    d2p[same] = np.float32(1e12)
                for hi_k, h in enumerate((0.65, 1.15, 2.00)):
                    W = np.exp(np.maximum(np.float32(-0.5 / (h ** 2)) * d2p, np.float32(-25.0)))
                    if same is not None:
                        W[same] = np.float32(0.0)
                    ws_b = np.sum(W, axis=1, dtype=np.float64)
                    pe = np.clip((W @ ys + 2.5 * p1_c) / (ws_b + 2.5), 1e-6, 1.0 - 1e-6)
                    pr[ie[st:en], hi_k] = np.log(pe / (1.0 - pe)) - plo_c
                    if hi_k == 1:
                        pr[ie[st:en], 3] = np.log1p(ws_b * n_scale)
                if k_big >= 10:
                    pk = np.argpartition(d2p, k_big - 1, axis=1)[:, :k_big]
                    dk_u = np.take_along_axis(d2p, pk, axis=1)
                    ord_k = np.argsort(dk_u, axis=1)
                    pk_s = np.take_along_axis(pk, ord_k, axis=1)
                    dk_s = np.sqrt(np.take_along_axis(d2p, pk_s, axis=1))
                    row_b = np.arange(en - st)[:, None]
                    for c_o, kv in ((4, min(40, k_big)), (6, k_big)):
                        rk = np.maximum(dk_s[:, kv - 1 : kv], np.float32(1e-4))
                        wk = (np.float32(1.0) - np.minimum(dk_s[:, :kv] / rk, np.float32(1.0)) ** 3) ** 3
                        W_k = np.zeros((en - st, len(it)), dtype=np.float32)
                        W_k[np.repeat(row_b, kv, axis=1), pk_s[:, :kv]] = wk
                        pr_k, lw_k = self._eval_dirichlet_lwnb_pair(
                            W_k, ys, y0s, oh3_s1, oh3_s0, p1_c, plo_c, off3, cols3_te, ie[st:en], p1_g3, p0_g3, llr_g3, 12.0, 2.5, 1.0
                        )
                        pr[ie[st:en], c_o] = pr_k
                        pr[ie[st:en], c_o + 1] = lw_k

                u_d = np.minimum(((d_te_s[st:en, None] - d_tr_s[None, :]) / np.float32(0.18)) ** 2, np.float32(25.0))
                u_r = np.minimum(((r_te_s[st:en, None] - r_tr_s[None, :]) / np.float32(0.06)) ** 2, np.float32(25.0))
                u_rv = np.minimum(((rv_te_s[st:en, None] - rv_tr_s[None, :]) / np.float32(0.055)) ** 2, np.float32(25.0))
                u_sr = np.minimum(((sr_te_s[st:en, None] - sr_tr_s[None, :]) / np.float32(0.45)) ** 2, np.float32(25.0))
                u_st = np.minimum(((st_te_s[st:en, None] - st_tr_s[None, :]) / np.float32(0.70)) ** 2, np.float32(25.0))
                for pk_i, d2 in enumerate((u_d + u_r, u_d + u_rv, u_d + u_sr, u_r + u_st)):
                    for hk_i, ch in enumerate(c_bws_2d):
                        W2 = np.exp(np.maximum(ch * d2, np.float32(-25.0)))
                        if same is not None:
                            W2[same] = np.float32(0.0)
                        ci2 = pk_i * 3 + hk_i
                        ws2[ci2][st:en] = np.sum(W2, axis=1, dtype=np.float64)
                        wy2[ci2][st:en] = (W2 @ ys).astype(np.float64)
            for ci2 in range(12):
                pe2 = np.clip((wy2[ci2] + 3.0 * p1_l) / (ws2[ci2] + 3.0), 1e-6, 1.0 - 1e-6)
                k2f[ie, ci2] = np.log(pe2 / (1.0 - pe2)) - plo_l

        return bk, lw, pr, k2f, wd_te

    @staticmethod
    def _extract_core_28_from_base(
        base_arr: np.ndarray, kde_full: np.ndarray, strict_w: np.ndarray
    ) -> np.ndarray:
        """Mengekstrak 28 sinyal LLR & Parzen KDE inti dari basis kalibrasi untuk gate channel dominan (BC=4,5)."""
        damp1 = 1.0 / (1.0 + 0.50 * strict_w[:, None])
        m_raw = base_arr[:, 1:92] / damp1
        m_imp = base_arr[:, 92:139] / damp1
        extra_18 = kde_full[:, 28:46]
        return np.column_stack([
            base_arr[:, 0],
            np.mean(m_raw[:, 0:10], axis=1),
            np.mean(m_raw[:, 10:21], axis=1),
            np.mean(m_raw[:, 21:48], axis=1),
            np.mean(m_raw[:, 48:64], axis=1),
            np.mean(m_raw[:, 64:91], axis=1),
            np.mean(m_imp[:, 0:10], axis=1),
            np.mean(m_imp[:, 10:20], axis=1),
            np.mean(m_imp[:, 20:32], axis=1),
            np.mean(m_imp[:, 32:], axis=1),
            kde_full[:, 0], kde_full[:, 3], kde_full[:, 5], kde_full[:, 7], kde_full[:, 10],
            kde_full[:, 12], kde_full[:, 15], kde_full[:, 23], kde_full[:, 24], kde_full[:, 25],
            extra_18[:, 0], extra_18[:, 3], extra_18[:, 4], extra_18[:, 5],
            extra_18[:, 9], extra_18[:, 13], extra_18[:, 15], extra_18[:, 16],
        ])

    def _build_v20_cll_basis_and_prior(
        self,
        m_base_in: np.ndarray,
        l2_base_in: np.ndarray,
        base_cur: np.ndarray,
        n_base_cols: int,
        d_p1: int,
        d_p2: int,
        k1d_blocks: List[Tuple[np.ndarray, float, float]],
        k2d: np.ndarray,
        coh_11: np.ndarray,
        clq_v19: np.ndarray,
        bk_24: np.ndarray,
        lw_17: np.ndarray,
        pr_14: np.ndarray,
        k2f_12: np.ndarray,
        bn38_hnb: np.ndarray,
        bn43_hnb: np.ndarray,
        bn38_sp2: np.ndarray,
        bn43_sp2: np.ndarray,
        bn38_a1de: np.ndarray,
        bn43_a1de: np.ndarray,
        bn38_k1: np.ndarray,
        bn43_k1: np.ndarray,
        s_cur: Dict[str, np.ndarray],
        ctx_cur: Dict[str, np.ndarray],
        kde_full_cur: np.ndarray,
        wd_cur: np.ndarray,
        strict_w: np.ndarray,
        l2_mult: float = 3.85,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Menyatukan basis V20 (True-LOO + Smooth-K1 + A1DE + SP-2DE-15 + Across-Month LWNB + BC45 Gates)."""
        m_base = m_base_in.copy()
        d1 = 1.0 / (1.0 + 0.50 * strict_w[:, None])
        d2 = 1.0 / (1.0 + 1.50 * strict_w[:, None])

        off1 = n_base_cols + d_p1
        off2 = n_base_cols + 8 * d_p1 + d_p1
        off3 = n_base_cols + 22 * d_p1 + d_p2
        off4 = n_base_cols + 22 * d_p1 + 9 * d_p2
        m_base[:, off1 : off1 + d_p1] = bn38_k1 * d1
        m_base[:, off2 : off2 + d_p1] = bn38_k1 * d2
        m_base[:, off3 : off3 + d_p2] = bn43_k1 * d1
        m_base[:, off4 : off4 + d_p2] = bn43_k1 * d2

        parts = [m_base]
        l2_parts = [l2_base_in * l2_mult]

        knots = (-1.0, 0.0, 1.0)
        for mat, r_lin, r_h in k1d_blocks + [
            (k2d, 240.0, 650.0),
            (coh_11, 320.0, 750.0),
            (bk_24, 220.0, 650.0),
            (lw_17, 240.0, 680.0),
            (pr_14, 240.0 * 1.25, 680.0 * 1.25),
            (k2f_12, 240.0, 680.0),
        ]:
            h_mat = np.hstack([mat] + [np.maximum(0.0, mat - k) for k in knots] + [np.maximum(0.0, k - mat) for k in knots])
            n0 = mat.shape[1]
            parts.append(h_mat)
            l2_parts.append(
                np.concatenate([
                    np.full(n0, r_lin * l2_mult, dtype=np.float64),
                    np.full(h_mat.shape[1] - n0, r_h * l2_mult, dtype=np.float64),
                ])
            )

        for bn_mat, reg_b in (
            (bn38_hnb, 1100.0),
            (bn43_hnb, 1200.0),
            (bn38_sp2, 1100.0),
            (bn43_sp2, 1200.0),
            (bn38_a1de, 750.0),
            (bn43_a1de, 750.0),
        ):
            parts.append(bn_mat)
            l2_parts.append(np.full(bn_mat.shape[1], reg_b * l2_mult, dtype=np.float64))

        clq_d = np.hstack([clq_v19 * d1, clq_v19 * d2])
        parts.append(clq_d)
        l2_parts.append(np.full(clq_d.shape[1], 550.0 * l2_mult, dtype=np.float64))

        # Gated interactions untuk segmen mayoritas (BC=4 Offline Agent, BC=5 OTA) & isolated bookings
        g0 = (strict_w / (strict_w + 0.5))[:, None]
        iso = 1.0 - g0
        bc4 = (s_cur["bc"] == 4).astype(np.float64)[:, None]
        bc5 = (s_cur["bc"] == 5).astype(np.float64)[:, None]
        sr0 = (ctx_cur["sr_imp"] == 0).astype(np.float64)[:, None]
        sun = (wd_cur == 6).astype(np.float64)[:, None]
        dba_na = np.isnan(s_cur["dba_raw"]).astype(np.float64)[:, None]
        sigs = np.column_stack([
            base_cur[:, 0], base_cur[:, 318], base_cur[:, 319], base_cur[:, 320], base_cur[:, 321],
            np.mean(bk_24[:, :6], axis=1), lw_17[:, 1], lw_17[:, 5], lw_17[:, 9], np.mean(bn38_a1de, axis=1),
            pr_14[:, 1], pr_14[:, 5], pr_14[:, 11], k2f_12[:, 1], k2f_12[:, 4], sun[:, 0],
        ])
        bc45_sum = np.hstack([bc4 * sigs, bc5 * sigs, iso * sigs, sr0 * sigs, dba_na * sigs[:, :6], bc5 * sr0 * sigs[:, :6]])
        parts.append(bc45_sum)
        l2_parts.append(np.full(bc45_sum.shape[1], 850.0 * l2_mult, dtype=np.float64))

        core28 = self._extract_core_28_from_base(base_cur, kde_full_cur, strict_w)
        c45 = np.hstack([bc4 * core28, bc5 * core28])
        parts.append(c45)
        l2_parts.append(np.full(c45.shape[1], 750.0 * l2_mult, dtype=np.float64))

        return np.hstack(parts), np.concatenate(l2_parts)

    def _fit_cll_irls_weights(
        self,
        M: np.ndarray,
        y: np.ndarray,
        l2_reg: Optional[float] = None,
        non_negative: bool = True,
    ) -> np.ndarray:
        """
        Mengoptimasi parameter pembobotan Conditional Log-Likelihood (CLL / WANBIA)
        menggunakan algoritma Newton-Raphson / IRLS atau Two-Loop L-BFGS murni NumPy
        [Zaidi et al., JMLR 2013; Greiner & Zhou, AAAI 2002].
        """
        reg_val = self.cll_l2_reg if l2_reg is None else float(l2_reg)
        n, d = M.shape
        if not non_negative and d > 250:
            return self._fit_cll_lbfgs_weights(M, y, l2_reg=reg_val)

        X_design = np.column_stack([np.ones(n, dtype=np.float64), M])
        theta = np.zeros(d + 1, dtype=np.float64)
        p1 = min(max(float(np.mean(y)), 1e-6), 1.0 - 1e-6)
        theta[0] = math.log(p1 / (1.0 - p1))
        if non_negative:
            theta[1:] = 1.0 / max(d, 1)

        reg_vec = np.full(d + 1, reg_val, dtype=np.float64)
        reg_vec[0] = 0.0
        y_float = y.astype(np.float64)

        for _ in range(self.cll_n_iter):
            z = np.clip(X_design @ theta, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            w_diag = np.maximum(p * (1.0 - p), 1e-5)
            grad = X_design.T @ (p - y_float) + reg_vec * theta
            H = (X_design.T * w_diag) @ X_design + np.diag(reg_vec)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            theta = theta - step
            if non_negative:
                theta[1:] = np.maximum(theta[1:], 0.0)

        return theta

    def _select_and_fit_best_cll_weights(
        self, M_oof: np.ndarray, y_bin: np.ndarray, non_negative: bool = True
    ) -> np.ndarray:
        """
        Mencari penalti regularisasi L2 terbaik dari self.cll_l2_grid secara internal
        dan mengembalikan vektor bobot CLL / WANBIA (b*, w*) yang paling terkalibrasi.
        """
        if not self.cll_l2_grid or len(self.cll_l2_grid) <= 1:
            self.best_cll_l2_ = self.cll_l2_reg
            return self._fit_cll_irls_weights(M_oof, y_bin, l2_reg=self.cll_l2_reg, non_negative=non_negative)

        n = len(y_bin)
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        mid = n // 2
        idx_a, idx_b = perm[:mid], perm[mid:]

        best_l2 = self.cll_l2_reg
        best_score = -np.inf

        for l2_cand in self.cll_l2_grid:
            th_a = self._fit_cll_irls_weights(M_oof[idx_a], y_bin[idx_a], l2_reg=l2_cand, non_negative=non_negative)
            th_b = self._fit_cll_irls_weights(M_oof[idx_b], y_bin[idx_b], l2_reg=l2_cand, non_negative=non_negative)

            z_val = np.empty(n, dtype=np.float64)
            z_val[idx_b] = np.clip(th_a[0] + M_oof[idx_b] @ th_a[1:], -30.0, 30.0)
            z_val[idx_a] = np.clip(th_b[0] + M_oof[idx_a] @ th_b[1:], -30.0, 30.0)
            p_val = 1.0 / (1.0 + np.exp(-z_val))

            acc_v = float(np.mean((p_val >= 0.440).astype(np.int64) == y_bin))
            ll_v = float(
                np.mean(
                    y_bin * np.log(np.clip(p_val, 1e-6, 1.0))
                    + (1 - y_bin) * np.log(np.clip(1.0 - p_val, 1e-6, 1.0))
                )
            )
            score = acc_v + 0.25 * ll_v
            if score > best_score:
                best_score = score
                best_l2 = float(l2_cand)

        self.best_cll_l2_ = best_l2
        return self._fit_cll_irls_weights(M_oof, y_bin, l2_reg=best_l2, non_negative=non_negative)

    def _fit_and_apply_numeric_discretization(self, X_arr: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Mengubah atribut numerik menjadi kategori interval kuantil (discretization) sesuai
        acuan Slide IF3170 2024 "Bayesian Network as Knowledge Representation" halaman 27-42,
        sehingga seluruh atribut dapat dihitung probabilitasnya dari frekuensi data +
        smoothing +1 dan diklasifikasikan dengan Maximum A Posteriori (MAP).
        """
        n_samples, n_features = X_arr.shape
        X_disc = np.empty((n_samples, n_features), dtype=object)
        if fit:
            self.numeric_disc_edges_ = {}
            self.numeric_disc_medians_ = {}

        for j in range(n_features):
            col = X_arr[:, j]
            is_num = True
            try:
                col_f = col.astype(np.float64)
            except (ValueError, TypeError):
                is_num = False
                col_f = np.array([], dtype=np.float64)

            if not is_num:
                X_disc[:, j] = np.asarray(col, dtype=str)
                continue

            nan_mask = np.isnan(col_f)
            valid_vals = col_f[~nan_mask]
            if fit:
                med_val = float(np.median(valid_vals)) if valid_vals.size > 0 else 0.0
                self.numeric_disc_medians_[j] = med_val
                unique_vals = np.unique(valid_vals) if valid_vals.size > 0 else np.array([0.0])
                if len(unique_vals) <= self.discretize_bins:
                    edges = (
                        np.array([], dtype=np.float64)
                        if len(unique_vals) <= 1
                        else (unique_vals[:-1] + unique_vals[1:]) / 2.0
                    )
                else:
                    qs = np.linspace(0.0, 100.0, self.discretize_bins + 1)[1:-1]
                    edges = np.unique(np.percentile(valid_vals, qs))
                self.numeric_disc_edges_[j] = edges
            else:
                med_val = self.numeric_disc_medians_.get(j, 0.0)
                edges = self.numeric_disc_edges_.get(j, np.array([], dtype=np.float64))

            col_filled = np.where(nan_mask, med_val, col_f)
            bins_j = (
                np.searchsorted(edges, col_filled, side="right")
                if edges.size > 0
                else np.zeros(n_samples, dtype=np.int64)
            )
            if np.any(nan_mask):
                nan_bin = len(edges) + 1
                bins_j = np.where(nan_mask, nan_bin, bins_j)
            X_disc[:, j] = np.array([f"BIN_{int(b):03d}" for b in bins_j], dtype=object)

        return X_disc

    def discretize_matrix(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Mengembalikan matriks integer hasil diskritisasi [0 .. n_categories_j - 1]
        berdasarkan kategori/bin yang telah dipelajari saat fit() (untuk verifikasi
        ekuivalensi 100% terhadap sklearn.naive_bayes.CategoricalNB).
        """
        if not self.is_fitted_:
            raise RuntimeError("Panggil fit(X, y) terlebih dahulu sebelum discretize_matrix(X).")
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if self.discretize_numeric:
            X_proc = self._fit_and_apply_numeric_discretization(X_arr, fit=False)
        else:
            X_proc = X_arr
        n_samples, n_features = X_proc.shape
        X_int = np.zeros((n_samples, n_features), dtype=np.int64)
        for j in range(n_features):
            if int(j) in self.cat_log_probs_:
                info = self.cat_log_probs_[int(j)]
                val_to_idx = info["val_to_idx"]
                n_cats = info["n_categories"]
                col_str = np.asarray(X_proc[:, j], dtype=str)
                X_int[:, j] = np.array([val_to_idx.get(v, n_cats - 1) for v in col_str], dtype=np.int64)
        return X_int

    def fit(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> "NaiveBayes":
        """
        Melatih model Naive Bayes:
        - Menghitung Prior Probability P(Y=c) dari frekuensi kelas.
        - Jika discretize_numeric=True (Implementasi Wajib Slide IF3170 Hal. 27-42),
          mengubah atribut numerik menjadi kategori (discretization) lalu menghitung
          seluruh P(X_j | Y=c) dari frekuensi data + Laplace Smoothing (+1).
        - Jika use_tan=True (Implementasi Bonus Optimasi), mempelajari struktur pohon
          Chow-Liu TAN + Multi-Resolution Factored Bayesian Cliques + Conditional
          Parzen-Window KDE + Discriminative CLL / WANBIA Weighting (100% tanpa ensemble).
        """
        if hasattr(X, "columns"):
            self.feature_names_in_ = [str(col) for col in X.columns]
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            self.feature_names_in_ = [f"Feature_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y).ravel()
        n_samples, n_features = X_arr.shape
        if n_samples != y_arr.shape[0]:
            raise ValueError(f"Jumlah baris X ({n_samples}) tidak sama dengan panjang y ({y_arr.shape[0]}).")

        self.n_features_in_ = n_features
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)

        # 1. Hitung Prior Probability P(Y=c) dan Log-Prior dari frekuensi kelas
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        for c_idx, c_val in enumerate(self.classes_):
            self.class_count_[c_idx] = float(np.sum(y_arr == c_val))

        self.class_prior_ = self.class_count_ / float(n_samples)
        self.log_prior_ = np.log(np.clip(self.class_prior_, 1e-300, 1.0))
        p1_global = float(self.class_prior_[1]) if n_classes > 1 else 0.5
        self.prior_log_odds_ = math.log(max(p1_global, 1e-6) / max(1.0 - p1_global, 1e-6))

        # 2. Diskritisasi atribut numerik jika discretize_numeric=True (Slide IF3170 Hal. 27-42)
        if self.discretize_numeric and not self.use_tan:
            X_proc = self._fit_and_apply_numeric_discretization(X_arr, fit=True)
            self.categorical_mask_ = np.ones(n_features, dtype=bool)
        else:
            X_proc = X_arr
            self.categorical_mask_ = self._detect_feature_types(X_arr)

        self.continuous_indices_ = np.where(~self.categorical_mask_)[0]
        self.categorical_indices_ = np.where(self.categorical_mask_)[0]

        # 3. Pelajari parameter Gaussian untuk fitur kontinu (jika discretize_numeric=False)
        n_cont = len(self.continuous_indices_)
        if n_cont > 0:
            X_cont = X_proc[:, self.continuous_indices_].astype(np.float64)
            global_vars = np.nanvar(X_cont, axis=0)
            max_var = float(np.nanmax(global_vars)) if global_vars.size > 0 else 1.0
            if np.isnan(max_var) or max_var <= 0.0:
                max_var = 1.0
            self.epsilon_ = self.var_smoothing * max_var

            self.theta_ = np.zeros((n_classes, n_cont), dtype=np.float64)
            self.var_ = np.zeros((n_classes, n_cont), dtype=np.float64)

            global_means = np.nanmean(X_cont, axis=0)
            global_means = np.where(np.isnan(global_means), 0.0, global_means)

            for c_idx, c_val in enumerate(self.classes_):
                mask_c = y_arr == c_val
                X_c = X_cont[mask_c]
                if X_c.shape[0] > 0:
                    mean_c = np.nanmean(X_c, axis=0)
                    var_c = np.nanvar(X_c, axis=0)
                    mean_c = np.where(np.isnan(mean_c), global_means, mean_c)
                    var_c = np.where(np.isnan(var_c), global_vars, var_c)
                else:
                    mean_c = global_means.copy()
                    var_c = global_vars.copy()

                self.theta_[c_idx] = mean_c
                self.var_[c_idx] = var_c + self.epsilon_
        else:
            self.theta_ = np.zeros((n_classes, 0), dtype=np.float64)
            self.var_ = np.zeros((n_classes, 0), dtype=np.float64)

        # 4. Pelajari tabel probabilitas dari frekuensi data + Laplace Smoothing (+1)
        self.cat_log_probs_ = {}
        alpha_cat = (
            self.marginal_alpha
            if (self.use_tan and self.tan_mode in ("dirichlet_jm", "cll_v3", "cll_v4", "cll_v6", "cll_kdb_kde"))
            else self.laplace_smoothing
        )
        for j in self.categorical_indices_:
            col_str = np.asarray(X_proc[:, j], dtype=str)
            categories = np.unique(col_str)
            n_cats = len(categories)
            val_to_idx = {val: idx for idx, val in enumerate(categories)}

            log_prob_table = np.zeros((n_classes, n_cats + 1), dtype=np.float64)
            mapped_col = np.array([val_to_idx[v] for v in col_str], dtype=np.int64)

            for c_idx, c_val in enumerate(self.classes_):
                mask_c = y_arr == c_val
                n_c = self.class_count_[c_idx]
                counts = np.bincount(mapped_col[mask_c], minlength=n_cats).astype(np.float64)
                probs = (counts + alpha_cat) / (n_c + alpha_cat * n_cats)
                unseen_prob = alpha_cat / (n_c + alpha_cat * (n_cats + 1))

                log_prob_table[c_idx, :n_cats] = np.log(np.clip(probs, 1e-300, None))
                log_prob_table[c_idx, n_cats] = np.log(max(unseen_prob, 1e-300))

            self.cat_log_probs_[int(j)] = {
                "categories": categories,
                "val_to_idx": val_to_idx,
                "n_categories": n_cats,
                "log_prob_table": log_prob_table,
            }

        # 5. Jika use_tan=True (Bonus Optimasi Model), pelajari struktur TAN + Factored Cliques + Parzen KDE + CLL
        self.cll_col_mean_ = None
        self.cll_col_std_ = None
        self.cll_col_mask_ = None
        self.cll_train_oof_auc_ = 0.0
        self.cll_train_oof_ll_ = 0.0
        self.hnb_sp2_path1_tables_ = None
        self.hnb_sp2_path2_tables_ = None
        self.clq_v19_tables_ = []
        self.parzen_rvrm_tr_ = None
        if self.use_tan and n_features > 1:
            if self.tan_mode in ("cll_v3", "cll_v4", "cll_v6", "cll_kdb_kde") and self.feature_names_in_ is not None:
                tan_set = set(DEFAULT_TAN_TREE_FEATURES)
                sub_idx = [idx for idx, name in enumerate(self.feature_names_in_) if name in tan_set]
                if len(sub_idx) < 2:
                    sub_idx = list(range(n_features))
                self.tan_feature_indices_ = np.array(sub_idx, dtype=np.int64)
            else:
                self.tan_feature_indices_ = np.arange(n_features, dtype=np.int64)

            self.tan_feature_names_ = [self.feature_names_in_[i] for i in self.tan_feature_indices_]

            if self.tan_mode in ("cll_v3", "cll_v4", "cll_v6", "cll_kdb_kde") and n_classes == 2:
                is_kde_mode = self.tan_mode in ("cll_v4", "cll_v6", "cll_kdb_kde")
                is_calibrated_cll = self.tan_mode in ("cll_v6", "cll_kdb_kde")
                y_bin_all = (y_arr == self.classes_[1]).astype(np.int64)
                cliques_all = self._extract_multi_resolution_cliques(X_arr)
                n_raw_cliques = len(cliques_all)

                pos_idx = np.where(y_bin_all == 1)[0]
                neg_idx = np.where(y_bin_all == 0)[0]
                rng = np.random.RandomState(self.random_state)
                pos_folds = np.array_split(rng.permutation(pos_idx), 5)
                neg_folds = np.array_split(rng.permutation(neg_idx), 5)

                if is_kde_mode:
                    s_tr_full = self._extract_raw_struct(X_arr)
                    kde_loo_full, ctx_tr_loo, strict_w_tr = self._compute_parzen_kde_and_context_imputation(
                        s_tr_full, y_bin_all, s_tr_full, is_self_loo=True, extended_v6=is_calibrated_cll
                    )
                    if is_calibrated_cll:
                        imp_cliques_all = self._extract_imputed_cliques(
                            ctx_tr_loo, s_te=s_tr_full, kde_full=kde_loo_full, s_tr_ref=s_tr_full
                        )
                    else:
                        imp_cliques_all = self._extract_imputed_cliques(ctx_tr_loo)
                    n_imp_cliques = len(imp_cliques_all)
                else:
                    s_tr_full = None
                    kde_loo_full = None
                    ctx_tr_loo = None
                    imp_cliques_all = []
                    n_imp_cliques = 0
                    strict_w_tr = np.zeros(n_samples, dtype=np.float64)

                tan_oof = np.zeros(n_samples, dtype=np.float64)
                if is_calibrated_cll:
                    rng_10f = np.random.RandomState(self.random_state)
                    pos_10f = np.array_split(rng_10f.permutation(pos_idx), 10)
                    neg_10f = np.array_split(rng_10f.permutation(neg_idx), 10)
                    self.clique_tables_, M_raw_oof, self.prior_log_odds_ = self._fit_and_10fold_oof_cliques_llr_tables(
                        cliques_all, y_bin_all, pos_10f, neg_10f
                    )
                    self.imputed_clique_tables_, M_imp_oof, _ = self._fit_and_10fold_oof_cliques_llr_tables(
                        imp_cliques_all, y_bin_all, pos_10f, neg_10f
                    )
                else:
                    M_raw_oof = np.zeros((n_samples, n_raw_cliques), dtype=np.float64)
                    M_imp_oof = np.zeros((n_samples, n_imp_cliques), dtype=np.float64)

                X_tan_all = X_arr[:, self.tan_feature_indices_]
                for k in range(5):
                    in_va = np.concatenate([pos_folds[k], neg_folds[k]])
                    in_tr_mask = np.ones(n_samples, dtype=bool)
                    in_tr_mask[in_va] = False
                    in_tr = np.where(in_tr_mask)[0]
                    y_in_tr = y_bin_all[in_tr]

                    # A. Komponen TAN Chow-Liu pada inner fold
                    X_tan_tr_disc = self._discretize_subset_matrix(
                        X_tan_all[in_tr], self.tan_feature_indices_, fit=True
                    )
                    X_tan_va_disc = self._discretize_subset_matrix(
                        X_tan_all[in_va], self.tan_feature_indices_, fit=False
                    )
                    self.cmi_matrix_, self.mi_per_class_, self.tan_marginals_ = self._compute_cmi_and_marginals(
                        X_tan_tr_disc, y_arr[in_tr]
                    )
                    self.tan_root_, self.tan_parents_, self.tan_tree_edges_ = self._build_chow_liu_mwst(
                        self.cmi_matrix_
                    )
                    self._fit_tan_parameters(X_tan_tr_disc, y_arr[in_tr])

                    in_counts = np.array(
                        [np.sum(y_arr[in_tr] == c) for c in self.classes_], dtype=np.float64
                    )
                    in_priors = in_counts / float(len(in_tr))
                    in_log_prior = np.log(np.clip(in_priors, 1e-300, 1.0))
                    in_prior_lo = float(in_log_prior[1] - in_log_prior[0])

                    jll_va_tan = self._eval_tan_dirichlet_jll(X_tan_va_disc, in_log_prior)
                    tan_oof[in_va] = (jll_va_tan[:, 1] - jll_va_tan[:, 0]) - in_prior_lo

                    if not is_calibrated_cll:
                        # B. Komponen Raw Multi-Resolution Cliques (Path A)
                        sub_c_tr = [(c[in_tr], m) for c, m in cliques_all]
                        sub_c_va = [(c[in_va], m) for c, m in cliques_all]
                        tabs_in, _ = self._fit_cliques_llr_tables(sub_c_tr, y_in_tr)
                        M_raw_oof[in_va] = self._transform_cliques_llr_matrix(sub_c_va, tabs_in)

                        # C. Komponen Context-Imputed & Cohort Cliques (Path B)
                        if is_kde_mode:
                            sub_imp_tr = [(c[in_tr], m) for c, m in imp_cliques_all]
                            sub_imp_va = [(c[in_va], m) for c, m in imp_cliques_all]
                            tabs_imp_in, _ = self._fit_cliques_llr_tables(sub_imp_tr, y_in_tr)
                            M_imp_oof[in_va] = self._transform_cliques_llr_matrix(sub_imp_va, tabs_imp_in)

                if is_calibrated_cll:
                    base_tr = self._build_calibrated_cll_basis(
                        tan_oof,
                        M_raw_oof,
                        M_imp_oof,
                        kde_loo_full,
                        strict_w_tr,
                        s_cur=s_tr_full,
                        ctx_cur=ctx_tr_loo,
                        s_tr_ref=s_tr_full,
                        ctx_tr_ref=ctx_tr_loo,
                        is_self_loo=True,
                    )
                    if self.tan_mode == "cll_kdb_kde":
                        v10_tr, sub12_tr = self._compute_soft_and_subspace_parzen_kde(
                            s_tr_full,
                            y_bin_all,
                            s_tr_full,
                            ctx_tr_loo,
                            ctx_tr_loo,
                            is_self_loo=True,
                        )
                        dag38_oof, mb38_oof, self.kdb_path1_tables_ = self._fit_and_strict_oof_kdb_depnet(
                            X_arr.astype(np.float64),
                            y_bin_all,
                            pos_folds,
                            neg_folds,
                            max_bins=16,
                            include_markov_blanket=True,
                        )
                        F43_tr = self._build_non_duplicate_ctx_43_features(
                            X_arr,
                            s_tr_full,
                            ctx_tr_loo,
                            s_tr_full,
                            ctx_tr_ref=ctx_tr_loo,
                            is_self_loo=True,
                        )
                        dag43_oof, _, self.kdb_path2_tables_ = self._fit_and_strict_oof_kdb_depnet(
                            F43_tr,
                            y_bin_all,
                            pos_folds,
                            neg_folds,
                            max_bins=24,
                            include_markov_blanket=False,
                        )
                        d_p1 = X_arr.shape[1]
                        d_p2 = F43_tr.shape[1]
                        M_base_tr, l2_base_vec = self._build_kdb_kde_cll_basis_and_prior(
                            base_tr,
                            v10_tr,
                            sub12_tr,
                            [
                                (dag38_oof, d_p1, 850.0),
                                (mb38_oof, d_p1, 1050.0),
                                (dag43_oof, d_p2, 950.0),
                            ],
                            strict_w_tr,
                            kde_loo_full,
                        )

                        col_map_tr = {str(nm): idx for idx, nm in enumerate(self.feature_names_in_ or [])}
                        if "rate_vs_room_month" in col_map_tr:
                            rvrm_tr = np.nan_to_num(
                                X_arr[:, col_map_tr["rate_vs_room_month"]].astype(np.float64), nan=1.0
                            )
                        else:
                            rvrm_tr = np.ones(n_samples, dtype=np.float64)
                        self.parzen_rvrm_tr_ = rvrm_tr.astype(np.float32)

                        bn38_k1_oof, bn38_hnb_oof, bn38_sp2_oof, bn38_a1de_oof, self.hnb_sp2_path1_tables_ = (
                            self._fit_and_strict_oof_hnb_sp2de(
                                X_arr.astype(np.float64),
                                y_bin_all,
                                pos_10f,
                                neg_10f,
                                max_bins=16,
                                sp_max_pairs=15,
                                top_a1de=14,
                                smooth_k1=True,
                            )
                        )
                        self.hnb_sp2_path1_tables_["is_v20"] = True
                        bn43_k1_oof, bn43_hnb_oof, bn43_sp2_oof, bn43_a1de_oof, self.hnb_sp2_path2_tables_ = (
                            self._fit_and_strict_oof_hnb_sp2de(
                                F43_tr,
                                y_bin_all,
                                pos_10f,
                                neg_10f,
                                max_bins=24,
                                sp_max_pairs=6,
                                top_a1de=14,
                                smooth_k1=True,
                            )
                        )
                        self.hnb_sp2_path2_tables_["is_v20"] = True
                        clq_v19_all = self._extract_v19_orthogonal_cliques(s_tr_full, ctx_tr_loo)
                        self.clq_v19_tables_, clq_v19_oof, _ = self._fit_and_10fold_oof_cliques_llr_tables(
                            clq_v19_all, y_bin_all, pos_10f, neg_10f
                        )
                        k1d_tr, k2d_tr, coh_tr = self._compute_v19_continuous_kde_and_cohort(
                            s_tr_full,
                            rvrm_tr,
                            y_bin_all,
                            s_tr_full,
                            rvrm_tr,
                            ctx_tr_loo,
                            ctx_tr_loo,
                            is_self_loo=True,
                        )
                        bk_tr, lw_tr, pr_tr, k2f_tr, wd_tr = self._compute_v20_continuous_kde_and_lwnb(
                            s_tr_full,
                            ctx_tr_loo,
                            y_bin_all,
                            rvrm_tr,
                            s_tr_full,
                            ctx_tr_loo,
                            rvrm_tr,
                            is_self_loo=True,
                        )
                        M_design_tr, l2_group_vec = self._build_v20_cll_basis_and_prior(
                            M_base_tr,
                            l2_base_vec,
                            base_tr,
                            base_tr.shape[1],
                            d_p1,
                            d_p2,
                            k1d_tr,
                            k2d_tr,
                            coh_tr,
                            clq_v19_oof,
                            bk_tr,
                            lw_tr,
                            pr_tr,
                            k2f_tr,
                            bn38_hnb_oof,
                            bn43_hnb_oof,
                            bn38_sp2_oof,
                            bn43_sp2_oof,
                            bn38_a1de_oof,
                            bn43_a1de_oof,
                            bn38_k1_oof,
                            bn43_k1_oof,
                            s_tr_full,
                            ctx_tr_loo,
                            kde_loo_full,
                            wd_tr,
                            strict_w_tr,
                            l2_mult=3.85,
                        )
                        self.cll_col_mean_ = np.mean(M_design_tr, axis=0)
                        col_std = np.std(M_design_tr, axis=0)
                        self.cll_col_mask_ = col_std > 1e-3
                        self.cll_col_std_ = np.where(self.cll_col_mask_, col_std, 1.0)
                        kept_mask = self.cll_col_mask_
                        M_tr_scaled = np.clip(
                            (M_design_tr[:, kept_mask] - self.cll_col_mean_[kept_mask])
                            / self.cll_col_std_[kept_mask],
                            -5.0,
                            5.0,
                        )
                        l2_kept = l2_group_vec[kept_mask]
                        self.best_cll_l2_ = float(np.mean(l2_kept))
                        self.cll_theta_ = self._fit_cll_lbfgs_weights(
                            M_tr_scaled, y_bin_all, l2_reg=l2_kept
                        )
                        z_tr_oof = np.clip(
                            self.cll_theta_[0] + M_tr_scaled @ self.cll_theta_[1:], -30.0, 30.0
                        )
                        p_tr_oof = np.clip(1.0 / (1.0 + np.exp(-z_tr_oof)), 1e-12, 1.0 - 1e-12)
                        self.cll_train_oof_ll_ = float(
                            -np.mean(y_bin_all * np.log(p_tr_oof) + (1.0 - y_bin_all) * np.log(1.0 - p_tr_oof))
                        )
                        # Fast vectorized exact OOF ROC-AUC
                        ord_p = np.argsort(p_tr_oof, kind="mergesort")
                        y_ord = y_bin_all[ord_p]
                        n1_tot = float(np.sum(y_bin_all))
                        n0_tot = float(len(y_bin_all) - n1_tot)
                        if n1_tot > 0.0 and n0_tot > 0.0:
                            ranks = np.arange(1, len(y_ord) + 1, dtype=np.float64)
                            self.cll_train_oof_auc_ = float(
                                (np.sum(ranks[y_ord == 1]) - 0.5 * n1_tot * (n1_tot + 1.0)) / (n1_tot * n0_tot)
                            )
                        self.parzen_ctx_tr_loo_ = ctx_tr_loo
                    else:
                        self.cll_col_mean_ = np.mean(base_tr, axis=0)
                        col_std = np.std(base_tr, axis=0)
                        self.cll_col_mask_ = None
                        self.cll_col_std_ = np.where(col_std > 1e-8, col_std, 1.0)
                        M_tr_scaled = (base_tr - self.cll_col_mean_) / self.cll_col_std_
                        self.cll_theta_ = self._select_and_fit_best_cll_weights(
                            M_tr_scaled, y_bin_all, non_negative=False
                        )
                        self.kdb_path1_tables_ = None
                        self.kdb_path2_tables_ = None
                        self.hnb_sp2_path1_tables_ = None
                        self.hnb_sp2_path2_tables_ = None
                        self.clq_v19_tables_ = []
                        self.parzen_rvrm_tr_ = None
                        self.parzen_ctx_tr_loo_ = None
                    self.parzen_train_struct_ = s_tr_full
                    self.parzen_y_train_ = y_bin_all
                elif is_kde_mode:
                    damp_tr = 1.0 / (1.0 + 0.50 * strict_w_tr[:, None])
                    M_inner_oof = np.column_stack([
                        tan_oof, M_raw_oof * damp_tr, M_imp_oof * damp_tr, kde_loo_full[:, :18]
                    ])
                    self.cll_theta_ = self._select_and_fit_best_cll_weights(
                        M_inner_oof, y_bin_all, non_negative=True
                    )
                    self.clique_tables_, self.prior_log_odds_ = self._fit_cliques_llr_tables(cliques_all, y_bin_all)
                    self.imputed_clique_tables_, _ = self._fit_cliques_llr_tables(imp_cliques_all, y_bin_all)
                    self.parzen_train_struct_ = s_tr_full
                    self.parzen_y_train_ = y_bin_all
                else:
                    M_inner_oof = np.column_stack([tan_oof, M_raw_oof])
                    self.cll_theta_ = self._fit_cll_irls_weights(
                        M_inner_oof, y_bin_all, l2_reg=self.cll_l2_reg, non_negative=True
                    )
                    self.clique_tables_, self.prior_log_odds_ = self._fit_cliques_llr_tables(cliques_all, y_bin_all)
                    self.imputed_clique_tables_ = []
                    self.parzen_train_struct_ = None
                    self.parzen_y_train_ = None

            # Fit struktur dan parameter pohon TAN pada 100% data X_arr
            X_tan_full = X_arr[:, self.tan_feature_indices_]
            X_disc = self._discretize_subset_matrix(X_tan_full, self.tan_feature_indices_, fit=True)
            self.cmi_matrix_, self.mi_per_class_, self.tan_marginals_ = self._compute_cmi_and_marginals(X_disc, y_arr)
            self.tan_root_, self.tan_parents_, self.tan_tree_edges_ = self._build_chow_liu_mwst(self.cmi_matrix_)
            self._fit_tan_parameters(X_disc, y_arr)
        else:
            self.tan_feature_indices_ = None
            self.tan_feature_names_ = None
            self.cmi_matrix_ = None
            self.mi_per_class_ = None
            self.tan_root_ = None
            self.tan_parents_ = np.full(n_features, -1, dtype=np.int64)
            self.tan_class_parents_ = None
            self.tan_tree_edges_ = []
            self.tan_cpt_log_probs_ = {}
            self.clique_tables_ = []
            self.imputed_clique_tables_ = []
            self.parzen_train_struct_ = None
            self.parzen_ctx_tr_loo_ = None
            self.parzen_y_train_ = None
            self.kdb_path1_tables_ = None
            self.kdb_path2_tables_ = None
            self.hnb_sp2_path1_tables_ = None
            self.hnb_sp2_path2_tables_ = None
            self.clq_v19_tables_ = []
            self.parzen_rvrm_tr_ = None
            self.cll_theta_ = None
            self.cll_col_mean_ = None
            self.cll_col_std_ = None
            self.cll_col_mask_ = None

        self.is_fitted_ = True
        return self

    def _compute_joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Menghitung matriks unnormalized log-posterior berukuran (n_samples, n_classes)."""
        if not self.is_fitted_:
            raise RuntimeError("Model NaiveBayes belum di-fit! Panggil model.fit(X, y) terlebih dahulu.")

        if hasattr(X, "to_numpy"):
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)

        n_samples, n_features = X_arr.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"Jumlah fitur input ({n_features}) tidak sesuai dengan jumlah fitur saat fit ({self.n_features_in_})."
            )

        n_classes = len(self.classes_)

        # --- Mode Calibrated Discriminative Bayesian Network ('cll_kdb_kde' / 'cll_v6'):
        #     TAN Chow-Liu + Factored Bayesian Cliques + Parzen KDE + Calibrated CLL (0 Trees, 0 Ensemble) ---
        if (
            self.use_tan
            and self.tan_mode in ("cll_v6", "cll_kdb_kde")
            and self.cll_theta_ is not None
            and self.cll_col_mean_ is not None
            and self.cll_col_std_ is not None
            and len(self.clique_tables_) > 0
            and self.parzen_train_struct_ is not None
        ):
            X_tan_sub = X_arr[:, self.tan_feature_indices_]
            X_tan_disc = self._discretize_subset_matrix(X_tan_sub, self.tan_feature_indices_, fit=False)
            jll_tan = self._eval_tan_dirichlet_jll(X_tan_disc, self.log_prior_)
            tan_llr = (jll_tan[:, 1] - jll_tan[:, 0]) - float(self.log_prior_[1] - self.log_prior_[0])

            s_te = self._extract_raw_struct(X_arr)
            kde_te_full, ctx_te, strict_w_te = self._compute_parzen_kde_and_context_imputation(
                self.parzen_train_struct_, self.parzen_y_train_, s_te, is_self_loo=False, extended_v6=True
            )
            raw_cliques_te = self._extract_multi_resolution_cliques(X_arr)
            M_raw = self._transform_cliques_llr_matrix(raw_cliques_te, self.clique_tables_)
            imp_cliques_te = self._extract_imputed_cliques(
                ctx_te, s_te=s_te, kde_full=kde_te_full, s_tr_ref=self.parzen_train_struct_
            )
            M_imp = self._transform_cliques_llr_matrix(imp_cliques_te, self.imputed_clique_tables_)

            is_v20 = bool(
                self.hnb_sp2_path1_tables_ is not None
                and self.hnb_sp2_path1_tables_.get("is_v20", False)
                and self.parzen_ctx_tr_loo_ is not None
            )
            base_te = self._build_calibrated_cll_basis(
                tan_llr,
                M_raw,
                M_imp,
                kde_te_full,
                strict_w_te,
                s_cur=s_te,
                ctx_cur=ctx_te,
                s_tr_ref=self.parzen_train_struct_,
                ctx_tr_ref=self.parzen_ctx_tr_loo_ if is_v20 else None,
                is_self_loo=False,
            )
            if (
                self.tan_mode == "cll_kdb_kde"
                and self.kdb_path1_tables_ is not None
                and self.kdb_path2_tables_ is not None
                and self.parzen_ctx_tr_loo_ is not None
            ):
                v10_te, sub12_te = self._compute_soft_and_subspace_parzen_kde(
                    self.parzen_train_struct_,
                    self.parzen_y_train_,
                    s_te,
                    self.parzen_ctx_tr_loo_,
                    ctx_te,
                    is_self_loo=False,
                )
                dag38_te, mb38_te = self._eval_kdb_depnet_tables(
                    self.kdb_path1_tables_, X_arr.astype(np.float64)
                )
                F43_te = self._build_non_duplicate_ctx_43_features(
                    X_arr,
                    s_te,
                    ctx_te,
                    self.parzen_train_struct_,
                    ctx_tr_ref=self.parzen_ctx_tr_loo_ if is_v20 else None,
                    is_self_loo=False,
                )
                dag43_te, _ = self._eval_kdb_depnet_tables(self.kdb_path2_tables_, F43_te)
                d_p1 = X_arr.shape[1]
                d_p2 = F43_te.shape[1]
                M_base_te, _ = self._build_kdb_kde_cll_basis_and_prior(
                    base_te,
                    v10_te,
                    sub12_te,
                    [
                        (dag38_te, d_p1, 850.0),
                        (mb38_te, d_p1, 1050.0),
                        (dag43_te, d_p2, 950.0),
                    ],
                    strict_w_te,
                    kde_te_full,
                )
                if (
                    self.hnb_sp2_path1_tables_ is not None
                    and self.hnb_sp2_path2_tables_ is not None
                    and len(self.clq_v19_tables_) > 0
                    and self.parzen_rvrm_tr_ is not None
                ):
                    col_map_te = {str(nm): idx for idx, nm in enumerate(self.feature_names_in_ or [])}
                    if "rate_vs_room_month" in col_map_te:
                        rvrm_te = np.nan_to_num(
                            X_arr[:, col_map_te["rate_vs_room_month"]].astype(np.float64), nan=1.0
                        )
                    else:
                        rvrm_te = np.ones(n_samples, dtype=np.float64)
                    clq_v19_te_m = self._transform_cliques_llr_matrix(
                        self._extract_v19_orthogonal_cliques(s_te, ctx_te), self.clq_v19_tables_
                    )
                    k1d_te, k2d_te, coh_te = self._compute_v19_continuous_kde_and_cohort(
                        self.parzen_train_struct_,
                        self.parzen_rvrm_tr_.astype(np.float64),
                        self.parzen_y_train_,
                        s_te,
                        rvrm_te,
                        self.parzen_ctx_tr_loo_,
                        ctx_te,
                        is_self_loo=False,
                    )
                    if is_v20:
                        bn38_k1_te, bn38_hnb_te, bn38_sp2_te, bn38_a1de_te = self._eval_hnb_sp2de_tables(
                            self.hnb_sp2_path1_tables_, X_arr.astype(np.float64), return_a1de=True
                        )
                        bn43_k1_te, bn43_hnb_te, bn43_sp2_te, bn43_a1de_te = self._eval_hnb_sp2de_tables(
                            self.hnb_sp2_path2_tables_, F43_te, return_a1de=True
                        )
                        bk_te, lw_te, pr_te, k2f_te, wd_te = self._compute_v20_continuous_kde_and_lwnb(
                            self.parzen_train_struct_,
                            self.parzen_ctx_tr_loo_,
                            self.parzen_y_train_,
                            self.parzen_rvrm_tr_.astype(np.float64),
                            s_te,
                            ctx_te,
                            rvrm_te,
                            is_self_loo=False,
                        )
                        M_design_te, _ = self._build_v20_cll_basis_and_prior(
                            M_base_te,
                            np.zeros(M_base_te.shape[1], dtype=np.float64),
                            base_te,
                            base_te.shape[1],
                            d_p1,
                            d_p2,
                            k1d_te,
                            k2d_te,
                            coh_te,
                            clq_v19_te_m,
                            bk_te,
                            lw_te,
                            pr_te,
                            k2f_te,
                            bn38_hnb_te,
                            bn43_hnb_te,
                            bn38_sp2_te,
                            bn43_sp2_te,
                            bn38_a1de_te,
                            bn43_a1de_te,
                            bn38_k1_te,
                            bn43_k1_te,
                            s_te,
                            ctx_te,
                            kde_te_full,
                            wd_te,
                            strict_w_te,
                            l2_mult=3.85,
                        )
                    else:
                        bn38_k1_te, bn38_hnb_te, bn38_sp2_te = self._eval_hnb_sp2de_tables(
                            self.hnb_sp2_path1_tables_, X_arr.astype(np.float64)
                        )
                        bn43_k1_te, bn43_hnb_te, bn43_sp2_te = self._eval_hnb_sp2de_tables(
                            self.hnb_sp2_path2_tables_, F43_te
                        )
                        M_design_te, _ = self._build_v19_cll_basis_and_prior(
                            M_base_te,
                            np.zeros(M_base_te.shape[1], dtype=np.float64),
                            base_te.shape[1],
                            d_p1,
                            d_p2,
                            k1d_te,
                            k2d_te,
                            coh_te,
                            clq_v19_te_m,
                            bn38_hnb_te,
                            bn43_hnb_te,
                            bn38_sp2_te,
                            bn43_sp2_te,
                            bn38_k1_te,
                            bn43_k1_te,
                            strict_w_te,
                            l2_mult=2.02,
                        )
                else:
                    M_design_te = M_base_te
            else:
                M_design_te = base_te
            if self.cll_col_mask_ is not None:
                m_k = self.cll_col_mask_
                M_te_scaled = np.clip(
                    (M_design_te[:, m_k] - self.cll_col_mean_[m_k]) / self.cll_col_std_[m_k],
                    -5.0,
                    5.0,
                )
            else:
                M_te_scaled = (M_design_te - self.cll_col_mean_) / self.cll_col_std_
            z = np.clip(self.cll_theta_[0] + M_te_scaled @ self.cll_theta_[1:], -30.0, 30.0)
            return np.column_stack([np.zeros(n_samples, dtype=np.float64), z])

        # --- Mode TAN v4 ('cll_v4'): Dual-Path Cliques + Conditional Parzen KDE + TAN v2 ---
        if (
            self.use_tan
            and self.tan_mode == "cll_v4"
            and self.cll_theta_ is not None
            and len(self.clique_tables_) > 0
            and self.parzen_train_struct_ is not None
        ):
            X_tan_sub = X_arr[:, self.tan_feature_indices_]
            X_tan_disc = self._discretize_subset_matrix(X_tan_sub, self.tan_feature_indices_, fit=False)
            jll_tan = self._eval_tan_dirichlet_jll(X_tan_disc, self.log_prior_)
            tan_llr = (jll_tan[:, 1] - jll_tan[:, 0]) - float(self.log_prior_[1] - self.log_prior_[0])

            raw_cliques_te = self._extract_multi_resolution_cliques(X_arr)
            M_raw = self._transform_cliques_llr_matrix(raw_cliques_te, self.clique_tables_)

            s_te = self._extract_raw_struct(X_arr)
            kde_te, ctx_te, strict_w_te = self._compute_parzen_kde_and_context_imputation(
                self.parzen_train_struct_, self.parzen_y_train_, s_te, is_self_loo=False
            )
            imp_cliques_te = self._extract_imputed_cliques(ctx_te)
            M_imp = self._transform_cliques_llr_matrix(imp_cliques_te, self.imputed_clique_tables_)

            damp_te = 1.0 / (1.0 + 0.50 * strict_w_te[:, None])
            M_all = np.column_stack([tan_llr, M_raw * damp_te, M_imp * damp_te, kde_te])
            z = np.clip(self.cll_theta_[0] + M_all @ self.cll_theta_[1:], -30.0, 30.0)
            return np.column_stack([np.zeros(n_samples, dtype=np.float64), z])

        # --- Mode TAN v3 ('cll_v3'): Discriminative CLL-Weighted TAN + Multi-Resolution Cliques ---
        if (
            self.use_tan
            and self.tan_mode == "cll_v3"
            and self.cll_theta_ is not None
            and len(self.clique_tables_) > 0
        ):
            X_tan_sub = X_arr[:, self.tan_feature_indices_]
            X_tan_disc = self._discretize_subset_matrix(X_tan_sub, self.tan_feature_indices_, fit=False)
            jll_tan = self._eval_tan_dirichlet_jll(X_tan_disc, self.log_prior_)
            tan_llr = (jll_tan[:, 1] - jll_tan[:, 0]) - float(self.log_prior_[1] - self.log_prior_[0])

            cliques_te = self._extract_multi_resolution_cliques(X_arr)
            M_cliques = self._transform_cliques_llr_matrix(cliques_te, self.clique_tables_)

            M_all = np.column_stack([tan_llr, M_cliques])
            z = np.clip(self.cll_theta_[0] + M_all @ self.cll_theta_[1:], -30.0, 30.0)
            return np.column_stack([np.zeros(n_samples, dtype=np.float64), z])

        # --- Mode TAN v2 ('dirichlet_jm'): inferensi murni dengan Hierarchical Dirichlet + JM TAN ---
        if self.use_tan and self.tan_mode == "dirichlet_jm" and self.tan_class_parents_ is not None:
            X_tan_sub = X_arr[:, self.tan_feature_indices_]
            X_disc = self._discretize_subset_matrix(X_tan_sub, self.tan_feature_indices_, fit=False)
            return self._eval_tan_dirichlet_jll(X_disc, self.log_prior_)

        # --- Perhitungan Naive Bayes Standar (Slide IF3170 Hal. 27-42: Discretization + Frekuensi + Smoothing +1 / Gaussian) ---
        if self.discretize_numeric and not self.use_tan:
            X_proc = self._fit_and_apply_numeric_discretization(X_arr, fit=False)
        else:
            X_proc = X_arr

        joint_log_likelihood = np.tile(self.log_prior_, (n_samples, 1))

        n_cont = len(self.continuous_indices_)
        if n_cont > 0:
            X_cont = X_proc[:, self.continuous_indices_].astype(np.float64)
            for c_idx in range(n_classes):
                mean_c = self.theta_[c_idx]
                var_c = self.var_[c_idx]
                X_clean = np.where(np.isnan(X_cont), mean_c, X_cont)
                log_coeff = -0.5 * np.log(2.0 * math.pi * var_c)
                quad_term = -0.5 * ((X_clean - mean_c) ** 2) / var_c
                joint_log_likelihood[:, c_idx] += np.sum(log_coeff + quad_term, axis=1)

        for j in self.categorical_indices_:
            info = self.cat_log_probs_[int(j)]
            val_to_idx = info["val_to_idx"]
            n_cats = info["n_categories"]
            log_prob_table = info["log_prob_table"]

            col_str = np.asarray(X_proc[:, j], dtype=str)
            mapped_idx = np.array([val_to_idx.get(v, n_cats) for v in col_str], dtype=np.int64)
            joint_log_likelihood += log_prob_table[:, mapped_idx].T

        # --- Mode TAN v1 ('legacy_blend') untuk kompatibilitas mundur ---
        if (
            self.use_tan
            and self.tan_mode == "legacy_blend"
            and self.tan_parents_ is not None
            and np.any(self.tan_parents_ >= 0)
        ):
            X_tan_sub = X_arr[:, self.tan_feature_indices_]
            X_disc = self._discretize_subset_matrix(X_tan_sub, self.tan_feature_indices_, fit=False)
            tan_log_likelihood = np.tile(self.log_prior_, (n_samples, 1))

            for i in range(len(self.tan_feature_indices_)):
                p = int(self.tan_parents_[i])
                if p < 0:
                    if self.categorical_mask_[i]:
                        info = self.cat_log_probs_[i]
                        col_str = np.asarray(X_arr[:, i], dtype=str)
                        mapped_idx = np.array(
                            [info["val_to_idx"].get(v, info["n_categories"]) for v in col_str],
                            dtype=np.int64,
                        )
                        tan_log_likelihood += info["log_prob_table"][:, mapped_idx].T
                    else:
                        i_cont_pos = int(np.where(self.continuous_indices_ == i)[0][0])
                        xi = np.asarray(X_arr[:, i], dtype=np.float64)
                        for c_idx in range(n_classes):
                            mu = self.theta_[c_idx, i_cont_pos]
                            var = self.var_[c_idx, i_cont_pos]
                            xi_clean = np.where(np.isnan(xi), mu, xi)
                            tan_log_likelihood[:, c_idx] += -0.5 * np.log(2.0 * math.pi * var) - 0.5 * (
                                (xi_clean - mu) ** 2
                            ) / var
                else:
                    cpt_log = self.tan_cpt_log_probs_[i]
                    idx_p = np.clip(X_disc[:, p], 0, cpt_log.shape[1] - 1)
                    idx_i = np.clip(X_disc[:, i], 0, cpt_log.shape[2] - 1)
                    for c_idx in range(n_classes):
                        tan_log_likelihood[:, c_idx] += cpt_log[c_idx, idx_p, idx_i]

            w = np.clip(self.tan_blending, 0.0, 1.0)
            joint_log_likelihood = (1.0 - w) * joint_log_likelihood + w * tan_log_likelihood

        return joint_log_likelihood

    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Menghitung estimasi probabilitas posterior P(Y=c | X) untuk setiap sampel
        dengan normalisasi Log-Sum-Exp.
        """
        jll = self._compute_joint_log_likelihood(X)
        log_prob_x = self._logsumexp(jll, axis=1, keepdims=True)
        log_posterior = jll - log_prob_x
        proba = np.exp(log_posterior)
        row_sums = np.sum(proba, axis=1, keepdims=True)
        proba = np.divide(proba, np.where(row_sums > 0, row_sums, 1.0))
        return proba

    def predict(self, X: Union[np.ndarray, Any], threshold: Optional[float] = None) -> np.ndarray:
        """Menghasilkan prediksi label kelas diskrit untuk setiap sampel di X."""
        proba = self.predict_proba(X)
        th = self.decision_threshold if threshold is None else float(threshold)
        if len(self.classes_) == 2 and th != 0.5:
            return np.where(proba[:, 1] >= th, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(proba, axis=1)]

    def save(self, filepath: str) -> None:
        """Menyimpan objek model NaiveBayes ke dalam file .pkl."""
        abs_path = os.path.abspath(filepath)
        parent_dir = os.path.dirname(abs_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        used_custom_io = False
        for mod_name in ("src.utils.model_io", "utils.model_io"):
            try:
                mod = __import__(mod_name, fromlist=["save_model"])
                if hasattr(mod, "save_model") and callable(mod.save_model):
                    mod.save_model(self, filepath)
                    used_custom_io = True
                    break
            except Exception:
                continue

        if not used_custom_io:
            with open(filepath, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> "NaiveBayes":
        """Memuat kembali objek model NaiveBayes dari file .pkl."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File model tidak ditemukan di: {filepath}")

        for mod_name in ("src.utils.model_io", "utils.model_io"):
            try:
                mod = __import__(mod_name, fromlist=["load_model"])
                if hasattr(mod, "load_model") and callable(mod.load_model):
                    obj = mod.load_model(filepath)
                    if isinstance(obj, cls):
                        return obj
            except Exception:
                continue

        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj

    def plot_network_graph(
        self,
        feature_names: Optional[List[str]] = None,
        target_name: str = "is_canceled",
        save_path: str = "output/figures/naive_bayes_graph.png",
    ) -> str:
        """
        Fitur Bonus Visualisasi Graf (1 Poin):
        Memvisualisasikan struktur Directed Acyclic Graph (DAG) dari Bayesian Network /
        Naive Bayes (baik struktur Naive Bayes standar maupun struktur Tree-Augmented
        Naive Bayes / TAN dengan bobot Conditional Mutual Information).
        """
        if not self.is_fitted_:
            raise RuntimeError("Model harus di-fit terlebih dahulu sebelum memanggil plot_network_graph().")

        import math
        import textwrap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, FancyArrowPatch

        if self.use_tan and self.tan_feature_names_ is not None:
            feature_names = list(self.tan_feature_names_)
        elif feature_names is None:
            if self.feature_names_in_ is not None and len(self.feature_names_in_) == self.n_features_in_:
                feature_names = list(self.feature_names_in_)
            else:
                feature_names = [f"X_{i}" for i in range(self.n_features_in_)]

        n_features = len(feature_names)
        abs_save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(abs_save_path), exist_ok=True)

        primary_purple = "#5b21b6"
        target_edge_col = "#2e1065"
        light_purple = "#f3e8ff"
        node_border_purple = "#7c3aed"
        tan_edge_color = "#d95f02"
        tan_root_fill = "#ffedd5"
        class_edge_color = "#8b5cf6"

        has_tan = self.use_tan and len(self.tan_tree_edges_) > 0

        def _format_circle_label(fname: str, is_root: bool = False) -> str:
            known_labels = {
                "hub_lead_channel_req": "Hub:\nLead×Chan\n×Req",
                "time_ym_wknd": "Time:\nYearMo\n×Wknd",
                "product_room_meal_rate": "Product:\nRoom×Meal\n×Rate",
                "stay_nights_wknd": "Stay:\nNights\n×Wknd",
                "party_adults_child_park": "Party:\nAdult×Chd\n×Park",
                "history_ret_cancel_comp": "History:\nRet×Canc\n×Comp",
                "days_before_arrival_is_missing": "Missing:\nDays\nBefore",
                "nightly_rate_idr_is_missing": "Missing:\nNightly\nRate",
                "num_adults_is_missing": "Missing:\nNum\nAdults",
                "special_requests_is_missing": "Missing:\nSpecial\nReq",
                "booking_group_size": "Booking\nGroup\nSize",
                "arrival_day": "Arrival\nDay",
                "total_spend_idr": "Total\nSpend\n(IDR)",
                "lead_to_stay_ratio": "Lead/Stay\nRatio",
                "rate_vs_room_month": "Rate vs\nRoom-Mo",
                "days_before_arrival": "Days\nBefore\nArrival",
                "nightly_rate_idr": "Nightly\nRate\n(IDR)",
                "prior_completed_bookings": "Prior\nCompleted",
                "prior_cancellations": "Prior\nCancels",
                "is_returning_guest": "Returning\nGuest",
                "special_requests": "Special\nRequests",
            }
            if fname in known_labels:
                txt = known_labels[fname]
            else:
                clean = fname.replace("booking_channel_", "Ch:\n").replace("meal_package_", "Meal:\n")
                clean = clean.replace("room_category_", "Room:\n").replace("_is_missing", "\n(Miss)")
                clean = clean.replace("_", " ")
                lines: List[str] = []
                for part in clean.split("\n"):
                    wrapped = textwrap.wrap(part, width=10)
                    lines.extend(wrapped if wrapped else [""])
                txt = "\n".join(lines[:3])
            if is_root:
                txt = f"{txt}\n[TAN Root]"
            return txt

        def _compute_tidy_tan_layout() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
            """
            Menghitung koordinat hierarkis tanpa perpotongan edge untuk pohon TAN Chow-Liu:
            - Target berada di lapisan teratas (y_top)
            - TAN Root di lapisan ke-2, lalu anak & cucu disusun proporsional lebar subtree.
            """
            children: Dict[int, List[int]] = {i: [] for i in range(n_features)}
            in_deg = np.zeros(n_features, dtype=int)
            edge_weights: Dict[Tuple[int, int], float] = {}
            for p_idx, c_idx, w in self.tan_tree_edges_:
                if 0 <= p_idx < n_features and 0 <= c_idx < n_features:
                    children[p_idx].append(c_idx)
                    in_deg[c_idx] += 1
                    edge_weights[(p_idx, c_idx)] = float(w)

            if self.tan_root_ is not None and 0 <= int(self.tan_root_) < n_features:
                root_idx = int(self.tan_root_)
            else:
                roots = [i for i in range(n_features) if in_deg[i] == 0]
                root_idx = roots[0] if roots else 0

            # Urutkan anak berdasarkan ukuran subtree agar cabang terlebar tampil rapi
            subtree_w: Dict[int, int] = {}
            depth_map: Dict[int, int] = {}
            visited = set()

            def _measure(u: int, d: int) -> int:
                visited.add(u)
                depth_map[u] = d
                kids = [v for v in children[u] if v not in visited]
                children[u] = kids
                if not kids:
                    subtree_w[u] = 1
                    return 1
                total = sum(_measure(v, d + 1) for v in kids)
                # Urutkan anak: daun / subtree kecil di kiri, subtree lebih dalam di kanan
                kids.sort(key=lambda v: (subtree_w[v], edge_weights.get((u, v), 0.0)))
                subtree_w[u] = max(1, total)
                return subtree_w[u]

            _measure(root_idx, 1)
            # Jika ada node terisolasi di luar komponen utama, tempatkan di akhir
            for idx in range(n_features):
                if idx not in visited:
                    depth_map[idx] = 1
                    subtree_w[idx] = 1

            total_leaves = max(1, subtree_w[root_idx])
            max_tree_depth = max(depth_map.values()) if depth_map else 1

            node_r = 0.35 if total_leaves <= 10 else max(0.24, 3.3 / total_leaves)
            slot_step = max(2.0 * node_r + 0.14, 0.84)
            total_span = max(1, total_leaves - 1) * slot_step

            x_coords = np.zeros(n_features, dtype=np.float64)

            def _assign_x(u: int, x_left: float, x_right: float) -> None:
                kids = children[u]
                if not kids:
                    x_coords[u] = 0.5 * (x_left + x_right)
                    return
                total_w = sum(subtree_w[v] for v in kids)
                cur = x_left
                span = x_right - x_left
                for v in kids:
                    frac = subtree_w[v] / max(total_w, 1)
                    nxt = cur + span * frac
                    _assign_x(v, cur, nxt)
                    cur = nxt
                # Pusatkan parent terhadap posisi anak paling kiri dan paling kanan
                x_coords[u] = 0.5 * (x_coords[kids[0]] + x_coords[kids[-1]])

            _assign_x(root_idx, -0.5 * total_span - 0.5 * slot_step, 0.5 * total_span + 0.5 * slot_step)
            # Pusatkan TAN Root tepat di tengah sumbu x di bawah Target
            x_shift = x_coords[root_idx]
            x_coords -= x_shift

            y_top_target = 2.28
            y_root = 1.08
            y_bottom = -2.35
            if max_tree_depth <= 1:
                layer_ys = {1: 0.0}
            else:
                ys = np.linspace(y_root, y_bottom, max_tree_depth)
                layer_ys = {d + 1: float(ys[d]) for d in range(max_tree_depth)}

            feat_pos = np.column_stack([x_coords, [layer_ys[depth_map[i]] for i in range(n_features)]])
            target_pos = np.array([0.0, y_top_target], dtype=np.float64)

            x_pad = node_r + 0.32
            x_lim = np.array([float(np.min(x_coords)) - x_pad, float(np.max(x_coords)) + x_pad])
            y_lim = np.array([-2.90, 2.90])
            return target_pos, feat_pos, x_lim, y_lim, node_r, 0.46, float(root_idx)

        if has_tan:
            t_pos_tan, f_pos_tan, xlim_tan, ylim_tan, r_tan_node, r_tan_target, tan_root_idx = _compute_tidy_tan_layout()
            tan_width_ratio = float((xlim_tan[1] - xlim_tan[0]) / 5.70)
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12.2 * (1.0 + tan_width_ratio), 11.2),
                dpi=200,
                gridspec_kw={"width_ratios": [1.0, max(1.05, tan_width_ratio)], "wspace": 0.06},
            )
            subtitle_tan = f"Tree-Augmented Naive Bayes (TAN - Chow-Liu MWST)\n({n_features} Core Features, {len(self.tan_tree_edges_)} Conditional Dependency Edges)"
            panels = [
                (axes[0], "Standard Naive Bayes Structure\n(Conditional Independence Assumption)", False),
                (axes[1], subtitle_tan, True),
            ]
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12.5, 11.5), dpi=200)
            panels = [(ax, f"Bayesian Network Structure (Naive Bayes • {n_features} Features)", False)]

        text_fit_records: List[Tuple[Any, Any, Circle, float]] = []

        for ax, title, draw_tan in panels:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color="#240b46")
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

            if draw_tan:
                target_pos = t_pos_tan
                feat_pos = f_pos_tan
                node_r = r_tan_node
                target_r = r_tan_target
                ax.set_xlim(float(xlim_tan[0]), float(xlim_tan[1]))
                ax.set_ylim(float(ylim_tan[0]), float(ylim_tan[1]))
            else:
                star_r = max(2.12, n_features * 0.155)
                node_r = min(0.37, max(0.20, 0.43 * (2.0 * math.pi * star_r / max(n_features, 1))))
                target_r = max(0.46, node_r * 1.25)
                target_pos = np.array([0.0, 0.0], dtype=np.float64)
                angles = np.pi / 2.0 - np.linspace(0.0, 2.0 * np.pi, n_features, endpoint=False)
                feat_pos = np.column_stack([star_r * np.cos(angles), star_r * np.sin(angles)])
                lim_box = star_r + node_r + 0.36
                ax.set_xlim(-lim_box, lim_box)
                ax.set_ylim(-lim_box, lim_box)

            # 1. Buat patch lingkaran geometris sejati terlebih dahulu agar panah berhenti tepat di batas keliling
            target_circle = Circle(
                tuple(target_pos),
                radius=target_r,
                facecolor=primary_purple,
                edgecolor=target_edge_col,
                linewidth=2.4,
                zorder=6,
            )
            ax.add_patch(target_circle)

            feat_circles: List[Circle] = []
            for i in range(n_features):
                is_root = draw_tan and (self.tan_root_ == i)
                c_patch = Circle(
                    tuple(feat_pos[i]),
                    radius=node_r * (1.07 if is_root else 1.0),
                    facecolor=tan_root_fill if is_root else light_purple,
                    edgecolor=tan_edge_color if is_root else node_border_purple,
                    linewidth=2.5 if is_root else 1.7,
                    zorder=6,
                )
                ax.add_patch(c_patch)
                feat_circles.append(c_patch)

            # 2. Gambar panah hubungan Target -> Fitur
            for i in range(n_features):
                if draw_tan:
                    dx = float(feat_pos[i, 0] - target_pos[0])
                    is_root = (self.tan_root_ == i)
                    if is_root:
                        rad_val = 0.0
                    elif abs(dx) < 0.25:
                        rad_val = 0.24
                    else:
                        rad_val = float(np.clip(-0.07 * dx, -0.24, 0.24))
                    conn_style = f"arc3,rad={rad_val:.3f}"
                else:
                    conn_style = "arc3,rad=0.0"

                arrow = FancyArrowPatch(
                    posA=tuple(target_pos),
                    posB=tuple(feat_pos[i]),
                    patchA=target_circle,
                    patchB=feat_circles[i],
                    connectionstyle=conn_style,
                    arrowstyle="-|>,head_length=6.5,head_width=4.0",
                    color=class_edge_color,
                    alpha=0.28 if draw_tan else 0.78,
                    linestyle="--" if draw_tan else "-",
                    linewidth=1.15 if draw_tan else 1.7,
                    shrinkA=1.0,
                    shrinkB=2.0,
                    zorder=2,
                )
                ax.add_patch(arrow)

            # 3. Gambar panah dependensi kondisional TAN antarfitur + bobot I(X_i; X_j | Y)
            if draw_tan and len(self.tan_tree_edges_) > 0:
                max_w = max([w for _, _, w in self.tan_tree_edges_] + [1e-6])
                for p_idx, c_idx, weight in self.tan_tree_edges_:
                    if p_idx >= n_features or c_idx >= n_features:
                        continue
                    lw = 1.6 + 2.6 * (weight / max_w)
                    dx_edge = float(feat_pos[c_idx, 0] - feat_pos[p_idx, 0])
                    rad_tan = 0.0 if abs(dx_edge) < 0.1 else (0.06 if dx_edge < 0 else -0.06)
                    tan_arrow = FancyArrowPatch(
                        posA=tuple(feat_pos[p_idx]),
                        posB=tuple(feat_pos[c_idx]),
                        patchA=feat_circles[p_idx],
                        patchB=feat_circles[c_idx],
                        connectionstyle=f"arc3,rad={rad_tan:.3f}",
                        arrowstyle="-|>,head_length=8,head_width=5",
                        color=tan_edge_color,
                        alpha=0.95,
                        linewidth=lw,
                        shrinkA=1.0,
                        shrinkB=2.5,
                        zorder=4,
                    )
                    ax.add_patch(tan_arrow)

                    if weight >= 0.015:
                        # Tempatkan label bobot CMI sedikit di atas titik tengah agar bebas tabrakan
                        mid = 0.48 * feat_pos[p_idx] + 0.52 * feat_pos[c_idx]
                        ax.text(
                            mid[0],
                            mid[1],
                            f"{weight:.3f}",
                            fontsize=8.0,
                            fontweight="bold",
                            color="#7c2d12",
                            ha="center",
                            va="center",
                            bbox=dict(
                                boxstyle="round,pad=0.20",
                                fc="#fff7ed",
                                ec=tan_edge_color,
                                alpha=0.95,
                                lw=0.9,
                            ),
                            zorder=5,
                        )

            # 4. Tambahkan teks di dalam setiap bulatan (ukuran font akan dikalibrasi otomatis di langkah ke-5)
            t_txt = ax.text(
                target_pos[0],
                target_pos[1],
                f"Target\n({target_name})",
                fontsize=10.0,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                linespacing=1.18,
                zorder=7,
            )
            text_fit_records.append((ax, t_txt, target_circle, 10.0))

            for i, fname in enumerate(feature_names):
                is_root = draw_tan and (self.tan_root_ == i)
                label_txt = _format_circle_label(fname, is_root=is_root)
                base_fs = 8.4 if not is_root else 8.1
                f_txt = ax.text(
                    feat_pos[i, 0],
                    feat_pos[i, 1],
                    label_txt,
                    fontsize=base_fs,
                    fontweight="bold",
                    color="#431407" if is_root else "#1e1b4b",
                    ha="center",
                    va="center",
                    linespacing=1.12,
                    zorder=7,
                )
                text_fit_records.append((ax, f_txt, feat_circles[i], base_fs))

        legend_handles = [
            mpatches.Patch(facecolor=primary_purple, edgecolor=target_edge_col, label=f"Class Variable ({target_name})"),
            mpatches.Patch(facecolor=light_purple, edgecolor=node_border_purple, label="Predictor Feature ($X_i$)"),
            Line2D([0], [0], color=class_edge_color, lw=2.0, linestyle="--" if has_tan else "-", label="Class-to-Feature Dependency $P(X_i \\mid Y)$"),
        ]
        if has_tan:
            legend_handles.extend([
                mpatches.Patch(facecolor=tan_root_fill, edgecolor=tan_edge_color, label="TAN Chow-Liu Root Feature ($X_{root}$)"),
                Line2D([0], [0], color=tan_edge_color, lw=2.8, label="TAN Tree Edge $P(X_i \\mid X_{\\pi(i)}, Y)$ with $I(X_i; X_j \\mid Y)$"),
            ])

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 3),
            fontsize=10.0,
            frameon=True,
            facecolor="#fafafa",
            edgecolor="#d4d4d8",
            bbox_to_anchor=(0.5, 0.015),
        )
        fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.10, wspace=0.06)

        # 5. Kalibrasi ukuran font secara geometris agar 100% tulisan masuk rapi ke dalam batas lingkaran
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, txt_artist, circ_patch, base_fs in text_fit_records:
            cx, cy = circ_patch.center
            p_center = ax.transData.transform((cx, cy))
            p_edge = ax.transData.transform((cx + circ_patch.radius, cy))
            circle_r_px = float(np.hypot(p_edge[0] - p_center[0], p_edge[1] - p_center[1]))
            max_allowed_r_px = 0.85 * circle_r_px

            bbox = txt_artist.get_window_extent(renderer=renderer)
            half_diag_px = 0.5 * math.hypot(bbox.width, bbox.height)
            if half_diag_px > max_allowed_r_px and half_diag_px > 1e-6:
                scaled_fs = max(4.2, base_fs * (max_allowed_r_px / half_diag_px))
                txt_artist.set_fontsize(scaled_fs)

        fig.savefig(abs_save_path, dpi=200, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        return abs_save_path
