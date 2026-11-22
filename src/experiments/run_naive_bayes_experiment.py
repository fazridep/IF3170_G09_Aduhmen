import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import CategoricalNB, GaussianNB

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.naive_bayes import NaiveBayes
from src.preprocessing import DataPreprocessor

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Mengunci seluruh generator angka acak untuk menjamin reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def _make_baseline_preprocessor(seed: int = SEED) -> DataPreprocessor:
    """Membuat instance preprocessor untuk baseline GaussianNB (one-hot + standard + MI)."""
    return DataPreprocessor(
        mode="processed_numeric",
        add_missing_indicators=False,
        add_domain_features=True,
        add_composite_interactions=False,
        rare_category_min_count=0,
        encoding_method="onehot",
        scaling_method="standard",
        feature_selection_method="mutual_info",
        n_features_to_select=16,
        use_pca=False,
        random_state=seed,
    )


def _make_optimized_preprocessor(seed: int = SEED) -> DataPreprocessor:
    """Membuat instance preprocessor untuk NaiveBayes (Slide Discretized & Optimized TAN/CLL)."""
    return DataPreprocessor(
        mode="processed_numeric",
        add_missing_indicators=True,
        add_domain_features=True,
        add_composite_interactions=True,
        composite_lead_bins=14,
        composite_rate_bins=10,
        max_children_clip=4.0,
        rare_category_min_count=0,
        encoding_method="ordinal",
        scaling_method="none",
        feature_selection_method="tan_v3",
        use_pca=False,
        random_state=seed,
    )


def _make_slide_categorical_model(seed: int = SEED) -> NaiveBayes:
    """
    Membuat instance NaiveBayes Wajib sesuai Slide IF3170 2024 Hal. 27-42 & QnA Asisten:
    - Mengubah atribut numerik menjadi kategori (discretization) dengan 10 interval kuantil.
    - Menghitung probabilitas dari frekuensi data dengan Laplace Smoothing +1 (laplace_smoothing=1.0).
    - Klasifikasi dengan Maximum A Posteriori (MAP).
    """
    return NaiveBayes(
        laplace_smoothing=1.0,
        discretize_numeric=True,
        discretize_bins=10,
        use_tan=False,
        decision_threshold=0.5,
        random_state=seed,
    )


def _make_optimized_model(decision_threshold: float = 0.5, seed: int = SEED) -> NaiveBayes:
    """
    Membuat instance model tunggal NaiveBayes Teroptimasi (Bonus Optimasi Model - 100% Tanpa Ensemble):
    - Tree-Augmented Naive Bayes (TAN) via Algoritma Chow-Liu MWST (Friedman et al., 1997; Chow & Liu, 1968).
    - Hierarchical Dirichlet 8-Level k-Dependence Bayesian Network (k=0..5) & 3-Level Markov Blanket
      Dependency Network dengan Strict Inner-Fold OOF Isolation (Sahami, KDD 1996; MacKay & Peto, 1995;
      Heckerman et al., JMLR 2000).
    - Semi-Naive Multi-Resolution Factored Bayesian Contingency Cliques (91 Raw + 47 Context Cliques).
    - Flexible Naive Bayes dengan Multi-Bandwidth & Subspace Parzen-Window KDE (John & Langley, UAI 1995;
      Frank, Hall, & Pfahringer, UAI 2003).
    - Discriminative Conditional Log-Likelihood (CLL) / WANBIA Parameter Weighting dengan Structured
      Group-Prior Gaussian MAP & Two-Loop L-BFGS (Zaidi et al., JMLR 2013; Liu & Nocedal, 1989).
    """
    return NaiveBayes(
        var_smoothing=1e-9,
        laplace_smoothing=5.0,
        discretize_numeric=False,
        use_tan=True,
        categorical_features="auto",
        tan_bins=16,
        tan_blending=0.65,
        class_specific_tree=True,
        marginal_alpha=1.0,
        tan_mode="cll_kdb_kde",
        decision_threshold=decision_threshold,
        cll_l2_reg=450.0,
        cll_l2_grid=[450.0],
        cll_n_iter=15,
        random_state=seed,
    )


def compute_metrics_row(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_pos: np.ndarray,
    fit_ms: float,
) -> Dict[str, Any]:
    """Menghitung metrik lengkap evaluasi klasifikasi dan confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Binary F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba_pos),
        "Fit (ms/fold)": fit_ms,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Cost (2*FP+FN)": 2 * fp + fn,
    }


def main() -> None:
    seed_everything(SEED)
    data_dir = ROOT_DIR / "data"
    output_dir = ROOT_DIR / "output"
    models_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    submissions_dir = output_dir / "submissions"
    for d in (output_dir, models_dir, figures_dir, submissions_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("EKSPERIMEN NAIVE BAYES FROM SCRATCH (WAJIB SLIDE IF3170 & BONUS OPTIMASI TAN/CLL - TANPA ENSEMBLE)")
    print("=" * 104)

    # 1. Load Dataset
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    y_all = train_df["is_canceled"].to_numpy(dtype=np.int64)
    n_samples = len(train_df)
    target_prior_pos = float(np.mean(y_all))

    print(f"\n[1] Dataset Berhasil Dimuat:")
    print(f"    - Train Set : {train_df.shape[0]:,} baris x {train_df.shape[1]} kolom (Proporsi Positif = {target_prior_pos:.4f})")
    print(f"    - Test Set  : {test_df.shape[0]:,} baris x {test_df.shape[1]} kolom")

    # 2. Outer Stratified 5-Fold Cross-Validation (Zero Data Leakage)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_sk_gnb = np.zeros(n_samples, dtype=np.float64)
    oof_scratch_gnb = np.zeros(n_samples, dtype=np.float64)
    oof_sk_cat = np.zeros(n_samples, dtype=np.float64)
    oof_scratch_cat = np.zeros(n_samples, dtype=np.float64)
    oof_nb_opt = np.zeros(n_samples, dtype=np.float64)
    fold_va_indices = []
    train_auc_opt, val_auc_opt = [], []
    train_ll_opt, val_ll_opt = [], []

    fit_times_sk_g, fit_times_sc_g = [], []
    fit_times_sk_c, fit_times_sc_c = [], []
    fit_times_opt = []

    print("\n[2] Menjalankan Outer Stratified 5-Fold Cross-Validation (SEED = 42, Zero Data Leakage)...")

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_all), start=1):
        fold_va_indices.append(va_idx)
        df_tr_fold = train_df.iloc[tr_idx].reset_index(drop=True)
        df_va_fold = train_df.iloc[va_idx].reset_index(drop=True)

        df_tr_clean = DataPreprocessor.clean_training_dataframe(
            df_tr_fold, target_col="is_canceled", drop_duplicates=True, fix_invalid_guests=True
        )
        _, X_tr_raw, y_tr_fold = DataPreprocessor.extract_ids_and_target(df_tr_clean)
        _, X_va_raw, _ = DataPreprocessor.extract_ids_and_target(df_va_fold)

        # A. Baseline GaussianNB (scikit-learn vs From Scratch)
        prep_base = _make_baseline_preprocessor(SEED)
        X_tr_base = prep_base.fit_transform(X_tr_raw, y_tr_fold, return_dataframe=False)
        X_va_base = prep_base.transform(X_va_raw, return_dataframe=False)

        t0 = time.perf_counter()
        sk_g = GaussianNB(var_smoothing=1e-9).fit(X_tr_base, y_tr_fold)
        fit_times_sk_g.append((time.perf_counter() - t0) * 1000.0)
        oof_sk_gnb[va_idx] = sk_g.predict_proba(X_va_base)[:, 1]

        t0 = time.perf_counter()
        nb_g = NaiveBayes(var_smoothing=1e-9, discretize_numeric=False, use_tan=False, categorical_features=None).fit(
            X_tr_base, y_tr_fold
        )
        fit_times_sc_g.append((time.perf_counter() - t0) * 1000.0)
        oof_scratch_gnb[va_idx] = nb_g.predict_proba(X_va_base)[:, 1]

        # B. Implementasi Wajib Slide IF3170 Hal. 27-42:
        #    Discretization + Frekuensi Data + Smoothing +1 + Klasifikasi MAP (Scratch vs sklearn CategoricalNB)
        prep_opt = _make_optimized_preprocessor(SEED)
        X_tr_opt = prep_opt.fit_transform(X_tr_raw, y_tr_fold, return_dataframe=True)
        X_va_opt = prep_opt.transform(X_va_raw, return_dataframe=True)

        t0 = time.perf_counter()
        nb_cat = _make_slide_categorical_model(seed=SEED).fit(X_tr_opt, y_tr_fold)
        fit_times_sc_c.append((time.perf_counter() - t0) * 1000.0)
        oof_scratch_cat[va_idx] = nb_cat.predict_proba(X_va_opt)[:, 1]

        X_tr_disc_int = nb_cat.discretize_matrix(X_tr_opt)
        X_va_disc_int = nb_cat.discretize_matrix(X_va_opt)
        min_cats = [nb_cat.cat_log_probs_[j]["n_categories"] for j in range(X_tr_disc_int.shape[1])]
        t0 = time.perf_counter()
        sk_cat = CategoricalNB(alpha=1.0, min_categories=min_cats).fit(X_tr_disc_int, y_tr_fold)
        fit_times_sk_c.append((time.perf_counter() - t0) * 1000.0)
        oof_sk_cat[va_idx] = sk_cat.predict_proba(X_va_disc_int)[:, 1]

        # C. Implementasi Bonus Optimasi Naive Bayes (TAN + k-DB + MB + HNB + SP-2DE + 1D/2D KDE + CLL)
        t0 = time.perf_counter()
        nb_opt = _make_optimized_model(seed=SEED).fit(X_tr_opt, y_tr_fold)
        fit_times_opt.append((time.perf_counter() - t0) * 1000.0)
        va_proba_opt = nb_opt.predict_proba(X_va_opt)[:, 1]
        oof_nb_opt[va_idx] = va_proba_opt

        if getattr(nb_opt, "cll_train_oof_auc_", 0.0) > 0.0:
            tr_auc_f = float(nb_opt.cll_train_oof_auc_)
            tr_ll_f = float(nb_opt.cll_train_oof_ll_)
        else:
            tr_proba_opt = nb_opt.predict_proba(X_tr_opt)[:, 1]
            tr_auc_f = roc_auc_score(y_tr_fold, tr_proba_opt)
            tr_ll_f = log_loss(y_tr_fold, tr_proba_opt)
        va_auc_f = roc_auc_score(y_all[va_idx], va_proba_opt)
        va_ll_f = log_loss(y_all[va_idx], va_proba_opt)
        train_auc_opt.append(tr_auc_f)
        val_auc_opt.append(va_auc_f)
        train_ll_opt.append(tr_ll_f)
        val_ll_opt.append(va_ll_f)

        print(
            f"    - Fold {fold_idx}/5 selesai | Slide-CatNB Acc={accuracy_score(y_all[va_idx], oof_scratch_cat[va_idx]>=0.5):.5f} "
            f"| OptNB Acc@0.5={accuracy_score(y_all[va_idx], va_proba_opt>=0.500):.5f} "
            f"AUC(tr/va)={tr_auc_f:.4f}/{va_auc_f:.4f} LL(tr/va)={tr_ll_f:.4f}/{va_ll_f:.4f}",
            flush=True,
        )

    match_g_pct = float(np.mean((oof_sk_gnb >= 0.5) == (oof_scratch_gnb >= 0.5))) * 100.0
    max_diff_g = float(np.max(np.abs(oof_sk_gnb - oof_scratch_gnb)))
    match_c_pct = float(np.mean((oof_sk_cat >= 0.5) == (oof_scratch_cat >= 0.5))) * 100.0
    max_diff_c = float(np.max(np.abs(oof_sk_cat - oof_scratch_cat)))
    print(f"    - Ekuivalensi OOF Scratch Slide-CatNB vs sklearn CategoricalNB(alpha=1.0) : {match_c_pct:.4f}% (MaxDiff = {max_diff_c:.3e})")
    print(f"    - Ekuivalensi OOF Scratch GaussianNB  vs sklearn GaussianNB               : {match_g_pct:.4f}% (MaxDiff = {max_diff_g:.3e})")
    print(
        f"    - Audit Overfitting OptNB (Rata-rata 5-Fold Train vs Val)                 : "
        f"AUC gap = {np.mean(train_auc_opt) - np.mean(val_auc_opt):.5f} "
        f"({np.mean(train_auc_opt):.4f} -> {np.mean(val_auc_opt):.4f}), "
        f"LogLoss gap = {np.mean(val_ll_opt) - np.mean(train_ll_opt):.5f}"
    )

    # 3. Kalibrasi Decision Threshold pada 5-Fold OOF Predictions (+ Audit Nested OOF Threshold)
    thresholds = np.round(np.arange(0.35, 0.651, 0.005), 3)
    smooth_k = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    f1_curve = np.array(
        [f1_score(y_all, (oof_nb_opt >= th).astype(int), average="macro", zero_division=0) for th in thresholds],
        dtype=np.float64,
    )
    acc_curve = np.array(
        [accuracy_score(y_all, (oof_nb_opt >= th).astype(int)) for th in thresholds],
        dtype=np.float64,
    )
    f1_pad = np.pad(f1_curve, 2, mode="edge")
    f1_smooth = np.convolve(f1_pad, smooth_k, mode="valid")
    best_sm_idx = int(np.argmax(f1_smooth))
    best_raw_idx = int(np.argmax(f1_curve))
    threshold_f1 = float(thresholds[best_sm_idx])
    threshold_f1_raw = float(thresholds[best_raw_idx])
    threshold_acc = float(thresholds[int(np.argmax(acc_curve))])

    nested_preds_f1 = np.zeros(n_samples, dtype=int)
    nested_th_list = []
    for va_idx_k in fold_va_indices:
        other_mask = np.ones(n_samples, dtype=bool)
        other_mask[va_idx_k] = False
        f1_k_curve = np.array(
            [
                f1_score(y_all[other_mask], (oof_nb_opt[other_mask] >= th).astype(int), average="macro", zero_division=0)
                for th in thresholds
            ],
            dtype=np.float64,
        )
        f1_k_sm = np.convolve(np.pad(f1_k_curve, 2, mode="edge"), smooth_k, mode="valid")
        best_th_k = float(thresholds[int(np.argmax(f1_k_sm))])
        nested_th_list.append(best_th_k)
        nested_preds_f1[va_idx_k] = (oof_nb_opt[va_idx_k] >= best_th_k).astype(int)

    threshold_prior = float(np.round(np.quantile(oof_nb_opt, 1.0 - target_prior_pos), 4))

    print("\n[3] Kalibrasi Decision Threshold untuk NaiveBayes Teroptimasi (5-Fold OOF):")
    for label_th, th_v in [
        ("threshold=0.500 (MAP Default)", 0.500),
        ("threshold_f1    (Smoothed Plateau F1)", threshold_f1),
        ("threshold_raw   (Raw Argmax Macro F1)", threshold_f1_raw),
        ("threshold_prior (Prior-Matched 32.77%)", threshold_prior),
        ("threshold_acc   (Maksimum Accuracy)", threshold_acc),
    ]:
        p_th = (oof_nb_opt >= th_v).astype(int)
        print(
            f"    - {label_th:<38} : Th = {th_v:.4f} | "
            f"Acc = {accuracy_score(y_all, p_th):.5f} | "
            f"Macro F1 = {f1_score(y_all, p_th, average='macro'):.5f} | "
            f"Binary F1 = {f1_score(y_all, p_th):.5f} | "
            f"PosRate = {p_th.mean()*100:.2f}%"
        )
    print(
        f"    - Nested Out-Of-Fold Th_F1 (0-Leakage) : Ths = {nested_th_list} | "
        f"Acc = {accuracy_score(y_all, nested_preds_f1):.5f} | "
        f"Macro F1 = {f1_score(y_all, nested_preds_f1, average='macro'):.5f} | "
        f"PosRate = {nested_preds_f1.mean()*100:.2f}%"
    )

    # 4. Tabel Komparasi Metrik Evaluasi 5-Fold Out-Of-Fold
    comparison_rows = [
        compute_metrics_row(
            "sklearn GaussianNB (Baseline)",
            y_all,
            (oof_sk_gnb >= 0.50).astype(int),
            oof_sk_gnb,
            float(np.mean(fit_times_sk_g)),
        ),
        compute_metrics_row(
            "Scratch NaiveBayes (Gaussian)",
            y_all,
            (oof_scratch_gnb >= 0.50).astype(int),
            oof_scratch_gnb,
            float(np.mean(fit_times_sc_g)),
        ),
        compute_metrics_row(
            "sklearn CategoricalNB (alpha=1.0, Discretized)",
            y_all,
            (oof_sk_cat >= 0.50).astype(int),
            oof_sk_cat,
            float(np.mean(fit_times_sk_c)),
        ),
        compute_metrics_row(
            "Scratch NaiveBayes (Wajib Slide: Disc + Smooth +1 + MAP)",
            y_all,
            (oof_scratch_cat >= 0.50).astype(int),
            oof_scratch_cat,
            float(np.mean(fit_times_sc_c)),
        ),
        compute_metrics_row(
            f"Scratch NaiveBayes (Bonus TAN+CLL - Th_F1={threshold_f1:.3f})",
            y_all,
            (oof_nb_opt >= threshold_f1).astype(int),
            oof_nb_opt,
            float(np.mean(fit_times_opt)),
        ),
        compute_metrics_row(
            f"Scratch NaiveBayes (Bonus TAN+CLL - Th_Acc={threshold_acc:.3f})",
            y_all,
            (oof_nb_opt >= threshold_acc).astype(int),
            oof_nb_opt,
            float(np.mean(fit_times_opt)),
        ),
    ]
    comp_df = pd.DataFrame(comparison_rows)

    print("\n" + "=" * 104)
    print("TABEL KOMPARASI METRIK EVALUASI (5-FOLD OUT-OF-FOLD CROSS-VALIDATION - 27.373 SAMPEL)")
    print("=" * 104)
    metric_cols = ["Model", "Accuracy", "Macro F1", "Binary F1", "Precision", "Recall", "ROC-AUC", "Fit (ms/fold)"]
    print(comp_df[metric_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "-" * 104)
    print("RINCIAN CONFUSION MATRIX & TOTAL EXPECTED COST [TN, FP, FN, TP]:")
    print("-" * 104)
    print(comp_df[["Model", "TN", "FP", "FN", "TP", "Cost (2*FP+FN)"]].to_string(index=False))

    # 5. Pelatihan 1 Instance Model Tunggal NaiveBayes Teroptimasi pada 100% Data Train
    print("\n[5] Melatih 1 Model Tunggal NaiveBayes (TAN + Factored Cliques + Parzen KDE + CLL, Tanpa Ensemble) pada 100% Data Train...")
    full_train_clean = DataPreprocessor.clean_training_dataframe(
        train_df, target_col="is_canceled", drop_duplicates=True, fix_invalid_guests=True
    )
    _, X_full_raw, y_full = DataPreprocessor.extract_ids_and_target(full_train_clean)
    test_ids, X_test_raw, _ = DataPreprocessor.extract_ids_and_target(test_df)

    final_prep = _make_optimized_preprocessor(SEED)
    X_full_opt = final_prep.fit_transform(X_full_raw, y_full, return_dataframe=True)
    X_test_opt = final_prep.transform(X_test_raw, return_dataframe=True)

    final_nb = _make_optimized_model(decision_threshold=threshold_f1, seed=SEED)
    final_nb.fit(X_full_opt, y_full)

    model_v6_path = models_dir / "naive_bayes_model_v6.pkl"
    final_nb.save(str(model_v6_path))
    loaded_nb = NaiveBayes.load(str(model_v6_path))

    test_proba_before = final_nb.predict_proba(X_test_opt)
    test_preds_before = final_nb.predict(X_test_opt, threshold=threshold_f1)
    test_proba_after = loaded_nb.predict_proba(X_test_opt)
    test_preds_after = loaded_nb.predict(X_test_opt, threshold=threshold_f1)

    assert np.array_equal(test_preds_before, test_preds_after), "Prediksi sebelum dan sesudah load tidak identik!"
    assert np.allclose(test_proba_before, test_proba_after, atol=1e-15), "Probabilitas sebelum dan sesudah load berbeda!"

    sub_std_path = submissions_dir / "submission_naive_bayes.csv"
    df_sub = pd.DataFrame({
        "id": test_ids.astype(str),
        "is_canceled": test_preds_before.astype(int),
    })
    df_sub.to_csv(sub_std_path, index=False)

    graph_path = final_nb.plot_network_graph(save_path=str(figures_dir / "naive_bayes_graph.png"))

    n_raw_c = len(final_nb.clique_tables_) + len(getattr(final_nb, "clq_v19_tables_", []))
    n_imp_c = len(final_nb.imputed_clique_tables_)
    n_cll_dim = len(final_nb.cll_theta_) - 1 if final_nb.cll_theta_ is not None else 0
    l2_norm_w = float(np.linalg.norm(final_nb.cll_theta_[1:])) if final_nb.cll_theta_ is not None else 0.0
    print(f"    - Penalti L2 CLL Terbaik         : mean lambda = {final_nb.best_cll_l2_:.2f} (||w||_2 = {l2_norm_w:.4f})")
    print(f"    - Tabel Frekuensi Cliques        : {n_raw_c} Raw/Orth Cliques + {n_imp_c} Context/Cohort Cliques")
    print(f"    - Dimensi Basis Log-Odds CLL     : {n_cll_dim} faktor Bayesian aktif (0 Pohon, 0 Ensemble)")
    print(f"    - File Model Disimpan            : {model_v6_path} ({model_v6_path.stat().st_size:,} bytes)")
    print(f"    - Verifikasi Save/Load Identik   : True (100% Identik)")
    print(f"    - Graf Jaringan NB/TAN           : {graph_path}")
    print(
        f"    - Submission (Th={threshold_f1:.3f})          : {sub_std_path} "
        f"[0={(df_sub['is_canceled']==0).sum():,} ({(df_sub['is_canceled']==0).mean()*100:.2f}%), "
        f"1={(df_sub['is_canceled']==1).sum():,} ({(df_sub['is_canceled']==1).mean()*100:.2f}%)]"
    )

    print("\n" + "=" * 104)
    print("EKSPERIMEN NAIVE BAYES & PEMBUATAN SUBMISSION SELESAI TANPA ERROR!")
    print("=" * 104)


if __name__ == "__main__":
    main()
