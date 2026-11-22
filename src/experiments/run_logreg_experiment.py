import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.logistic_regression import LogisticRegression
from src.preprocessing import DataPreprocessor

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Mengunci seluruh generator angka acak untuk menjamin reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def _make_baseline_preprocessor(seed: int = SEED) -> DataPreprocessor:
    """Membuat instance preprocessor standar (onehot + standard, tanpa ekspansi non-linear)."""
    return DataPreprocessor(
        mode="processed_numeric",
        add_missing_indicators=True,
        add_domain_features=True,
        add_composite_interactions=False,
        encoding_method="onehot",
        scaling_method="standard",
        rare_category_min_count=10,
        random_state=seed,
    )


def _make_optimized_preprocessor(seed: int = SEED) -> DataPreprocessor:
    """
    Membuat instance preprocessor teroptimasi untuk Logistic Regression:
    - Domain & Composite Interactions (38 fitur kanonik TAN_V3_FEATURE_SET)
    - Multi-Resolution Half-Step Shifted Cliques WoE (91 Raw + 30 Context-Imputed/Booking-Day)
    - Dual-Path Kernel-Weighted Context Imputation & Relative-Scale Parzen-Window KDE28 WoE
    - Unsupervised Cohort Z-Scores & Local Density Gating
    - Piecewise ReLU Hinge Splines + Degree-2 Polynomial Log-Odds Interactions (982 fitur)
    """
    return DataPreprocessor(
        mode="processed_numeric",
        add_missing_indicators=True,
        add_domain_features=True,
        add_composite_interactions=True,
        composite_lead_bins=14,
        composite_rate_bins=10,
        max_children_clip=4.0,
        rare_category_min_count=0,
        add_linear_nonlinear_features=True,
        encoding_method="ordinal",
        scaling_method="none",
        use_rff=False,
        random_state=seed,
    )


def _make_optimized_model(threshold: float = 0.5, seed: int = SEED) -> LogisticRegression:
    """Membuat instance model tunggal LogisticRegression teroptimasi berbasis NumPy L-BFGS."""
    return LogisticRegression(
        penalty="l2",
        C=0.01,
        optimizer="lbfgs",
        max_iter=250,
        tol=1e-7,
        lbfgs_memory=14,
        threshold=threshold,
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

    print("=" * 102)
    print("EKSPERIMEN LOGISTIC REGRESSION FROM SCRATCH vs SKLEARN & OPTIMASI FINAL (IF3170_G09_Aduhmen)")
    print("=" * 102)

    # 1. Load Dataset
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    y_all = train_df["is_canceled"].to_numpy(dtype=np.int64)
    n_samples = len(train_df)
    target_prior_pos = float(np.mean(y_all))

    print(f"\n[1] Dataset Berhasil Dimuat:")
    print(f"    - Train Set : {train_df.shape[0]:,} baris x {train_df.shape[1]} kolom (Proporsi Positif = {target_prior_pos:.4f})")
    print(f"    - Test Set  : {test_df.shape[0]:,} baris x {test_df.shape[1]} kolom")

    # 2. Komparasi Keempat Varian Regularisasi pada DataPreprocessor Standar (Fold-1 Parity Check)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    f1_tr_idx, f1_va_idx = next(skf.split(train_df, y_all))
    df_f1_tr = DataPreprocessor.clean_training_dataframe(train_df.iloc[f1_tr_idx].reset_index(drop=True))
    _, X_f1_tr_raw, y_f1_tr = DataPreprocessor.extract_ids_and_target(df_f1_tr)
    _, X_f1_va_raw, _ = DataPreprocessor.extract_ids_and_target(train_df.iloc[f1_va_idx].reset_index(drop=True))
    y_f1_va = y_all[f1_va_idx]

    prep_base_f1 = _make_baseline_preprocessor(SEED)
    X_f1_tr_base = prep_base_f1.fit_transform(X_f1_tr_raw, y_f1_tr, return_dataframe=False)
    X_f1_va_base = prep_base_f1.transform(X_f1_va_raw, return_dataframe=False)

    print(f"\n[2] Komparasi 4 Varian Regularisasi vs scikit-learn pada Fold-1 ({X_f1_tr_base.shape[1]} Fitur Baseline):")
    penalty_configs = [
        ("None", dict(penalty="none", optimizer="lbfgs", max_iter=200), dict(penalty=None, solver="lbfgs", C=1.0, max_iter=500)),
        ("L1 (C=1.0)", dict(penalty="l1", C=1.0, optimizer="adam", learning_rate=0.01, max_iter=60, batch_size=256), dict(penalty="l1", solver="saga", C=1.0, max_iter=500, random_state=SEED)),
        ("L2 (C=1.0)", dict(penalty="l2", C=1.0, optimizer="lbfgs", max_iter=200), dict(penalty="l2", solver="lbfgs", C=1.0, max_iter=500)),
        ("ElasticNet (C=1.0, l1=0.5)", dict(penalty="elasticnet", C=1.0, l1_ratio=0.5, optimizer="adam", learning_rate=0.01, max_iter=60, batch_size=256), dict(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=500, random_state=SEED)),
    ]
    for label, my_kw, sk_kw in penalty_configs:
        m_sc = LogisticRegression(**my_kw, random_state=SEED).fit(X_f1_tr_base, y_f1_tr)
        try:
            m_sk = SkLogisticRegression(**sk_kw).fit(X_f1_tr_base, y_f1_tr)
        except TypeError:
            # Kompatibilitas versi sklearn terbaru terkait parameter penalty=None / l1_ratio
            sk_alt = {k: v for k, v in sk_kw.items() if k != "penalty"}
            if sk_kw.get("penalty") is None:
                sk_alt["C"] = 1e12
            elif sk_kw.get("penalty") == "l1":
                sk_alt["l1_ratio"] = 1.0
            m_sk = SkLogisticRegression(**sk_alt).fit(X_f1_tr_base, y_f1_tr)
        p_sc = m_sc.predict_proba(X_f1_va_base)[:, 1]
        p_sk = m_sk.predict_proba(X_f1_va_base)[:, 1]
        match_lbl = np.mean((p_sc >= 0.5) == (p_sk >= 0.5)) * 100.0
        print(
            f"    - {label:<26} | Scratch Acc={accuracy_score(y_f1_va, p_sc>=0.5):.5f} AUC={roc_auc_score(y_f1_va, p_sc):.5f} "
            f"| Sklearn Acc={accuracy_score(y_f1_va, p_sk>=0.5):.5f} AUC={roc_auc_score(y_f1_va, p_sk):.5f} "
            f"| LabelMatch={match_lbl:.2f}%"
        )

    # 3. Outer Stratified 5-Fold Cross-Validation (Baseline vs Optimized LogisticRegression)
    oof_base_scratch = np.zeros(n_samples, dtype=np.float64)
    oof_opt_scratch = np.zeros(n_samples, dtype=np.float64)
    oof_opt_sklearn = np.zeros(n_samples, dtype=np.float64)
    fit_ms_base, fit_ms_opt_sc, fit_ms_opt_sk = [], [], []
    opt_dims = []

    print("\n[3] Menjalankan Outer Stratified 5-Fold Cross-Validation (SEED = 42, Zero Data Leakage)...")
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_all), start=1):
        df_tr_fold = train_df.iloc[tr_idx].reset_index(drop=True)
        df_va_fold = train_df.iloc[va_idx].reset_index(drop=True)
        df_tr_clean = DataPreprocessor.clean_training_dataframe(
            df_tr_fold, target_col="is_canceled", drop_duplicates=True, fix_invalid_guests=True
        )
        _, X_tr_raw, y_tr_fold = DataPreprocessor.extract_ids_and_target(df_tr_clean)
        _, X_va_raw, _ = DataPreprocessor.extract_ids_and_target(df_va_fold)

        # A. Baseline Standard One-Hot + StandardScaler (L2)
        prep_b = _make_baseline_preprocessor(SEED)
        X_tr_b = prep_b.fit_transform(X_tr_raw, y_tr_fold, return_dataframe=False)
        X_va_b = prep_b.transform(X_va_raw, return_dataframe=False)
        t0 = time.perf_counter()
        lr_b = LogisticRegression(penalty="l2", C=1.0, optimizer="lbfgs", max_iter=200, random_state=SEED).fit(X_tr_b, y_tr_fold)
        fit_ms_base.append((time.perf_counter() - t0) * 1000.0)
        oof_base_scratch[va_idx] = lr_b.predict_proba(X_va_b)[:, 1]

        # B. Optimized Preprocessing (Linear+NonLinear + LOO Freq + OOF TE + RFF1600)
        prep_opt = _make_optimized_preprocessor(SEED)
        X_tr_opt = prep_opt.fit_transform(X_tr_raw, y_tr_fold, return_dataframe=False)
        X_va_opt = prep_opt.transform(X_va_raw, return_dataframe=False)
        opt_dims.append(X_tr_opt.shape[1])

        t0 = time.perf_counter()
        lr_opt = _make_optimized_model(seed=SEED).fit(X_tr_opt, y_tr_fold)
        fit_ms_opt_sc.append((time.perf_counter() - t0) * 1000.0)
        oof_opt_scratch[va_idx] = lr_opt.predict_proba(X_va_opt)[:, 1]

        t0 = time.perf_counter()
        sk_opt = SkLogisticRegression(C=0.01, max_iter=500, solver="lbfgs").fit(X_tr_opt, y_tr_fold)
        fit_ms_opt_sk.append((time.perf_counter() - t0) * 1000.0)
        oof_opt_sklearn[va_idx] = sk_opt.predict_proba(X_va_opt)[:, 1]

        print(
            f"    - Fold {fold_idx}/5 (dim={X_tr_opt.shape[1]}) | Baseline AUC={roc_auc_score(y_all[va_idx], oof_base_scratch[va_idx]):.5f} "
            f"| Optimized Scratch Acc={accuracy_score(y_all[va_idx], oof_opt_scratch[va_idx]>=0.5):.5f} "
            f"AUC={roc_auc_score(y_all[va_idx], oof_opt_scratch[va_idx]):.5f}",
            flush=True,
        )

    match_opt_pct = float(np.mean((oof_opt_scratch >= 0.5) == (oof_opt_sklearn >= 0.5))) * 100.0
    max_diff_opt = float(np.max(np.abs(oof_opt_scratch - oof_opt_sklearn)))
    print(f"    - Kecocokan Prediksi OOF Optimized Scratch vs sklearn L-BFGS : {match_opt_pct:.4f}%")
    print(f"    - Selisih Maksimum Probabilitas OOF |P_scratch - P_sklearn|  : {max_diff_opt:.3e}")

    # 4. Kalibrasi Decision Threshold pada 5-Fold OOF Predictions
    thresholds = np.round(np.arange(0.35, 0.651, 0.005), 3)
    threshold_f1, best_f1 = 0.50, -1.0
    threshold_acc, best_acc = 0.50, -1.0
    for th in thresholds:
        preds = (oof_opt_scratch >= th).astype(int)
        acc_val = accuracy_score(y_all, preds)
        f1_val = f1_score(y_all, preds, average="macro", zero_division=0)
        if f1_val > best_f1:
            best_f1, threshold_f1 = f1_val, float(th)
        if acc_val > best_acc:
            best_acc, threshold_acc = acc_val, float(th)

    print("\n[4] Kalibrasi Decision Threshold Logistic Regression Teroptimasi (5-Fold OOF):")
    for label_th, th_v in [
        ("threshold=0.50  (Default)", 0.50),
        ("threshold_acc   (Maksimum Accuracy)", threshold_acc),
        ("threshold_f1    (Maksimum Macro F1)", threshold_f1),
    ]:
        p_th = (oof_opt_scratch >= th_v).astype(int)
        print(
            f"    - {label_th:<38} : Th = {th_v:.3f} | "
            f"Acc = {accuracy_score(y_all, p_th):.5f} | "
            f"Macro F1 = {f1_score(y_all, p_th, average='macro'):.5f} | "
            f"ROC-AUC = {roc_auc_score(y_all, oof_opt_scratch):.5f} | "
            f"PosRate = {p_th.mean()*100:.2f}%"
        )

    # 5. Tabel Komparasi Metrik Evaluasi 5-Fold Out-Of-Fold
    comp_rows = [
        compute_metrics_row(
            "Scratch LogReg (Baseline Standard)",
            y_all,
            (oof_base_scratch >= 0.50).astype(int),
            oof_base_scratch,
            float(np.mean(fit_ms_base)),
        ),
        compute_metrics_row(
            "sklearn LogReg (Optimized Preprocessing)",
            y_all,
            (oof_opt_sklearn >= 0.50).astype(int),
            oof_opt_sklearn,
            float(np.mean(fit_ms_opt_sk)),
        ),
        compute_metrics_row(
            "Scratch LogReg (Optimized - Th=0.500)",
            y_all,
            (oof_opt_scratch >= 0.50).astype(int),
            oof_opt_scratch,
            float(np.mean(fit_ms_opt_sc)),
        ),
        compute_metrics_row(
            f"Scratch LogReg (Optimized - Th={threshold_acc:.3f})",
            y_all,
            (oof_opt_scratch >= threshold_acc).astype(int),
            oof_opt_scratch,
            float(np.mean(fit_ms_opt_sc)),
        ),
    ]
    comp_df = pd.DataFrame(comp_rows)

    print("\n" + "=" * 102)
    print(f"TABEL KOMPARASI METRIK 5-FOLD OOF (dim rata-rata fitur teroptimasi ~ {int(np.mean(opt_dims))} fitur)")
    print("=" * 102)
    metric_cols = ["Model", "Accuracy", "Macro F1", "Binary F1", "Precision", "Recall", "ROC-AUC", "Fit (ms/fold)"]
    print(comp_df[metric_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "-" * 102)
    print("RINCIAN CONFUSION MATRIX & TOTAL EXPECTED COST [TN, FP, FN, TP]:")
    print("-" * 102)
    print(comp_df[["Model", "TN", "FP", "FN", "TP", "Cost (2*FP+FN)"]].to_string(index=False))

    # 6. Pelatihan Model Tunggal Final pada 100% Data Train & Simpan Artefak ke output/models & output/submissions
    print("\n[5] Melatih 1 Model Tunggal LogisticRegression Teroptimasi pada 100% Data Train...")
    full_train_clean = DataPreprocessor.clean_training_dataframe(
        train_df, target_col="is_canceled", drop_duplicates=True, fix_invalid_guests=True
    )
    _, X_full_raw, y_full = DataPreprocessor.extract_ids_and_target(full_train_clean)
    test_ids, X_test_raw, _ = DataPreprocessor.extract_ids_and_target(test_df)

    final_prep = _make_optimized_preprocessor(SEED)
    X_full_mat = final_prep.fit_transform(X_full_raw, y_full, return_dataframe=False)
    X_test_mat = final_prep.transform(X_test_raw, return_dataframe=False)
    print(f"    - Dimensi Matriks Final : train {X_full_mat.shape}, test {X_test_mat.shape}")

    final_lr = _make_optimized_model(threshold=threshold_acc, seed=SEED)
    final_lr.fit(X_full_mat, y_full)

    model_path = models_dir / "logreg_kaggle.pkl"
    final_lr.save(str(model_path))
    loaded_lr = LogisticRegression.load(str(model_path))

    test_proba_before = final_lr.predict_proba(X_test_mat)
    test_preds_before = final_lr.predict(X_test_mat, threshold=threshold_acc)
    test_proba_after = loaded_lr.predict_proba(X_test_mat)
    test_preds_after = loaded_lr.predict(X_test_mat)

    assert np.array_equal(test_preds_before, test_preds_after), "Prediksi LogReg sebelum & sesudah load tidak identik!"
    assert np.allclose(test_proba_before, test_proba_after, atol=1e-15), "Probabilitas LogReg sebelum & sesudah load berbeda!"

    sub_path = submissions_dir / "submission_LR.csv"
    df_sub = pd.DataFrame({
        "id": test_ids.astype(str),
        "is_canceled": test_preds_before.astype(int),
    })
    df_sub.to_csv(sub_path, index=False)

    x_dba = pd.to_numeric(X_full_raw["days_before_arrival"], errors="coerce").fillna(58.0).to_numpy()
    contour_video_path = final_lr.animate_loss_contour(
        x_dba,
        y_full,
        feature_name="days_before_arrival (Standardized)",
        save_path=str(figures_dir / "logreg_contour.gif"),
    )

    print(f"    - Iterasi L-BFGS Konvergen   : {final_lr.n_iter_} iterasi")
    print(f"    - File Model Disimpan        : {model_path} ({model_path.stat().st_size:,} bytes)")
    print(f"    - Verifikasi Save/Load       : True (100% Identik)")
    print(f"    - Video Kontur Log-Loss      : {contour_video_path}")
    print(
        f"    - File Submisi (Th={threshold_acc:.3f})   : {sub_path} "
        f"[0={(df_sub['is_canceled']==0).sum():,} ({(df_sub['is_canceled']==0).mean()*100:.2f}%), "
        f"1={(df_sub['is_canceled']==1).sum():,} ({(df_sub['is_canceled']==1).mean()*100:.2f}%)]"
    )
    print("\n" + "=" * 102)
    print("EKSPERIMEN LOGISTIC REGRESSION & PEMBUATAN SUBMISI SELESAI TANPA ERROR!")
    print("=" * 102)


if __name__ == "__main__":
    main()