"""Eksperimen End-to-End Decision Tree Learning (DTL / CART) From Scratch.

Menjalankan:
1. Cleaning & preprocessing jalur `dtl_ready` (memakai DataPreprocessor tim tanpa
   mengubah src/preprocessing.py; fitur numerik, kategorikal, dan nilai NaN dipertahankan).
2. Stratified 5-Fold CV & Minimal Cost-Complexity Pruning (ccp_alpha).
3. Komparasi DTL from scratch terhadap 3 konfigurasi scikit-learn wajib:
   - ID3  : DecisionTreeClassifier(criterion='entropy')
   - C4.5 : DecisionTreeClassifier(criterion='entropy', ccp_alpha=...)
   - CART : DecisionTreeClassifier(criterion='gini')
4. Pelatihan model final, uji save/load via src/utils/model_io.py, visualisasi
   top-N tree, grafik perbandingan metrik/runtime/confusion matrix, dan ekspor
   output/submissions/submission_dtl.csv.
"""
from __future__ import annotations

import copy
import hashlib
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.dtl import DecisionTree
from src.preprocessing import DataPreprocessor
from src.utils.model_io import load_model, save_model

SEED = 42
TARGET = "is_canceled"
CAT_COLS = ["meal_package", "room_category", "booking_channel"]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "output" / "models"
FIGURES_DIR = ROOT_DIR / "output" / "figures"
SUBMISSIONS_DIR = ROOT_DIR / "output" / "submissions"

BEST_DTL_PARAMS: Dict[str, Any] = dict(
    criterion="entropy",
    max_depth=22,
    min_samples_leaf=5,
    min_samples_split=10,
    n_bins=64,
    ccp_alpha=0.0001,
)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def stratified_kfold_indices(y: np.ndarray, k: int = 5, seed: int = SEED) -> np.ndarray:
    """Membagi indeks menjadi k fold terstratifikasi secara deterministik."""
    rng = np.random.RandomState(seed)
    fold = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = np.arange(len(idx)) % k
    return fold


class DTLFeatureBuilder:
    """Preprocessing khusus DTL di atas keluaran DataPreprocessor(mode='dtl_ready').

    Mempertahankan kolom numerik, kolom kategorikal, serta nilai NaN asli agar
    ditangani langsung oleh mekanisme internal kelas DecisionTree. Seluruh kamus
    kategori hanya dipelajari pada data latih (.fit()).
    """

    def fit(self, df: pd.DataFrame) -> "DTLFeatureBuilder":
        self.cats_: Dict[str, List[Any]] = {c: sorted(df[c].dropna().unique()) for c in CAT_COLS}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        d = df.copy()
        d.loc[d["nightly_rate_idr"] <= 0, "nightly_rate_idr"] = np.nan
        d.loc[d["num_adults"] <= 0, "num_adults"] = np.nan
        d["total_nights"] = d["weekend_nights"] + d["weekday_nights"]
        d["total_guests"] = d["num_adults"] + d["num_children"].fillna(0)
        d["total_price"] = d["nightly_rate_idr"] * d["total_nights"]
        d["month_index"] = (d["arrival_year"] - 2023) * 12 + d["arrival_month"]
        dt = pd.to_datetime(
            dict(
                year=d["arrival_year"],
                month=d["arrival_month"],
                day=d["arrival_day"].clip(1, 28),
            ),
            errors="coerce",
        )
        d["arrival_dow"] = dt.dt.dayofweek
        hist = d["prior_cancellations"] + d["prior_completed_bookings"]
        d["has_history"] = (hist > 0).astype(int)
        d["cancel_ratio"] = d["prior_cancellations"] / hist.replace(0, np.nan)
        for c in CAT_COLS:
            d[c] = d[c].map({v: i for i, v in enumerate(self.cats_[c])})  # unseen/null -> NaN
        self.feature_names_ = [c for c in d.columns if c not in ("id", TARGET)]
        self.cat_idx_ = [self.feature_names_.index(c) for c in CAT_COLS]
        return d[self.feature_names_].to_numpy(dtype=float)


def prepare_dtl_folds(
    train_df: pd.DataFrame, n_splits: int = 5, seed: int = SEED
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]]]:
    """Membersihkan duplikat di train lalu membangun fold Stratified K-Fold bebas leakage."""
    clean_df = DataPreprocessor.clean_training_dataframe(
        train_df, target_col=TARGET, drop_duplicates=True, fix_invalid_guests=False
    )
    y = clean_df[TARGET].to_numpy(dtype=int)
    fold = stratified_kfold_indices(y, k=n_splits, seed=seed)
    folds = []
    for k in range(n_splits):
        a, b = fold != k, fold == k
        base_prep = DataPreprocessor(mode="dtl_ready", random_state=seed)
        df_tr_raw = base_prep.fit_transform(clean_df[a], return_dataframe=True)
        df_va_raw = base_prep.transform(clean_df[b], return_dataframe=True)
        fb = DTLFeatureBuilder().fit(df_tr_raw)
        folds.append((fb.transform(df_tr_raw), y[a], fb.transform(df_va_raw), y[b], fb.cat_idx_))
    return clean_df, y, fold, folds


def run_cv_tuning(
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]],
    seed: int = SEED,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluasi hyperparameter dasar + Minimal Cost-Complexity Pruning (tree reuse)."""
    rows: List[Dict[str, Any]] = []
    for crit in ("gini", "entropy"):
        for leaf in (5, 8, 20):
            p = dict(criterion=crit, max_depth=12, min_samples_leaf=leaf, min_samples_split=2 * leaf, n_bins=64)
            va, tr_s = [], []
            for Xa, ya, Xb, yb, ci in folds:
                m = DecisionTree(categorical_features=ci, random_state=seed, **p).fit(Xa, ya)
                va.append(f1_score(yb, m.predict(Xb), average="macro"))
                tr_s.append(f1_score(ya, m.predict(Xa), average="macro"))
            m_va, s_va, m_tr = float(np.mean(va)), float(np.std(va)), float(np.mean(tr_s))
            rows.append({**p, "ccp_alpha": 0.0, "cv_macroF1": m_va, "cv_std": s_va, "train_macroF1": m_tr, "gap": m_tr - m_va})

    for leaf in (5, 8):
        base_trees = [
            DecisionTree(
                criterion="entropy",
                max_depth=22,
                min_samples_leaf=leaf,
                min_samples_split=2 * leaf,
                ccp_alpha=0.0,
                n_bins=64,
                categorical_features=ci,
                random_state=seed,
            ).fit(Xa, ya)
            for Xa, ya, Xb, yb, ci in folds
        ]
        for alpha in (0.00010, 0.00012, 0.00015):
            va, tr_s = [], []
            for t_tree, (Xa, ya, Xb, yb, _) in zip(base_trees, folds):
                pruned = copy.deepcopy(t_tree)
                pruned._prune(alpha)
                va.append(f1_score(yb, pruned.predict(Xb), average="macro"))
                tr_s.append(f1_score(ya, pruned.predict(Xa), average="macro"))
            m_va, s_va, m_tr = float(np.mean(va)), float(np.std(va)), float(np.mean(tr_s))
            rows.append(
                dict(
                    criterion="entropy",
                    max_depth=22,
                    min_samples_leaf=leaf,
                    min_samples_split=2 * leaf,
                    n_bins=64,
                    ccp_alpha=alpha,
                    cv_macroF1=m_va,
                    cv_std=s_va,
                    train_macroF1=m_tr,
                    gap=m_tr - m_va,
                )
            )

    res = pd.DataFrame(rows).sort_values("cv_macroF1", ascending=False).reset_index(drop=True)
    top = res.iloc[0]
    ok = res[res["cv_macroF1"] >= top["cv_macroF1"] - 0.0005]
    sel = ok.sort_values("gap").iloc[0]
    best_params = dict(
        criterion=str(sel["criterion"]),
        max_depth=int(sel["max_depth"]),
        min_samples_leaf=int(sel["min_samples_leaf"]),
        min_samples_split=int(sel["min_samples_split"]),
        n_bins=int(sel["n_bins"]),
        ccp_alpha=float(sel["ccp_alpha"]),
    )
    return res, best_params


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _save_comparison_plots(summary_df: pd.DataFrame, y: np.ndarray, oof_preds: Dict[str, np.ndarray], show: bool = False) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    colors = ["#2563eb", "#0d9488", "#d97706", "#4f46e5"]

    # 1. Perbandingan metrik
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [("accuracy", "Accuracy"), ("precision_macro", "Precision (Macro)"),
               ("recall_macro", "Recall (Macro)"), ("f1_macro", "Macro F1-Score")]
    labels = summary_df["model"].tolist()
    x = np.arange(len(labels))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        vals = summary_df[col].to_numpy(dtype=float)
        bars = ax.bar(x, vals, color=colors[: len(vals)], alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8.5)
        ax.set_ylim(0.75, 0.92)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("Decision Tree: From Scratch vs scikit-learn (5-Fold CV)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dtl_vs_sklearn_metrics.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 2. Perbandingan waktu latih
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    vals = summary_df["fit_time_s"].to_numpy(dtype=float)
    bars = ax.bar(x, vals, color=colors[: len(vals)], alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=8.5)
    ax.set_title("Decision Tree 5-Fold Training Time (seconds)", fontweight="bold")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dtl_vs_sklearn_runtime.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 3. Confusion matrices (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    for ax, (name, yp) in zip(axes.ravel(), oof_preds.items()):
        cm = confusion_matrix(y, yp, labels=[0, 1])
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Canceled (0)", "Canceled (1)"], fontsize=8.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Not Canceled (0)", "Canceled (1)"], fontsize=8.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        thr = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > thr else "#111827", fontweight="bold")
    fig.suptitle("Decision Tree Confusion Matrices (5-Fold OOF)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dtl_confusion_matrices.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_sklearn_comparison(
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]],
    y: np.ndarray,
    fold: np.ndarray,
    final_params: Dict[str, Any],
    seed: int = SEED,
    save_figures: bool = True,
    show: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Membandingkan DTL from scratch terhadap ID3, C4.5, dan CART milik scikit-learn."""
    sk_alpha = 0.0005
    sk_cfg = {
        "DTL ID3 (entropy)": dict(criterion="entropy"),
        "DTL C4.5 (entropy+ccp)": dict(criterion="entropy", ccp_alpha=sk_alpha),
        "DTL CART (gini)": dict(criterion="gini"),
    }
    common = dict(
        max_depth=final_params["max_depth"] if final_params["ccp_alpha"] == 0 else 12,
        min_samples_leaf=final_params["min_samples_leaf"],
        random_state=seed,
    )

    records: List[Dict[str, Any]] = []
    oof_preds: Dict[str, np.ndarray] = {}

    # 1. From scratch (final)
    oof_p = np.zeros(len(y), dtype=int)
    oof_prob = np.zeros(len(y), dtype=float)
    t_fit, t_pred = 0.0, 0.0
    for k, (Xa, ya, Xb, yb, ci) in enumerate(folds):
        t0 = time.perf_counter()
        m = DecisionTree(categorical_features=ci, random_state=seed, **final_params).fit(Xa, ya)
        t_fit += time.perf_counter() - t0
        t0 = time.perf_counter()
        prob = m.predict_proba(Xb)
        pred = (prob >= 0.5).astype(int)
        t_pred += time.perf_counter() - t0
        oof_p[fold == k] = pred
        oof_prob[fold == k] = prob
    oof_preds["DTL From Scratch (Final)"] = oof_p
    met = _compute_metrics(y, oof_p, oof_prob)
    records.append(
        dict(
            model="DTL Final (Scratch)",
            implementation="from_scratch",
            **met,
            fit_time_s=t_fit,
            predict_time_s=t_pred,
            total_time_s=t_fit + t_pred,
        )
    )

    # 2. Tiga konfigurasi scikit-learn (ID3, C4.5, CART)
    for name, cfg in sk_cfg.items():
        oof_p = np.zeros(len(y), dtype=int)
        oof_prob = np.zeros(len(y), dtype=float)
        t_fit, t_pred = 0.0, 0.0
        for k, (Xa, ya, Xb, yb, _) in enumerate(folds):
            med = np.nanmedian(Xa, axis=0)
            Xa_imp = np.where(np.isnan(Xa), med, Xa)
            Xb_imp = np.where(np.isnan(Xb), med, Xb)
            t0 = time.perf_counter()
            m = DecisionTreeClassifier(**common, **cfg).fit(Xa_imp, ya)
            t_fit += time.perf_counter() - t0
            t0 = time.perf_counter()
            prob = m.predict_proba(Xb_imp)[:, 1]
            pred = (prob >= 0.5).astype(int)
            t_pred += time.perf_counter() - t0
            oof_p[fold == k] = pred
            oof_prob[fold == k] = prob
        oof_preds[f"sklearn {name}"] = oof_p
        met = _compute_metrics(y, oof_p, oof_prob)
        records.append(
            dict(
                model=name,
                implementation="scikit_learn",
                **met,
                fit_time_s=t_fit,
                predict_time_s=t_pred,
                total_time_s=t_fit + t_pred,
            )
        )

    summary_df = pd.DataFrame(records)
    if save_figures:
        _save_comparison_plots(summary_df, y, oof_preds, show=show)
    return summary_df, oof_preds


def train_final_and_export(
    clean_df: pd.DataFrame,
    test_df: pd.DataFrame,
    final_params: Dict[str, Any],
    seed: int = SEED,
    show_tree: bool = False,
) -> Tuple[DecisionTree, Path, Path, Path]:
    """Melatih model final di seluruh data latih bersih, save/load, plot tree, dan ekspor CSV."""
    y_full = clean_df[TARGET].to_numpy(dtype=int)
    base_prep = DataPreprocessor(mode="dtl_ready", random_state=seed)
    df_tr_raw = base_prep.fit_transform(clean_df, return_dataframe=True)
    df_te_raw = base_prep.transform(test_df, return_dataframe=True)

    fb = DTLFeatureBuilder().fit(df_tr_raw)
    X_tr = fb.transform(df_tr_raw)
    X_te = fb.transform(df_te_raw)

    model = DecisionTree(categorical_features=fb.cat_idx_, random_state=seed, **final_params).fit(X_tr, y_full)

    # Verifikasi save & load melalui src/utils/model_io.py
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(save_model(model, MODELS_DIR / "dtl_kaggle.pkl"))
    loaded = load_model(model_path)
    pred = loaded.predict(X_te)
    assert np.array_equal(pred, model.predict(X_te)), "Prediksi setelah load_model tidak identik!"
    assert not np.isnan(loaded.predict_proba(X_te)).any(), "Probabilitas prediksi mengandung NaN!"

    # Bonus visualisasi top-N tree
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tree_path = Path(
        model.plot_tree(
            top_n=3,
            filepath=str(FIGURES_DIR / "dtl_tree.png"),
            feature_names=fb.feature_names_,
            show=show_tree,
        )
    )

    # Ekspor submission CSV
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "submission_dtl.csv"
    sub = pd.DataFrame({"id": test_df["id"].astype(str), TARGET: pred.astype(int)})
    sub.to_csv(sub_path, index=False)

    return loaded, model_path, tree_path, sub_path


def main() -> None:
    seed_everything(SEED)
    train_df = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(DATA_DIR / "test.csv", dtype={"id": str})
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", dtype={"id": str})

    clean_df, y, fold, folds = prepare_dtl_folds(train_df, n_splits=5, seed=SEED)
    print(f"Train bersih: {clean_df.shape} | rasio cancel: {y.mean():.4f}")

    skip_grid = "--fast" in sys.argv
    if skip_grid:
        final_params = dict(BEST_DTL_PARAMS)
        print("Mode --fast: menggunakan BEST_DTL_PARAMS =", final_params)
    else:
        res, final_params = run_cv_tuning(folds, seed=SEED)
        print("\n=== Hasil 5-Fold CV & Minimal Cost-Complexity Pruning ===")
        print(res.round(4).to_string(index=False))
        print("\nParameter final terpilih:", final_params)

    print("\n=== Komparasi From Scratch vs scikit-learn (ID3, C4.5, CART) ===")
    cmp_df, _ = run_sklearn_comparison(folds, y, fold, final_params, seed=SEED, save_figures=True, show=False)
    cols = ["model", "implementation", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "fit_time_s"]
    print(cmp_df[cols].round(4).to_string(index=False))

    model, model_path, tree_path, sub_path = train_final_and_export(clean_df, test_df, final_params, seed=SEED)
    sub = pd.read_csv(sub_path, dtype={"id": str})
    assert len(sub) == len(sample_sub) and set(sub["id"]) == set(sample_sub["id"])

    print(f"\nKedalaman final : {model.get_depth()} | Jumlah daun: {model.count_leaves()}")
    print(f"Model tersimpan : {model_path.relative_to(ROOT_DIR)}")
    print(f"Gambar tree     : {tree_path.relative_to(ROOT_DIR)}")
    print(f"Submisi         : {sub_path.relative_to(ROOT_DIR)} ({len(sub)} baris, rasio cancel={sub[TARGET].mean():.4f})")
    print(f"MD5 submisi     : {hashlib.md5(sub_path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
