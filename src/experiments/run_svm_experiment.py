from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiments.knn_common import SEED, add_date_features, make_preprocessor, weight_vector
from src.experiments.run_knn_experiment import FEATURE_WEIGHTS
from src.models.svm import SVM
from src.preprocessing import DataPreprocessor
from src.utils.model_io import load_model

MODELS_DIR = ROOT_DIR / "output" / "models"
SUBMISSIONS_DIR = ROOT_DIR / "output" / "submissions"
FIGURES_DIR = ROOT_DIR / "output" / "figures"
CV_DIR = ROOT_DIR / "output" / "svm_cv"

# gamma = gamma_mult / (n_fitur * Var(X)). Param Gaussian dari grid search holdout 80/20.
KERNEL_CONFIGS = {
    "linear": dict(kernel="linear", C=1.0),
    "polynomial": dict(kernel="polynomial", C=1.0, degree=3, coef0=1.0, gamma_mult=1.0),
    "gaussian": dict(kernel="gaussian", C=10.0, gamma_mult=3.0),
}
KAGGLE_KERNEL = "gaussian"

# Threshold tuning untuk F1 (Lipton, Elkan & Narayanaswamy, ECML PKDD 2014) https://link.springer.com/chapter/10.1007/978-3-662-44851-9_15
# dipilih dari OOF 5-fold Gaussian; F1 datar di -0.35..-0.10, ambil tengahnya.
THRESHOLD = -0.30

# Bobot fitur dari search_knn_weights.py. Di kernel Gaussian ini sama dengan gamma per fitur
# (Chapelle dkk., "Choosing Multiple Parameters for SVM", Machine Learning 46, 2002).
USE_FEATURE_WEIGHTS = True


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def preprocess(train_part, other_part):
    prep = make_preprocessor()
    X_train = prep.fit_transform(add_date_features(train_part), train_part["is_canceled"].to_numpy())
    X_other = prep.transform(add_date_features(other_part))
    columns = prep.final_feature_names_
    if USE_FEATURE_WEIGHTS:
        w = weight_vector(columns, FEATURE_WEIGHTS)
        X_train, X_other = X_train * w, X_other * w
    return X_train, X_other, columns


def resolve_params(cfg, X_train):
    params = {k: v for k, v in cfg.items() if k != "gamma_mult"}
    if "gamma_mult" in cfg:
        params["gamma"] = cfg["gamma_mult"] / (X_train.shape[1] * X_train.var())
    return params


def make_scratch(cfg, X_train):
    return SVM(**resolve_params(cfg, X_train))


def make_sklearn(cfg, X_train):
    p = resolve_params(cfg, X_train)
    sk_kernel = {"linear": "linear", "polynomial": "poly", "gaussian": "rbf"}[p["kernel"]]
    return SVC(kernel=sk_kernel, C=p["C"], gamma=p.get("gamma", "scale"),
               degree=p.get("degree", 3), coef0=p.get("coef0", 0.0), tol=1e-3, cache_size=500)


def macro_f1(y, scores, threshold):
    return f1_score(y, (scores >= threshold).astype(int), average="macro")


def best_threshold(y, scores, n_grid=401):
    grid = np.unique(np.round(np.quantile(scores, np.linspace(0.05, 0.95, n_grid)), 3))
    f1s = np.array([macro_f1(y, scores, t) for t in grid])
    k = int(np.argmax(f1s))
    return float(grid[k]), float(f1s[k])


def train_final_and_submit(train_df, test_df, kernel=KAGGLE_KERNEL, threshold=THRESHOLD):
    train_clean = DataPreprocessor.clean_training_dataframe(train_df)
    X_train, X_test, _ = preprocess(train_clean, test_df)
    y_train = train_clean["is_canceled"].to_numpy()

    start = time.time()
    model = make_scratch(KERNEL_CONFIGS[kernel], X_train)
    model.threshold = threshold
    model.fit(X_train, y_train)
    print(f"Latih {model} selesai dalam {time.time() - start:.1f} s "
          f"(iterasi SMO={model.estimators_[0].n_iter_}, #SV={model.n_support_[0]})")

    # hasil load harus memprediksi sama persis
    model_path = MODELS_DIR / "svm_kaggle.pkl"
    model.save(str(model_path))
    reloaded = SVM.load(str(model_path))
    reloaded_io = load_model(model_path)
    pred = model.predict(X_test)
    assert np.array_equal(pred, reloaded.predict(X_test)), "Prediksi model hasil load berbeda!"
    assert np.array_equal(pred, reloaded_io.predict(X_test)), "Prediksi model hasil load berbeda!"

    submission = pd.DataFrame({"id": test_df["id"], "is_canceled": pred.astype(int)})
    sub_path = SUBMISSIONS_DIR / "submission_svm.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False, lineterminator="\n")
    print(f"Model  : {model_path.relative_to(ROOT_DIR)}")
    print(f"Submisi: {sub_path.relative_to(ROOT_DIR)}  (positif={pred.mean():.3f})")
    print(f"md5    : {hashlib.md5(sub_path.read_bytes()).hexdigest()}")
    return sub_path


def cross_validate(train_df, kernels, n_folds):
    y = train_df["is_canceled"].to_numpy()
    impls = {"Scratch": make_scratch, "sklearn": make_sklearn}
    oof = {(k, m): np.zeros(len(y)) for k in kernels for m in impls}
    seconds = {(k, m): 0.0 for k in kernels for m in impls}
    n_sv = {(k, m): [] for k in kernels for m in impls}

    folds = StratifiedKFold(n_folds, shuffle=True, random_state=SEED).split(train_df, y)
    for f, (tr_idx, va_idx) in enumerate(folds, 1):
        fold_train = DataPreprocessor.clean_training_dataframe(train_df.iloc[tr_idx])
        X_tr, X_va, _ = preprocess(fold_train, train_df.iloc[va_idx])
        y_tr = fold_train["is_canceled"].to_numpy()
        for k in kernels:
            for m, make in impls.items():
                start = time.time()
                model = make(KERNEL_CONFIGS[k], X_tr).fit(X_tr, y_tr)
                oof[(k, m)][va_idx] = model.decision_function(X_va)
                seconds[(k, m)] += time.time() - start
                n_sv[(k, m)].append(int(np.sum(model.n_support_)))
                print(f"  fold {f}/{n_folds} {k:<10s} {m:<7s} "
                      f"F1={macro_f1(y[va_idx], oof[(k, m)][va_idx], 0.0):.4f} "
                      f"({time.time() - start:.1f} s)", flush=True)

    CV_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in kernels:
        s_scr, s_skl = oof[(k, "Scratch")], oof[(k, "sklearn")]
        agree = float(np.mean((s_scr >= 0) == (s_skl >= 0)))
        for m in impls:
            s = oof[(k, m)]
            t_best, f1_best = best_threshold(y, s)
            tn, fp, fn, tp = confusion_matrix(y, (s >= 0).astype(int)).ravel()
            rows.append({
                "Kernel": k, "Model": m,
                "Macro F1 (t=0)": macro_f1(y, s, 0.0),
                "Threshold*": t_best,
                "Macro F1 (t*)": f1_best,
                "Accuracy (t=0)": accuracy_score(y, (s >= 0).astype(int)),
                "Waktu fit+predict (detik)": seconds[(k, m)],
                "Rata-rata #SV": float(np.mean(n_sv[(k, m)])),
                "Kesepakatan prediksi Scratch vs sklearn": agree,
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            })
        pd.DataFrame([r for r in rows if r["Kernel"] == k]).to_csv(CV_DIR / f"results_{k}.csv", index=False)
    return pd.DataFrame(rows)


def load_cv_results():
    files = sorted(CV_DIR.glob("results_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    order = {k: i for i, k in enumerate(KERNEL_CONFIGS)}
    return df.sort_values(["Kernel", "Model"], key=lambda c: c.map(order) if c.name == "Kernel" else c)


def plot_cv_results(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    kernels = list(dict.fromkeys(df["Kernel"]))
    x = np.arange(len(kernels))
    colors = {"Scratch": "#2c6fbb", "sklearn": "#e67e22"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, col in zip(axes[:2], ["Macro F1 (t=0)", "Macro F1 (t*)"]):
        for i, m in enumerate(["Scratch", "sklearn"]):
            vals = [df[(df.Kernel == k) & (df.Model == m)][col].iloc[0] for k in kernels]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=m, color=colors[m])
            ax.bar_label(bars, fmt="%.4f", fontsize=8)
        ax.set_xticks(x, kernels)
        ax.set_ylim(min(df[col]) - 0.05, max(df[col]) + 0.02)
        ax.set_title(col.replace("t*", "threshold terbaik OOF"))
        ax.legend(loc="lower right")
    ax = axes[2]
    for i, m in enumerate(["Scratch", "sklearn"]):
        vals = [df[(df.Kernel == k) & (df.Model == m)]["Waktu fit+predict (detik)"].iloc[0] for k in kernels]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=m, color=colors[m])
        ax.bar_label(bars, fmt="%.0f", fontsize=8)
    ax.set_xticks(x, kernels)
    ax.set_title("Total waktu fit + predict seluruh fold (detik)")
    ax.legend()
    fig.suptitle("SVM from scratch vs scikit-learn (Stratified K-Fold, out-of-fold)")
    fig.tight_layout()
    out = FIGURES_DIR / "svm_vs_sklearn_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(kernels), figsize=(4.2 * len(kernels), 7.5), squeeze=False)
    for c, k in enumerate(kernels):
        for r, m in enumerate(["Scratch", "sklearn"]):
            row = df[(df.Kernel == k) & (df.Model == m)].iloc[0]
            cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
            ax = axes[r, c]
            ax.imshow(cm, cmap="Blues")
            for (a, b), v in np.ndenumerate(cm):
                ax.text(b, a, f"{v}", ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black")
            ax.set_xticks([0, 1], ["pred 0", "pred 1"])
            ax.set_yticks([0, 1], ["true 0", "true 1"])
            ax.set_title(f"{k} - {m}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "svm_confusion_matrices.png", dpi=150)
    plt.close(fig)
    print(f"Grafik: {out.relative_to(ROOT_DIR)}, output/figures/svm_confusion_matrices.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="jalankan perbandingan K-Fold scratch vs sklearn")
    parser.add_argument("--kernels", nargs="+", default=list(KERNEL_CONFIGS), choices=list(KERNEL_CONFIGS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--plot-only", action="store_true", help="hanya gambar ulang grafik dari output/svm_cv")
    parser.add_argument("--skip-submit", action="store_true", help="lewati pelatihan model final & submisi")
    args = parser.parse_args()

    seed_everything()
    pd.set_option("display.width", 200)
    if args.plot_only:
        df = load_cv_results()
        print(df.to_string(index=False, float_format="%.4f"))
        plot_cv_results(df)
        return

    train_df = pd.read_csv(ROOT_DIR / "data" / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(ROOT_DIR / "data" / "test.csv", dtype={"id": str})

    if not args.skip_submit:
        train_final_and_submit(train_df, test_df)

    if args.cv:
        cross_validate(train_df, args.kernels, args.folds)
        df = load_cv_results()
        print(df.to_string(index=False, float_format="%.4f"))
        plot_cv_results(df)


if __name__ == "__main__":
    main()
