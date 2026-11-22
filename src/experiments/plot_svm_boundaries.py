from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.experiments.knn_common import SEED, add_date_features, make_preprocessor
from src.models.svm import SVM
from src.preprocessing import DataPreprocessor

FIGURES_DIR = ROOT_DIR / "output" / "figures"

FEATURE_PAIR = ("days_before_arrival", "special_requests")
KERNEL_SETTINGS = [
    ("Linear", dict(kernel="linear", C=1.0, class_weight="balanced")),
    ("Polynomial (degree=3)", dict(kernel="polynomial", C=1.0, degree=3, coef0=1.0, gamma="scale",
                                   class_weight="balanced")),
    ("Gaussian (RBF)", dict(kernel="gaussian", C=1.0, gamma="scale", class_weight="balanced")),
]


def pca_2d(X):
    mean = X.mean(axis=0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    explained = (s ** 2) / np.sum(s ** 2)
    Z = (X - mean) @ vt[:2].T
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)    # samakan skala sumbu
    return Z, explained[:2]


def stratified_sample(y, n, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        k = int(round(n * len(cls) / len(y)))
        idx.append(rng.choice(cls, size=min(k, len(cls)), replace=False))
    return np.sort(np.concatenate(idx))


def plot_kernel(ax, Z, y, title, params, jitter_y=False, seed=SEED):
    model = SVM(**params).fit(Z, y)
    pad = 0.5
    x_min, x_max = Z[:, 0].min() - pad, Z[:, 0].max() + pad
    y_min, y_max = Z[:, 1].min() - pad, Z[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    f = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    lim = np.percentile(np.abs(f), 98)
    ax.contourf(xx, yy, f, levels=np.linspace(-lim, lim, 21), cmap="RdBu_r", alpha=0.25, extend="both")
    ax.contour(xx, yy, f, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1, 1.8, 1])

    # jitter cuma buat tampilan fitur diskrit
    jit = np.random.default_rng(seed).uniform(-0.08, 0.08, len(Z)) if jitter_y else np.zeros(len(Z))
    colors = np.where(y == 1, "#c0392b", "#2c6fbb")
    ax.scatter(Z[:, 0], Z[:, 1] + jit, c=colors, s=9, alpha=0.75, linewidths=0)
    sv_idx = model.estimators_[0].support_
    ax.scatter(Z[sv_idx, 0], Z[sv_idx, 1] + jit[sv_idx], s=38, facecolors="none", edgecolors="k",
               linewidths=0.6)

    est = model.estimators_[0]
    ax.set_title(f"{title}\n#SV={len(est.support_)} (bounded={est.n_bounded_sv_}), "
                 f"margin 2/||w||={model.margin_width():.3f}", fontsize=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["features", "pca"], default="features")
    parser.add_argument("--n-samples", type=int, default=1500)
    args = parser.parse_args()

    train_df = pd.read_csv(ROOT_DIR / "data" / "train.csv", dtype={"id": str})
    train_clean = DataPreprocessor.clean_training_dataframe(train_df)
    y = train_clean["is_canceled"].to_numpy()
    prep = make_preprocessor()
    X = prep.fit_transform(add_date_features(train_clean), y)

    if args.mode == "pca":
        Z, explained = pca_2d(X)
        labels = ("PC1", "PC2")
        info = f"2 komponen PCA (variansi terjelaskan PC1={explained[0]:.1%}, PC2={explained[1]:.1%})"
    else:
        cols = [prep.final_feature_names_.index(c) for c in FEATURE_PAIR]
        Z = X[:, cols]
        labels = tuple(f"{c} (standardized)" for c in FEATURE_PAIR)
        info = f"fitur {FEATURE_PAIR[0]} vs {FEATURE_PAIR[1]}"
    idx = stratified_sample(y, args.n_samples, SEED)
    Z, y = Z[idx], y[idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for ax, (title, params) in zip(axes, KERNEL_SETTINGS):
        plot_kernel(ax, Z, y, title, params, jitter_y=args.mode == "features")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#2c6fbb", label="is_canceled = 0"),
        plt.Line2D([], [], marker="o", ls="", color="#c0392b", label="is_canceled = 1"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="support vector"),
        plt.Line2D([], [], color="k", lw=1.8, label="bidang pemisah f(x)=0"),
        plt.Line2D([], [], color="k", lw=1, ls="--", label="margin f(x)=±1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"SVM from scratch pada {info}; {len(y)} sampel, class_weight=balanced", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / ("svm_decision_boundaries.png" if args.mode == "features"
                         else "svm_decision_boundaries_pca.png")
    fig.savefig(out, dpi=150)
    print(f"Gambar disimpan: {out.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
