from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRIMARY_COLOR = "#7a3db8"
SECONDARY_COLOR = "#2d8a8a"
ACCENT_ORANGE = "#d97706"
ACCENT_SLATE = "#4b5563"
ACCENT_RED = "#b91c1c"
SERIES_COLORS = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_ORANGE, ACCENT_SLATE, "#2563eb", ACCENT_RED]

ArrayLike = Union[np.ndarray, Sequence[Any], pd.Series]


def _to_1d_array(arr: Iterable[Any]) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim != 1:
        out = out.ravel()
    return out


def _extract_positive_scores(y_score: Union[np.ndarray, Sequence[float]], pos_index: int = 1) -> np.ndarray:
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim == 2:
        if scores.shape[1] == 1:
            return scores[:, 0]
        return scores[:, pos_index]
    return scores.ravel()


def confusion_matrix(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    labels: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    yt = _to_1d_array(y_true)
    yp = _to_1d_array(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"Mismatched lengths: y_true={yt.shape[0]}, y_pred={yp.shape[0]}")

    if labels is None:
        unique_vals = np.unique(np.concatenate([yt, yp]))
        label_list = [0, 1] if set(unique_vals.tolist()).issubset({0, 1}) else unique_vals.tolist()
    else:
        label_list = list(labels)

    n_labels = len(label_list)
    t_idx = np.full(yt.shape[0], -1, dtype=np.int64)
    p_idx = np.full(yp.shape[0], -1, dtype=np.int64)
    for k, lbl in enumerate(label_list):
        t_idx[yt == lbl] = k
        p_idx[yp == lbl] = k
    valid = (t_idx >= 0) & (p_idx >= 0)
    flat = np.bincount(t_idx[valid] * n_labels + p_idx[valid], minlength=n_labels * n_labels)
    return flat.reshape(n_labels, n_labels).astype(int)


def accuracy_score(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    yt = _to_1d_array(y_true)
    yp = _to_1d_array(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"Mismatched lengths: y_true={yt.shape[0]}, y_pred={yp.shape[0]}")
    if yt.shape[0] == 0:
        return 0.0
    return float(np.mean(yt == yp))


def _per_class_prf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[Any],
    zero_division: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    support = cm.sum(axis=1).astype(float)
    prec = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1.0), zero_division)
    rec = np.where((tp + fn) > 0, tp / np.maximum(tp + fn, 1.0), zero_division)
    denom = prec + rec
    f1 = np.where(denom > 0, 2.0 * prec * rec / np.maximum(denom, 1e-15), zero_division)
    return prec, rec, f1, support


def _resolve_labels(yt: np.ndarray, yp: np.ndarray, pos_label: Any) -> List[Any]:
    unique_vals = np.unique(np.concatenate([yt, yp])).tolist()
    if set(unique_vals).issubset({0, 1}):
        return [0, 1]
    if pos_label not in unique_vals:
        unique_vals.append(pos_label)
    return unique_vals


def _average(values: np.ndarray, support: np.ndarray, labels: List[Any], average: str, pos_label: Any) -> float:
    if average == "binary":
        return float(values[labels.index(pos_label)])
    if average == "macro":
        return float(np.mean(values))
    if average == "weighted":
        total = support.sum()
        return float(np.sum(values * support) / total) if total > 0 else 0.0
    raise ValueError(f"Unsupported average={average!r}. Use 'binary', 'macro', or 'weighted'.")


def precision_score(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    average: str = "binary",
    pos_label: Any = 1,
    zero_division: float = 0.0,
) -> float:
    yt, yp = _to_1d_array(y_true), _to_1d_array(y_pred)
    labels = _resolve_labels(yt, yp, pos_label)
    prec, _, _, support = _per_class_prf(yt, yp, labels, zero_division=zero_division)
    return _average(prec, support, labels, average, pos_label)


def recall_score(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    average: str = "binary",
    pos_label: Any = 1,
    zero_division: float = 0.0,
) -> float:
    yt, yp = _to_1d_array(y_true), _to_1d_array(y_pred)
    labels = _resolve_labels(yt, yp, pos_label)
    _, rec, _, support = _per_class_prf(yt, yp, labels, zero_division=zero_division)
    return _average(rec, support, labels, average, pos_label)


def f1_score(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    average: str = "binary",
    pos_label: Any = 1,
    zero_division: float = 0.0,
) -> float:
    yt, yp = _to_1d_array(y_true), _to_1d_array(y_pred)
    labels = _resolve_labels(yt, yp, pos_label)
    _, _, f1, support = _per_class_prf(yt, yp, labels, zero_division=zero_division)
    return _average(f1, support, labels, average, pos_label)


def roc_auc_score(
    y_true: Iterable[Any],
    y_score: Union[np.ndarray, Sequence[float]],
    pos_label: Any = 1,
) -> float:
    yt = _to_1d_array(y_true)
    scores = _extract_positive_scores(y_score)
    if yt.shape[0] != scores.shape[0]:
        raise ValueError(f"Mismatched lengths: y_true={yt.shape[0]}, y_score={scores.shape[0]}")
    y_bin = (yt == pos_label).astype(int)
    n_pos = int(np.sum(y_bin == 1))
    n_neg = int(np.sum(y_bin == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    _, first_idx, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    avg_ranks = first_idx + (counts + 1) / 2.0
    ranks = np.repeat(avg_ranks, counts)
    pos_rank_sum = float(np.sum(ranks[y_bin[order] == 1]))
    return float((pos_rank_sum - 0.5 * n_pos * (n_pos + 1)) / (n_pos * n_neg))


def roc_curve(
    y_true: Iterable[Any],
    y_score: Union[np.ndarray, Sequence[float]],
    pos_label: Any = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = (_to_1d_array(y_true) == pos_label).astype(int)
    scores = _extract_positive_scores(y_score)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted, y_sorted = scores[order], yt[order]
    distinct = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[distinct]
    fps = (distinct + 1) - tps
    n_pos, n_neg = max(int(yt.sum()), 1), max(int(len(yt) - yt.sum()), 1)
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[np.inf, s_sorted[distinct]]
    return fpr, tpr, thresholds


def log_loss(
    y_true: Iterable[Any],
    y_proba: Union[np.ndarray, Sequence[float]],
    eps: Optional[float] = None,
    pos_label: Any = 1,
) -> float:
    eps = float(np.finfo(np.float64).eps) if eps is None else float(eps)
    yt = (_to_1d_array(y_true) == pos_label).astype(float)
    probs = np.clip(_extract_positive_scores(y_proba), eps, 1.0 - eps)
    return float(-np.mean(yt * np.log(probs) + (1.0 - yt) * np.log(1.0 - probs)))


def compute_classification_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    y_proba: Optional[Union[np.ndarray, Sequence[float]]] = None,
    pos_label: Any = 1,
) -> Dict[str, Any]:
    yt = _to_1d_array(y_true)
    yp = _to_1d_array(y_pred)
    other_labels = [lbl for lbl in np.unique(np.concatenate([yt, yp])) if lbl != pos_label]
    neg_label = 0 if pos_label == 1 else (other_labels[0] if other_labels else (1 if pos_label == 0 else 0))
    cm = confusion_matrix(yt, yp, labels=[neg_label, pos_label])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    return {
        "accuracy": accuracy_score(yt, yp),
        "precision_binary": precision_score(yt, yp, average="binary", pos_label=pos_label),
        "recall_binary": recall_score(yt, yp, average="binary", pos_label=pos_label),
        "f1_binary": f1_score(yt, yp, average="binary", pos_label=pos_label),
        "precision_macro": precision_score(yt, yp, average="macro", pos_label=pos_label),
        "recall_macro": recall_score(yt, yp, average="macro", pos_label=pos_label),
        "f1_macro": f1_score(yt, yp, average="macro", pos_label=pos_label),
        "roc_auc": roc_auc_score(yt, y_proba, pos_label=pos_label) if y_proba is not None else float("nan"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_model(
    model_name: str,
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    y_proba: Optional[Union[np.ndarray, Sequence[float]]] = None,
    fit_time: Optional[float] = None,
    predict_time: Optional[float] = None,
    implementation: str = "from_scratch",
    notes: str = "",
) -> Dict[str, Any]:
    metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)
    fit_t = float(fit_time) if fit_time is not None else float("nan")
    pred_t = float(predict_time) if predict_time is not None else float("nan")
    total_t = fit_t + pred_t if not (np.isnan(fit_t) or np.isnan(pred_t)) else float("nan")
    record: Dict[str, Any] = {
        "model": model_name,
        "implementation": implementation,
        **metrics,
        "fit_time_s": fit_t,
        "predict_time_s": pred_t,
        "total_time_s": total_t,
    }
    if notes:
        record["notes"] = notes
    return record


def _positive_scores_of(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        try:
            return _extract_positive_scores(model.predict_proba(X))
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).ravel()
    return None


def benchmark_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    implementation: str = "from_scratch",
    notes: str = "",
    fit: bool = True,
) -> Dict[str, Any]:
    fit_t = float("nan")
    if fit:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_t = time.perf_counter() - t0
    t1 = time.perf_counter()
    y_pred = np.asarray(model.predict(X_eval)).ravel()
    y_score = _positive_scores_of(model, X_eval)
    pred_t = time.perf_counter() - t1
    record = evaluate_model(
        model_name, y_eval, y_pred, y_proba=y_score, fit_time=fit_t, predict_time=pred_t,
        implementation=implementation, notes=notes,
    )
    return {"record": record, "model": model, "y_pred": y_pred, "y_score": y_score}


def summarize_models(
    results: Sequence[Dict[str, Any]],
    sort_by: Optional[str] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    df = pd.DataFrame(list(results))
    if df.empty:
        return df
    preferred_order = [
        "model", "implementation", "accuracy", "precision_binary", "recall_binary", "f1_binary",
        "precision_macro", "recall_macro", "f1_macro", "roc_auc", "fit_time_s", "predict_time_s",
        "total_time_s", "tn", "fp", "fn", "tp",
    ]
    ordered = [c for c in preferred_order if c in df.columns]
    df = df[ordered + [c for c in df.columns if c not in ordered]]
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending, kind="mergesort").reset_index(drop=True)
    return df


def compare_scratch_vs_sklearn(
    results: Sequence[Dict[str, Any]],
    metrics: Sequence[str] = ("accuracy", "f1_binary", "f1_macro", "roc_auc"),
) -> pd.DataFrame:
    df = summarize_models(results)
    if df.empty or "implementation" not in df.columns:
        return pd.DataFrame()
    scratch = df[df["implementation"] == "from_scratch"].drop_duplicates(subset=["model"], keep="last").set_index("model")
    sklearn = df[df["implementation"] == "scikit-learn"].drop_duplicates(subset=["model"], keep="last").set_index("model")
    rows: List[Dict[str, Any]] = []
    for model_name in [m for m in scratch.index if m in sklearn.index]:
        row: Dict[str, Any] = {"model": model_name}
        for m in metrics:
            if m in df.columns:
                s_val, k_val = float(scratch.loc[model_name, m]), float(sklearn.loc[model_name, m])
                row[f"{m}_scratch"], row[f"{m}_sklearn"], row[f"{m}_diff"] = s_val, k_val, s_val - k_val
        if "fit_time_s" in df.columns:
            t_s, t_k = float(scratch.loc[model_name, "fit_time_s"]), float(sklearn.loc[model_name, "fit_time_s"])
            row["fit_time_scratch_s"], row["fit_time_sklearn_s"] = t_s, t_k
            row["fit_time_ratio"] = t_s / t_k if t_k > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_agreement(y_pred_a: Iterable[Any], y_pred_b: Iterable[Any]) -> float:
    a, b = _to_1d_array(y_pred_a), _to_1d_array(y_pred_b)
    return float(np.mean(a == b)) if a.shape[0] else float("nan")


def theoretical_cost_threshold(fp_cost: float = 2.0, fn_cost: float = 1.0) -> float:
    if fp_cost < 0 or fn_cost < 0 or (fp_cost + fn_cost) == 0:
        raise ValueError("Costs must be non-negative with positive sum.")
    return float(fp_cost / (fp_cost + fn_cost))


def compute_expected_cost(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    fp_cost: float = 2.0,
    fn_cost: float = 1.0,
    pos_label: Any = 1,
) -> Dict[str, float]:
    yt, yp = _to_1d_array(y_true), _to_1d_array(y_pred)
    cm = confusion_matrix(yt, yp, labels=[0 if pos_label == 1 else 1 - pos_label, pos_label])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    total_cost = float(fp_cost * fp + fn_cost * fn)
    n = len(yt)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "total_cost": total_cost, "avg_cost": total_cost / n if n > 0 else 0.0}


def find_optimal_cost_threshold(
    y_true: Iterable[Any],
    y_proba: Union[np.ndarray, Sequence[float]],
    fp_cost: float = 2.0,
    fn_cost: float = 1.0,
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, pd.DataFrame]:
    yt = _to_1d_array(y_true)
    probs = _extract_positive_scores(y_proba)
    tau_star = theoretical_cost_threshold(fp_cost=fp_cost, fn_cost=fn_cost)
    grid = (
        np.unique(np.concatenate([np.linspace(0.05, 0.95, 181), [0.5, tau_star]]))
        if thresholds is None else np.asarray(list(thresholds), dtype=float)
    )
    rows: List[Dict[str, Any]] = []
    for t in grid:
        yp = (probs >= t).astype(int)
        rows.append({
            "threshold": float(t),
            **compute_expected_cost(yt, yp, fp_cost=fp_cost, fn_cost=fn_cost),
            "accuracy": accuracy_score(yt, yp),
            "precision_binary": precision_score(yt, yp, average="binary"),
            "recall_binary": recall_score(yt, yp, average="binary"),
            "f1_binary": f1_score(yt, yp, average="binary"),
            "f1_macro": f1_score(yt, yp, average="macro"),
        })
    curve_df = pd.DataFrame(rows)
    tied = curve_df[curve_df["total_cost"] == curve_df["total_cost"].min()].copy()
    tied["_dist"] = (tied["threshold"] - tau_star).abs()
    tied = tied.sort_values(["f1_macro", "_dist"], ascending=[False, True], kind="mergesort")
    best_threshold = float(tied.iloc[0]["threshold"])
    return best_threshold, curve_df


def evaluate_cost_sensitive(
    y_true: Iterable[Any],
    y_proba: Union[np.ndarray, Sequence[float]],
    fp_cost: float = 2.0,
    fn_cost: float = 1.0,
    model_name: str = "Model",
    tuned_threshold: Optional[float] = None,
    baseline_pred: Optional[Iterable[Any]] = None,
    extra_strategies: Optional[Dict[str, Iterable[Any]]] = None,
) -> Dict[str, Any]:
    yt = _to_1d_array(y_true)
    probs = _extract_positive_scores(y_proba)
    tau_star = theoretical_cost_threshold(fp_cost=fp_cost, fn_cost=fn_cost)
    searched_t, curve_df = find_optimal_cost_threshold(yt, probs, fp_cost=fp_cost, fn_cost=fn_cost)
    if tuned_threshold is None:
        best_t, tuned_label = searched_t, f"Empirical min cost on this set (t={searched_t:.3f})"
    else:
        best_t, tuned_label = float(tuned_threshold), f"Tuned threshold (t={float(tuned_threshold):.3f})"

    if baseline_pred is None:
        base_label, base_pred = "Baseline (t=0.500)", (probs >= 0.5).astype(int)
    else:
        base_label, base_pred = "Baseline (default rule)", _to_1d_array(baseline_pred).astype(int)

    strategies: List[Tuple[str, float, np.ndarray]] = [
        (base_label, 0.5, base_pred),
        (f"Bayes threshold (tau*={tau_star:.3f})", tau_star, (probs >= tau_star).astype(int)),
        (tuned_label, best_t, (probs >= best_t).astype(int)),
    ]
    for label, preds in (extra_strategies or {}).items():
        strategies.append((label, float("nan"), _to_1d_array(preds).astype(int)))

    summary_rows: List[Dict[str, Any]] = []
    preds_by_strategy: Dict[str, np.ndarray] = {}
    for label, thr, yp in strategies:
        preds_by_strategy[label] = yp
        cost_info = compute_expected_cost(yt, yp, fp_cost=fp_cost, fn_cost=fn_cost)
        summary_rows.append({
            "model": model_name,
            "strategy": label,
            "threshold": thr,
            "total_cost": cost_info["total_cost"],
            "avg_cost": cost_info["avg_cost"],
            **compute_classification_metrics(yt, yp, y_proba=probs),
        })
    summary_df = pd.DataFrame(summary_rows)
    base_cost = float(summary_df.loc[0, "total_cost"])
    summary_df["cost_reduction_pct"] = (
        (base_cost - summary_df["total_cost"]) / base_cost * 100.0 if base_cost > 0 else 0.0
    )
    theo_cost = float(summary_df.loc[1, "total_cost"])
    emp_cost = float(summary_df.loc[2, "total_cost"])
    best_row = summary_df.loc[int(summary_df["total_cost"].idxmin())]
    return {
        "model_name": model_name,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "tau_star": tau_star,
        "best_threshold": best_t,
        "baseline_cost": base_cost,
        "theoretical_cost": theo_cost,
        "empirical_cost": emp_cost,
        "cost_reduction_theoretical_pct": ((base_cost - theo_cost) / base_cost * 100.0) if base_cost > 0 else 0.0,
        "cost_reduction_empirical_pct": ((base_cost - emp_cost) / base_cost * 100.0) if base_cost > 0 else 0.0,
        "best_strategy": str(best_row["strategy"]),
        "summary_df": summary_df,
        "curve_df": curve_df,
        "y_true": yt,
        "preds_by_strategy": preds_by_strategy,
    }


def _finalize_figure(fig: plt.Figure, save_path: Optional[Union[str, Path]], show: bool) -> plt.Figure:
    fig.tight_layout()
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_metric_comparison(
    summary_df: pd.DataFrame,
    metrics: Sequence[str] = ("accuracy", "precision_macro", "recall_macro", "f1_macro"),
    title: str = "From Scratch vs scikit-learn Metrics",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    available = [m for m in metrics if m in summary_df.columns]
    if not available:
        raise ValueError("None of the requested metric columns exist in summary_df.")
    ncols = 2 if len(available) > 1 else 1
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    models = list(dict.fromkeys(summary_df["model"].tolist()))
    implementations = list(dict.fromkeys(summary_df["implementation"].tolist()))
    x = np.arange(len(models))
    width = 0.72 / max(len(implementations), 1)

    for idx, metric in enumerate(available):
        ax = axes[idx // ncols][idx % ncols]
        for j, impl in enumerate(implementations):
            sub = summary_df[summary_df["implementation"] == impl].set_index("model")
            vals = [float(sub.loc[m, metric]) if m in sub.index else np.nan for m in models]
            offset = (j - (len(implementations) - 1) / 2.0) * width
            bars = ax.bar(x + offset, vals, width=width * 0.92, label=impl, color=SERIES_COLORS[j % len(SERIES_COLORS)], alpha=0.9)
            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0.0, 1.15)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=max(1, len(implementations)))
    for idx in range(len(available), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    return _finalize_figure(fig, save_path, show)


def plot_confusion_matrix(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    labels: Sequence[Any] = (0, 1),
    class_names: Sequence[str] = ("Not Canceled (0)", "Canceled (1)"),
    normalize: bool = False,
    title: str = "Confusion Matrix",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        disp = np.where(row_sums > 0, cm / np.maximum(row_sums, 1), 0.0)
    else:
        disp = cm.astype(float)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
    else:
        fig = ax.figure
    im = ax.imshow(disp, interpolation="nearest", cmap="Purples")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    thresh = disp.max() / 2.0 if disp.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:,}\n({disp[i, j]:.1%})" if normalize else f"{cm[i, j]:,}"
            ax.text(j, i, text, ha="center", va="center", color="white" if disp[i, j] > thresh else "#1f2937", fontsize=10, fontweight="bold")
    if standalone:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return _finalize_figure(fig, save_path, show)
    return fig


def plot_confusion_matrices(
    items: Sequence[Tuple[str, Iterable[Any], Iterable[Any]]],
    labels: Sequence[Any] = (0, 1),
    class_names: Sequence[str] = ("Not Canceled (0)", "Canceled (1)"),
    normalize: bool = False,
    title: str = "Confusion Matrix Comparison",
    ncols: int = 3,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    n = len(items)
    if n == 0:
        raise ValueError("items cannot be empty.")
    cols = min(ncols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.3 * rows), squeeze=False)
    for idx, (subtitle, yt, yp) in enumerate(items):
        plot_confusion_matrix(yt, yp, labels=labels, class_names=class_names, normalize=normalize, title=subtitle, ax=axes[idx // cols][idx % cols])
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    return _finalize_figure(fig, save_path, show)


def plot_runtime_comparison(
    summary_df: pd.DataFrame,
    time_col: str = "fit_time_s",
    title: str = "Training Time Comparison (seconds)",
    log_scale: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    if time_col not in summary_df.columns:
        raise ValueError(f"Column {time_col!r} not found in summary_df.")
    fig, ax = plt.subplots(figsize=(max(8.5, 1.6 * summary_df["model"].nunique() + 4), 4.8))
    models = list(dict.fromkeys(summary_df["model"].tolist()))
    implementations = list(dict.fromkeys(summary_df["implementation"].tolist()))
    x = np.arange(len(models))
    width = 0.72 / max(len(implementations), 1)
    for j, impl in enumerate(implementations):
        sub = summary_df[summary_df["implementation"] == impl].set_index("model")
        vals = [float(sub.loc[m, time_col]) if m in sub.index else np.nan for m in models]
        offset = (j - (len(implementations) - 1) / 2.0) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92, label=impl, color=SERIES_COLORS[j % len(SERIES_COLORS)], alpha=0.9)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.2f}s", ha="center", va="bottom", fontsize=8)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Seconds", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    return _finalize_figure(fig, save_path, show)


def plot_roc_curves(
    items: Sequence[Tuple[str, Iterable[Any], Union[np.ndarray, Sequence[float]]]],
    title: str = "ROC Curves (validation set)",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    styles = ["-", "--", "-.", ":"]
    for k, (label, yt, score) in enumerate(items):
        fpr, tpr, _ = roc_curve(yt, score)
        auc = roc_auc_score(yt, score)
        ax.plot(fpr, tpr, lw=1.9, color=SERIES_COLORS[k % len(SERIES_COLORS)], ls=styles[(k // len(SERIES_COLORS)) % len(styles)], label=f"{label} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.0, ls="--", label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.9)
    return _finalize_figure(fig, save_path, show)


def plot_cost_sensitive_analysis(
    cs_result: Dict[str, Any],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    curve_df = cs_result["curve_df"]
    summary_df = cs_result["summary_df"]
    tau_star, best_t = cs_result["tau_star"], cs_result["best_threshold"]
    fp_cost, fn_cost = cs_result["fp_cost"], cs_result["fn_cost"]
    y_true, preds_map = cs_result["y_true"], cs_result["preds_by_strategy"]
    keys = list(preds_map.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    ax0 = axes[0][0]
    ax0.plot(curve_df["threshold"], curve_df["total_cost"], color=PRIMARY_COLOR, lw=2.2, label=f"Total cost = {fp_cost:g}·FP + {fn_cost:g}·FN")
    ax0.plot(curve_df["threshold"], fp_cost * curve_df["fp"], color=ACCENT_RED, lw=1.2, ls="--", alpha=0.8, label=f"{fp_cost:g}·FP part")
    ax0.plot(curve_df["threshold"], fn_cost * curve_df["fn"], color=SECONDARY_COLOR, lw=1.2, ls="--", alpha=0.8, label=f"{fn_cost:g}·FN part")
    ax0.axvline(0.50, color=ACCENT_SLATE, ls="--", lw=1.4, label="t = 0.50")
    ax0.axvline(tau_star, color=SECONDARY_COLOR, ls="-.", lw=1.8, label=f"Bayes tau* = {tau_star:.3f}")
    ax0.axvline(best_t, color=ACCENT_ORANGE, ls=":", lw=2.2, label=f"Tuned t = {best_t:.3f}")
    ax0.set_title("Validation cost vs decision threshold", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Threshold on P(is_canceled = 1)")
    ax0.set_ylabel("Cost")
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="upper center", fontsize=8, frameon=True, framealpha=0.92, ncol=2)

    ax1 = axes[0][1]
    labels = summary_df["strategy"].tolist()
    fp_part = fp_cost * summary_df["fp"].to_numpy(dtype=float)
    fn_part = fn_cost * summary_df["fn"].to_numpy(dtype=float)
    ypos = np.arange(len(labels))[::-1]
    ax1.barh(ypos, fp_part, color=ACCENT_RED, alpha=0.85, label=f"{fp_cost:g}·FP (overbooking risk)")
    ax1.barh(ypos, fn_part, left=fp_part, color=SECONDARY_COLOR, alpha=0.85, label=f"{fn_cost:g}·FN (empty room)")
    for yv, total, red in zip(ypos, fp_part + fn_part, summary_df["cost_reduction_pct"].to_numpy(dtype=float)):
        ax1.text(total, yv, f"  {total:,.0f} ({-red:+.1f}%)" if abs(red) > 1e-9 else f"  {total:,.0f}", va="center", fontsize=9)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("Total cost on validation set")
    ax1.set_xlim(0, float(np.max(fp_part + fn_part)) * 1.3)
    ax1.set_title("Total cost per strategy (lower is better)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.5, frameon=False)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    best_key = cs_result.get("best_strategy", keys[-1])
    base_key = keys[0]
    base_row = summary_df[summary_df["strategy"] == base_key].iloc[0]
    best_row = summary_df[summary_df["strategy"] == best_key].iloc[0]
    plot_confusion_matrix(y_true, preds_map[base_key], title=f"{base_key}\ncost = {base_row['total_cost']:,.0f}", ax=axes[1][0])
    plot_confusion_matrix(
        y_true, preds_map[best_key],
        title=f"{best_key}\ncost = {best_row['total_cost']:,.0f} ({-best_row['cost_reduction_pct']:+.1f}%)",
        ax=axes[1][1],
    )
    fig.suptitle(
        title or f"Cost-sensitive classification: {cs_result['model_name']} (C_FP = {fp_cost:g}, C_FN = {fn_cost:g})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    return _finalize_figure(fig, save_path, show)


def plot_svm_2d_kernels(
    X_2d: np.ndarray,
    y: np.ndarray,
    models: Dict[str, Any],
    feature_names: Tuple[str, str] = ("Component 1", "Component 2"),
    title: str = "SVM Decision Boundary, Margin & Support Vectors (2D)",
    grid_steps: int = 300,
    max_scatter: int = 1500,
    random_state: int = 42,
    axis_unscale: Optional[Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    X_arr = np.asarray(X_2d, dtype=float)
    y_arr = _to_1d_array(y)
    if X_arr.shape[1] != 2:
        raise ValueError(f"X_2d must have exactly 2 columns, got shape {X_arr.shape}")
    lo = np.percentile(X_arr, 1.0, axis=0)
    hi = np.percentile(X_arr, 99.0, axis=0)
    pad = 0.12 * (hi - lo + 1e-6)
    x_min, y_min = lo - pad
    x_max, y_max = hi + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_steps), np.linspace(y_min, y_max, grid_steps))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    rng = np.random.default_rng(random_state)
    scatter_idx = rng.choice(len(X_arr), size=max_scatter, replace=False) if len(X_arr) > max_scatter else np.arange(len(X_arr))

    def _unscaled_ticks(ax_obj: plt.Axes, axis: str, params: Optional[Tuple[float, float]], vmin: float, vmax: float) -> None:
        if params is None:
            return
        mean, std = float(params[0]), float(params[1]) or 1.0
        orig = MaxNLocator(nbins=6, integer=True).tick_values(mean + std * vmin, mean + std * vmax)
        orig = [v for v in orig if mean + std * vmin <= v <= mean + std * vmax]
        pos = [(v - mean) / std for v in orig]
        labels = [f"{v:,.0f}" for v in orig]
        if axis == "x":
            ax_obj.set_xticks(pos)
            ax_obj.set_xticklabels(labels)
        else:
            ax_obj.set_yticks(pos)
            ax_obj.set_yticklabels(labels)

    items = list(models.items())
    fig, axes = plt.subplots(1, len(items), figsize=(5.6 * len(items), 5.3), squeeze=False)
    for idx, (label, model) in enumerate(items):
        ax = axes[0][idx]
        Z = model.decision_function(grid_points).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-1e12, 0.0, 1e12], colors=["#e6f4f4", "#f1e6fb"], alpha=0.8)
        ax.contour(xx, yy, Z, levels=[-1.0, 1.0], colors=[SECONDARY_COLOR, PRIMARY_COLOR], linestyles="--", linewidths=1.3)
        ax.contour(xx, yy, Z, levels=[0.0], colors="#111827", linewidths=2.0)
        xs, ys = X_arr[scatter_idx], y_arr[scatter_idx]
        ax.scatter(xs[ys == 0, 0], xs[ys == 0, 1], c=SECONDARY_COLOR, s=11, alpha=0.55, edgecolors="none")
        ax.scatter(xs[ys == 1, 0], xs[ys == 1, 1], c=PRIMARY_COLOR, s=11, alpha=0.55, edgecolors="none")

        sv = getattr(model, "support_vectors_", None)
        n_sv = 0 if sv is None else len(sv)
        n_free = n_bounded = 0
        if n_sv:
            alphas = getattr(model, "alphas_", None)
            c_sv = getattr(model, "support_C_", None)
            bounded = alphas >= c_sv * (1.0 - 1e-6) if (alphas is not None and c_sv is not None) else np.zeros(n_sv, dtype=bool)
            n_bounded, n_free = int(bounded.sum()), int((~bounded).sum())
            ax.scatter(sv[bounded, 0], sv[bounded, 1], s=24, facecolors="none", edgecolors=ACCENT_SLATE, linewidths=0.5, alpha=0.5)
            ax.scatter(sv[~bounded, 0], sv[~bounded, 1], s=70, facecolors="none", edgecolors=ACCENT_ORANGE, linewidths=1.8)
        acc = accuracy_score(y_arr, model.predict(X_arr))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{label}\ntrain acc = {acc:.3f} | SV = {n_sv} (free {n_free}, bounded {n_bounded})", fontsize=10.5, fontweight="bold")
        ax.set_xlabel(feature_names[0], fontsize=9.5)
        if idx == 0:
            ax.set_ylabel(feature_names[1], fontsize=9.5)
        if axis_unscale is not None:
            _unscaled_ticks(ax, "x", axis_unscale[0], x_min, x_max)
            _unscaled_ticks(ax, "y", axis_unscale[1], y_min, y_max)
        ax.grid(True, linestyle=":", alpha=0.35)

    handles = [
        Line2D([], [], ls="", marker="o", ms=5, color=SECONDARY_COLOR, alpha=0.7, label="Not canceled (0)"),
        Line2D([], [], ls="", marker="o", ms=5, color=PRIMARY_COLOR, alpha=0.7, label="Canceled (1)"),
        Line2D([], [], color="#111827", lw=2.0, label="f(x) = 0 (separating surface)"),
        Line2D([], [], color=ACCENT_SLATE, lw=1.3, ls="--", label="f(x) = ±1 (margin)"),
        Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec=ACCENT_ORANGE, mew=1.8, label="Free SV (0 < alpha < C, on the margin)"),
        Line2D([], [], ls="", marker="o", ms=6, mfc="none", mec=ACCENT_SLATE, mew=0.8, alpha=0.7, label="Bounded SV (alpha = C, slack > 0)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    return _finalize_figure(fig, save_path, show)


def plot_svm_convergence(
    histories: Dict[str, Dict[str, np.ndarray]],
    title: str = "SMO convergence (working-set selection strategies)",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    for k, (label, hist) in enumerate(histories.items()):
        if not hist:
            continue
        color = SERIES_COLORS[k % len(SERIES_COLORS)]
        axes[0].plot(hist["step"], np.maximum(hist["kkt_violation"], 1e-6), color=color, lw=1.7, label=label)
        axes[1].plot(hist["step"], hist["dual_objective"], color=color, lw=1.7, label=label)
    axes[0].set_yscale("log")
    axes[0].axhline(1e-3, color="#9ca3af", ls="--", lw=1.0, label="tol = 1e-3")
    axes[0].set_title("Maximal KKT violation  b_low - b_up", fontsize=11, fontweight="bold")
    axes[1].set_title("Dual objective W(alpha)", fontsize=11, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("SMO pair updates")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=8.5)
    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.03)
    return _finalize_figure(fig, save_path, show)
