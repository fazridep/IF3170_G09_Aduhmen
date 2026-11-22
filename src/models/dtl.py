"""Decision Tree Learning (CART / Entropy) from scratch - hanya NumPy + pustaka standar.

Mengacu pada salindia kuliah IF3170 (N. U. Maulidevi) serta literatur standar berikut:
 [1] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification and Regression
     Trees (CART)," Wadsworth, 1984.
 [2] N. U. Maulidevi, "IF3270/IF3170 DTL: Basic Algorithm, Continuous-valued Attribute,
     Missing Attribute Values, Overfitting," ITB, 2021/2026.
 [3] J. R. Quinlan, "C4.5: Programs for Machine Learning," Morgan Kaufmann, 1993.
 [4] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD, 2016
     (Section 3.3 & 3.4: Weighted Quantile Sketch & Sparsity-aware Split Finding).
 [5] GeeksforGeeks, "CART (Classification And Regression Tree) in Machine Learning,"
     https://www.geeksforgeeks.org/machine-learning/cart-classification-and-regression-tree-in-machine-learning/
"""
from __future__ import annotations

import copy
import os
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _impurity(p1: np.ndarray | float, w: np.ndarray | float, criterion: str):
    """Menghitung Gini Impurity (2*p*(1-p)) atau Entropy biner [1][2][5]."""
    p = p1 / np.maximum(w, 1e-12)
    if criterion == "gini":
        return 2.0 * p * (1.0 - p)
    q = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q))


class DecisionTree:
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        n_bins: int = 64,
        categorical_features: Sequence[int] = (),
        class_weight: Optional[Dict[int, float]] = None,
        null_handling: str = "learned",
        random_state: int = 42,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.n_bins = n_bins
        self.categorical_features = tuple(categorical_features)
        self.class_weight = class_weight
        self.null_handling = null_handling  # 'learned' | 'c45'
        self.random_state = random_state

    # ------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        w = np.ones(len(y), dtype=float)
        if self.class_weight:
            w = np.where(y == 1, self.class_weight[1], self.class_weight[0]).astype(float)
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_ = X.shape[1]
        self.total_w_ = float(w.sum())
        self.nodes_: list[Dict[str, Any]] = []
        self._grow(X, y, w, np.arange(len(y)), 0)
        self.n_nodes_before_prune_ = len(self.nodes_)
        if self.ccp_alpha > 0:
            self._prune(self.ccp_alpha)
        return self

    def _make_node(self, p1: float, w: float, n: int) -> int:
        self.nodes_.append(
            dict(
                leaf=True,
                feat=-1,
                thr=0.0,
                cats=None,
                left=-1,
                right=-1,
                nan_left=True,
                wl_frac=0.5,
                p=p1 / max(w, 1e-12),
                w=w,
                n=n,
                gain=0.0,
            )
        )
        return len(self.nodes_) - 1

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, idx: np.ndarray
    ) -> Tuple[Optional[Tuple[Any, ...]], float]:
        yy, ww = y[idx], w[idx]
        tot = float(ww.sum())
        p_all = float((ww * yy).sum())
        best = None
        best_gain = 0.0
        for f in range(X.shape[1]):
            x = X[idx, f]
            ok = ~np.isnan(x)
            n_ok = int(ok.sum())
            if n_ok < 2 * self.min_samples_leaf:
                continue
            xv, wv, yv = x[ok], ww[ok], yy[ok]
            is_cat = f in self.categorical_features
            if is_cat:
                # Sumber: Breiman et al. (1984) Ch. 4.2.2 [1] — Optimal Binary Subset Split untuk target biner.
                # Cara kerja: kategori diurutkan menurut proporsi kelas positif P(y=1|x=c), sehingga
                # pencarian subset biner terbaik x in S cukup mengevaluasi prefix terurut O(K log K).
                cats = np.unique(xv)
                if len(cats) < 2:
                    continue
                code = np.searchsorted(cats, xv)
                Wc = np.bincount(code, wv, len(cats))
                Pc = np.bincount(code, wv * yv, len(cats))
                order = np.argsort(Pc / np.maximum(Wc, 1e-12), kind="stable")
                rank = np.empty(len(cats), dtype=int)
                rank[order] = np.arange(len(cats))
                b = rank[code]
                nb = len(cats)
            else:
                # Sumber: Quantile/Histogram Candidate Split (terinspirasi Chen & Guestrin, KDD 2016 [4]).
                # Cara kerja: kandidat ambang batas numerik diambil dari kuantil `n_bins`, lalu agregasi
                # bobot kiri/kanan dihitung sekaligus memakai `np.bincount` + `np.cumsum`.
                edges = np.unique(np.quantile(xv, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                if len(edges) == 0:
                    continue
                b = np.searchsorted(edges, xv, side="left")
                nb = len(edges) + 1
            W = np.bincount(b, wv, nb)
            P = np.bincount(b, wv * yv, nb)
            N = np.bincount(b, minlength=nb)
            Wl, Pl, Nl = np.cumsum(W)[:-1], np.cumsum(P)[:-1], np.cumsum(N)[:-1]
            Wt, Pt = W.sum(), P.sum()
            Wr, Pr, Nr = Wt - Wl, Pt - Pl, n_ok - Nl
            if self.null_handling == "c45":
                # Sumber: Quinlan, C4.5 (1993) [3] & Slide IF3170 Missing Attribute Values [2].
                # Cara kerja: impurity gain dihitung pada sampel non-null lalu dikali fraksi non-null (Wt/tot).
                valid = (Nl >= self.min_samples_leaf) & (Nr >= self.min_samples_leaf)
                g = (
                    _impurity(Pt, Wt, self.criterion) * Wt
                    - _impurity(Pl, Wl, self.criterion) * Wl
                    - _impurity(Pr, Wr, self.criterion) * Wr
                ) / tot
                variants = [(valid, g, Wl >= Wr, Wl / Wt)]
            else:
                # Sumber: Sparsity-aware Default Direction (Chen & Guestrin, XGBoost, KDD 2016 [4]).
                # Cara kerja: baris bernilai NaN dicoba diarahkan ke cabang kiri maupun kanan, lalu dipilih
                # arah default yang menghasilkan impurity gain terbesar untuk dipakai saat training & prediksi.
                Wn, Pn, Nn = tot - Wt, p_all - Pt, len(idx) - n_ok
                parent = _impurity(p_all, tot, self.criterion) * tot
                variants = []
                for nl in (True, False):
                    a_w, a_p, a_n = (Wl + Wn, Pl + Pn, Nl + Nn) if nl else (Wl, Pl, Nl)
                    b_w, b_p, b_n = (Wr, Pr, Nr) if nl else (Wr + Wn, Pr + Pn, Nr + Nn)
                    valid = (a_n >= self.min_samples_leaf) & (b_n >= self.min_samples_leaf)
                    g = (
                        parent
                        - _impurity(a_p, a_w, self.criterion) * a_w
                        - _impurity(b_p, b_w, self.criterion) * b_w
                    ) / tot
                    variants.append((valid, g, nl, 1.0 if nl else 0.0))
            for valid, g, nl, frac in variants:
                if not valid.any():
                    continue
                gain = np.where(valid, g, -1.0)
                j = int(np.argmax(gain))
                if gain[j] > best_gain:
                    best_gain = float(gain[j])
                    nlj = bool(nl[j]) if hasattr(nl, "__len__") else bool(nl)
                    frj = float(frac[j]) if hasattr(frac, "__len__") else float(frac)
                    thr = None if is_cat else float(edges[j])
                    lc = cats[order[: j + 1]] if is_cat else None
                    best = (f, thr, lc, nlj, frj)
        return best, best_gain

    def _grow(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, idx: np.ndarray, depth: int) -> int:
        wt = float(w[idx].sum())
        p1 = float((w[idx] * y[idx]).sum())
        nid = self._make_node(p1, wt, len(idx))
        if depth >= self.max_depth or len(idx) < self.min_samples_split or p1 <= 0 or p1 >= wt:
            return nid
        best, gain = self._best_split(X, y, w, idx)
        if best is None or gain * wt / self.total_w_ < self.min_impurity_decrease:
            return nid
        f, thr, cats, nan_left, frac = best
        x = X[idx, f]
        nan = np.isnan(x)
        go_left = np.isin(x, cats) if cats is not None else (x <= thr)
        go_left = np.where(nan, nan_left, go_left)
        node = self.nodes_[nid]
        node.update(
            leaf=False,
            feat=f,
            thr=thr,
            cats=cats,
            nan_left=bool(nan_left),
            wl_frac=float(frac),
            gain=float(gain),
        )
        node["left"] = self._grow(X, y, w, idx[go_left], depth + 1)
        node["right"] = self._grow(X, y, w, idx[~go_left], depth + 1)
        return nid

    # ------------------------------------------------ cost-complexity pruning
    def _risk(self, nid: int) -> float:
        nd = self.nodes_[nid]
        return float(nd["w"] * min(nd["p"], 1.0 - nd["p"]))

    def _sub(self, nid: int) -> Tuple[float, int]:
        nd = self.nodes_[nid]
        if nd["leaf"]:
            return self._risk(nid), 1
        a, la = self._sub(nd["left"])
        b, lb = self._sub(nd["right"])
        return a + b, la + lb

    def _prune(self, alpha: float) -> None:
        """Minimal Cost-Complexity Pruning (Bonus Optimasi DTL).

        Sumber: Breiman et al., CART (1984) Ch. 3 [1] (Weakest-Link Pruning).
        Cara kerja: secara iteratif menghitung risiko sub-pohon R(T_t) dan jumlah daun |leaves(T_t)|
        secara bottom-up dynamic programming O(N), lalu memangkas node internal dengan nilai
        g(t) = (R(t) - R(T_t)) / (total_w * (|leaves(T_t)| - 1)) terkecil selama g(t) <= alpha.
        """
        n = len(self.nodes_)
        node_risk = np.array([nd["w"] * min(nd["p"], 1.0 - nd["p"]) for nd in self.nodes_], dtype=float)
        sub_risk = np.zeros(n, dtype=float)
        sub_leaves = np.zeros(n, dtype=int)
        while True:
            stack, order = [0], []
            while stack:
                i = stack.pop()
                order.append(i)
                nd = self.nodes_[i]
                if not nd["leaf"]:
                    stack.append(nd["left"])
                    stack.append(nd["right"])
            for i in reversed(order):
                nd = self.nodes_[i]
                if nd["leaf"]:
                    sub_risk[i], sub_leaves[i] = node_risk[i], 1
                else:
                    l, r = nd["left"], nd["right"]
                    sub_risk[i] = sub_risk[l] + sub_risk[r]
                    sub_leaves[i] = sub_leaves[l] + sub_leaves[r]
            best_g, best_id = np.inf, None
            for i in order:
                if not self.nodes_[i]["leaf"]:
                    g = (node_risk[i] - sub_risk[i]) / self.total_w_ / (sub_leaves[i] - 1)
                    if g < best_g:
                        best_g, best_id = g, i
            if best_id is None or best_g > alpha:
                return
            self.nodes_[best_id]["leaf"] = True

    def cost_complexity_path(self) -> np.ndarray:
        """Daftar alpha efektif untuk analisis cost-complexity pruning [1]."""
        t = copy.deepcopy(self)
        alphas = []
        while not t.nodes_[0]["leaf"]:
            best_g, best_id, stack = np.inf, None, [0]
            while stack:
                nid = stack.pop()
                nd = t.nodes_[nid]
                if nd["leaf"]:
                    continue
                r, lv = t._sub(nid)
                g = (t._risk(nid) - r) / t.total_w_ / (lv - 1)
                if g < best_g:
                    best_g, best_id = g, nid
                stack += [nd["left"], nd["right"]]
            alphas.append(best_g)
            t.nodes_[best_id]["leaf"] = True
        return np.array(alphas, dtype=float)

    # ------------------------------------------------------------ predict
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Vektor probabilitas kelas positif P(y=1 | x) berdimensi (n_samples,)."""
        X = np.asarray(X, dtype=float)
        out = np.zeros(len(X), dtype=float)
        stack = [(0, np.arange(len(X)), np.ones(len(X), dtype=float))]
        while stack:
            nid, idx, wt = stack.pop()
            if len(idx) == 0:
                continue
            nd = self.nodes_[nid]
            if nd["leaf"]:
                out[idx] += wt * nd["p"]
                continue
            x = X[idx, nd["feat"]]
            nan = np.isnan(x)
            left = np.isin(x, nd["cats"]) if nd["cats"] is not None else (x <= nd["thr"])
            wl = np.where(nan, nd["wl_frac"], left.astype(float))
            wr = np.where(nan, 1.0 - nd["wl_frac"], (~left).astype(float))
            ml, mr = wl > 0, wr > 0
            stack.append((nd["left"], idx[ml], (wt * wl)[ml]))
            stack.append((nd["right"], idx[mr], (wt * wr)[mr]))
        return out

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_depth(self) -> int:
        def d(i: int) -> int:
            nd = self.nodes_[i]
            return 0 if nd["leaf"] else 1 + max(d(nd["left"]), d(nd["right"]))

        return d(0)

    def count_leaves(self) -> int:
        return self._sub(0)[1]

    # ------------------------------------------------------- save / load
    def save(self, filepath: str) -> None:
        """Menyimpan model DecisionTree ke file .pkl (menggunakan src.utils.model_io)."""
        for mod_name in ("src.utils.model_io", "utils.model_io"):
            try:
                mod = __import__(mod_name, fromlist=["save_model"])
                if hasattr(mod, "save_model") and callable(mod.save_model):
                    mod.save_model(self, filepath)
                    return
            except Exception:
                continue
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> "DecisionTree":
        """Memuat kembali model DecisionTree dari file .pkl."""
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
        if not isinstance(obj, cls):
            raise TypeError(f"Objek pada {filepath} bukan {cls.__name__}")
        return obj

    # ---------------------------------------------------------- visual
    def plot_tree(
        self,
        top_n: int = 3,
        filepath: str = "output/figures/dtl_tree.png",
        feature_names: Optional[Sequence[str]] = None,
        show: bool = False,
    ) -> str:
        """Gambar percabangan tree sampai top_n tingkat teratas (Bonus Visualisasi DTL)."""
        import matplotlib.pyplot as plt

        pos: Dict[int, Tuple[float, float]] = {}
        labels: Dict[int, str] = {}
        counter = [0]

        def layout(nid: int, depth: int) -> float:
            nd = self.nodes_[nid]
            stop = nd["leaf"] or depth >= top_n
            if stop:
                x = float(counter[0])
                counter[0] += 1
                labels[nid] = f"P(cancel)={nd['p']:.2f}\nn={nd['n']}"
            else:
                xl = layout(nd["left"], depth + 1)
                xr = layout(nd["right"], depth + 1)
                x = (xl + xr) / 2.0
                name = feature_names[nd["feat"]] if feature_names else f"X{nd['feat']}"
                cond = (
                    f"{name} in {len(nd['cats'])} kategori"
                    if nd["cats"] is not None
                    else f"{name} <= {nd['thr']:.4g}"
                )
                labels[nid] = f"{cond}\nn={nd['n']}  p={nd['p']:.2f}"
            pos[nid] = (x, float(-depth))
            return x

        layout(0, 0)
        fig, ax = plt.subplots(figsize=(max(8.5, 2.6 * counter[0]), 2.4 * (top_n + 1)))
        for nid, (x, yy) in pos.items():
            nd = self.nodes_[nid]
            if not nd["leaf"] and -yy < top_n:
                for c, tag in ((nd["left"], "ya"), (nd["right"], "tidak")):
                    ax.plot([x, pos[c][0]], [yy, pos[c][1]], color="gray", zorder=0)
                    ax.text((x + pos[c][0]) / 2.0, (yy + pos[c][1]) / 2.0, tag, fontsize=7.5, color="gray")
        for nid, (x, yy) in pos.items():
            leafish = self.nodes_[nid]["leaf"] or -yy >= top_n
            col = plt.cm.RdYlGn_r(self.nodes_[nid]["p"])
            ax.text(
                x,
                yy,
                labels[nid],
                ha="center",
                va="center",
                fontsize=7.5,
                bbox=dict(boxstyle="round", fc=col if leafish else "white", ec="black", alpha=0.88),
            )
        ax.axis("off")
        ax.set_title(f"Decision Tree Learning (CART) - {top_n} Tingkat Teratas", fontsize=12, fontweight="bold")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return filepath
