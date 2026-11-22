from __future__ import annotations

import os
import pickle
import warnings
from collections import OrderedDict

import numpy as np

KERNELS = ("linear", "polynomial", "gaussian")
_KERNEL_ALIASES = {"linear": "linear", "poly": "polynomial", "polynomial": "polynomial",
                   "rbf": "gaussian", "gaussian": "gaussian"}
_TAU = 1e-12


class Kernel:
    def __init__(self, name="gaussian", gamma=1.0, degree=3, coef0=1.0):
        self.name = _KERNEL_ALIASES[name]
        self.gamma = float(gamma)
        self.degree = int(degree)
        self.coef0 = float(coef0)

    def __call__(self, A, B, sq_A=None, sq_B=None):
        dot = A @ B.T
        if self.name == "linear":
            return dot
        if self.name == "polynomial":
            return (self.gamma * dot + self.coef0) ** self.degree
        if sq_A is None:
            sq_A = np.einsum("ij,ij->i", A, A)
        if sq_B is None:
            sq_B = np.einsum("ij,ij->i", B, B)
        dist_sq = np.maximum(sq_A[:, None] + sq_B[None, :] - 2.0 * dot, 0.0)
        return np.exp(-self.gamma * dist_sq)

    def diagonal(self, A, sq_A):
        if self.name == "linear":
            return sq_A.copy()
        if self.name == "polynomial":
            return (self.gamma * sq_A + self.coef0) ** self.degree
        return np.ones(A.shape[0])


class _KernelRowCache:
    # cache baris kernel (Chang & Lin, LIBSVM, 2011)

    def __init__(self, X, y, kernel, sq_norms, max_mb):
        self.X, self.y, self.kernel, self.sq = X, y, kernel, sq_norms
        self.max_rows = max(2, int(max_mb * 1024 * 1024 // (8 * X.shape[0])))
        self.rows = OrderedDict()

    def get(self, i):
        row = self.rows.get(i)
        if row is not None:
            self.rows.move_to_end(i)
            return row
        k = self.kernel(self.X, self.X[i:i + 1], self.sq, self.sq[i:i + 1]).ravel()
        row = (self.y[i] * self.y) * k
        self.rows[i] = row
        if len(self.rows) > self.max_rows:
            self.rows.popitem(last=False)
        return row


def _solve_dual_smo(X, y, C_vec, kernel, tol, max_iter, cache_mb, shrinking, verbose):
    # SMO (Platt, 1998): min 1/2 a'Qa - sum(a), s.t. y'a = 0, 0 <= a <= C. G = Qa - 1
    n = X.shape[0]
    sq = np.einsum("ij,ij->i", X, X)
    QD = kernel.diagonal(X, sq)               # K(x_i, x_i)
    cache = _KernelRowCache(X, y, kernel, sq, cache_mb)

    alpha = np.zeros(n)
    G = -np.ones(n)                           # gradien saat alpha = 0
    is_pos = y > 0
    active = np.arange(n)                     # indeks yang belum di-shrink
    shrink_every = min(n, 1000)
    unshrunk = False
    counter = shrink_every
    it = 0

    def violating_pair(idx):
        a, c, yy, g = alpha[idx], C_vec[idx], y[idx], G[idx]
        pos = is_pos[idx]
        # I_up: (y=+1, a<C) / (y=-1, a>0); I_low: (y=+1, a>0) / (y=-1, a<C)
        up = np.where(pos, a < c, a > 0)
        low = np.where(pos, a > 0, a < c)
        minus_yG = -yy * g
        return up, low, minus_yG

    def reconstruct_gradient():
        # hitung ulang G penuh setelah unshrink
        sv = np.flatnonzero(alpha > 0)
        G[:] = -1.0
        for s in range(0, len(sv), 512):
            blk = sv[s:s + 512]
            K = kernel(X, X[blk], sq, sq[blk])
            G[:] += (y[:, None] * K) @ (y[blk] * alpha[blk])

    while max_iter < 0 or it < max_iter:
        # shrinking (Joachims, 1999; Chang & Lin, 2011)
        if shrinking:
            counter -= 1
            if counter == 0:
                counter = shrink_every
                up, low, myG = violating_pair(active)
                gmax = myG[up].max() if up.any() else -np.inf
                gmin = myG[low].min() if low.any() else np.inf
                if not unshrunk and gmax - gmin <= 10 * tol:
                    unshrunk = True
                    reconstruct_gradient()
                    active = np.arange(n)
                    up, low, myG = violating_pair(active)
                    gmax = myG[up].max() if up.any() else -np.inf
                    gmin = myG[low].min() if low.any() else np.inf
                # buang sementara alpha di batas yang sudah aman
                a, c = alpha[active], C_vec[active]
                at_upper, at_lower = a >= c, a <= 0
                pos = is_pos[active]
                shrink = np.zeros(len(active), dtype=bool)
                shrink |= at_upper & pos & (myG > gmax)
                shrink |= at_upper & ~pos & (myG < gmin)
                shrink |= at_lower & pos & (myG < gmin)
                shrink |= at_lower & ~pos & (myG > gmax)
                if shrink.any() and (~shrink).sum() >= 2:
                    active = active[~shrink]

        # pilih pasangan (i, j) orde-dua (Fan, Chen & Lin, JMLR 2005)
        up, low, myG = violating_pair(active)
        if not up.any() or not low.any():
            gap = -np.inf
        else:
            ui = np.flatnonzero(up)
            i_loc = ui[np.argmax(myG[ui])]
            gmax = myG[i_loc]
            gmin = myG[low].min()
            gap = gmax - gmin
        if gap < tol:
            if shrinking and len(active) < n:
                # cek ulang di semua variabel sebelum berhenti
                reconstruct_gradient()
                active = np.arange(n)
                unshrunk = True
                counter = shrink_every
                continue
            break

        i = active[i_loc]
        Q_i = cache.get(i)
        # j: kandidat yang paling menurunkan objektif
        cand = low & (myG < gmax)
        cl = np.flatnonzero(cand)
        t_idx = active[cl]
        grad_diff = gmax - myG[cl]                                   # b_it > 0
        quad = QD[i] + QD[t_idx] - 2.0 * y[i] * y[t_idx] * Q_i[t_idx]  # a_it = K_ii + K_tt - 2 K_it
        quad = np.where(quad > 0, quad, _TAU)
        j = t_idx[np.argmin(-(grad_diff ** 2) / quad)]
        Q_j = cache.get(j)

        # update analitik dua alpha (Platt, 1998)
        C_i, C_j = C_vec[i], C_vec[j]
        old_ai, old_aj = alpha[i], alpha[j]
        ai, aj = old_ai, old_aj
        if y[i] != y[j]:
            q = QD[i] + QD[j] + 2.0 * Q_i[j]
            q = q if q > 0 else _TAU
            delta = (-G[i] - G[j]) / q
            diff = ai - aj
            ai += delta
            aj += delta
            if diff > 0:
                if aj < 0:
                    aj, ai = 0.0, diff
            elif ai < 0:
                ai, aj = 0.0, -diff
            if diff > C_i - C_j:
                if ai > C_i:
                    ai, aj = C_i, C_i - diff
            elif aj > C_j:
                aj, ai = C_j, C_j + diff
        else:
            q = QD[i] + QD[j] - 2.0 * Q_i[j]
            q = q if q > 0 else _TAU
            delta = (G[i] - G[j]) / q
            total = ai + aj
            ai -= delta
            aj += delta
            if total > C_i:
                if ai > C_i:
                    ai, aj = C_i, total - C_i
            elif aj < 0:
                aj, ai = 0.0, total
            if total > C_j:
                if aj > C_j:
                    aj, ai = C_j, total - C_j
            elif ai < 0:
                ai, aj = 0.0, total
        alpha[i], alpha[j] = ai, aj

        # update G hanya di indeks aktif
        d_i, d_j = ai - old_ai, aj - old_aj
        G[active] += Q_i[active] * d_i + Q_j[active] * d_j
        it += 1
        if verbose and it % 5000 == 0:
            print(f"  [SMO] iter={it:>7d}  gap={gap:.5f}  active={len(active)}")

    if shrinking and len(active) < n:
        reconstruct_gradient()

    # b dari rata-rata SV bebas (0 < a < C)
    yG = y * G
    free = (alpha > 0) & (alpha < C_vec)
    if free.any():
        rho = yG[free].mean()
    else:
        up = np.where(is_pos, alpha < C_vec, alpha > 0)
        low = np.where(is_pos, alpha > 0, alpha < C_vec)
        ub = yG[up].min() if up.any() else np.inf
        lb = yG[low].max() if low.any() else -np.inf
        rho = 0.5 * (ub + lb) if np.isfinite(ub + lb) else 0.0
    return alpha, -rho, it


class _BinarySVM:
    def __init__(self, kernel, C, class_weight, tol, max_iter, cache_mb, shrinking, verbose):
        self.kernel = kernel
        self.C = C
        self.class_weight = class_weight
        self.tol = tol
        self.max_iter = max_iter
        self.cache_mb = cache_mb
        self.shrinking = shrinking
        self.verbose = verbose

    def fit(self, X, y_pm):
        n_pos, n_neg = np.sum(y_pm > 0), np.sum(y_pm < 0)
        w_pos, w_neg = 1.0, 1.0
        if self.class_weight == "balanced":
            # C per kelas (Veropoulos dkk., 1999): C_k = C * n / (2 n_k)
            w_pos, w_neg = len(y_pm) / (2.0 * n_pos), len(y_pm) / (2.0 * n_neg)
        elif isinstance(self.class_weight, dict):
            w_pos, w_neg = self.class_weight.get(1, 1.0), self.class_weight.get(-1, 1.0)
        C_vec = np.where(y_pm > 0, self.C * w_pos, self.C * w_neg)

        alpha, b, n_iter = _solve_dual_smo(X, y_pm, C_vec, self.kernel, self.tol, self.max_iter,
                                           self.cache_mb, self.shrinking, self.verbose)
        sv = alpha > 1e-8
        self.support_ = np.flatnonzero(sv)
        self.support_vectors_ = X[sv].copy()
        self.dual_coef_ = (alpha[sv] * y_pm[sv])           # alpha_i * y_i
        self.alpha_ = alpha[sv]
        self.n_bounded_sv_ = int(np.sum(alpha[sv] >= C_vec[sv] - 1e-8))
        self.intercept_ = b
        self.n_iter_ = n_iter
        self.converged_ = self.max_iter < 0 or n_iter < self.max_iter
        if not self.converged_:
            warnings.warn(f"SMO berhenti di max_iter={self.max_iter} sebelum konvergen")
        self._sv_sq = np.einsum("ij,ij->i", self.support_vectors_, self.support_vectors_)
        # kernel linear: w = sum alpha_i y_i x_i
        self.coef_ = self.dual_coef_ @ self.support_vectors_ if self.kernel.name == "linear" else None
        return self

    def decision_function(self, X, batch_size=2048):
        if self.coef_ is not None:
            return X @ self.coef_ + self.intercept_
        out = np.empty(X.shape[0])
        for s in range(0, X.shape[0], batch_size):
            xb = X[s:s + batch_size]
            K = self.kernel(xb, self.support_vectors_, None, self._sv_sq)
            out[s:s + batch_size] = K @ self.dual_coef_ + self.intercept_
        return out


class SVM:
    def __init__(self, kernel="gaussian", C=1.0, gamma="scale", degree=3, coef0=1.0,
                 class_weight=None, tol=1e-3, max_iter=-1, cache_size=500,
                 shrinking=True, threshold=0.0, verbose=False):
        if _KERNEL_ALIASES.get(kernel) is None:
            raise ValueError(f"kernel tidak dikenal: {kernel} (pilih {KERNELS})")
        if C <= 0:
            raise ValueError("C harus > 0")
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.class_weight = class_weight
        self.tol = tol
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.shrinking = shrinking
        self.threshold = threshold
        self.verbose = verbose

    def _resolve_gamma(self, X):
        if self.gamma == "scale":
            var = X.var()
            return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        if np.isnan(X).any():
            raise ValueError("SVM tidak bisa menerima NaN (lakukan imputasi terlebih dahulu)")
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Butuh minimal 2 kelas")
        self.n_features_in_ = X.shape[1]
        self.gamma_ = self._resolve_gamma(X)
        self.kernel_ = Kernel(self.kernel, self.gamma_, self.degree, self.coef0)

        def make():
            return _BinarySVM(self.kernel_, self.C, None, self.tol, self.max_iter,
                              self.cache_size, self.shrinking, self.verbose)

        if len(self.classes_) == 2:
            # classes_[1] -> +1, classes_[0] -> -1
            y_pm = np.where(y == self.classes_[1], 1.0, -1.0)
            est = make()
            est.class_weight = self._binary_weight(y_pm)
            self.estimators_ = [est.fit(X, y_pm)]
        else:
            # One-vs-All: satu SVM per kelas, ambil skor terbesar
            self.estimators_ = []
            for c in self.classes_:
                y_pm = np.where(y == c, 1.0, -1.0)
                est = make()
                est.class_weight = self._binary_weight(y_pm)
                self.estimators_.append(est.fit(X, y_pm))
        return self

    def _binary_weight(self, y_pm):
        if self.class_weight is None or self.class_weight == "balanced":
            return self.class_weight
        if len(self.classes_) == 2:
            return {1: self.class_weight.get(self.classes_[1], 1.0),
                    -1: self.class_weight.get(self.classes_[0], 1.0)}
        return None

    def decision_function(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        scores = np.column_stack([e.decision_function(X) for e in self.estimators_])
        return scores[:, 0] if len(self.classes_) == 2 else scores

    def predict(self, X, threshold=None):
        scores = self.decision_function(X)
        # threshold tuning F1 (Lipton, Elkan & Narayanaswamy, ECML PKDD 2014)
        if len(self.classes_) == 2:
            t = self.threshold if threshold is None else threshold
            return self.classes_[(scores >= t).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]

    @property
    def support_vectors_(self):
        return self.estimators_[0].support_vectors_ if len(self.estimators_) == 1 else \
            [e.support_vectors_ for e in self.estimators_]

    @property
    def n_support_(self):
        return [len(e.support_) for e in self.estimators_]

    @property
    def coef_(self):
        return self.estimators_[0].coef_ if len(self.estimators_) == 1 else \
            [e.coef_ for e in self.estimators_]

    @property
    def intercept_(self):
        return self.estimators_[0].intercept_ if len(self.estimators_) == 1 else \
            np.array([e.intercept_ for e in self.estimators_])

    def margin_width(self):
        # 2 / ||w||
        widths = []
        for e in self.estimators_:
            K = self.kernel_(e.support_vectors_, e.support_vectors_, e._sv_sq, e._sv_sq)
            w_norm_sq = float(e.dual_coef_ @ K @ e.dual_coef_)
            widths.append(2.0 / np.sqrt(w_norm_sq) if w_norm_sq > 0 else np.inf)
        return widths[0] if len(widths) == 1 else widths

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"{filepath} bukan model {cls.__name__}")
        return model

    def __repr__(self):
        return (f"SVM(kernel={self.kernel!r}, C={self.C}, gamma={self.gamma!r}, degree={self.degree}, "
                f"coef0={self.coef0}, class_weight={self.class_weight!r})")
