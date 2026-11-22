import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np


class LogisticRegression:

    VALID_PENALTIES = ("none", "l1", "l2", "elasticnet")
    VALID_OPTIMIZERS = ("stochastic_gradient_ascent", "adam", "lbfgs")

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        l1_ratio: float = 0.5,
        optimizer: str = "stochastic_gradient_ascent",
        learning_rate: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-6,
        batch_size: int = 1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        lbfgs_memory: int = 12,
        threshold: float = 0.5,
        random_state: int = 42,
    ):
        self.penalty = str(penalty).lower()
        self.C = float(C)
        self.l1_ratio = float(l1_ratio)
        self.optimizer = str(optimizer).lower()
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.batch_size = int(batch_size)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.lbfgs_memory = int(lbfgs_memory)
        self.threshold = float(threshold)
        self.random_state = int(random_state)

        if self.penalty not in self.VALID_PENALTIES:
            raise ValueError(f"penalty harus salah satu dari {self.VALID_PENALTIES}")
        if self.optimizer not in self.VALID_OPTIMIZERS:
            raise ValueError(f"optimizer harus salah satu dari {self.VALID_OPTIMIZERS}")
        if self.optimizer == "lbfgs" and self.penalty in ("l1", "elasticnet"):
            raise ValueError(
                "Optimizer 'lbfgs' hanya mendukung penalty terdiferensialkan ('l2' atau 'none'); "
                "gunakan 'adam' atau 'stochastic_gradient_ascent' untuk L1/ElasticNet."
            )
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError("l1_ratio harus berada pada [0, 1]")
        if self.C <= 0.0:
            raise ValueError("C harus bernilai positif (> 0)")
        if self.batch_size < 1:
            raise ValueError("batch_size minimal 1")

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.classes_: Optional[np.ndarray] = None
        self.lambda_: float = 0.0
        self.n_iter_: int = 0
        self.history_: Dict[str, List[float]] = {}

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def _linear(X: np.ndarray, w: np.ndarray, b0: float) -> np.ndarray:
        return X @ w + b0

    def _l1_l2_strength(self) -> Tuple[float, float]:
        lam = self.lambda_
        if self.penalty == "l1":
            return lam, 0.0
        if self.penalty == "l2":
            return 0.0, lam
        if self.penalty == "elasticnet":
            return lam * self.l1_ratio, lam * (1.0 - self.l1_ratio)
        return 0.0, 0.0

    def _penalty_value(self, w: np.ndarray) -> float:
        l1, l2 = self._l1_l2_strength()
        return float(l1 * np.sum(np.abs(w)) + 0.5 * l2 * np.dot(w, w))

    def _penalty_gradient(self, w: np.ndarray) -> np.ndarray:
        l1, l2 = self._l1_l2_strength()
        g = l2 * w
        if l1 > 0.0:
            g = g + l1 * np.sign(w)
        return g

    def _mean_log_likelihood(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b0: float) -> float:
        z = np.clip(self._linear(X, w, b0), -500.0, 500.0)
        # Secara aljabar identik dengan y*log(p) + (1-y)*log(1-p), stabil via logaddexp
        ll_vec = y * z - np.logaddexp(0.0, z)
        return float(np.mean(ll_vec))

    def _likelihood_gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b0: float) -> Tuple[np.ndarray, float]:
        p = self._sigmoid(self._linear(X, w, b0))
        r = (y - p) / len(y)
        return X.T @ r, float(np.sum(r))

    def _record(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b0: float) -> float:
        ll = self._mean_log_likelihood(X, y, w, b0)
        obj = ll - self._penalty_value(w)
        self.history_["log_likelihood"].append(ll)
        self.history_["loss"].append(-obj)
        self.history_["theta_0"].append(float(b0))
        self.history_["theta_1"].append(float(w[0]) if w.size > 0 else 0.0)
        return obj

    def _fit_stochastic_gradient_ascent(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b0: float
    ) -> Tuple[np.ndarray, float]:
        """
        p_i = 1 / (1 + e^(-b^T x_i))
        b_j = b_j + eta * ((y_i - p_i) * x_ij - dR/db_j)
        """
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        eta = self.learning_rate
        bs = min(self.batch_size, n)
        prev = self._record(X, y, w, b0)
        for epoch in range(1, self.max_iter + 1):
            order = rng.permutation(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                gw, gb0 = self._likelihood_gradient(X[idx], y[idx], w, b0)
                w = w + eta * (gw - self._penalty_gradient(w))
                b0 = b0 + eta * gb0
            cur = self._record(X, y, w, b0)
            self.n_iter_ = epoch
            if abs(cur - prev) < self.tol:
                break
            prev = cur
        return w, b0

    def _fit_adam(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b0: float
    ) -> Tuple[np.ndarray, float]:
        """
        Adam Optimizer untuk memaksimalkan penalized log-likelihood J(b) = LL(b) / n - R(w).
        Sumber: Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
        [https://arxiv.org/abs/1412.6980]. Penjelasan tambahan:
        [https://www.geeksforgeeks.org/deep-learning/adam-optimizer/],
        [https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/],
        [https://www.nature.com/articles/s41599-025-04505-8].
        Cara kerja, untuk gradien mini-batch g pada langkah ke-t:
        m = beta1 * m + (1 - beta1) * g            (momen pertama: rata-rata bergerak gradien)
        v = beta2 * v + (1 - beta2) * g^2          (momen kedua: rata-rata bergerak kuadrat gradien)
        m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)   (koreksi bias karena m, v mulai dari 0)
        b <- b + eta * m_hat / (sqrt(v_hat) + epsilon)
        Setiap parameter, termasuk intersep b0, memiliki m dan v sendiri sehingga langkahnya
        menyesuaikan besar gradien masing-masing. Tanda + dipakai karena J dimaksimalkan,
        sedangkan sumber menulis tanda - untuk meminimalkan loss.
        """
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        eta = self.learning_rate
        bs = min(self.batch_size, n)
        m_w = np.zeros_like(w)
        v_w = np.zeros_like(w)
        m_b, v_b, t = 0.0, 0.0, 0
        prev = self._record(X, y, w, b0)
        for epoch in range(1, self.max_iter + 1):
            order = rng.permutation(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                gw, gb0 = self._likelihood_gradient(X[idx], y[idx], w, b0)
                gw = gw - self._penalty_gradient(w)
                t += 1
                m_w = self.beta1 * m_w + (1.0 - self.beta1) * gw
                v_w = self.beta2 * v_w + (1.0 - self.beta2) * gw * gw
                m_hat = m_w / (1.0 - self.beta1 ** t)
                v_hat = v_w / (1.0 - self.beta2 ** t)
                w = w + eta * m_hat / (np.sqrt(v_hat) + self.epsilon)
                m_b = self.beta1 * m_b + (1.0 - self.beta1) * gb0
                v_b = self.beta2 * v_b + (1.0 - self.beta2) * gb0 * gb0
                mb_hat = m_b / (1.0 - self.beta1 ** t)
                vb_hat = v_b / (1.0 - self.beta2 ** t)
                b0 = b0 + eta * mb_hat / (np.sqrt(vb_hat) + self.epsilon)
            cur = self._record(X, y, w, b0)
            self.n_iter_ = epoch
            if abs(cur - prev) < self.tol:
                break
            prev = cur
        return w, b0

    def _fit_lbfgs(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b0: float
    ) -> Tuple[np.ndarray, float]:
        """
        Two-Loop Limited-Memory BFGS untuk meminimalkan -J(b), yaitu negatif log-likelihood
        ditambah penalti L2. Sumber: rekursi dua loop dan penanganan kondisi kelengkungan pada
        [https://en.wikipedia.org/wiki/Limited-memory_BFGS]; konsep BFGS (aproksimasi invers
        Hessian, arah turun, line search, pembaruan parameter) pada
        [https://www.geeksforgeeks.org/deep-learning/second-order-optimization-methods/].
        Cara kerja: invers Hessian H didekati dari `lbfgs_memory` pasangan terakhir
        s = x_baru - x dan y = g_baru - g tanpa membentuk matriks Hessian.
        1. q = g. Untuk pasangan dari yang terbaru: a_i = rho_i * s_i^T q, q = q - a_i * y_i,
           dengan rho_i = 1 / (y_i^T s_i).
        2. r = gamma * q, dengan gamma = s^T y / y^T y dari pasangan terbaru (H0 = gamma * I).
        3. Untuk pasangan dari yang terlama: b_i = rho_i * y_i^T r, r = r + s_i * (a_i - b_i).
        4. Arah d = -r. Langkah dipilih dengan Armijo backtracking: step dibagi dua sampai
           f(x + step * d) <= f(x) + c1 * step * g^T d.
        5. Pasangan (s, y) disimpan hanya jika s^T y > 0 agar H tetap definit positif.
        Hanya berlaku untuk penalti yang terdiferensialkan (L2 atau tanpa penalti).
        """
        n = X.shape[0]
        _, l2 = self._l1_l2_strength()
        m_mem = max(1, self.lbfgs_memory)

        def _loss_and_grad(w_cur: np.ndarray, b_cur: float) -> Tuple[float, np.ndarray, float, float]:
            z = np.clip(X @ w_cur + b_cur, -500.0, 500.0)
            ll_vec = y * z - np.logaddexp(0.0, z)
            ll = float(np.mean(ll_vec))
            loss = -ll + 0.5 * l2 * float(np.dot(w_cur, w_cur))
            p = 1.0 / (1.0 + np.exp(-z))
            r = (p - y) / n
            gw = X.T @ r + l2 * w_cur
            gb = float(np.sum(r))
            return loss, gw, gb, ll

        loss, gw, gb, ll = _loss_and_grad(w, b0)
        self.history_["log_likelihood"].append(ll)
        self.history_["loss"].append(loss)
        self.history_["theta_0"].append(float(b0))
        self.history_["theta_1"].append(float(w[0]) if w.size > 0 else 0.0)

        s_hist: List[np.ndarray] = []
        y_hist: List[np.ndarray] = []
        rho_hist: List[float] = []

        x = np.append(w, b0)
        g = np.append(gw, gb)

        for it in range(1, self.max_iter + 1):
            if float(np.linalg.norm(g, ord=np.inf)) < self.tol:
                break

            q = g.copy()
            alphas: List[float] = []
            for s_i, y_i, rho_i in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
                al = rho_i * float(np.dot(s_i, q))
                alphas.append(al)
                q = q - al * y_i

            if y_hist:
                ys = float(np.dot(y_hist[-1], s_hist[-1]))
                yy = float(np.dot(y_hist[-1], y_hist[-1]))
                gamma = ys / max(yy, 1e-16)
            else:
                gamma = 1.0 / max(float(np.linalg.norm(g)), 1.0)

            r_dir = gamma * q
            for s_i, y_i, rho_i, al in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
                bt = rho_i * float(np.dot(y_i, r_dir))
                r_dir = r_dir + (al - bt) * s_i
            direction = -r_dir

            slope = float(np.dot(g, direction))
            if slope >= 0.0:
                direction = -g
                slope = float(np.dot(g, direction))

            step = 1.0
            c1 = 1e-4
            accepted = False
            x_new = x
            loss_new, gw_new, gb_new, ll_new = loss, gw, gb, ll
            for _ in range(30):
                x_new = x + step * direction
                loss_new, gw_new, gb_new, ll_new = _loss_and_grad(x_new[:-1], float(x_new[-1]))
                if loss_new <= loss + c1 * step * slope:
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                break

            g_new = np.append(gw_new, gb_new)
            s_vec = x_new - x
            y_vec = g_new - g
            sy = float(np.dot(s_vec, y_vec))
            if sy > 1e-12:
                if len(s_hist) >= m_mem:
                    s_hist.pop(0)
                    y_hist.pop(0)
                    rho_hist.pop(0)
                s_hist.append(s_vec)
                y_hist.append(y_vec)
                rho_hist.append(1.0 / sy)

            rel_imp = abs(loss - loss_new) / max(abs(loss), 1.0)
            x, g, loss = x_new, g_new, loss_new
            self.n_iter_ = it
            self.history_["log_likelihood"].append(ll_new)
            self.history_["loss"].append(loss_new)
            self.history_["theta_0"].append(float(x[-1]))
            self.history_["theta_1"].append(float(x[0]) if x.size > 1 else 0.0)

            if rel_imp < self.tol * 1e-2:
                break

        return x[:-1], float(x[-1])

    def fit(self, X, y) -> "LogisticRegression":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y).ravel()
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError("Logistic Regression ini hanya mendukung klasifikasi biner.")
        y01 = (y_arr == self.classes_[1]).astype(np.float64)
        n, d = X_arr.shape

        self.lambda_ = 0.0 if self.penalty == "none" else 1.0 / (self.C * n)
        w = np.zeros(d, dtype=np.float64)
        b0 = 0.0
        self.history_ = {"log_likelihood": [], "loss": [], "theta_0": [], "theta_1": []}
        self.n_iter_ = 0

        if self.optimizer == "stochastic_gradient_ascent":
            w, b0 = self._fit_stochastic_gradient_ascent(X_arr, y01, w, b0)
        elif self.optimizer == "adam":
            w, b0 = self._fit_adam(X_arr, y01, w, b0)
        else:
            w, b0 = self._fit_lbfgs(X_arr, y01, w, b0)

        self.coef_ = w
        self.intercept_ = float(b0)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model belum di-fit. Panggil fit(X, y) terlebih dahulu.")
        X_arr = np.asarray(X, dtype=np.float64)
        p1 = self._sigmoid(self._linear(X_arr, self.coef_, self.intercept_))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold: Optional[float] = None) -> np.ndarray:
        th = self.threshold if threshold is None else float(threshold)
        p1 = self.predict_proba(X)[:, 1]
        return np.where(p1 >= th, self.classes_[1], self.classes_[0])

    def save(self, filepath: str) -> None:
        folder = os.path.dirname(os.path.abspath(filepath))
        if folder:
            os.makedirs(folder, exist_ok=True)

        for mod_name in ("src.utils.model_io", "utils.model_io"):
            try:
                mod = __import__(mod_name, fromlist=["save_model"])
                if hasattr(mod, "save_model") and callable(mod.save_model):
                    mod.save_model(self, filepath)
                    return
            except Exception:
                continue

        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> "LogisticRegression":
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
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Objek pada {filepath} bukan {cls.__name__}")
        return model

    def animate_loss_contour(
        self,
        X,
        y,
        feature_idx: Optional[int] = None,
        feature_name: Optional[str] = None,
        save_path: str = "output/figures/logreg_contour.gif",
        n_steps: int = 45,
        fps: int = 12,
        grid_size: int = 80,
    ) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y).ravel()
        classes = np.unique(y_arr)
        if len(classes) != 2:
            raise ValueError("Visualisasi kontur membutuhkan target biner (2 kelas).")
        y01 = (y_arr == classes[1]).astype(np.float64)
        n_samples, n_features = X_arr.shape

        if feature_idx is None:
            if n_features == 1:
                best_j = 0
            else:
                y_c = y01 - np.mean(y01)
                y_norm = float(np.sqrt(np.sum(y_c ** 2))) + 1e-12
                corrs = np.zeros(n_features, dtype=np.float64)
                for j in range(n_features):
                    col = X_arr[:, j]
                    c_c = col - np.mean(col)
                    c_norm = float(np.sqrt(np.sum(c_c ** 2)))
                    if c_norm > 1e-12:
                        corrs[j] = abs(float(np.sum(c_c * y_c)) / (c_norm * y_norm))
                best_j = int(np.argmax(corrs))
        else:
            best_j = int(np.clip(feature_idx, 0, n_features - 1))

        x1_raw = X_arr[:, best_j]
        x1_mean = float(np.mean(x1_raw))
        x1_std = float(np.std(x1_raw))
        x1 = (x1_raw - x1_mean) / (x1_std if x1_std > 1e-12 else 1.0)
        feat_label = feature_name if feature_name is not None else f"Fitur Terkuat (X[:, {best_j}])"

        lam = 0.0 if self.penalty == "none" else 1.0 / (self.C * n_samples)
        l1_coeff, l2_coeff = 0.0, 0.0
        if self.penalty == "l1":
            l1_coeff = lam
        elif self.penalty == "l2":
            l2_coeff = lam
        elif self.penalty == "elasticnet":
            l1_coeff = lam * self.l1_ratio
            l2_coeff = lam * (1.0 - self.l1_ratio)

        def eval_loss_and_grad(t0_val: float, t1_val: float) -> Tuple[float, float, float]:
            z = np.clip(t0_val + t1_val * x1, -500.0, 500.0)
            ll = float(np.mean(y01 * z - np.logaddexp(0.0, z)))
            pen = l1_coeff * abs(t1_val) + 0.5 * l2_coeff * (t1_val ** 2)
            loss_v = -ll + pen
            p = 1.0 / (1.0 + np.exp(-z))
            r = (p - y01) / n_samples
            g0 = float(np.sum(r))
            g1 = float(np.dot(x1, r)) + l2_coeff * t1_val + (l1_coeff * float(np.sign(t1_val)) if l1_coeff > 0 else 0.0)
            return loss_v, g0, g1

        opt_lr = LogisticRegression(
            penalty="l2" if self.penalty != "none" else "none",
            C=self.C,
            optimizer="lbfgs",
            max_iter=200,
            tol=1e-9,
            random_state=self.random_state,
        ).fit(x1.reshape(-1, 1), y01)
        t0_star = float(opt_lr.intercept_)
        t1_star = float(opt_lr.coef_[0])

        if (
            n_features == 1
            and self.coef_ is not None
            and len(self.coef_) == 1
            and len(self.history_.get("theta_0", [])) >= 10
            and len(self.history_.get("theta_1", [])) == len(self.history_.get("theta_0", []))
        ):
            t0_path = np.asarray(self.history_["theta_0"], dtype=np.float64)
            t1_path = np.asarray(self.history_["theta_1"], dtype=np.float64)
            loss_path = np.asarray(self.history_["loss"], dtype=np.float64)
        else:
            t0_cur = float(-t0_star * 0.85 if abs(t0_star) > 0.35 else 0.85)
            t1_cur = float(-t1_star * 0.65 if abs(t1_star) > 0.35 else -0.85)
            m0, m1 = 0.0, 0.0
            v0, v1 = 0.0, 0.0
            b1, b2, eps = 0.82, 0.99, 1e-8
            lr_step = 0.14

            t0_list: List[float] = []
            t1_list: List[float] = []
            loss_list: List[float] = []
            rng = np.random.default_rng(self.random_state)
            batch_sz = min(max(512, n_samples // 12), n_samples)

            for step_i in range(1, n_steps + 1):
                loss_v, g0_full, g1_full = eval_loss_and_grad(t0_cur, t1_cur)
                t0_list.append(t0_cur)
                t1_list.append(t1_cur)
                loss_list.append(loss_v)

                idx_b = rng.choice(n_samples, size=batch_sz, replace=False)
                zb = np.clip(t0_cur + t1_cur * x1[idx_b], -500.0, 500.0)
                pb = 1.0 / (1.0 + np.exp(-zb))
                rb = (pb - y01[idx_b]) / len(idx_b)
                g0_b = 0.65 * g0_full + 0.35 * float(np.sum(rb))
                g1_b = 0.65 * g1_full + 0.35 * (
                    float(np.dot(x1[idx_b], rb))
                    + l2_coeff * t1_cur
                    + (l1_coeff * float(np.sign(t1_cur)) if l1_coeff > 0 else 0.0)
                )

                m0 = b1 * m0 + (1.0 - b1) * g0_b
                m1 = b1 * m1 + (1.0 - b1) * g1_b
                v0 = b2 * v0 + (1.0 - b2) * (g0_b ** 2)
                v1 = b2 * v1 + (1.0 - b2) * (g1_b ** 2)
                m0_h = m0 / (1.0 - b1 ** step_i)
                m1_h = m1 / (1.0 - b1 ** step_i)
                v0_h = v0 / (1.0 - b2 ** step_i)
                v1_h = v1 / (1.0 - b2 ** step_i)

                decay = 1.0 / (1.0 + 0.015 * step_i)
                t0_cur -= lr_step * decay * m0_h / (np.sqrt(v0_h) + eps)
                t1_cur -= lr_step * decay * m1_h / (np.sqrt(v1_h) + eps)

            loss_final, _, _ = eval_loss_and_grad(t0_star, t1_star)
            t0_list.append(t0_star)
            t1_list.append(t1_star)
            loss_list.append(loss_final)

            t0_path = np.asarray(t0_list, dtype=np.float64)
            t1_path = np.asarray(t1_list, dtype=np.float64)
            loss_path = np.asarray(loss_list, dtype=np.float64)

            if not self.history_.get("theta_0"):
                self.history_ = {
                    "log_likelihood": (-loss_path).tolist(),
                    "loss": loss_path.tolist(),
                    "theta_0": t0_path.tolist(),
                    "theta_1": t1_path.tolist(),
                }

        total_pts = len(t0_path)
        if total_pts > n_steps + 1:
            sel_idx = np.unique(np.round(np.linspace(0, total_pts - 1, n_steps + 1)).astype(int))
            t0_path = t0_path[sel_idx]
            t1_path = t1_path[sel_idx]
            loss_path = loss_path[sel_idx]
            total_pts = len(t0_path)

        t0_min, t0_max = float(min(np.min(t0_path), t0_star)), float(max(np.max(t0_path), t0_star))
        t1_min, t1_max = float(min(np.min(t1_path), t1_star)), float(max(np.max(t1_path), t1_star))
        span0 = max(t0_max - t0_min, 0.8)
        span1 = max(t1_max - t1_min, 0.8)
        pad0, pad1 = 0.35 * span0, 0.35 * span1

        g_t0 = np.linspace(t0_min - pad0, t0_max + pad0, grid_size)
        g_t1 = np.linspace(t1_min - pad1, t1_max + pad1, grid_size)
        T0, T1 = np.meshgrid(g_t0, g_t1)

        if n_samples > 2500:
            rng_sub = np.random.default_rng(self.random_state)
            sub_i = rng_sub.choice(n_samples, size=2500, replace=False)
            x1_eval, y_eval = x1[sub_i], y01[sub_i]
        else:
            x1_eval, y_eval = x1, y01

        t0_flat = T0.ravel()
        t1_flat = T1.ravel()
        n_grid_pts = len(t0_flat)
        ll_flat = np.zeros(n_grid_pts, dtype=np.float64)
        chunk_sz = 400
        for s_idx in range(0, n_grid_pts, chunk_sz):
            e_idx = min(s_idx + chunk_sz, n_grid_pts)
            z_chunk = np.clip(
                t0_flat[s_idx:e_idx, None] + t1_flat[s_idx:e_idx, None] * x1_eval[None, :],
                -500.0,
                500.0,
            )
            ll_flat[s_idx:e_idx] = np.mean(y_eval[None, :] * z_chunk - np.logaddexp(0.0, z_chunk), axis=1)

        PEN_grid = l1_coeff * np.abs(t1_flat) + 0.5 * l2_coeff * (t1_flat ** 2)
        Loss_grid = (-ll_flat + PEN_grid).reshape(grid_size, grid_size)

        fig, (ax_c, ax_l) = plt.subplots(
            1, 2, figsize=(14.2, 5.7), dpi=110, gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.34}
        )
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.13, top=0.88, wspace=0.34)
        fig.patch.set_facecolor("white")

        levels = np.linspace(float(np.min(Loss_grid)), float(np.percentile(Loss_grid, 96)), 22)
        cf = ax_c.contourf(T0, T1, Loss_grid, levels=levels, cmap="YlGnBu_r", alpha=0.88, extend="max")
        cs = ax_c.contour(T0, T1, Loss_grid, levels=levels[::2], colors="#1e293b", linewidths=0.75, alpha=0.65)
        ax_c.clabel(cs, inline=True, fontsize=7.5, fmt="%.3f")
        cbar = fig.colorbar(cf, ax=ax_c, fraction=0.046, pad=0.03)
        cbar.set_label("Regularized Log-Loss $J(\\theta_0, \\theta_1)$", fontsize=9.5, fontweight="bold")

        ax_c.plot(
            t0_path[0], t1_path[0], marker="o", markersize=8.5, color="#dc2626",
            markeredgecolor="white", markeredgewidth=1.4, label=f"Start $({t0_path[0]:.2f}, {t1_path[0]:.2f})$", zorder=6
        )
        ax_c.plot(
            t0_star, t1_star, marker="*", markersize=14.0, color="#facc15",
            markeredgecolor="#713f12", markeredgewidth=1.3, label=f"Optimum $({t0_star:.2f}, {t1_star:.2f})$", zorder=7
        )

        traj_line, = ax_c.plot([], [], color="#e11d48", linewidth=2.4, alpha=0.95, label="Lintasan $(\\theta_0, \\theta_1)$", zorder=5)
        traj_pts, = ax_c.plot([], [], "o", color="#fb7185", markersize=4.0, alpha=0.85, zorder=5)
        head_pt, = ax_c.plot([], [], "o", color="#be123c", markersize=9.5, markeredgecolor="white", markeredgewidth=1.6, zorder=8)

        ax_c.set_title(f"Kontur Log-Loss & Lintasan Parameter $(\\theta_0, \\theta_1)$\n[{feat_label}]", fontsize=11.5, fontweight="bold", pad=10)
        ax_c.set_xlabel("Intercept / Bias ($\\theta_0$)", fontsize=10.5, fontweight="bold")
        ax_c.set_ylabel("Bobot Fitur ($\\theta_1$)", fontsize=10.5, fontweight="bold")
        ax_c.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
        ax_c.grid(True, linestyle=":", alpha=0.35, color="#475569")

        status_box = ax_c.text(
            0.03, 0.04, "", transform=ax_c.transAxes, fontsize=9.0, fontweight="bold",
            color="#0f172a", bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffbeb", edgecolor="#f59e0b", alpha=0.94),
            zorder=9,
        )

        iters = np.arange(total_pts)
        ax_l.plot(iters, loss_path, color="#cbd5e1", linewidth=1.8, linestyle="--", label="Lintasan Penuh")
        loss_line, = ax_l.plot([], [], color="#2563eb", linewidth=2.5, label="Log-Loss Saat Ini")
        loss_head, = ax_l.plot([], [], "o", color="#1d4ed8", markersize=8.5, markeredgecolor="white", markeredgewidth=1.5, zorder=6)

        y_min_l, y_max_l = float(np.min(loss_path)), float(np.max(loss_path))
        y_span_l = max(y_max_l - y_min_l, 1e-3)
        ax_l.set_xlim(0, max(total_pts - 1, 1))
        ax_l.set_ylim(y_min_l - 0.08 * y_span_l, y_max_l + 0.12 * y_span_l)
        ax_l.set_title("Konvergensi Fungsi Objektif (Log-Loss) Selama Training", fontsize=11.5, fontweight="bold", pad=10)
        ax_l.set_xlabel("Iterasi Training ($t$)", fontsize=10.5, fontweight="bold")
        ax_l.set_ylabel("Regularized Log-Loss $J(\\theta^{(t)})$", fontsize=10.5, fontweight="bold")
        ax_l.grid(True, linestyle="--", alpha=0.45)
        ax_l.legend(loc="upper right", fontsize=8.8)

        def _init_anim():
            traj_line.set_data([], [])
            traj_pts.set_data([], [])
            head_pt.set_data([], [])
            loss_line.set_data([], [])
            loss_head.set_data([], [])
            status_box.set_text("")
            return traj_line, traj_pts, head_pt, loss_line, loss_head, status_box

        def _update_anim(frame_idx: int):
            cur_t0 = t0_path[: frame_idx + 1]
            cur_t1 = t1_path[: frame_idx + 1]
            cur_loss = loss_path[: frame_idx + 1]
            cur_it = iters[: frame_idx + 1]

            traj_line.set_data(cur_t0, cur_t1)
            traj_pts.set_data(cur_t0[:-1], cur_t1[:-1])
            head_pt.set_data([cur_t0[-1]], [cur_t1[-1]])

            loss_line.set_data(cur_it, cur_loss)
            loss_head.set_data([cur_it[-1]], [cur_loss[-1]])

            status_box.set_text(
                f"Iterasi : {frame_idx}/{total_pts - 1}\n"
                f"θ₀ = {cur_t0[-1]:+.4f} | θ₁ = {cur_t1[-1]:+.4f}\n"
                f"Log-Loss = {cur_loss[-1]:.5f}"
            )
            return traj_line, traj_pts, head_pt, loss_line, loss_head, status_box

        anim = animation.FuncAnimation(
            fig, _update_anim, init_func=_init_anim, frames=total_pts, interval=max(int(1000 / max(fps, 1)), 40), blit=True
        )

        abs_save = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(abs_save), exist_ok=True)
        if abs_save.lower().endswith(".mp4") and animation.writers.is_available("ffmpeg"):
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(abs_save, writer=writer)
        else:
            if abs_save.lower().endswith(".mp4"):
                abs_save = os.path.splitext(abs_save)[0] + ".gif"
            writer = animation.PillowWriter(fps=fps)
            anim.save(abs_save, writer=writer)

        plt.close(fig)
        return abs_save