import os
import pickle

import numpy as np


class KNN:
    def __init__(self, n_neighbors=5, metric="euclidean", p=2, weights="uniform",
                 feature_weights=None, batch_size=256):
        if metric not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError(f"metric tidak dikenal: {metric}")
        if weights not in ("uniform", "distance"):
            raise ValueError(f"weights tidak dikenal: {weights}")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        self.feature_weights = feature_weights
        self.batch_size = batch_size

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        if np.isnan(X).any():
            raise ValueError("KNN tidak bisa menerima NaN")

        self.classes_, self.y_train_ = np.unique(y, return_inverse=True)

        # Pembobotan fitur (Mitchell, Machine Learning, 1997, Bab 8.2) https://www.cs.cmu.edu/~tom/files/MachineLearningTomMitchell.pdf
        # fitur tak relevan mendominasi jarak, jadi tiap sumbu fitur diberi bobot yang dipilih lewat cross-validation.
        if self.feature_weights is None:
            self.feature_weights_ = np.ones(X.shape[1])
        else:
            self.feature_weights_ = np.asarray(self.feature_weights, dtype=np.float64)
        self.used_features_ = self.feature_weights_ != 0

        self.X_train_ = self._apply_weights(X)
        return self

    def _apply_weights(self, X):
        return (X * self.feature_weights_)[:, self.used_features_]

    def _compute_distances(self, X_query):
        X_train = self.X_train_

        if self.metric == "euclidean":
            query_sq = np.sum(X_query ** 2, axis=1)[:, None]
            train_sq = np.sum(X_train ** 2, axis=1)[None, :]
            dist_sq = query_sq - 2 * (X_query @ X_train.T) + train_sq
            return np.sqrt(np.maximum(dist_sq, 0))

        p = 1 if self.metric == "manhattan" else self.p
        total = np.zeros((X_query.shape[0], X_train.shape[0]))
        for j in range(X_train.shape[1]):
            total += np.abs(X_query[:, j, None] - X_train[None, :, j]) ** p
        return total if p == 1 else total ** (1 / p)

    def kneighbors(self, X):
        X_query = self._apply_weights(np.asarray(X, dtype=np.float64))

        all_dist, all_idx = [], []
        for start in range(0, len(X_query), self.batch_size):
            dist = self._compute_distances(X_query[start:start + self.batch_size])
            nearest = np.argsort(dist, axis=1, kind="stable")[:, :self.n_neighbors]
            all_idx.append(nearest)
            all_dist.append(np.take_along_axis(dist, nearest, axis=1))

        return np.vstack(all_dist), np.vstack(all_idx)

    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        neighbor_labels = self.y_train_[idx]

        if self.weights == "uniform":
            votes = np.ones_like(dist)
        else:
            # Pembobotan jarak (Dudani, IEEE Trans. SMC, 1976)
            # tetangga diberi bobot menurut jaraknya, sehingga tetangga yang lebih dekat lebih berpengaruh.
            votes = np.zeros_like(dist)
            exact = (dist == 0).any(axis=1)
            votes[exact] = dist[exact] == 0
            votes[~exact] = 1 / dist[~exact]

        proba = np.zeros((len(dist), len(self.classes_)))
        for c in range(len(self.classes_)):
            proba[:, c] = np.sum(votes * (neighbor_labels == c), axis=1)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X, threshold=None):
        proba = self.predict_proba(X)
        # Threshold tuning untuk F1 (Lipton, Elkan & Narayanaswamy, ECML PKDD 2014) https://link.springer.com/chapter/10.1007/978-3-662-44851-9_15
        # ambang optimal untuk F1 umumnya bukan 0.5, sehingga ambang dipilih dari data validasi.
        if threshold is not None and len(self.classes_) == 2:
            return self.classes_[(proba[:, 1] >= threshold).astype(int)]
        return self.classes_[np.argmax(proba, axis=1)]

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"{filepath} bukan model {cls.__name__}")
        return model
