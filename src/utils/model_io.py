"""Utilitas Save & Load Model (dipakai bersama oleh seluruh model di src/models/).

Berbasis modul standar Python `pickle` (HIGHEST_PROTOCOL). Direktori tujuan
(`output/models/` secara default jika filepath tidak menyertakan folder)
dibuat secara otomatis.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Union

DEFAULT_MODELS_DIR = os.path.join("output", "models")


def save_model(model_obj: Any, filepath: Union[str, Path]) -> str:
    """Menyimpan objek model yang sudah di-fit ke file .pkl."""
    path_str = str(filepath)
    if not os.path.dirname(path_str):
        path_str = os.path.join(DEFAULT_MODELS_DIR, path_str)
    parent = os.path.dirname(os.path.abspath(path_str))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path_str, "wb") as f:
        pickle.dump(model_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path_str


def load_model(filepath: Union[str, Path]) -> Any:
    """Memuat kembali objek model dari file .pkl dan siap dipakai untuk predict()."""
    path_str = str(filepath)
    if not os.path.exists(path_str) and not os.path.dirname(path_str):
        candidate = os.path.join(DEFAULT_MODELS_DIR, path_str)
        if os.path.exists(candidate):
            path_str = candidate
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"File model tidak ditemukan di: {path_str}")
    with open(path_str, "rb") as f:
        return pickle.load(f)
