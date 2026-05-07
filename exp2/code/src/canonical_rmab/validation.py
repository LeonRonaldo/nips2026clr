from __future__ import annotations

import numpy as np


def as_float_array(value: object, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def validate_probability_vector(value: object, name: str, tol: float = 1e-10) -> np.ndarray:
    vector = as_float_array(value, name, ndim=1)
    if vector.size == 0:
        raise ValueError(f"{name} must be nonempty.")
    if np.any(vector < -tol):
        raise ValueError(f"{name} has negative entries.")
    clipped = np.maximum(vector, 0.0)
    total = float(clipped.sum())
    if total <= 0.0:
        raise ValueError(f"{name} has zero mass.")
    normalized = clipped / total
    if abs(float(vector.sum()) - 1.0) > tol:
        raise ValueError(f"{name} must sum to one, got {vector.sum()}.")
    return normalized


def normalize_probability_vector(value: object, name: str) -> np.ndarray:
    vector = as_float_array(value, name, ndim=1)
    if np.any(vector < -1e-12):
        raise ValueError(f"{name} has negative entries.")
    clipped = np.maximum(vector, 0.0)
    total = float(clipped.sum())
    if total <= 0.0:
        raise ValueError(f"{name} has zero mass.")
    return clipped / total


def validate_row_stochastic(value: object, name: str, tol: float = 1e-10) -> np.ndarray:
    matrix = as_float_array(value, name, ndim=2)
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty.")
    if np.any(matrix < -tol):
        raise ValueError(f"{name} has negative entries.")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, np.ones(matrix.shape[0]), atol=tol, rtol=0.0):
        raise ValueError(f"{name} rows must sum to one, got {row_sums}.")
    return np.maximum(matrix, 0.0)


def normalize_rows(value: object, name: str) -> np.ndarray:
    matrix = as_float_array(value, name, ndim=2)
    if np.any(matrix < -1e-12):
        raise ValueError(f"{name} has negative entries.")
    clipped = np.maximum(matrix, 0.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"{name} has a zero-mass row.")
    return clipped / row_sums

