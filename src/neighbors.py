from __future__ import annotations

import numpy as np

TOP = 5
COLUMNS = ["ближайший", "похожесть", "среднее по соседям", "средняя похожесть", "сколько близких"]


def tanimoto(left, right):
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    shared = left @ right.T
    sizes_left = left.sum(axis=1)[:, None]
    sizes_right = right.sum(axis=1)[None, :]
    return shared / np.maximum(sizes_left + sizes_right - shared, 1e-9)


def read_across(bits, reference_bits, reference_values, skip_self=False):
    similarity = tanimoto(bits, reference_bits)
    if skip_self:
        np.fill_diagonal(similarity, -1.0)
    scale = np.log1p(reference_values)
    order = np.argsort(-similarity, axis=1)[:, :TOP]
    rows = np.arange(len(bits))[:, None]
    best = similarity[rows, order]
    values = scale[order]
    weights = np.clip(best, 0.0, None) + 1e-6
    return np.column_stack([
        values[:, 0],
        best[:, 0],
        (values * weights).sum(axis=1) / weights.sum(axis=1),
        best.mean(axis=1),
        (similarity > 0.7).sum(axis=1),
    ]).astype(np.float64)
