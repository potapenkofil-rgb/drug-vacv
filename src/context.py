from __future__ import annotations

import numpy as np
import pandas as pd

SMOOTHING = 5.0


def title_kind(title):
    name = str(title).strip().lower()
    if name.startswith("sid"):
        return "pubchem"
    if name.isdigit():
        return "номер"
    if len(name) <= 5:
        return "короткое"
    return "название"


def labels_of(table, with_title=False):
    strain = table["strain"].astype(str).str.strip().str.lower()
    culture = table["culture"].astype(str).str.strip().str.lower()
    columns = {
        "опыт": strain + " | " + culture,
        "штамм": strain,
        "клетки": culture,
    }
    if with_title:
        columns["подпись"] = table["title"].map(title_kind)
    return pd.DataFrame(columns)


def learn_encoding(labels, values):
    scale = np.log1p(values)
    overall = float(np.mean(scale))
    tables = {}
    for column in labels.columns:
        frame = pd.DataFrame({"label": labels[column].to_numpy(), "value": scale})
        grouped = frame.groupby("label")["value"].agg(["sum", "count"])
        smoothed = (grouped["sum"] + overall * SMOOTHING) / (grouped["count"] + SMOOTHING)
        tables[column] = (smoothed.to_dict(), grouped["count"].to_dict())
    return tables, overall


def apply_encoding(labels, tables, overall):
    columns = []
    for column in labels.columns:
        means, counts = tables[column]
        values = labels[column].to_numpy()
        columns.append(np.array([means.get(name, overall) for name in values]))
        columns.append(np.array([counts.get(name, 0) for name in values], dtype=float))
    return np.column_stack(columns)
