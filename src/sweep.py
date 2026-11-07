from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import assay_matrix, assay_vocabulary, build_matrix
from train import PROCESSED, REPORTS, cross_check, from_model_scale, quality, read_config

VARIANTS = [
    ("как сейчас", {}),
    ("отпечаток 1024", {"morgan": {"radius": 2, "size": 1024}}),
    ("отпечаток 4096", {"morgan": {"radius": 2, "size": 4096}}),
    ("радиус 3", {"morgan": {"radius": 3, "size": 2048}}),
    ("глубина 4", {"regressor": {"depth": 4}}),
    ("глубина 8", {"regressor": {"depth": 8}}),
    ("деревьев 400", {"regressor": {"iterations": 400}}),
    ("деревьев 2500, шаг 0.015", {"regressor": {"iterations": 2500, "learning_rate": 0.015}}),
    ("сильная регуляризация", {"regressor": {"l2_leaf_reg": 20.0}}),
    ("без условий опыта", {"use_assay_context": False}),
]


def merge(base, changes):
    out = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict):
            out[key].update(value)
        else:
            out[key] = value
    return out


def main():
    base = read_config()
    base["cv"]["repeats"] = 1
    table = pd.read_csv(PROCESSED / "train.csv")
    selectivity = table["si_vacv"].to_numpy(dtype=float)
    structures = table["structure_key"].astype(str).to_numpy()
    from_organizers = (table["source"] == "organizers").to_numpy()
    checked = selectivity[from_organizers]

    cache = {}
    results = []
    for name, changes in VARIANTS:
        config = merge(base, changes)
        key = (config["morgan"]["radius"], config["morgan"]["size"], bool(config.get("use_assay_context")))
        if key not in cache:
            numbers = build_matrix(table["smiles_canonical"], key[0], key[1])
            if key[2]:
                vocabulary = assay_vocabulary(table["strain"], table["culture"])
                numbers = np.hstack([numbers, assay_matrix(table["strain"], table["culture"], vocabulary)])
            cache[key] = numbers
        numbers = cache[key]

        raw = cross_check(numbers, selectivity, structures, config, "log", from_organizers)
        predicted = from_model_scale(raw[from_organizers], "log")
        values = quality(checked, predicted)
        values["вариант"] = name
        results.append(values)
        print(f"{name}: корреляция {values['pearson']:.3f}, Spearman {values['spearman']:.3f}")

    report = pd.DataFrame(results).set_index("вариант")[["pearson", "spearman", "mae", "mae_log"]]
    report = report.sort_values("pearson", ascending=False)
    print()
    print(report.round(4).to_string())
    (REPORTS / "sweep.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
