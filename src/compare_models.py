from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_matrix
from train import cross_check, from_model_scale, mean_correction, quality, read_config

PROCESSED = Path("data/processed")
REPORTS = Path("reports")


def main():
    config = read_config()
    table = pd.read_csv(PROCESSED / "train.csv")
    selectivity = table["si_vacv"].to_numpy(dtype=float)
    structures = table["structure_key"].astype(str).to_numpy()
    numbers = build_matrix(table["smiles_canonical"], config["morgan"]["radius"], config["morgan"]["size"])

    from_organizers = (table["source"] == "organizers").to_numpy()
    checked = selectivity[from_organizers]
    results = {"среднее по train": quality(checked, np.full(len(checked), checked.mean()))}
    for target in ("raw", "log"):
        on_model_scale = cross_check(numbers, selectivity, structures, config, target, from_organizers)[from_organizers]
        results[f"CatBoost по цели {target}"] = quality(checked, from_model_scale(on_model_scale, target))
        correction = mean_correction(checked, on_model_scale, target, True)
        if correction != 1.0:
            results[f"CatBoost по цели {target} с поправкой"] = quality(
                checked, from_model_scale(on_model_scale, target, correction))
        print(f"посчитан вариант {target}")

    report = pd.DataFrame(results).T[["mse", "mae", "r2", "mae_log", "spearman"]]
    print()
    print(report.round(3).to_string())
    (REPORTS / "model_comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    best = report["mse"].idxmin()
    print(f"\nпо метрике соревнования MSE лучший вариант: {best}")


if __name__ == "__main__":
    main()
