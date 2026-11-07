from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold

import train
from features import build_matrix
from train import PROCESSED, MODELS, REPORTS, new_model, print_quality, quality, read_config

PARTS = MODELS / "parts"


def new_classifier(settings, seed):
    return CatBoostClassifier(
        iterations=settings["iterations"],
        depth=settings["depth"],
        learning_rate=settings["learning_rate"],
        l2_leaf_reg=settings["l2_leaf_reg"],
        random_seed=seed,
        thread_count=train.THREADS,
        verbose=0,
        allow_writing_files=False,
    )


def fit_on_measured(numbers, values, rows, settings, seed):
    known = rows & np.isfinite(values) & (values > 0)
    model = new_model(settings, seed)
    model.fit(numbers[known], np.log10(values[known]))
    return model


def predict_selectivity(models, numbers):
    activity, toxicity, classifier = models
    ratio = 10.0 ** (toxicity.predict(numbers) - activity.predict(numbers))
    return np.clip(classifier.predict_proba(numbers)[:, 1] * ratio, 0.0, None)


def fit_all(numbers, table, rows, settings, seed):
    activity = fit_on_measured(numbers, table["ic50_um_vacv"].to_numpy(dtype=float), rows, settings, seed)
    toxicity = fit_on_measured(numbers, table["cc50_um"].to_numpy(dtype=float), rows, settings, seed)
    classifier = new_classifier(settings, seed)
    classifier.fit(numbers[rows], (table["si_vacv"].to_numpy(dtype=float)[rows] > 0).astype(int))
    return activity, toxicity, classifier


def scale_to_mean(actual, predicted):
    return float(np.mean((actual + 1.0) / (predicted + 1.0)))


def main():
    config = read_config()
    seed = config["seed"]
    PARTS.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(PROCESSED / "train.csv")
    selectivity = table["si_vacv"].to_numpy(dtype=float)
    structures = table["structure_key"].astype(str).to_numpy()
    numbers = build_matrix(table["smiles_canonical"], config["morgan"]["radius"], config["morgan"]["size"])
    from_organizers = (table["source"] == "organizers").to_numpy()

    measured_ic50 = int(np.isfinite(table["ic50_um_vacv"]).sum())
    measured_cc50 = int(np.isfinite(table["cc50_um"]).sum())
    print(f"молекул: {len(table)}, с измеренным IC50: {measured_ic50}, с измеренным CC50: {measured_cc50}")

    main_rows = np.where(from_organizers)[0]
    extra_rows = np.where(~from_organizers)[0]
    repeats = config["cv"].get("repeats", 1)
    predicted = np.zeros((repeats, len(table)))
    for repeat in range(repeats):
        splitter = GroupKFold(n_splits=config["cv"]["folds"], shuffle=True, random_state=seed + repeat)
        for learn_part, check_part in splitter.split(main_rows, selectivity[main_rows], structures[main_rows]):
            learn_rows = np.zeros(len(table), dtype=bool)
            learn_rows[np.concatenate([main_rows[learn_part], extra_rows])] = True
            models = fit_all(numbers, table, learn_rows, config["regressor"], seed)
            predicted[repeat, main_rows[check_part]] = predict_selectivity(models, numbers[main_rows[check_part]])

    checked = selectivity[from_organizers]
    raw = predicted.mean(axis=0)[from_organizers]
    correction = scale_to_mean(checked, raw)
    with_correction = np.clip((raw + 1.0) * correction - 1.0, 0.0, None)

    values = quality(checked, with_correction)
    (REPORTS / "cv_metrics_parts.json").write_text(json.dumps({
        "rows": len(table),
        "rows_checked": int(from_organizers.sum()),
        "measured_ic50": measured_ic50,
        "measured_cc50": measured_cc50,
        "correction": correction,
        "model": values,
        "model_without_correction": quality(checked, raw),
        "constant_mean": quality(checked, np.full(len(checked), checked.mean())),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    everything = np.ones(len(table), dtype=bool)
    activity, toxicity, classifier = fit_all(numbers, table, everything, config["regressor"], seed)
    activity.save_model(str(PARTS / "ic50.cbm"))
    toxicity.save_model(str(PARTS / "cc50.cbm"))
    classifier.save_model(str(PARTS / "active.cbm"))
    (MODELS / "parts_config.json").write_text(json.dumps({
        "approach": "parts",
        "morgan": config["morgan"],
        "target": "log",
        "correction": correction,
    }, ensure_ascii=False), encoding="utf-8")

    print(f"поправка на среднее: {correction:.2f}")
    print_quality("модель по частям   ", values)
    print_quality("без поправки       ", quality(checked, raw))
    print_quality("среднее по train   ", quality(checked, np.full(len(checked), checked.mean())))


if __name__ == "__main__":
    main()
