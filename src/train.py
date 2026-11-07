from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

from features import assay_matrix, assay_vocabulary, build_matrix

PROCESSED = Path("data/processed")
MODELS = Path("models")
REPORTS = Path("reports")


THREADS = 5


def read_config(path="configs/baseline.yaml"):
    global THREADS
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    THREADS = int(config.get("threads", 5))
    return config


def new_model(settings, seed):
    return CatBoostRegressor(
        iterations=settings["iterations"],
        depth=settings["depth"],
        learning_rate=settings["learning_rate"],
        l2_leaf_reg=settings["l2_leaf_reg"],
        loss_function="RMSE",
        random_seed=seed,
        thread_count=THREADS,
        verbose=0,
        allow_writing_files=False,
    )


def to_model_scale(selectivity, target):
    return np.log1p(selectivity) if target == "log" else selectivity


def from_model_scale(value, target, correction=1.0):
    back = np.exp(value) * correction - 1.0 if target == "log" else value * correction
    return np.clip(back, 0.0, None)


def mean_correction(actual, predicted_on_log_scale, target, enabled):
    if target != "log" or not enabled:
        return 1.0
    return float(np.mean(np.exp(np.log1p(actual) - predicted_on_log_scale)))


def quality(actual, predicted):
    error = predicted - actual
    error_on_log = np.log1p(predicted) - np.log1p(actual)
    spread = np.sum((actual - actual.mean()) ** 2)
    return {
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error ** 2) / spread) if spread else float("nan"),
        "mae_log": float(np.mean(np.abs(error_on_log))),
        "spearman": float(pd.Series(actual).corr(pd.Series(predicted), method="spearman")),
    }


def cross_check(numbers, selectivity, structures, config, target, checkable=None):
    folds = config["cv"]["folds"]
    repeats = config["cv"].get("repeats", 1)
    if checkable is None:
        checkable = np.ones(len(selectivity), dtype=bool)
    main_rows = np.where(checkable)[0]
    extra_rows = np.where(~checkable)[0]
    predicted = np.zeros((repeats, len(selectivity)))
    for repeat in range(repeats):
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=config["seed"] + repeat)
        for learn_part, check_part in splitter.split(main_rows, selectivity[main_rows], structures[main_rows]):
            learn_rows = np.concatenate([main_rows[learn_part], extra_rows])
            check_rows = main_rows[check_part]
            model = new_model(config["regressor"], config["seed"])
            model.fit(numbers[learn_rows], to_model_scale(selectivity[learn_rows], target))
            predicted[repeat, check_rows] = model.predict(numbers[check_rows])
    return predicted.mean(axis=0)


def print_quality(title, values):
    print(f"{title}: MSE {values['mse']:,.0f} | MAE {values['mae']:.2f} | R2 {values['r2']:.3f} "
          f"| MAE на логарифме {values['mae_log']:.3f} | Spearman {values['spearman']:.3f}")


def main():
    config = read_config()
    target = config["target"]
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(PROCESSED / "train.csv")
    selectivity = table["si_vacv"].to_numpy(dtype=float)
    structures = table["structure_key"].astype(str).to_numpy()
    numbers = build_matrix(table["smiles_canonical"], config["morgan"]["radius"], config["morgan"]["size"])
    vocabulary = []
    if config.get("use_assay_context"):
        vocabulary = assay_vocabulary(table["strain"], table["culture"])
        numbers = np.hstack([numbers, assay_matrix(table["strain"], table["culture"], vocabulary)])
    from_organizers = (table["source"] == "organizers").to_numpy()
    print(f"молекул: {len(table)} (из них внешних {int((~from_organizers).sum())}), "
          f"признаков у каждой: {numbers.shape[1]}, цель: {target}")

    on_model_scale = cross_check(numbers, selectivity, structures, config, target, from_organizers)
    checked = selectivity[from_organizers]
    correction = mean_correction(checked, on_model_scale[from_organizers], target,
                                 config.get("mean_correction", False))
    predicted = from_model_scale(on_model_scale[from_organizers], target, correction)
    plain = from_model_scale(on_model_scale[from_organizers], target)
    values = quality(checked, predicted)
    constant = quality(checked, np.full(len(checked), checked.mean()))
    print(f"поправка на среднее: {correction:.2f}")

    pd.DataFrame({
        "structure_key": structures[from_organizers],
        "si_vacv": checked,
        "prediction": predicted,
    }).to_csv(REPORTS / "oof_predictions.csv", index=False, encoding="utf-8")
    (REPORTS / "cv_metrics.json").write_text(json.dumps({
        "folds": config["cv"]["folds"],
        "repeats": config["cv"].get("repeats", 1),
        "rows": len(table),
        "rows_checked": int(from_organizers.sum()),
        "seed": config["seed"],
        "target": target,
        "mean_correction": correction,
        "model": values,
        "model_without_correction": quality(checked, plain),
        "constant_mean": constant,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    model = new_model(config["regressor"], config["seed"])
    model.fit(numbers, to_model_scale(selectivity, target))
    model.save_model(str(MODELS / "model.cbm"))
    (MODELS / "feature_config.json").write_text(
        json.dumps({"morgan": config["morgan"], "target": target, "correction": correction,
                    "assay_vocabulary": vocabulary}, ensure_ascii=False), encoding="utf-8")

    print()
    print_quality("модель с поправкой ", values)
    print_quality("модель без поправки", quality(checked, plain))
    print_quality("среднее по train   ", constant)
    print(f"\nмодель сохранена: {MODELS / 'model.cbm'}")


if __name__ == "__main__":
    main()
