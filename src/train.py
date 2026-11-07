from __future__ import annotations

import argparse
import json
from pathlib import Path

import quiet

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

from context import apply_encoding, labels_of, learn_encoding
from features import DESCRIPTOR_NAMES, assay_matrix, assay_vocabulary, build_matrix
from neighbors import read_across

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


def target_ceiling(target):
    if not target.startswith("logcap"):
        return None
    tail = target[len("logcap"):]
    return float(tail) if tail else 5000.0


def to_model_scale(selectivity, target):
    limit = target_ceiling(target)
    if limit is not None:
        return np.log1p(np.clip(selectivity, 0.0, limit))
    return np.log1p(selectivity) if target == "log" else selectivity


def from_model_scale(value, target, correction=1.0):
    on_log = target == "log" or target.startswith("logcap")
    back = np.expm1(value) * correction if on_log else value * correction
    return np.clip(back, 0.0, None)


def mean_correction(actual, predicted_on_log_scale, target, mode, limit=None):
    if not (target == "log" or target.startswith("logcap")) or mode in (None, False, "none"):
        return 1.0
    if mode is True or mode == "smearing":
        return float(np.mean(np.exp(np.log1p(actual) - predicted_on_log_scale)))
    wanted = actual if limit is None else np.clip(actual, 0.0, limit)
    plain = from_model_scale(predicted_on_log_scale, target)
    weight = float(np.sum(plain ** 2))
    return float(np.sum(wanted * plain) / weight) if weight > 0 else 1.0


def quality(actual, predicted):
    error = predicted - actual
    error_on_log = np.log1p(predicted) - np.log1p(actual)
    spread = np.sum((actual - actual.mean()) ** 2)
    capped = pd.Series(np.clip(actual, 0.0, 5000.0))
    return {
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error ** 2) / spread) if spread else float("nan"),
        "mae_log": float(np.mean(np.abs(error_on_log))),
        "pearson": float(pd.Series(actual).corr(pd.Series(predicted), method="pearson")),
        "pearson_capped": float(capped.corr(pd.Series(predicted), method="pearson")),
        "spearman": float(pd.Series(actual).corr(pd.Series(predicted), method="spearman")),
    }


def base_features(table, config):
    shape = config["morgan"]
    numbers = build_matrix(table["smiles_canonical"], shape["radius"], shape["size"],
                           bool(shape.get("chirality")), bool(shape.get("counts")))
    bits = numbers[:, len(DESCRIPTOR_NAMES):]
    vocabulary = []
    if config.get("use_assay_context"):
        vocabulary = assay_vocabulary(table["strain"], table["culture"],
                                      keep=config.get("assay_vocabulary_size", 25))
        numbers = np.hstack([numbers, assay_matrix(table["strain"], table["culture"], vocabulary)])
    return numbers, bits, vocabulary


def add_extras(numbers, bits, labels, selectivity, learn_rows, apply_rows, config,
               reference_rows=None):
    if reference_rows is None:
        reference_rows = learn_rows
    columns = [numbers[apply_rows]]
    if config.get("use_neighbours"):
        same = len(apply_rows) == len(reference_rows) and np.array_equal(apply_rows, reference_rows)
        columns.append(read_across(bits[apply_rows], bits[reference_rows],
                                   selectivity[reference_rows], skip_self=same))
    if config.get("use_context_encoding"):
        tables, overall = learn_encoding(labels.iloc[learn_rows], selectivity[learn_rows])
        columns.append(apply_encoding(labels.iloc[apply_rows], tables, overall))
    return np.hstack(columns)


def cross_check(numbers, bits, labels, selectivity, structures, config, target, checkable=None,
                keep=None):
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
            learn_numbers = add_extras(numbers, bits, labels, selectivity, learn_rows, learn_rows, config)
            check_numbers = add_extras(numbers, bits, labels, selectivity, learn_rows, check_rows, config)
            model = new_model(config["regressor"], config["seed"])
            model.fit(learn_numbers, to_model_scale(selectivity[learn_rows], target))
            predicted[repeat, check_rows] = model.predict(check_numbers)
            if keep is not None:
                keep.append(model)
    return predicted.mean(axis=0)


def print_quality(title, values):
    print(f"{title}: MSE {values['mse']:,.0f} | MAE {values['mae']:.2f} | R2 {values['r2']:.3f} "
          f"| корреляция при потолке 5000 {values['pearson_capped']:.3f} "
          f"| Spearman {values['spearman']:.3f}")


def main():
    quiet.step_aside()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--table", default="train.csv")
    args = parser.parse_args()
    config = read_config(args.config)
    target = config["target"]
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(PROCESSED / args.table)
    selectivity = table["si_vacv"].to_numpy(dtype=float)
    structures = table["structure_key"].astype(str).to_numpy()
    labels = labels_of(table)
    numbers, bits, vocabulary = base_features(table, config)
    from_organizers = (table["source"] == "organizers").to_numpy()
    print(f"молекул: {len(table)} (из них внешних {int((~from_organizers).sum())}), "
          f"признаков у каждой: {numbers.shape[1]}, цель: {target}")
    print(f"соседи: {bool(config.get('use_neighbours'))}, "
          f"кодирование опыта: {bool(config.get('use_context_encoding'))}")

    fold_models = []
    on_model_scale = cross_check(numbers, bits, labels, selectivity, structures, config,
                                 target, from_organizers, keep=fold_models)
    checked = selectivity[from_organizers]
    highest = float(config.get("ceiling") or np.percentile(selectivity, 99))
    correction = mean_correction(checked, on_model_scale[from_organizers], target,
                                 config.get("correction", config.get("mean_correction", False)),
                                 limit=highest)
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

    everything = np.arange(len(table))
    full_numbers = add_extras(numbers, bits, labels, selectivity, everything, everything, config)
    model = new_model(config["regressor"], config["seed"])
    model.fit(full_numbers, to_model_scale(selectivity, target))
    model.save_model(str(MODELS / "model.cbm"))
    folds_here = MODELS / "folds"
    for old in folds_here.glob("*.cbm"):
        old.unlink()
    folds_here.mkdir(parents=True, exist_ok=True)
    for number, piece in enumerate(fold_models):
        piece.save_model(str(folds_here / f"model_{number:02d}.cbm"))
    tables, overall = learn_encoding(labels, selectivity)
    (MODELS / "feature_config.json").write_text(
        json.dumps({"morgan": config["morgan"], "target": target, "correction": correction,
                    "assay_vocabulary": vocabulary,
                    "use_neighbours": bool(config.get("use_neighbours")),
                    "use_context_encoding": bool(config.get("use_context_encoding")),
                    "context_tables": {name: [means, counts] for name, (means, counts) in tables.items()},
                    "context_overall": overall,
                    "ceiling": highest,
                    "folds": len(fold_models)}, ensure_ascii=False), encoding="utf-8")

    print()
    print_quality("модель с поправкой ", values)
    print_quality("модель без поправки", quality(checked, plain))
    print_quality("среднее по train   ", constant)
    print(f"\nмодель сохранена: {MODELS / 'model.cbm'}")


if __name__ == "__main__":
    main()
