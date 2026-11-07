from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

from context import apply_encoding, labels_of, learn_encoding
from features import DESCRIPTOR_NAMES, assay_matrix, assay_vocabulary, build_matrix
from neighbors import read_across
from train import PROCESSED, REPORTS, read_config

TEST_VARIANCE = 2018936.88
CEILING = 5000.0

PLAN = {
    "база": {},
    "соседи": {"neighbours": True},
    "контекст": {"context": True},
    "подпись": {"context": True, "title": True},
    "корень": {"scale": "sqrt"},
    "четверть": {"scale": "quart"},
    "вес слабый": {"weight": 0.15},
    "вес сильный": {"weight": 0.35},
    "глубина 4": {"regressor": {"depth": 4}},
    "глубина 8": {"regressor": {"depth": 8}},
    "регуляризация 20": {"regressor": {"l2_leaf_reg": 20.0}},
    "деревьев 2500": {"regressor": {"iterations": 2500, "learning_rate": 0.015}},
    "радиус 3": {"morgan": {"radius": 3, "size": 2048}},
    "отпечаток 4096": {"morgan": {"radius": 2, "size": 4096}},
    "три зерна": {"seeds": 3},
    "соседи и корень": {"neighbours": True, "scale": "sqrt"},
    "глубина 4 и корень": {"scale": "sqrt", "regressor": {"depth": 4}},
}


def slug(name):
    return name.replace(" ", "_")


def boosting(settings, seed, threads):
    return CatBoostRegressor(
        iterations=settings["iterations"],
        depth=settings["depth"],
        learning_rate=settings["learning_rate"],
        l2_leaf_reg=settings["l2_leaf_reg"],
        loss_function="RMSE",
        random_seed=seed,
        thread_count=threads,
        verbose=0,
        allow_writing_files=False,
    )


def forward(selectivity, scale):
    if scale == "sqrt":
        return np.sqrt(np.clip(selectivity, 0.0, CEILING))
    if scale == "quart":
        return np.clip(selectivity, 0.0, CEILING) ** 0.25
    return np.log1p(selectivity)


def backward(value, scale):
    if scale == "sqrt":
        return np.clip(value, 0.0, None) ** 2
    if scale == "quart":
        return np.clip(value, 0.0, None) ** 4
    return np.clip(np.exp(value) - 1.0, 0.0, None)


def extend(numbers, bits, labels, selectivity, learn_rows, apply_rows, options):
    columns = [numbers[apply_rows]]
    if options.get("neighbours"):
        same = np.array_equal(apply_rows, learn_rows)
        columns.append(read_across(bits[apply_rows], bits[learn_rows],
                                   selectivity[learn_rows], skip_self=same))
    if options.get("context"):
        tables, overall = learn_encoding(labels.iloc[learn_rows], selectivity[learn_rows])
        columns.append(apply_encoding(labels.iloc[apply_rows], tables, overall))
    return np.hstack(columns)


def run(name, options, numbers, bits, labels, selectivity, structures, checkable, config):
    repeats = config["cv"].get("repeats", 1)
    seeds = [config["seed"] + 101 * step for step in range(options.get("seeds", 1))]
    settings = dict(config["regressor"], **options.get("regressor", {}))
    scale = options.get("scale", "log")
    power = options.get("weight")
    main_rows = np.where(checkable)[0]
    extra_rows = np.where(~checkable)[0]
    collected = np.zeros((repeats, len(selectivity)))
    for repeat in range(repeats):
        splitter = GroupKFold(n_splits=config["cv"]["folds"], shuffle=True,
                              random_state=config["seed"] + repeat)
        for learn_part, check_part in splitter.split(main_rows, selectivity[main_rows], structures[main_rows]):
            learn_rows = np.concatenate([main_rows[learn_part], extra_rows])
            check_rows = main_rows[check_part]
            learn_numbers = extend(numbers, bits, labels, selectivity, learn_rows, learn_rows, options)
            check_numbers = extend(numbers, bits, labels, selectivity, learn_rows, check_rows, options)
            goal = forward(selectivity[learn_rows], scale)
            weights = None if power is None else (1.0 + selectivity[learn_rows]) ** power
            parts = []
            for seed in seeds:
                model = boosting(settings, seed, config["threads"])
                model.fit(learn_numbers, goal, sample_weight=weights)
                parts.append(backward(model.predict(check_numbers), scale))
            collected[repeat, check_rows] = np.mean(parts, axis=0)

    predicted = collected.mean(axis=0)[checkable]
    actual = selectivity[checkable]
    pd.DataFrame({"structure_key": structures[checkable], "si_vacv": actual,
                  "на логарифме": np.log1p(predicted), "прогноз": predicted}).to_csv(
        REPORTS / f"oof_{os.environ.get('TAG', '')}{slug(name)}.csv", index=False, encoding="utf-8")
    values = {"вариант": name}
    for cap in (CEILING, 2500.0):
        goal = pd.Series(np.clip(actual, 0.0, cap))
        values[f"потолок {cap:.0f}"] = float(goal.corr(pd.Series(predicted)))
    values["ранги"] = float(pd.Series(actual).corr(pd.Series(predicted), method="spearman"))
    values["ожидаемый MSE"] = TEST_VARIANCE * (1 - values["потолок 5000"] ** 2)
    print(f"{name}: потолок5000 {values['потолок 5000']:.3f} | потолок2500 "
          f"{values['потолок 2500']:.3f} | ранги {values['ранги']:.3f} "
          f"| ожидаемый MSE {values['ожидаемый MSE']:,.0f}", flush=True)
    return values


def main():
    wanted = sys.argv[1:] or list(PLAN)
    config = read_config(os.environ.get("CONFIG", "configs/baseline.yaml"))
    config["cv"]["repeats"] = int(os.environ.get("REPEATS", 2))
    config["threads"] = int(config.get("threads", 5))
    table = pd.read_csv(PROCESSED / os.environ.get("TABLE", "train.csv"))
    selectivity = table["si_vacv"].to_numpy(dtype=float)
    structures = table["structure_key"].astype(str).to_numpy()
    checkable = (table["source"] == "organizers").to_numpy()
    vocabulary = assay_vocabulary(table["strain"], table["culture"], keep=60)
    cache = {}

    def prepared(shape):
        if shape not in cache:
            built = build_matrix(table["smiles_canonical"], shape[0], shape[1])
            extra = assay_matrix(table["strain"], table["culture"], vocabulary)
            cache[shape] = (np.hstack([built, extra]), built[:, len(DESCRIPTOR_NAMES):])
        return cache[shape]

    print(f"молекул {len(table)}, опытов в словаре {len(vocabulary)}, "
          f"повторов {config['cv']['repeats']}", flush=True)
    results = []
    for name in wanted:
        options = PLAN[name]
        shape = options.get("morgan", config["morgan"])
        numbers, bits = prepared((shape["radius"], shape["size"]))
        labels = labels_of(table, with_title=bool(options.get("title")))
        results.append(run(name, options, numbers, bits, labels, selectivity,
                           structures, checkable, config))
    (REPORTS / "search.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(pd.DataFrame(results).set_index("вариант").sort_values("потолок 5000", ascending=False)
          .round(4).to_string())


if __name__ == "__main__":
    main()
