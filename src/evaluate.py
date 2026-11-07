from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from train import metrics

REPORTS = Path("reports")


def main():
    oof = pd.read_csv(REPORTS / "oof_predictions.csv")
    actual = oof["si_vacv"].to_numpy(dtype=float)
    rows = []
    for column, name in (("pred_two_stage", "двухэтапная модель"), ("pred_single", "одна регрессия")):
        m = metrics(actual, oof[column].to_numpy(dtype=float))
        m["модель"] = name
        rows.append(m)
    table = pd.DataFrame(rows).set_index("модель")
    print(table.round(4).to_string())

    zero = actual == 0
    print(f"\nдоля нулевых SI в обучении: {zero.mean():.1%}")
    print(f"средний прогноз на нулевых строках: {oof.loc[zero, 'pred_two_stage'].mean():.2f}")
    print(f"средний прогноз на ненулевых строках: {oof.loc[~zero, 'pred_two_stage'].mean():.2f}")

    path = REPORTS / "cv_metrics.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        print(f"\nфолдов: {saved['folds']}, строк: {saved['rows']}, seed: {saved['seed']}")


if __name__ == "__main__":
    main()
