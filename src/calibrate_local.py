from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path("reports")
COLUMN = "SI VACV"
SQUARE_MEAN = 2123511.05353
CONSTANT = 300.0
CONSTANT_SCORE = 2019483.47407


def target_stats():
    average = (SQUARE_MEAN + CONSTANT ** 2 - CONSTANT_SCORE) / (2 * CONSTANT)
    return average, SQUARE_MEAN - average ** 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--ceiling", type=float, default=5000.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = pd.read_csv(args.oof)
    truth = np.clip(reference.iloc[:, 1].to_numpy(dtype=float), 0.0, args.ceiling)
    checked = reference.iloc[:, 3].to_numpy(dtype=float)
    slope, shift = np.polyfit(checked, truth, 1)
    correlation = float(np.corrcoef(checked, truth)[0, 1])

    table = pd.read_csv(args.predictions)
    values = table[COLUMN].to_numpy(dtype=float)
    moved = slope * values + shift
    trimmed = int((moved > args.ceiling).sum())
    table[COLUMN] = np.clip(moved, 0.0, args.ceiling)
    table.to_csv(args.output, index=False, encoding="utf-8")

    average, variance = target_stats()
    expected = variance * (1 - correlation ** 2)
    report = {"oof": args.oof, "slope": float(slope), "shift": float(shift),
              "correlation": correlation, "expected_score": float(expected),
              "ceiling": args.ceiling, "trimmed": trimmed}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"calibration_local_{Path(args.predictions).stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"корреляция вне своей части разбиения: {correlation:.4f}")
    print(f"преобразование: {slope:.4f} * прогноз + {shift:.1f}")
    print(f"срезано потолком {args.ceiling:,.0f}: {trimmed} строк")
    print(f"ожидаемый результат: {expected:,.0f} (лучшая константа даёт {variance:,.0f})")
    print(f"прогноз от {table[COLUMN].min():.2f} до {table[COLUMN].max():,.1f}, "
          f"среднее {table[COLUMN].mean():.1f}")
    print(f"{args.output} записан")


if __name__ == "__main__":
    main()
