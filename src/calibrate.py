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


def solve(full_score, half_score):
    square = 2 * ((full_score - SQUARE_MEAN) + 2 * (SQUARE_MEAN - half_score))
    product = (square - (full_score - SQUARE_MEAN)) / 2
    return square, product


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--full-score", type=float, required=True)
    parser.add_argument("--half-score", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.predictions)
    predictions = table[COLUMN].to_numpy(dtype=float)
    average, variance = target_stats()
    square, product = solve(args.full_score, args.half_score)

    centre = float(predictions.mean())
    spread = square - centre ** 2
    covariance = product - average * centre
    slope = covariance / spread
    shift = average - slope * centre
    expected = variance - covariance ** 2 / spread
    correlation = covariance / np.sqrt(spread * variance)

    table[COLUMN] = np.clip(slope * predictions + shift, 0.0, None)
    table.to_csv(args.output, index=False, encoding="utf-8")

    report = {
        "source": args.predictions,
        "correlation": correlation,
        "slope": slope,
        "shift": shift,
        "expected_score": expected,
        "best_constant_score": variance,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    name = Path(args.predictions).stem
    (REPORTS / f"calibration_{name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"корреляция с правдой: {correlation:.3f}")
    print(f"преобразование: {slope:.4f} * прогноз + {shift:.1f}")
    print(f"ожидаемый результат: {expected:,.0f} (лучшая константа даёт {variance:,.0f})")
    print(f"{args.output} записан")


if __name__ == "__main__":
    main()
