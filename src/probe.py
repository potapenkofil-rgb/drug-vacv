from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

COLUMN = "SI VACV"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.predictions)
    table[COLUMN] = np.clip(table[COLUMN].to_numpy(dtype=float) * args.factor, 0.0, None)
    table.to_csv(args.output, index=False, encoding="utf-8")
    print(f"{args.output}: прогноз умножен на {args.factor}")


if __name__ == "__main__":
    main()
