from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train import REPORTS

CAP = 5000.0


def load():
    files = sorted(p for p in REPORTS.glob("oof_*.csv") if p.name != "oof_predictions.csv")
    collected = {}
    actual = None
    for path in files:
        table = pd.read_csv(path)
        if actual is None:
            actual = table["si_vacv"].to_numpy(dtype=float)
        collected[path.stem[4:].replace("_", " ")] = table["на логарифме"].to_numpy(dtype=float)
    return actual, collected


def score(actual, on_log):
    predicted = np.clip(np.exp(on_log) - 1.0, 0.0, None)
    goal = pd.Series(np.clip(actual, 0.0, CAP))
    return float(goal.corr(pd.Series(predicted)))


def steady(actual, on_log, parts=8, seed=0):
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(actual))
    pieces = np.array_split(order, parts)
    values = [score(actual[piece], on_log[piece]) for piece in pieces]
    return float(np.mean(values)), float(np.std(values) / np.sqrt(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    actual, collected = load()
    if not collected:
        raise SystemExit("нет файлов reports/oof_*.csv")

    rows = []
    for name, on_log in collected.items():
        middle, spread = steady(actual, on_log)
        rows.append({"вариант": name, "корреляция": score(actual, on_log),
                     "по кускам": middle, "разброс": spread})
    print(pd.DataFrame(rows).set_index("вариант").sort_values("корреляция", ascending=False).round(4).to_string())

    chosen = []
    best = -1.0
    for _ in range(args.steps):
        candidate, value = None, best
        for name, on_log in collected.items():
            mix = np.mean([collected[part] for part in chosen + [name]], axis=0)
            here = score(actual, mix)
            if here > value:
                candidate, value = name, here
        if candidate is None:
            break
        chosen.append(candidate)
        best = value
        print(f"смесь из {len(chosen)}: {' + '.join(chosen)} -> {best:.4f}")

    if chosen:
        mix = np.mean([collected[part] for part in chosen], axis=0)
        middle, spread = steady(actual, mix)
        print(f"\nлучшая смесь: {' + '.join(chosen)}")
        print(f"корреляция {best:.4f} (по кускам {middle:.4f} ± {spread:.4f})")
        Path(REPORTS / "blend_choice.txt").write_text("\n".join(chosen), encoding="utf-8")


if __name__ == "__main__":
    main()
