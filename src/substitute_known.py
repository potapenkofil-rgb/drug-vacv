from __future__ import annotations

import argparse

import pandas as pd

from train import PROCESSED

COLUMN = "SI VACV"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train = pd.read_csv(PROCESSED / "train.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    answer = pd.read_csv(args.predictions)

    organizers = train[train["source"] == "organizers"]
    known = organizers.groupby("structure_key")["si_vacv"].median()
    known = known[known > 0]
    found = test[["index", "structure_key"]].copy()
    found["known"] = found["structure_key"].map(known)
    found = found.dropna(subset=["known"]).set_index("index")["known"]

    before = answer.set_index("index")[COLUMN]
    changed = answer["index"].isin(found.index)
    answer.loc[changed, COLUMN] = answer.loc[changed, "index"].map(found).to_numpy()
    answer.to_csv(args.output, index=False, encoding="utf-8")

    moved = (before.loc[found.index] - found).abs()
    print(f"молекул теста с известной структурой: {len(found)}")
    print(f"средний сдвиг прогноза: {moved.mean():.1f}, наибольший: {moved.max():.1f}")
    print(f"{args.output} записан")


if __name__ == "__main__":
    main()
