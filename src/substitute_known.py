from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features import structure_key
from prepare_data import TARGET
from train import PROCESSED

RAW = Path("data/raw")
COLUMN = "SI VACV"


def known_values():
    table = pd.read_csv(RAW / "train_v2.csv")
    keys = [structure_key(s) for s in table["SMILES"]]
    values = pd.to_numeric(table[TARGET], errors="coerce")
    frame = pd.DataFrame({"structure_key": keys, "si_vacv": values}).dropna()
    frame = frame[frame["si_vacv"] > 0]
    counted = frame.groupby("structure_key")["si_vacv"].agg(["median", "size", "min", "max"])
    return counted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train = pd.read_csv(PROCESSED / "train.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    answer = pd.read_csv(args.predictions)

    counted = known_values()
    spread = counted[counted["max"] > counted["min"] * 1.5]
    known = counted["median"]
    found = test[["index", "structure_key"]].copy()
    found["known"] = found["structure_key"].map(known)
    found = found.dropna(subset=["known"]).set_index("index")["known"]

    before = answer.set_index("index")[COLUMN]
    changed = answer["index"].isin(found.index)
    answer.loc[changed, COLUMN] = answer.loc[changed, "index"].map(found).to_numpy()
    answer.to_csv(args.output, index=False, encoding="utf-8")

    moved = (before.loc[found.index] - found).abs()
    print(f"молекул теста с известной структурой: {len(found)}")
    print(f"структур с несколькими замерами: {int((counted['size'] > 1).sum())}, "
          f"из них расходятся больше чем в 1,5 раза: {len(spread)}")
    print(f"средний сдвиг прогноза: {moved.mean():.1f}, наибольший: {moved.max():.1f}")
    print(f"{args.output} записан")


if __name__ == "__main__":
    main()
