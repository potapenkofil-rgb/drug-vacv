from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features import canonical_smiles, structure_key

RAW = Path("data/raw")
EXTERNAL = Path("data/external")
PROCESSED = Path("data/processed")

TARGET = "SI VACV"


def selectivity(cc50, ic50):
    if pd.isna(cc50) or pd.isna(ic50) or float(ic50) == 0.0:
        return 0.0
    return float(cc50) / float(ic50)


def shape(frame, labelled):
    out = pd.DataFrame()
    if "index" in frame.columns:
        out["index"] = frame["index"]
    out["title"] = frame["Title"].astype(str)
    out["smiles_original"] = frame["SMILES"].astype(str)
    out["smiles_canonical"] = [canonical_smiles(s) for s in frame["SMILES"]]
    out["structure_key"] = [structure_key(s) for s in frame["SMILES"]]
    out["cc50_um"] = pd.to_numeric(frame["CC50_uM"], errors="coerce")
    out["ic50_um_vacv"] = pd.to_numeric(frame["IC50_uM_VACV"], errors="coerce")
    out["strain"] = frame["strain"].astype(str)
    out["culture"] = frame["Culture"].astype(str)
    out["source"] = "organizers"
    if labelled:
        out["si_vacv"] = pd.to_numeric(frame[TARGET], errors="coerce")
    out["structure_key"] = out["structure_key"].fillna(out["smiles_original"])
    return out


def load_external():
    files = sorted(EXTERNAL.glob("*.csv"))
    if not files:
        return None
    pieces = []
    needed = {"smiles", "cc50_um", "ic50_um", "si"}
    for path in files:
        piece = pd.read_csv(path)
        if not needed.issubset(piece.columns):
            raise ValueError(f"{path}: ожидались колонки {sorted(needed)}")
        print(f"  {path.name}: {len(piece)} записей")
        pieces.append(piece)
    frame = pd.concat(pieces, ignore_index=True)
    out = pd.DataFrame()
    out["title"] = frame.get("molecule_chembl_id", pd.Series(["external"] * len(frame)))
    out["smiles_original"] = frame["smiles"].astype(str)
    out["smiles_canonical"] = [canonical_smiles(s) for s in frame["smiles"]]
    out["structure_key"] = [structure_key(s) for s in frame["smiles"]]
    out["cc50_um"] = pd.to_numeric(frame["cc50_um"], errors="coerce")
    out["ic50_um_vacv"] = pd.to_numeric(frame["ic50_um"], errors="coerce")
    out["strain"] = frame.get("assay_description", pd.Series([""] * len(frame))).astype(str)
    out["culture"] = ""
    out["source"] = "chembl"
    out["si_vacv"] = pd.to_numeric(frame["si"], errors="coerce")
    out = out[out["smiles_canonical"].notna()]
    out["structure_key"] = out["structure_key"].fillna(out["smiles_original"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-external", action="store_true")
    args = parser.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train = shape(pd.read_csv(RAW / "train_v2.csv"), True)
    test = shape(pd.read_csv(RAW / "test.csv"), False)

    train["si_rule"] = [selectivity(c, i) for c, i in zip(train["cc50_um"], train["ic50_um_vacv"])]
    train["target_matches_rule"] = (train["si_vacv"] - train["si_rule"]).abs() < 0.01
    train = train[train["smiles_canonical"].notna()]
    test = test[test["smiles_canonical"].notna()]

    before = len(train)
    train = train.sort_values("si_vacv", ascending=False).drop_duplicates("structure_key", keep="first")
    dropped = before - len(train)

    added = 0
    removed_as_test = 0
    removed_out_of_range = 0
    if args.with_external:
        external = load_external()
        if external is not None:
            external = external.drop_duplicates("structure_key", keep="first")
            highest = train["si_vacv"].max()
            in_range = external["si_vacv"].between(0, highest) & external["si_vacv"].notna()
            removed_out_of_range = int((~in_range).sum())
            external = external[in_range]
            in_test = external["structure_key"].isin(set(test["structure_key"]))
            removed_as_test = int(in_test.sum())
            external = external[~in_test]
            external = external[~external["structure_key"].isin(set(train["structure_key"]))]
            train = pd.concat([train, external], ignore_index=True)
            added = len(external)

    train.to_csv(PROCESSED / "train.csv", index=False, encoding="utf-8")
    test.to_csv(PROCESSED / "test.csv", index=False, encoding="utf-8")
    print(f"train: {len(train)} строк (дубликатов структуры убрано {dropped}, из внешних источников добавлено {added})")
    if args.with_external:
        print(f"внешних молекул отброшено как совпадающие с test: {removed_as_test}")
        print(f"внешних молекул отброшено с SI вне диапазона обучения: {removed_out_of_range}")
    print(f"test: {len(test)} строк")
    print(f"целевая переменная совпадает с CC50/IC50: {int(train['target_matches_rule'].fillna(False).sum())} из {len(train)}")


if __name__ == "__main__":
    main()
