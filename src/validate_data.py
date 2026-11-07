from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from features import parse, structure_key

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")

REQUIRED_TRAIN = ["Title", "CC50_uM", "IC50_uM_VACV", "SI VACV", "SMILES", "strain", "Culture"]
REQUIRED_TEST = ["index", "Title", "CC50_uM", "IC50_uM_VACV", "SMILES", "strain", "Culture"]


def check_columns(frame, required, name, problems):
    missing = [c for c in required if c not in frame.columns]
    if missing:
        problems.append(f"{name}: нет колонок {missing}")


def smiles_report(frame, name, problems):
    bad = [s for s in frame["SMILES"] if parse(str(s)) is None]
    if bad:
        problems.append(f"{name}: не разбирается SMILES — {len(bad)} шт.")
    return len(bad)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    problems = []
    train = pd.read_csv(RAW / "train_v2.csv")
    test = pd.read_csv(RAW / "test.csv")
    submit = pd.read_csv(RAW / "sample_submit.csv")

    check_columns(train, REQUIRED_TRAIN, "train", problems)
    check_columns(test, REQUIRED_TEST, "test", problems)

    bad_train = smiles_report(train, "train", problems)
    bad_test = smiles_report(test, "test", problems)

    keys_train = [structure_key(str(s)) for s in train["SMILES"]]
    keys_test = [structure_key(str(s)) for s in test["SMILES"]]
    overlap = sorted(set(keys_train) & set(keys_test) - {None})

    cc50 = pd.to_numeric(train["CC50_uM"], errors="coerce")
    ic50 = pd.to_numeric(train["IC50_uM_VACV"], errors="coerce")
    target = pd.to_numeric(train["SI VACV"], errors="coerce")
    rule = (cc50 / ic50).where(cc50.notna() & ic50.notna() & (ic50 != 0), 0.0)
    mismatch = (rule - target).abs() > 0.01

    negative = int(((cc50 < 0) | (ic50 < 0)).sum())
    if negative:
        problems.append(f"train: отрицательные концентрации — {negative} шт.")
    if int(mismatch.sum()):
        problems.append(f"train: SI не равен CC50/IC50 — {int(mismatch.sum())} строк")
    if len(submit) != len(test):
        problems.append(f"sample_submit ({len(submit)}) и test ({len(test)}) разной длины")

    audit = {
        "train_rows": len(train),
        "test_rows": len(test),
        "submission_rows": len(submit),
        "submission_columns": list(submit.columns),
        "train_bad_smiles": bad_train,
        "test_bad_smiles": bad_test,
        "train_duplicate_structures": len(keys_train) - len(set(keys_train)),
        "test_duplicate_structures": len(keys_test) - len(set(keys_test)),
        "structures_in_both_train_and_test": len(overlap),
        "train_missing_ic50": int(ic50.isna().sum()),
        "train_missing_cc50": int(cc50.isna().sum()),
        "train_zero_target": int((target == 0).sum()),
        "train_zero_target_without_ic50": int(((target == 0) & ic50.isna()).sum()),
        "target_equals_cc50_over_ic50": int((~mismatch).sum()),
        "target_min": float(target.min()),
        "target_median": float(target.median()),
        "target_max": float(target.max()),
        "test_rows_with_both_concentrations": int((pd.to_numeric(test["CC50_uM"], errors="coerce").notna()
                                                   & pd.to_numeric(test["IC50_uM_VACV"], errors="coerce").notna()).sum()),
        "problems": problems,
    }

    (REPORTS / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, value in audit.items():
        if key != "problems":
            print(f"{key}: {value}")
    if problems:
        print("\nзамечания:")
        for p in problems:
            print(" -", p)
    else:
        print("\nзамечаний нет")


if __name__ == "__main__":
    main()
