from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from context import apply_encoding, labels_of
from features import DESCRIPTOR_NAMES, assay_matrix, build_matrix
from neighbors import read_across
from train import from_model_scale

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
MODELS = Path("models")
SUBMISSIONS = Path("submissions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", action="store_true")
    parser.add_argument("--output", default="submissions/submission.csv")
    args = parser.parse_args()
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    settings = json.loads((MODELS / "feature_config.json").read_text(encoding="utf-8"))
    morgan = settings["morgan"]
    test = pd.read_csv(PROCESSED / "test.csv")
    train = pd.read_csv(PROCESSED / "train.csv")
    sample = pd.read_csv(RAW / "sample_submit.csv")
    highest_known = float(settings.get("ceiling") or train["si_vacv"].max())

    numbers = build_matrix(test["smiles_canonical"], morgan["radius"], morgan["size"])
    bits = numbers[:, len(DESCRIPTOR_NAMES):]
    vocabulary = settings.get("assay_vocabulary") or []
    if vocabulary:
        numbers = np.hstack([numbers, assay_matrix(test["strain"], test["culture"], vocabulary)])
    columns = [numbers]
    if settings.get("use_neighbours"):
        learned = build_matrix(train["smiles_canonical"], morgan["radius"], morgan["size"])
        columns.append(read_across(bits, learned[:, len(DESCRIPTOR_NAMES):],
                                   train["si_vacv"].to_numpy(dtype=float)))
    if settings.get("use_context_encoding"):
        tables = {name: (pair[0], pair[1]) for name, pair in settings["context_tables"].items()}
        columns.append(apply_encoding(labels_of(test), tables, settings["context_overall"]))
    numbers = np.hstack(columns)

    pieces = sorted((MODELS / "folds").glob("*.cbm")) if args.folds else []
    if pieces:
        raw = []
        for path in pieces:
            piece = CatBoostRegressor()
            piece.load_model(str(path))
            raw.append(piece.predict(numbers))
        on_model_scale = np.mean(raw, axis=0)
        print(f"усреднено моделей по частям: {len(pieces)}")
    else:
        model = CatBoostRegressor()
        model.load_model(str(MODELS / "model.cbm"))
        on_model_scale = model.predict(numbers)
    predicted = from_model_scale(on_model_scale, settings["target"], settings["correction"])
    print(f"признаков у каждой молекулы: {numbers.shape[1]}")
    above_known = int((predicted > highest_known).sum())
    answer = pd.DataFrame({
        "index": test["index"].astype(int),
        "SI VACV": predicted.clip(0.0, highest_known),
    })

    answer = answer.set_index("index").reindex(sample["index"].astype(int)).reset_index()
    if answer["SI VACV"].isna().any():
        raise ValueError("для части строк sample_submit не нашлось прогноза")
    if list(answer.columns) != list(sample.columns):
        raise ValueError(f"колонки {list(answer.columns)} не совпадают с sample_submit {list(sample.columns)}")

    path = Path(args.output)
    answer.to_csv(path, index=False, encoding="utf-8")
    print(f"{path}: {len(answer)} строк")
    print(f"ограничено потолком ({highest_known:.0f}): {above_known} строк")
    print(f"прогноз от {answer['SI VACV'].min():.3f} до {answer['SI VACV'].max():.1f}, "
          f"медиана {answer['SI VACV'].median():.2f}")


if __name__ == "__main__":
    main()
