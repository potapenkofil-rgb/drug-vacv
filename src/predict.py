from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from features import assay_matrix, build_matrix
from train import from_model_scale

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
MODELS = Path("models")
SUBMISSIONS = Path("submissions")


def main():
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    settings = json.loads((MODELS / "feature_config.json").read_text(encoding="utf-8"))
    morgan = settings["morgan"]
    test = pd.read_csv(PROCESSED / "test.csv")
    sample = pd.read_csv(RAW / "sample_submit.csv")
    highest_known = float(pd.read_csv(PROCESSED / "train.csv")["si_vacv"].max())

    numbers = build_matrix(test["smiles_canonical"], morgan["radius"], morgan["size"])
    vocabulary = settings.get("assay_vocabulary") or []
    if vocabulary:
        numbers = np.hstack([numbers, assay_matrix(test["strain"], test["culture"], vocabulary)])
    if settings.get("approach") == "parts":
        from catboost import CatBoostClassifier

        from train_parts import PARTS, predict_selectivity

        activity, toxicity, classifier = CatBoostRegressor(), CatBoostRegressor(), CatBoostClassifier()
        activity.load_model(str(PARTS / "ic50.cbm"))
        toxicity.load_model(str(PARTS / "cc50.cbm"))
        classifier.load_model(str(PARTS / "active.cbm"))
        raw = predict_selectivity((activity, toxicity, classifier), numbers)
        predicted = np.clip((raw + 1.0) * settings["correction"] - 1.0, 0.0, None)
    else:
        model = CatBoostRegressor()
        model.load_model(str(MODELS / "model.cbm"))
        predicted = from_model_scale(model.predict(numbers), settings["target"], settings["correction"])
    print(f"способ: {settings.get('approach', 'direct')}")
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

    path = SUBMISSIONS / "submission.csv"
    answer.to_csv(path, index=False, encoding="utf-8")
    print(f"{path}: {len(answer)} строк")
    print(f"ограничено максимумом обучения ({highest_known:.0f}): {above_known} строк")
    print(f"прогноз от {answer['SI VACV'].min():.3f} до {answer['SI VACV'].max():.1f}, "
          f"медиана {answer['SI VACV'].median():.2f}")


if __name__ == "__main__":
    main()
