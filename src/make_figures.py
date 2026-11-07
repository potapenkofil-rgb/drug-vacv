from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED = Path("data/processed")
FIGURES = Path("docs/figures")

INK = "#16181d"
ACCENT = "#1f7a4d"
WARM = "#c4570f"
RED = "#a82a1f"
GRAY = "#9aa1ab"

TEST_SQUARE_MEAN = 2123511.05
TEST_AVERAGE = 323.38
TEST_VARIANCE = TEST_SQUARE_MEAN - TEST_AVERAGE ** 2
OUR_CORRELATION = 0.598


def style(axes, title, xlabel="", ylabel=""):
    axes.set_title(title, fontsize=12, color=INK, pad=12, loc="left")
    axes.set_xlabel(xlabel, fontsize=10, color=INK)
    axes.set_ylabel(ylabel, fontsize=10, color=INK)
    axes.grid(axis="y", color="#e4e6ea", linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#d7dae0")
    axes.tick_params(colors="#6f7681", labelsize=9)


def distribution(selectivity):
    figure, axes = plt.subplots(figsize=(8, 4.2))
    positive = selectivity[selectivity > 0]
    bins = np.logspace(np.log10(max(positive.min(), 1e-3)), np.log10(positive.max()), 40)
    axes.hist(positive, bins=bins, color=ACCENT, alpha=0.85)
    axes.set_xscale("log")
    axes.axvline(np.median(selectivity), color=WARM, linewidth=2, label=f"медиана {np.median(selectivity):.1f}")
    axes.axvline(selectivity.mean(), color=RED, linewidth=2, label=f"среднее {selectivity.mean():.0f}")
    zeros = int((selectivity == 0).sum())
    style(axes, f"Распределение SI в обучении: {zeros} нулей и хвост до {selectivity.max():.0f}",
          "SI, логарифмическая шкала", "молекул")
    axes.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / "si_distribution.png", dpi=140)
    plt.close(figure)


def variance_share(selectivity):
    squares = np.sort((selectivity - selectivity.mean()) ** 2)[::-1]
    share = np.cumsum(squares) / squares.sum() * 100
    figure, axes = plt.subplots(figsize=(8, 4.2))
    top = 30
    axes.plot(range(1, top + 1), share[:top], color=ACCENT, linewidth=2.5)
    axes.scatter([5], [share[4]], color=RED, zorder=5, s=60)
    axes.annotate(f"5 молекул — {share[4]:.1f}%", (5, share[4]), textcoords="offset points",
                  xytext=(14, -18), color=RED, fontsize=10)
    axes.set_ylim(0, 102)
    style(axes, "Откуда берётся ошибка: вклад молекул в разброс цели",
          "сколько самых больших молекул учтено", "доля разброса, %")
    figure.tight_layout()
    figure.savefig(FIGURES / "variance_share.png", dpi=140)
    plt.close(figure)


def leaderboard():
    labels = ["одни нули", "константа 300", "лучшая\nконстанта", "наша модель\nбез калибровки",
              "наша модель\nс калибровкой"]
    values = [2123511, 2019483, TEST_VARIANCE, 1467940, 1297590]
    colors = [GRAY, GRAY, WARM, "#7fb79a", ACCENT]
    figure, axes = plt.subplots(figsize=(8, 4.4))
    bars = axes.bar(labels, values, color=colors, width=0.62)
    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width() / 2, value + 30000, f"{value:,.0f}".replace(",", " "),
                  ha="center", fontsize=9.5, color=INK)
    axes.axhline(TEST_VARIANCE, color=WARM, linestyle="--", linewidth=1.2)
    axes.set_ylim(0, 2450000)
    style(axes, "Результаты на Kaggle, метрика MSE: меньше — лучше", "", "MSE")
    figure.tight_layout()
    figure.savefig(FIGURES / "leaderboard.png", dpi=140)
    plt.close(figure)


def correlation_curve():
    grid = np.linspace(0, 0.95, 200)
    figure, axes = plt.subplots(figsize=(8, 4.2))
    axes.plot(grid, TEST_VARIANCE * (1 - grid ** 2), color=ACCENT, linewidth=2.5)
    here = TEST_VARIANCE * (1 - OUR_CORRELATION ** 2)
    axes.scatter([OUR_CORRELATION], [here], color=RED, s=70, zorder=5)
    axes.annotate(f"мы: {OUR_CORRELATION}\n{here:,.0f}".replace(",", " "),
                  (OUR_CORRELATION, here), textcoords="offset points", xytext=(12, 14),
                  color=RED, fontsize=10)
    for mark in (0.7, 0.8):
        axes.scatter([mark], [TEST_VARIANCE * (1 - mark ** 2)], color=WARM, s=45, zorder=5)
        axes.annotate(f"{TEST_VARIANCE * (1 - mark ** 2):,.0f}".replace(",", " "),
                      (mark, TEST_VARIANCE * (1 - mark ** 2)), textcoords="offset points",
                      xytext=(8, 10), color=WARM, fontsize=9)
    style(axes, "Что даёт улучшение модели: MSE = разброс × (1 − корреляция²)",
          "корреляция прогноза с истиной", "ожидаемый MSE")
    figure.tight_layout()
    figure.savefig(FIGURES / "correlation_curve.png", dpi=140)
    plt.close(figure)


def predicted_against_actual():
    oof = pd.read_csv("reports/oof_predictions.csv")
    figure, axes = plt.subplots(figsize=(8, 4.6))
    actual = oof["si_vacv"].to_numpy(dtype=float)
    predicted = oof["prediction"].to_numpy(dtype=float)
    axes.scatter(actual + 1, predicted + 1, s=26, color=ACCENT, alpha=0.55, edgecolor="none")
    limit = [1, max(actual.max(), predicted.max()) * 1.3]
    axes.plot(limit, limit, color=GRAY, linestyle="--", linewidth=1.2)
    axes.set_xscale("log")
    axes.set_yscale("log")
    style(axes, "Прогноз против истины на отложенных молекулах",
          "настоящий SI + 1", "предсказанный SI + 1")
    figure.tight_layout()
    figure.savefig(FIGURES / "predicted_vs_actual.png", dpi=140)
    plt.close(figure)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(PROCESSED / "train.csv")
    selectivity = table.loc[table["source"] == "organizers", "si_vacv"].to_numpy(dtype=float)
    distribution(selectivity)
    variance_share(selectivity)
    leaderboard()
    correlation_curve()
    predicted_against_actual()
    print(f"готово: {len(list(FIGURES.glob('*.png')))} рисунков в {FIGURES}")


if __name__ == "__main__":
    main()
