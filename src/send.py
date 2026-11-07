from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

COMPETITION = "molecular-activity-prediction-hack"
REPORTS = Path("reports")
LOG = REPORTS / "submissions.json"


def history():
    done = subprocess.run(["python", "-m", "kaggle", "competitions", "submissions", COMPETITION, "-v"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [line for line in done.stdout.splitlines() if "," in line]
    if not lines:
        return None
    return pd.read_csv(io.StringIO("\n".join(lines)))


def score_of(name, table):
    if table is None or "fileName" not in table.columns:
        return None
    rows = table[table["fileName"] == name]
    if rows.empty:
        return None
    return pd.to_numeric(rows.iloc[0]["publicScore"], errors="coerce")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--wait", type=int, default=90)
    args = parser.parse_args()

    table = pd.read_csv(args.file)
    values = table["SI VACV"].to_numpy(dtype=float)
    print(f"{args.file}: строк {len(table)}, среднее {values.mean():.3f}, "
          f"средний квадрат {(values ** 2).mean():,.2f}")

    done = subprocess.run(["python", "-m", "kaggle", "competitions", "submit", COMPETITION,
                           "-f", args.file, "-m", args.message],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(done.stdout.strip() or done.stderr.strip())
    if "error" in (done.stdout + done.stderr).lower() and "successfully" not in done.stdout.lower():
        raise SystemExit("отправка не прошла")

    name = Path(args.file).name
    score = None
    for _ in range(args.wait // 10):
        time.sleep(10)
        score = score_of(name, history())
        if score is not None:
            break
    print(f"результат: {score}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    recorded = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    recorded.append({"file": args.file, "message": args.message, "score": score,
                   "mean": float(values.mean()), "square_mean": float((values ** 2).mean())})
    LOG.write_text(json.dumps(recorded, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
