from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from features import canonical_smiles

BASE = "https://www.ebi.ac.uk/chembl/api/data"
EXTERNAL = Path("data/external")
ACTIVITY_TYPES = ("IC50", "EC50")
TOXICITY_TYPES = ("CC50", "TC50", "GI50")
TO_MICROMOLAR = {"nM": 0.001, "uM": 1.0, "µM": 1.0, "mM": 1000.0, "M": 1000000.0}


def request_json(path, params, attempts=4):
    url = f"{BASE}/{path}.json?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "drug-vacv-hackathon"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                answer = json.loads(response.read().decode("utf-8"))
            time.sleep(0.5)
            return answer
        except Exception as error:
            if attempt == attempts:
                raise
            print(f"  повтор {attempt} после ошибки: {error}")
            time.sleep(3 * attempt)
    return {}


def request_pages(path, params, key, limit):
    collected = []
    while len(collected) < limit:
        page = dict(params, limit=min(200, limit - len(collected)), offset=len(collected))
        try:
            answer = request_json(path, page, attempts=2)
        except Exception:
            break
        chunk = answer.get(key, [])
        collected.extend(chunk)
        if len(chunk) < page["limit"]:
            break
    return collected


def targets_for(keyword):
    answer = request_json("target/search", {"q": keyword, "limit": 100})
    found = []
    for target in answer.get("targets", []):
        description = f"{target.get('organism') or ''} {target.get('pref_name') or ''}".lower()
        if keyword in description:
            found.append(target["target_chembl_id"])
    return sorted(set(found))


def micromolar(value, units):
    if value is None or units not in TO_MICROMOLAR:
        return None
    try:
        number = float(value) * TO_MICROMOLAR[units]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def as_table(records, wanted_types):
    rows = []
    for record in records:
        if record.get("standard_type") not in wanted_types:
            continue
        if (record.get("standard_relation") or "=") != "=":
            continue
        value = micromolar(record.get("standard_value"), record.get("standard_units"))
        if value is None or not record.get("canonical_smiles"):
            continue
        rows.append({
            "molecule_chembl_id": record.get("molecule_chembl_id"),
            "smiles": record["canonical_smiles"],
            "value_um": value,
            "assay_description": record.get("assay_description") or "",
            "document_chembl_id": record.get("document_chembl_id") or "",
        })
    return pd.DataFrame(rows)


def by_parts(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def doi_by_document(document_ids):
    found = {}
    unique = sorted({d for d in document_ids if d})
    for part in by_parts(unique, 10):
        answer = request_json("document", {"document_chembl_id__in": ",".join(part), "limit": 50})
        for document in answer.get("documents", []):
            found[document["document_chembl_id"]] = document.get("doi") or ""
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--organisms", default="vaccinia")
    parser.add_argument("--output", default="data/external/chembl_vacv.csv")
    args = parser.parse_args()

    EXTERNAL.mkdir(parents=True, exist_ok=True)
    keywords = [word.strip().lower() for word in args.organisms.split(",") if word.strip()]
    targets = sorted({name for word in keywords for name in targets_for(word)})
    print(f"мишеней по запросу {keywords} в ChEMBL: {len(targets)}")

    measurements = []
    for target in targets:
        found = request_pages("activity", {"target_chembl_id": target}, "activities", args.limit)
        measurements.extend(found)
        print(f"  {target}: {len(found)} измерений")

    activity = as_table(measurements, ACTIVITY_TYPES)
    if activity.empty:
        raise SystemExit("не нашлось ни одного IC50 или EC50 против VACV")
    molecules = sorted({m for m in activity["molecule_chembl_id"] if m})
    print(f"молекул с активностью против VACV: {len(molecules)}")

    toxicity_records = []
    for part in by_parts(molecules, 10):
        toxicity_records.extend(request_pages(
            "activity", {"molecule_chembl_id__in": ",".join(part)}, "activities", 3000))
    toxicity = as_table(toxicity_records, TOXICITY_TYPES)
    print(f"измерений цитотоксичности: {len(toxicity)}")

    activity_per_molecule = activity.groupby("molecule_chembl_id", as_index=False).agg(
        ic50_um=("value_um", "median"),
        smiles=("smiles", "first"),
        assay_description=("assay_description", "first"),
        document_chembl_id=("document_chembl_id", "first"))
    toxicity_per_molecule = toxicity.groupby("molecule_chembl_id", as_index=False).agg(
        cc50_um=("value_um", "median"))

    joined = activity_per_molecule.merge(toxicity_per_molecule, on="molecule_chembl_id")
    joined["smiles"] = [canonical_smiles(s) for s in joined["smiles"]]
    joined = joined[joined["smiles"].notna()]
    joined["si"] = joined["cc50_um"] / joined["ic50_um"]

    doi = doi_by_document(list(joined["document_chembl_id"]))
    joined["doi"] = [doi.get(d, "") for d in joined["document_chembl_id"]]
    joined["source"] = "ChEMBL"
    joined["source_url"] = "https://www.ebi.ac.uk/chembl/compound_report_card/" + joined["molecule_chembl_id"]
    joined["measurement_note"] = "только точные значения, медиана по молекуле"
    joined["retrieved_on"] = date.today().isoformat()

    columns = ["molecule_chembl_id", "smiles", "ic50_um", "cc50_um", "si", "assay_description",
               "doi", "source", "source_url", "measurement_note", "retrieved_on"]
    joined[columns].to_csv(args.output, index=False, encoding="utf-8")
    print(f"\n{args.output}: {len(joined)} молекул с IC50, CC50 и SI")
    print(f"из них с DOI: {int((joined['doi'] != '').sum())}")


if __name__ == "__main__":
    main()
