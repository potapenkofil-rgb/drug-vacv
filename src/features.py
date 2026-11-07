from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]


def parse(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def canonical_smiles(smiles):
    mol = parse(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def structure_key(smiles):
    mol = parse(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return Chem.MolToSmiles(mol)


def descriptor_row(mol):
    values = Descriptors.CalcMolDescriptors(mol)
    return np.array([values.get(name, np.nan) for name in DESCRIPTOR_NAMES], dtype=float)


def build_matrix(smiles_list, radius=2, size=2048, chirality=False, counts=False):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=size, includeChirality=chirality)
    rows = []
    for smiles in smiles_list:
        mol = parse(smiles)
        if mol is None:
            raise ValueError(f"не разбирается SMILES: {smiles}")
        if counts:
            fingerprint = np.array(generator.GetCountFingerprintAsNumPy(mol), dtype=float)
        else:
            fingerprint = np.array(generator.GetFingerprintAsNumPy(mol), dtype=float)
        rows.append(np.concatenate([descriptor_row(mol), fingerprint]))
    matrix = np.nan_to_num(np.vstack(rows), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(matrix, -1e6, 1e6)


def pair_matrix(smiles_list, size=2048, chirality=False):
    generator = rdFingerprintGenerator.GetAtomPairGenerator(
        fpSize=size, includeChirality=chirality)
    rows = []
    for smiles in smiles_list:
        mol = parse(smiles)
        if mol is None:
            raise ValueError(f"не разбирается SMILES: {smiles}")
        rows.append(np.array(generator.GetCountFingerprintAsNumPy(mol), dtype=float))
    return np.vstack(rows)


def feature_names(size=2048):
    return DESCRIPTOR_NAMES + [f"fp_{i}" for i in range(size)]


def assay_label(strain, culture):
    return f"{str(strain).strip().lower()} | {str(culture).strip().lower()}"


def assay_vocabulary(strains, cultures, keep=25):
    labels = [assay_label(s, c) for s, c in zip(strains, cultures)]
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts, key=lambda name: (-counts[name], name))
    return ordered[:keep]


def assay_matrix(strains, cultures, vocabulary):
    labels = [assay_label(s, c) for s, c in zip(strains, cultures)]
    matrix = np.zeros((len(labels), len(vocabulary) + 1), dtype=float)
    for row, label in enumerate(labels):
        if label in vocabulary:
            matrix[row, vocabulary.index(label)] = 1.0
        else:
            matrix[row, -1] = 1.0
    return matrix
