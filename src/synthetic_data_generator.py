"""
synthetic_data_generator.py

Generates a fully synthetic "OMIM-inspired CMT" tabular dataset and saves it as CSV.
"""

import numpy as np
import pandas as pd

GENE_VARIANTS = [f"GENE{idx:03d}" for idx in range(1, 21)]
MUTATION_TYPES = ["missense", "nonsense", "frameshift", "duplication", "deletion"]

def variant_risk_score(variant):
    base = (int(variant.replace("GENE","")) % 7) / 7.0
    return base * 0.9 + 0.05

def mutation_modifier(mtype):
    mods = {"missense": 0.0, "nonsense": 0.15, "frameshift": 0.25, "duplication": 0.08, "deletion": 0.20}
    return mods.get(mtype, 0.0)

def simulate_patient(idx, seed=None):
    rng = np.random.RandomState(seed + idx if seed is not None else None)
    gene = rng.choice(GENE_VARIANTS)
    mtype = rng.choice(MUTATION_TYPES, p=[0.45,0.15,0.15,0.15,0.10])
    risk = variant_risk_score(gene) + mutation_modifier(mtype) + rng.normal(scale=0.05)
    risk = np.clip(risk, 0.01, 0.99)
    age_of_onset = int(np.clip(rng.normal(loc=50 - 30 * risk, scale=8), 5, 85))
    motor_base = 20 + 60 * risk + rng.normal(scale=8)
    sensory_base = 15 + 65 * risk + rng.normal(scale=10)
    motor_score = int(np.clip(motor_base, 0, 100))
    sensory_score = int(np.clip(sensory_base, 0, 100))
    combined = 0.6 * motor_score + 0.4 * sensory_score
    if combined < 35:
        severity = "mild"
    elif combined < 65:
        severity = "moderate"
    else:
        severity = "severe"
    return {
        "patient_id": f"P{idx:06d}",
        "gene_variant_id": gene,
        "mutation_type": mtype,
        "age_of_onset": age_of_onset,
        "motor_score": motor_score,
        "sensory_score": sensory_score,
        "disease_severity": severity
    }

def generate_dataset(n=2000, seed=42, out_path=None):
    rows = [simulate_patient(i, seed=seed) for i in range(1, n+1)]
    df = pd.DataFrame(rows)
    if out_path:
        df.to_csv(out_path, index=False)
    return df

if __name__ == "__main__":
    df = generate_dataset(n=2000, seed=42, out_path="cmt_synthetic_dataset.csv")
    print("Generated dataset with", len(df), "rows")
