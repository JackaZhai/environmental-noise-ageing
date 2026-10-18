#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(20261004)
import argparse

parser = argparse.ArgumentParser(description="Generate synthetic data for tests or pipeline smoke checks")
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "tests" / "fixtures")
args = parser.parse_args()
root = args.output_dir.expanduser().resolve()
root.mkdir(parents=True, exist_ok=True)
systems = [
    "overall",
    "cardiovascular",
    "hepatic",
    "pulmonary",
    "renal",
    "metabolic",
    "musculoskeletal",
    "nervous",
]


def participants(n, offset=0):
    group = rng.choice(["NC", "ND", "NN"], n, p=[0.60, 0.22, 0.18])
    age = rng.uniform(45, 80, n)
    df = pd.DataFrame(
        {
            "eid": np.arange(1 + offset, n + 1 + offset),
            "chronological_age": age,
            "bmi": rng.normal(27, 3, n),
            "townsend": rng.normal(0, 2, n),
            "no2_2010": rng.normal(28, 4, n),
            "pm25_2010": rng.normal(10, 1, n),
            "sex_code": rng.choice(["Female", "Male"], n),
            "ethnicity_group": rng.choice(["White British", "Other"], n),
            "education_degree": rng.choice(["Degree", "No degree"], n),
            "smoking_status": rng.choice(["Never", "Previous", "Current"], n),
            "alcohol_frequency": rng.choice(["Never", "1-2 times/week"], n),
            "urban_rural_group": rng.choice(["Urban", "Village"], n),
            "brain_imaging_sample": rng.choice([0, 1], n, p=[0.08, 0.92]),
        }
    )
    df["brain_mri_nonmissing_fields"] = df.brain_imaging_sample * 3
    df["Lday"] = np.where(group == "NC", rng.uniform(48, 59, n), rng.uniform(60, 72, n))
    df["Lnight"] = np.where(
        group == "NN", rng.uniform(60, 67, n), rng.uniform(40, 59, n)
    )
    df["Lden"] = df.Lday + np.where(group == "NN", 3, -1)
    common = rng.normal(0, 1, n)
    ages = pd.DataFrame({"eid": df.eid, "chronological_age": age})
    for s in systems:
        raw = 0.08 * (age - 60) + common + rng.normal(0, 2, n) + 0.2 * (group == "NN")
        ages[s + "_biological_age"] = age + raw
        x = np.column_stack([np.ones(n), age])
        gap = raw - x @ np.linalg.lstsq(x, raw, rcond=None)[0]
        df[s + "_age_gap"] = gap
    df.loc[df.brain_imaging_sample == 0, "nervous_age_gap"] = np.nan
    return df, ages


df, ages = participants(900)
full, _ = participants(1700, 10000)
full = pd.concat([df, full], ignore_index=True)
for data in [df, full]:
    data["53-0.0"] = "2010-01-01"
    data["40000-0.0"] = None
    data["event_I10"] = pd.Timestamp("2010-01-01") + pd.to_timedelta(
        rng.exponential(4300, len(data)).astype(int), unit="D"
    )
    data["event_E11"] = pd.Timestamp("2010-01-01") + pd.to_timedelta(
        rng.exponential(5000, len(data)).astype(int), unit="D"
    )
for name, data in [
    ("participants", df),
    ("ages", ages),
    ("clinical_full", full),
    ("clinical_agegap", df),
]:
    data.to_csv(root / (name + ".csv"), index=False)
pd.DataFrame(
    {
        "icd10": ["I10", "E11"],
        "event_column": ["event_I10", "event_E11"],
        "system": ["cardiovascular", "metabolic"],
    }
).to_csv(root / "endpoints.csv", index=False)
proteins = pd.DataFrame({"eid": df.eid})
for i in range(12):
    proteins["P" + str(i)] = rng.normal(0, 1, len(df)) + 0.3 * df.overall_age_gap
proteins.loc[rng.choice(len(df), 40, replace=False), "P0"] = np.nan
proteins.to_csv(root / "proteins.csv", index=False)
pd.DataFrame(
    {
        "protein": ["P" + str(i) for i in range(12)],
        "gene": ["SYNTH" + str(i) for i in range(12)],
    }
).to_csv(root / "protein_mapping.csv", index=False)
(root / "gene_sets.gmt").write_text(
    "SYNTHETIC_GO\tSynthetic example only\t"
    + "\t".join(["SYNTH" + str(i) for i in range(8)])
    + "\nSYNTHETIC_KEGG\tSynthetic example only\t"
    + "\t".join(["SYNTH" + str(i) for i in range(4, 12)])
    + "\n"
)
(root / "profiles.jsonl").write_text(
    "\n".join(
        json.dumps(
            {
                "eid": int(row.eid),
                "chronological_age": row.chronological_age,
                "profile": {
                    "chronological_age": row.chronological_age,
                    "sex": row.sex_code,
                    "bmi": row.bmi,
                },
            }
        )
        for row in df.head(3).itertuples()
    )
    + "\n"
)
print(
    "Generated only synthetic participants and proteins; these do not reproduce manuscript results."
)

pd.DataFrame(
    {
        "eid": df.eid,
        "walking_pace": rng.normal(size=len(df)),
        "binary_trait": rng.integers(0, 2, len(df)),
    }
).to_csv(root / "traits.csv", index=False)
pd.DataFrame(
    {
        "target": ["overall_age_gap", "metabolic_age_gap", "overall_age_gap"],
        "marker": ["walking_pace", "bmi", "binary_trait"],
        "kind": ["continuous", "continuous", "binary"],
    }
).to_csv(root / "trait_specs.csv", index=False)
