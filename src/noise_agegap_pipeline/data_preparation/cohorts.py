from __future__ import annotations
import numpy as np
import pandas as pd
from ..common import read_table, require, unique_eid, csv, covariates, gap_columns


def define_exposures(df, threshold=60):
    df = df.copy()
    if "Lden" not in df and "L24h" in df:
        df["Lden"] = df["L24h"]
    if "Lden" in df and "L24h" in df:
        both = df[["Lden", "L24h"]].dropna()
        if not np.allclose(both.Lden, both.L24h):
            raise ValueError("Lden and legacy L24h disagree")
    require(df, ["Lday", "Lnight", "Lden"])
    df["noise_group"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[(df.Lday < threshold) & (df.Lnight < threshold), "noise_group"] = "NC"
    df.loc[(df.Lday >= threshold) & (df.Lnight < threshold), "noise_group"] = "ND"
    df.loc[(df.Lday >= threshold) & (df.Lnight >= threshold), "noise_group"] = "NN"
    # Low-day/high-night and missing exposures retain explicit missing classification.
    df["lowhigh_group"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[df.Lden.notna() & (df.Lden < threshold), "lowhigh_group"] = "Low"
    df.loc[df.Lden.notna() & (df.Lden >= threshold), "lowhigh_group"] = "High"
    return df


def merge_sources(tables):
    if not tables:
        raise ValueError("No input tables")
    result = tables[0].copy()
    unique_eid(result)
    for table in tables[1:]:
        unique_eid(table)
        duplicate = (set(result) & set(table)) - {"eid"}
        if duplicate:
            raise ValueError(
                f"Duplicate fields across source tables: {sorted(duplicate)}"
            )
        result = result.merge(table, on="eid", how="left", validate="one_to_one")
    return result


def recode_fields(df, fields, clinical=False):
    df = df.copy()
    mapping = fields["clinical_covariates" if clinical else "agegap_covariates"]
    for name, raw in {**fields["noise"], **mapping}.items():
        if name == "legacy_Lden_alias":
            continue
        if raw in df and (name not in df or clinical):
            df[name] = df[raw]
    if clinical:
        for raw, derived in [
            ("smoking_status_code", "smoking_status"),
            ("alcohol_code", "alcohol_frequency"),
            ("education_code", "education_degree"),
            ("ethnicity_code", "ethnicity_group"),
        ]:
            if mapping.get(raw) in df and derived in df:
                df = df.drop(columns=[derived])
    # Recode raw fields only; preprocessed categories are never silently reinterpreted.
    if "sex_code" in df and pd.api.types.is_numeric_dtype(df.sex_code):
        df["sex_code"] = df.sex_code.map(
            {0: "Female", 1: "Male", -1: "Unknown", -3: "Unknown"}
        )
    for src, dest, levels in [
        (
            "smoking_status_code",
            "smoking_status",
            {0: "Never", 1: "Previous", 2: "Current"},
        ),
        (
            "alcohol_code",
            "alcohol_frequency",
            {
                1: "Daily/almost daily",
                2: "3-4 times/week",
                3: "1-2 times/week",
                4: "1-3 times/month",
                5: "Special occasions only",
                6: "Never",
            },
        ),
    ]:
        if dest not in df and src in df:
            df[dest] = df[src].map(
                {**levels, **{x: "Unknown" for x in fields["missing_category_codes"]}}
            )
    for src, dest, reference, positive, other in [
        ("education_code", "education_degree", 1, "Degree", "No degree"),
        ("ethnicity_code", "ethnicity_group", 1001, "White British", "Other"),
    ]:
        if dest not in df and src in df:
            df[dest] = pd.Series(pd.NA, index=df.index, dtype="object")
            df.loc[df[src].notna(), dest] = np.where(
                df.loc[df[src].notna(), src] == reference, positive, other
            )
            df.loc[df[src].isin(fields["missing_category_codes"]), dest] = "Unknown"
    return df


def baseline(df, grouping, cfg):
    rows = []
    numeric, categorical = covariates(cfg, "Expanded")
    for variable in numeric + categorical + ["Lday", "Lnight", "Lden"]:
        if variable not in df:
            continue
        for group, block in df.dropna(subset=[grouping]).groupby(grouping):
            values = block[variable].dropna()
            if variable in categorical:
                for level, n in values.value_counts().items():
                    rows.append(
                        {
                            "grouping": grouping,
                            "group": group,
                            "variable": variable,
                            "level": level,
                            "n": n,
                            "nonmissing_n": len(values),
                            "percent": 100 * n / len(values),
                        }
                    )
            else:
                rows.append(
                    {
                        "grouping": grouping,
                        "group": group,
                        "variable": variable,
                        "n": len(values),
                        "mean": values.mean(),
                        "sd": values.std(),
                    }
                )
    return pd.DataFrame(rows)


def run(paths, cfg, fields):
    df = read_table(paths["analysis_input"])
    unique_eid(df)
    df = define_exposures(recode_fields(df, fields), cfg["threshold_db"])
    if "brain_imaging_sample" in df and "nervous_age_gap" in df:
        df.loc[df.brain_imaging_sample != 1, "nervous_age_gap"] = np.nan
    out = paths["output_dir"] / "data_preparation"
    outputs = []

    def save(frame, name):
        p = out / name
        csv(frame, p)
        outputs.append(p)

    save(df, "analysis_dataset.csv")
    rows = [
        {"cohort": "input", "outcome": "all", "model": "none", "n": len(df)},
        {
            "cohort": "valid_day_night_exposure",
            "outcome": "all",
            "model": "none",
            "n": df.noise_group.notna().sum(),
        },
        {
            "cohort": "unclassified_day_night_exposure",
            "outcome": "all",
            "model": "none",
            "n": df.noise_group.isna().sum(),
        },
    ]
    if cfg["input_agegap_state"] == "residualized":
        require(df, gap_columns(cfg), "saved residualized AgeGap input")
        for col in gap_columns(cfg):
            for model in cfg["models"]:
                num, cat = covariates(cfg, model)
                require(df, num + cat)
                rows.append(
                    {
                        "cohort": "agegap",
                        "outcome": col,
                        "model": model,
                        "n": len(df.dropna(subset=[col, "noise_group"] + num + cat)),
                    }
                )
    # Clinical populations are counted independently; never intersect with AgeGap eids.
    for key in ["clinical_noise_input", "clinical_agegap_input", "proteins"]:
        p = paths.get(key)
        if p is not None and p.is_file():
            ids = read_table(p)[["eid"]]
            unique_eid(ids, key)
            rows.append(
                {
                    "cohort": key,
                    "outcome": "pre_endpoint_exclusions",
                    "model": "none",
                    "n": len(ids),
                }
            )
    save(pd.DataFrame(rows), "cohort_flow.csv")
    save(
        pd.concat(
            [baseline(df, g, cfg) for g in ["noise_group", "lowhigh_group"]],
            ignore_index=True,
        ),
        "baseline.csv",
    )
    save(
        pd.DataFrame(
            {
                "variable": df.columns,
                "missing_n": df.isna().sum().values,
                "total_n": len(df),
            }
        ),
        "missingness.csv",
    )
    return outputs
