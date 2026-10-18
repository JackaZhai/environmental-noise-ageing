from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from ..common import read_table, require, unique_eid, gap_columns, csv


def residualize(df, systems, age="chronological_age"):
    df = df.copy()
    models = []
    for system in systems:
        bio = system + "_biological_age"
        require(df, [age, bio])
        valid = df[[age, bio]].notna().all(axis=1)
        if valid.sum() < 3 or df.loc[valid, age].nunique() < 2:
            raise ValueError(f"Insufficient age support for {system}")
        raw = df.loc[valid, bio] - df.loc[valid, age]
        if not np.isfinite(raw).all() or not np.isfinite(df.loc[valid, age]).all():
            raise ValueError("Nonfinite age inputs")
        x = np.column_stack([np.ones(valid.sum()), df.loc[valid, age]])
        beta = np.linalg.lstsq(x, raw, rcond=None)[0]
        df[system + "_raw_age_gap"] = np.nan
        df.loc[valid, system + "_raw_age_gap"] = raw
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            residual = raw - x @ beta
        if not np.isfinite(residual).all():
            raise FloatingPointError("Nonfinite residualized AgeGap")
        df[system + "_age_gap"] = np.nan
        df.loc[valid, system + "_age_gap"] = residual
        models.append(
            {
                "system": system,
                "intercept": beta[0],
                "age_slope": beta[1],
                "n": valid.sum(),
                "method": "full-analysis-sample linear age residualization",
            }
        )
    return df, pd.DataFrame(models)


def run(paths, cfg, fields):
    source = paths["output_dir"] / "data_preparation/analysis_dataset.csv"
    df = read_table(source if source.exists() else paths["analysis_input"])
    unique_eid(df)
    out = paths["output_dir"] / "agegap_phenotyping"
    outputs = []
    if cfg["input_agegap_state"] == "predicted":
        p = paths["output_dir"] / "age_inference/ages.csv"
        ages = read_table(p if p.exists() else paths["predicted_ages"])
        unique_eid(ages)
        bio_cols = [s + "_biological_age" for s in cfg["systems"]]
        df = df.drop(columns=[c for c in gap_columns(cfg) + bio_cols if c in df])
        df = df.merge(
            ages[["eid"] + bio_cols], on="eid", how="left", validate="one_to_one"
        )
        if "brain_imaging_sample" in df:
            df.loc[df.brain_imaging_sample != 1, "nervous_biological_age"] = np.nan
        df, models = residualize(df, cfg["systems"])
        p = out / "residualization_models.csv"
        csv(models, p)
        outputs.append(p)
    elif cfg["input_agegap_state"] != "residualized":
        raise ValueError("input_agegap_state must be predicted or residualized")
    require(df, gap_columns(cfg) + ["chronological_age"])
    summary = []
    for system in cfg["systems"]:
        gap = system + "_age_gap"
        block = df[[gap, "chronological_age"]].dropna()
        row = {
            "system": system,
            "n": len(block),
            "gap_mean": block[gap].mean(),
            "gap_sd": block[gap].std(),
            "residual_age_r": (
                pearsonr(block[gap], block.chronological_age).statistic
                if len(block) > 2 and block[gap].std() > 0
                else np.nan
            ),
            "agegap_source": cfg["input_agegap_state"],
        }
        bio = system + "_biological_age"
        if bio in df:
            b = df[[bio, "chronological_age"]].dropna()
            row.update(
                reconstruction_r=pearsonr(b[bio], b.chronological_age).statistic,
                reconstruction_mae=(b[bio] - b.chronological_age).abs().mean(),
            )
        summary.append(row)
    for name, frame in [
        ("phenotypes.csv", df),
        ("phenotype_summary.csv", pd.DataFrame(summary)),
    ]:
        p = out / name
        csv(frame, p)
        outputs.append(p)
    if paths.get("trait_data") and paths.get("trait_specs"):
        from .traits import evaluate_traits

        trait = read_table(paths["trait_data"])
        unique_eid(trait, "trait data")
        keep = [c for c in trait if c not in df or c == "eid"]
        merged = df.merge(trait[keep], on="eid", how="left", validate="one_to_one")
        p = out / "trait_associations.csv"
        csv(evaluate_traits(merged, read_table(paths["trait_specs"]), cfg), p)
        outputs.append(p)
    return outputs
