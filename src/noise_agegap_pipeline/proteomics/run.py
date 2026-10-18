from __future__ import annotations
import numpy as np
import pandas as pd
from ..common import read_table, unique_eid, require, csv, covariates, gap_columns
from ..data_preparation.cohorts import define_exposures
from . import core


def run(paths, cfg, fields):
    from ..common import load_analysis

    df = load_analysis(paths, cfg, fields)
    protein = read_table(paths["proteins"])
    unique_eid(protein, "protein matrix")
    columns = [c for c in protein if c != "eid"]
    spec = cfg["proteomics"]
    num, cat = covariates(cfg, spec["adjustment"])
    require(df, num + cat + gap_columns(cfg))
    # Restrict before constructing dummies. Missing categories must not become the reference category.
    df = df.dropna(subset=num + cat)
    merged = df[["eid"]].merge(protein, on="eid", how="inner", validate="one_to_one")
    kept, qc = core.select_proteins_by_missingness(
        merged, columns, spec["missing_threshold"]
    )
    if not kept:
        raise ValueError("No proteins retained at the configured missingness threshold")
    systems = [(s + "_age_gap", s.title()) for s in cfg["systems"]]
    age_screen = core.screen_protein_to_agegap(
        df, protein, [], kept, systems, num, cat, spec["min_n"]
    )
    noise_screens = []
    for group, ref, comparisons in [
        ("noise_group", "NC", ["ND", "NN"]),
        ("lowhigh_group", "Low", ["High"]),
    ]:
        sub = df.dropna(subset=[group]).copy()
        terms = []
        for comparison in comparisons:
            term = f"{comparison}_vs_{ref}"
            sub[term] = (sub[group] == comparison).astype(float)
            terms.append(term)
        noise_screens.append(
            core.screen_exposure_to_protein(
                sub, protein, terms, kept, num, cat, spec["min_n"]
            )
        )
    noise_screen = pd.concat(noise_screens, ignore_index=True)
    strata = []
    for group, levels in [
        ("noise_group", ["NC", "ND", "NN"]),
        ("lowhigh_group", ["Low", "High"]),
    ]:
        strata.append(
            core.screen_protein_to_agegap_by_stratum(
                df,
                protein,
                group,
                levels,
                kept,
                systems,
                num,
                cat,
                spec["stratum_min_n"],
            )
        )
    stratified = pd.concat(strata, ignore_index=True)
    out = paths["output_dir"] / "proteomics"
    outputs = []
    for name, frame in [
        ("protein_qc", qc),
        ("protein_agegap", age_screen),
        ("noise_protein", noise_screen),
        ("stratified_protein_agegap", stratified),
    ]:
        p = out / (name + ".csv")
        csv(frame, p)
        outputs.append(p)
    return outputs
