from __future__ import annotations
import numpy as np
import pandas as pd
from ..common import read_table, require, unique_eid, csv, adjust, gap_columns
from ..data_preparation.cohorts import define_exposures, recode_fields
from .outcomes import build_riskset
from .cox import fit_cox


def run(paths, cfg, fields):
    # These two sources are intentionally independent. No AgeGap restriction is applied to noise Cox.
    noise = read_table(paths["clinical_noise_input"])
    age = read_table(paths["clinical_agegap_input"])
    for name, df in [("clinical_noise_input", noise), ("clinical_agegap_input", age)]:
        unique_eid(df, name)
    noise = define_exposures(
        recode_fields(noise, fields, clinical=True), cfg["threshold_db"]
    )
    age = define_exposures(
        recode_fields(age, fields, clinical=True), cfg["threshold_db"]
    )
    if "brain_imaging_sample" in age and "nervous_age_gap" in age:
        age.loc[age.brain_imaging_sample != 1, "nervous_age_gap"] = np.nan
    endpoints = read_table(paths["endpoints"])
    require(endpoints, ["icd10", "event_column", "system"], "endpoint inventory")
    if endpoints.icd10.duplicated().any():
        raise ValueError("Duplicate ICD endpoint definitions")
    outputs = []
    noise_rows = []
    age_rows = []
    qcs = []
    ph = []
    for endpoint in endpoints.itertuples(index=False):
        column = endpoint.event_column
        for kind, data in [("noise", noise), ("agegap", age)]:
            if column not in data:
                qcs.append(
                    {
                        "analysis": kind,
                        "icd10": endpoint.icd10,
                        "status": "missing_event_column",
                    }
                )
                continue
            risk, qc = build_riskset(data, column, cfg)
            qcs.append(
                {"analysis": kind, "icd10": endpoint.icd10, "status": "ok", **qc}
            )
            for model in cfg["models"]:
                for group, ref, comparisons in [
                    ("noise_group", "NC", ["ND", "NN"]),
                    ("lowhigh_group", "Low", ["High"]),
                ]:
                    if kind == "noise":
                        for comparison in comparisons:
                            block = risk[risk[group].isin([ref, comparison])].copy()
                            block["exposure_indicator"] = (
                                block[group] == comparison
                            ).astype(int)
                            num, cat = (
                                cfg["models"][model]["numeric"],
                                cfg["models"][model]["categorical"],
                            )
                            require(block, num + cat, "noise Cox covariates")
                            complete = block.dropna(subset=num + cat)
                            counts = complete.groupby(group).event.sum()
                            if any(
                                counts.get(g, 0) < cfg["clinical"]["min_events"]
                                for g in [ref, comparison]
                            ):
                                result = {
                                    "model": model,
                                    "n": len(complete),
                                    "events": int(complete.event.sum()),
                                    "status": "not_estimated_low_group_events",
                                    "p_value": np.nan,
                                }
                                diagnostics = []
                            else:
                                result, diagnostics = fit_cox(
                                    block, "exposure_indicator", cfg, model
                                )
                            metadata = {
                                "icd10": endpoint.icd10,
                                "system": endpoint.system,
                                "comparison": f"{comparison} vs {ref}",
                            }
                            noise_rows.append({**metadata, **result})
                            for row in diagnostics:
                                ph.append(
                                    {
                                        "analysis": kind,
                                        **metadata,
                                        "model": model,
                                        **row,
                                    }
                                )
                    else:
                        gap = endpoint.system + "_age_gap"
                        if gap not in gap_columns(cfg) or gap not in risk:
                            age_rows.append(
                                {
                                    "icd10": endpoint.icd10,
                                    "gap_column": gap,
                                    "model": model,
                                    "stratum": "all",
                                    "status": "not_estimated_missing_gap",
                                    "p_value": np.nan,
                                }
                            )
                            continue
                        for stratum in [ref] + comparisons:
                            block = risk[risk[group] == stratum]
                            result, diagnostics = fit_cox(block, gap, cfg, model, True)
                            metadata = {
                                "icd10": endpoint.icd10,
                                "gap_column": gap,
                                "stratum": stratum,
                                "grouping": group,
                            }
                            age_rows.append({**metadata, **result})
                            for row in diagnostics:
                                ph.append(
                                    {
                                        "analysis": kind,
                                        **metadata,
                                        "model": model,
                                        **row,
                                    }
                                )
    noise_result = adjust(pd.DataFrame(noise_rows), ["model", "comparison"])
    age_result = adjust(pd.DataFrame(age_rows), ["model", "stratum", "gap_column"])
    out = paths["output_dir"] / "clinical_outcomes"
    for name, frame in [
        ("noise_cox", noise_result),
        ("agegap_cox", age_result),
        ("riskset_qc", pd.DataFrame(qcs)),
        ("ph_screen", pd.DataFrame(ph)),
        ("endpoint_inventory", endpoints),
    ]:
        p = out / (name + ".csv")
        csv(frame, p)
        outputs.append(p)
    return outputs
