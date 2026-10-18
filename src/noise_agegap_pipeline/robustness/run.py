from __future__ import annotations
from ..common import read_table, csv, json_write, require
from ..data_preparation.cohorts import define_exposures
from . import core


def run(paths, cfg, fields):
    from ..common import load_analysis

    df = load_analysis(paths, cfg, fields)
    require(df, core.REQUIRED_INPUT_COLUMNS, "robustness")
    same, diag = core.fit_same_sample_pollutant_models(df)
    interactions, marginals, int_diag = core.fit_pollutant_interactions(df)
    overlap, balance, ow_diag = core.fit_overlap_weighted_agegap(
        df,
        bootstrap_reps=cfg["ow_bootstrap"],
        seed=cfg["seed"],
        n_jobs=cfg["jobs"],
        include_urban_sensitivity=True,
        spline_df=cfg["spline_df"],
    )
    out = paths["output_dir"] / "robustness"
    outputs = []
    for name, frame in [
        ("same_sample_models", same),
        ("pollutant_interactions", interactions),
        ("pollutant_marginals", marginals),
        ("overlap_effects", overlap),
        ("overlap_balance", balance),
        ("main_effect_mde", core.build_main_effect_mde(same)),
        ("interaction_mde", core.build_interaction_mde(interactions)),
    ]:
        p = out / (name + ".csv")
        csv(frame, p)
        outputs.append(p)
    p = out / "diagnostics.json"
    json_write({"same_sample": diag, "interaction": int_diag, "overlap": ow_diag}, p)
    outputs.append(p)
    return outputs
