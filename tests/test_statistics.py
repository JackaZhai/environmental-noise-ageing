from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from noise_agegap_pipeline.common import read_config
from noise_agegap_pipeline.noise_agegap.models import fit_contrasts, fit_splines
from noise_agegap_pipeline.noise_agegap import legacy
from noise_agegap_pipeline.data_preparation.cohorts import define_exposures
from noise_agegap_pipeline.proteomics import core

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_primary_contrasts_agree_with_modular_model(synthetic_data_dir):
    df = define_exposures(pd.read_csv(synthetic_data_dir / "participants.csv"))
    cfg = read_config(ROOT / "config/analysis.yaml")
    for model in cfg["models"].values():
        model["categorical"].remove("ethnicity_group")
    actual = fit_contrasts(df, cfg)
    df["L24h"] = df.Lden
    expected = legacy.run_models(df)
    merged = actual.merge(
        expected,
        on=["model", "comparison", "gap_column"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    assert len(merged) == 48
    for metric in ["beta", "ci_low", "ci_high", "p_value", "bonferroni_p"]:
        np.testing.assert_allclose(
            merged[metric + "_new"], merged[metric + "_old"], rtol=1e-9, atol=1e-11
        )
    np.testing.assert_array_equal(merged.n, merged.model_n)


def test_protein_ols_matches_independent_statsmodels_fit():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(100, 2))
    y = 0.5 * x[:, 0] + rng.normal(size=100)
    result = core.fit_ols_terms(
        y, pd.DataFrame(x, columns=["protein", "covariate"]), ["protein"]
    )["protein"]
    expected = sm.OLS(y, sm.add_constant(x)).fit()
    assert np.isclose(result["beta"], expected.params[1])
    assert np.isclose(result["se"], expected.bse[1])


def test_protein_missingness_boundary():
    df = pd.DataFrame({"p1": [np.nan] * 3 + [1.0] * 7, "p2": [np.nan] * 4 + [1.0] * 6})
    kept, qc = core.select_proteins_by_missingness(df, ["p1", "p2"], 0.3)
    assert kept == ["p1"] and len(qc) == 2


def test_spline_tests_and_intervals_finite(synthetic_data_dir):
    df = define_exposures(pd.read_csv(synthetic_data_dir / "participants.csv"))
    cfg = read_config(ROOT / "config/analysis.yaml")
    results, curves = fit_splines(df, cfg)
    assert len(results) == 8 and len(curves) == 808
    assert np.isfinite(results.overall_p).all()
    assert (curves.ci_low <= curves.beta).all() and (
        curves.ci_high >= curves.beta
    ).all()
