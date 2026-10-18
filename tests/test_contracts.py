from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from noise_agegap_pipeline.common import read_config, unique_eid, adjust, resolve_paths
from noise_agegap_pipeline.data_preparation.cohorts import (
    define_exposures,
    merge_sources,
    recode_fields,
)
from noise_agegap_pipeline.agegap_phenotyping.phenotypes import residualize
from noise_agegap_pipeline.age_inference.inference import (
    parse_response,
    validate_profile,
)
from noise_agegap_pipeline.clinical_outcomes.outcomes import build_riskset
from noise_agegap_pipeline.bridge_enrichment.bridge import select_candidates
from noise_agegap_pipeline.correlation_structure.structure import pca

ROOT = Path(__file__).resolve().parents[1]


def test_threshold_missing_and_fourth_group():
    frame = pd.DataFrame(
        {
            "Lday": [59, 60, 60, 59, np.nan],
            "Lnight": [59, 59, 60, 60, 59],
            "Lden": [59, 60, np.nan, 65, np.nan],
        }
    )
    result = define_exposures(frame)
    assert list(result.noise_group[:3]) == ["NC", "ND", "NN"]
    assert result.noise_group[3:].isna().all()
    assert list(result.lowhigh_group[:2]) == ["Low", "High"]
    assert result.lowhigh_group.iloc[2] is pd.NA
    assert result.lowhigh_group.iloc[4] is pd.NA


def test_lden_alias_requires_agreement():
    with pytest.raises(ValueError, match="disagree"):
        define_exposures(
            pd.DataFrame({"Lday": [60], "Lnight": [60], "L24h": [60], "Lden": [62]})
        )


def test_unique_merge():
    with pytest.raises(ValueError):
        merge_sources([pd.DataFrame({"eid": [1, 1]})])
    with pytest.raises(ValueError):
        merge_sources(
            [
                pd.DataFrame({"eid": [1], "value": [2]}),
                pd.DataFrame({"eid": [1], "value": [3]}),
            ]
        )


def test_linear_residualization_preserves_missing_domain():
    rng = np.random.default_rng(5)
    age = np.linspace(40, 80, 100)
    df = pd.DataFrame(
        {
            "chronological_age": age,
            "overall_biological_age": age + 0.4 * age + rng.normal(size=100),
        }
    )
    df.loc[3, "overall_biological_age"] = np.nan
    result, models = residualize(df, ["overall"])
    valid = result.dropna()
    assert result.overall_age_gap.isna().sum() == 1
    assert abs(valid.overall_age_gap.mean()) < 1e-10
    assert (
        abs(np.corrcoef(valid.overall_age_gap, valid.chronological_age)[0, 1]) < 1e-10
    )
    assert len(models) == 1 and models.n.iloc[0] == 99


def test_categorical_unknown_and_negative_townsend():
    fields = read_config(ROOT / "config/fields.yaml")
    df = pd.DataFrame(
        {
            "eid": [1, 2, 3],
            "education_code": [1, -7, -3],
            "ethnicity_code": [1001, -1, 3001],
            "townsend": [-3.0, 2.0, np.nan],
        }
    )
    result = recode_fields(df, fields)
    assert list(result.education_degree) == ["Degree", "No degree", "Unknown"]
    assert list(result.ethnicity_group) == ["White British", "Unknown", "Other"]
    assert result.townsend.iloc[0] == -3


def test_response_and_noise_exclusion():
    systems = read_config(ROOT / "config/analysis.yaml")["systems"]
    obj = {s.title(): 60 for s in systems}
    assert len(parse_response("```json\n" + json.dumps(obj) + "\n```", systems)) == 8
    obj["Nervous"] = True
    with pytest.raises(ValueError):
        parse_response(json.dumps(obj), systems)
    fields = read_config(ROOT / "config/fields.yaml")
    for key in ["Lden", "24024-0.0", "noise_group"]:
        with pytest.raises(ValueError):
            validate_profile(
                {"nested": {key: 60}}, fields["noise_profile_excluded_fields"]
            )
    with pytest.raises(ValueError):
        validate_profile("free-text data", fields["noise_profile_excluded_fields"])


def test_riskset_prevalent_death_loss_administrative():
    cfg = read_config(ROOT / "config/analysis.yaml")
    cfg["clinical"]["loss_column"] = "loss"
    df = pd.DataFrame(
        {
            "eid": [1, 2, 3, 4, 5],
            "53-0.0": ["2010-01-01"] * 5,
            "event": [
                "2009-01-01",
                "2012-01-01",
                "2013-01-01",
                "2024-01-01",
                "2014-01-01",
            ],
            "40000-0.0": [None, None, "2011-01-01", None, None],
            "loss": [None, None, None, None, "2012-01-01"],
        }
    )
    block, qc = build_riskset(df, "event", cfg)
    assert list(block.eid) == [2, 3, 4, 5]
    assert list(block.event) == [1, 0, 0, 0]
    assert qc["prevalent_excluded_n"] == 1
    assert np.isclose(block.loc[block.eid == 3, "time_years"].iloc[0], 365 / 365.25)
    assert block.loc[block.eid == 4, "time_years"].iloc[0] < 13


def test_fdr_separate_families():
    frame = pd.DataFrame(
        {"family": ["a", "a", "b", "b"], "p_value": [0.01, 0.04, 0.2, 0.8]}
    )
    result = adjust(frame, ["family"])
    np.testing.assert_allclose(result.fdr_p, [0.02, 0.04, 0.4, 0.8])


def test_bridge_is_nominal_noise_and_overall_fdr():
    noise = pd.DataFrame(
        {
            "protein": ["p1", "p2", "p3"],
            "contrast": ["NN_vs_NC"] * 3,
            "beta": [1, -1, 1],
            "p_value": [0.01, 0.01, 0.2],
        }
    )
    age = pd.DataFrame(
        {
            "protein": ["p1", "p2", "p3"],
            "system_col": ["overall_age_gap"] * 3,
            "beta": [2, 2, 2],
            "fdr_p": [0.01, 0.08, 0.01],
        }
    )
    result = select_candidates(noise, age, {"noise_p": 0.05, "agegap_fdr": 0.05})
    assert set(result.loc[result.eligible, "protein"]) == {"p1"}
    assert result.loc[result.protein == "p2", "bridge_score"].iloc[0] == -2


def test_pca_respects_selected_systems(synthetic_data_dir):
    df = pd.read_csv(synthetic_data_dir / "participants.csv")
    cols = [
        s + "_age_gap"
        for s in read_config(ROOT / "config/analysis.yaml")["systems"]
        if s != "overall"
    ]
    scores, variance, loadings = pca(define_exposures(df), cols)
    assert "overall_age_gap" not in loadings.system.tolist()
    assert len(variance) == 7 and np.isclose(variance.variance_fraction.sum(), 1)


def test_source_data_allowlist_excludes_participants():
    from noise_agegap_pipeline.reporting.export import SOURCE_TABLES

    for value in SOURCE_TABLES:
        assert not any(
            x in value for x in ["pca_scores", "analysis_dataset", "phenotypes.csv"]
        )


def test_clinical_raw_baseline_overrides_imaging_covariates():
    fields = read_config(ROOT / "config/fields.yaml")
    data = pd.DataFrame(
        {
            "eid": [1],
            "chronological_age": [70.0],
            "21003-0.0": [55.0],
            "bmi": [30.0],
            "21001-0.0": [24.0],
            "smoking_status": ["Current"],
            "20116-0.0": [0],
        }
    )
    result = recode_fields(data, fields, clinical=True)
    assert result.chronological_age.iloc[0] == 55 and result.bmi.iloc[0] == 24
    assert result.smoking_status.iloc[0] == "Never"


def test_profile_whitelist_missingness_denominator_and_noise():
    from noise_agegap_pipeline.data_preparation.profiles import build_profiles

    fields = read_config(ROOT / "config/fields.yaml")
    data = pd.DataFrame(
        {
            "eid": [1, 2],
            "chronological_age": [60, 60],
            "a": [1, np.nan],
            "b": [np.nan, np.nan],
        }
    )
    records, audit = build_profiles(
        data, ["a", "b"], fields["noise_profile_excluded_fields"]
    )
    assert len(records) == 1 and list(audit.excluded_gt_60pct) == [False, True]
    with pytest.raises(ValueError):
        build_profiles(data, [], fields["noise_profile_excluded_fields"])
