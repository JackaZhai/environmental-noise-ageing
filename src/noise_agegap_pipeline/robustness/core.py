"""Core statistics for the legacy-AgeGap supplemental robustness analyses.

This module is deliberately side-effect free: it never searches for inputs and it
never writes results.  The command-line runner is responsible for passing the
single frozen input file and serialising the returned data frames/diagnostics.

The independent analysis unit throughout is one participant (``eid``).  All
bootstrap resampling therefore operates on participant rows, never on individual
variables or fitted residuals.
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Frozen input contract and analysis constants
# ---------------------------------------------------------------------------

EXPECTED_INPUT_SHA256 = "0e3588128566b120db0b30b74b95c9da5fc6e68cc777ee44ddc41d899666faf4"
EXPECTED_INPUT_ROWS = 50_627
EXPECTED_GROUP_COUNTS: Mapping[str, int] = {"NC": 45_055, "ND": 4_432, "NN": 1_140}
NOISE_GROUP_ORDER: tuple[str, ...] = ("NC", "ND", "NN")

AGEGAP_COLUMNS: tuple[str, ...] = (
    "overall_age_gap",
    "cardiovascular_age_gap",
    "hepatic_age_gap",
    "pulmonary_age_gap",
    "renal_age_gap",
    "metabolic_age_gap",
    "musculoskeletal_age_gap",
    "nervous_age_gap",
)
NERVOUS_GAP_COLUMN = "nervous_age_gap"
NON_NERVOUS_GAP_COLUMNS: tuple[str, ...] = tuple(c for c in AGEGAP_COLUMNS if c != NERVOUS_GAP_COLUMN)

BASE_CONTINUOUS_COVARIATES: tuple[str, ...] = (
    "chronological_age",
    "bmi",
    "townsend",
)
POLLUTANT_COLUMNS: tuple[str, ...] = ("no2_2010", "pm25_2010")
BASE_CATEGORICAL_COVARIATES: tuple[str, ...] = (
    "sex_code",
    "ethnicity_group",
    "education_degree",
    "smoking_status",
    "alcohol_frequency",
)
MODEL_D_REQUIRED_COLUMNS: tuple[str, ...] = (
    "noise_group",
    *BASE_CONTINUOUS_COVARIATES,
    *BASE_CATEGORICAL_COVARIATES,
    *POLLUTANT_COLUMNS,
)

BASE_ADJUSTMENT_FORMULA = (
    "chronological_age + C(sex_code) + C(ethnicity_group) + bmi + "
    "C(education_degree) + townsend + C(smoking_status) + C(alcohol_frequency)"
)
GROUP_TERM_PREFIX = "C(noise_group, Treatment(reference='NC'))"
GROUP_TERMS: Mapping[str, str] = {
    "ND vs NC": f"{GROUP_TERM_PREFIX}[T.ND]",
    "NN vs NC": f"{GROUP_TERM_PREFIX}[T.NN]",
}
SAME_SAMPLE_MODEL_FORMULAS: Mapping[str, str] = {
    "Model A": f"{{gap}} ~ {GROUP_TERM_PREFIX} + {BASE_ADJUSTMENT_FORMULA}",
    "Model B": f"{{gap}} ~ {GROUP_TERM_PREFIX} + {BASE_ADJUSTMENT_FORMULA} + no2_2010",
    "Model C": f"{{gap}} ~ {GROUP_TERM_PREFIX} + {BASE_ADJUSTMENT_FORMULA} + pm25_2010",
    "Model D": (
        f"{{gap}} ~ {GROUP_TERM_PREFIX} + {BASE_ADJUSTMENT_FORMULA} + "
        "pm25_2010 + no2_2010"
    ),
}

PROPENSITY_CONTINUOUS_COVARIATES: tuple[str, ...] = (
    "chronological_age",
    "bmi",
    "townsend",
    "no2_2010",
    "pm25_2010",
)
PROPENSITY_CATEGORICAL_COVARIATES: tuple[str, ...] = BASE_CATEGORICAL_COVARIATES
DEFAULT_SPLINE_DF = 4
DEFAULT_BALANCE_THRESHOLD = 0.10
DEFAULT_BOOTSTRAP_SUCCESS_THRESHOLD = 0.99

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "eid",
    *AGEGAP_COLUMNS,
    "brain_imaging_sample",
    "brain_mri_nonmissing_fields",
    "Lday",
    "Lnight",
    "urban_rural_group",
    *MODEL_D_REQUIRED_COLUMNS,
)

SAME_SAMPLE_RESULT_COLUMNS: tuple[str, ...] = (
    "system",
    "gap_column",
    "model",
    "comparison",
    "beta",
    "se",
    "ci_low",
    "ci_high",
    "p_value",
    "bonferroni_p",
    "model_n",
    "nc_n",
    "nd_n",
    "nn_n",
    "outcome_sd",
    "sample_eid_sha256",
    "model_a_beta",
    "beta_minus_model_a",
    "absolute_beta_difference_from_model_a",
    "percent_attenuation_from_model_a",
)

INTERACTION_RESULT_COLUMNS: tuple[str, ...] = (
    "system",
    "gap_column",
    "pollutant",
    "other_pollutant",
    "pollutant_mean",
    "pollutant_sd",
    "model_n",
    "nc_n",
    "nd_n",
    "nn_n",
    "outcome_sd",
    "sample_eid_sha256",
    "support_low",
    "support_high",
    "support_valid",
    "nd_beta",
    "nd_se",
    "nd_ci_low",
    "nd_ci_high",
    "nd_p_value",
    "nn_beta",
    "nn_se",
    "nn_ci_low",
    "nn_ci_high",
    "nn_p_value",
    "nn_bh_fdr_p",
    "global_wald_chi2",
    "global_wald_df",
    "global_p_value",
    "global_bh_fdr_p",
)

MARGINAL_RESULT_COLUMNS: tuple[str, ...] = (
    "system",
    "gap_column",
    "pollutant",
    "comparison",
    "point_label",
    "point_quantile",
    "pollutant_value",
    "pollutant_z",
    "estimate",
    "se",
    "ci_low",
    "ci_high",
    "p_value",
    "model_n",
    "support_low",
    "support_high",
    "within_common_support",
)

OVERLAP_EFFECT_COLUMNS: tuple[str, ...] = (
    "system",
    "gap_column",
    "analysis_variant",
    "analysis_set",
    "comparison",
    "estimate",
    "se",
    "ci_low",
    "ci_high",
    "p_value",
    "bootstrap_reps_requested",
    "bootstrap_reps_successful",
    "bootstrap_success_rate",
    "confirmatory_inference_valid",
    "model_n",
    "nc_n",
    "nn_n",
    "nc_ess",
    "nn_ess",
    "total_ess",
    "max_abs_weighted_smd",
    "balance_pass",
    "outcome_sd",
    "standardized_estimate",
    "seed",
)

OVERLAP_BALANCE_COLUMNS: tuple[str, ...] = (
    "analysis_variant",
    "analysis_set",
    "covariate",
    "basis_term",
    "variable_type",
    "unweighted_smd",
    "weighted_smd",
    "abs_weighted_smd",
    "balance_pass",
    "n",
    "nc_n",
    "nn_n",
    "nc_ess",
    "nn_ess",
    "total_ess",
    "ps_min",
    "ps_p01",
    "ps_median",
    "ps_p99",
    "ps_max",
    "weight_min",
    "weight_p01",
    "weight_median",
    "weight_p99",
    "weight_max",
)

MAIN_MDE_COLUMNS: tuple[str, ...] = (
    "system",
    "gap_column",
    "comparison",
    "model",
    "power",
    "alpha",
    "correction",
    "standard_error",
    "outcome_sd",
    "mde_agegap_years",
    "mde_standardized_sd",
    "model_n",
)

INTERACTION_MDE_COLUMNS: tuple[str, ...] = (
    "system",
    "gap_column",
    "pollutant",
    "term",
    "power",
    "alpha",
    "correction",
    "standard_error",
    "pollutant_scale",
    "outcome_sd",
    "mde_agegap_years_per_1sd_pollutant",
    "mde_standardized_outcome_sd_per_1sd_pollutant",
    "model_n",
)


class DataContractError(ValueError):
    """Raised when the frozen legacy analysis input fails its contract."""


def _system_label(gap_column: str) -> str:
    return gap_column.removesuffix("_age_gap").replace("_", " ").title()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eid_digest(values: Iterable[Any]) -> str:
    eids = pd.Series(list(values)).dropna().astype("int64").sort_values().astype(str)
    payload = "\n".join(eids.tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def validate_input_contract(
    df: pd.DataFrame,
    source_path: str | Path | None = None,
    *,
    enforce_hash: bool = True,
    expected_hash: str = EXPECTED_INPUT_SHA256,
    expected_rows: int = EXPECTED_INPUT_ROWS,
    expected_group_counts: Mapping[str, int] = EXPECTED_GROUP_COUNTS,
) -> dict[str, Any]:
    """Validate the immutable legacy-AgeGap analysis table.

    ``source_path`` is required when ``enforce_hash`` is true.  Tests using
    synthetic frames can set ``enforce_hash=False`` while retaining all schema
    and logical checks by overriding the expected row/group counts.
    """

    errors: list[str] = []
    file_hash: str | None = None
    resolved_path: str | None = None

    if source_path is not None:
        path = Path(source_path).expanduser().resolve()
        resolved_path = str(path)
        if not path.is_file():
            errors.append(f"input path is not a regular file: {path}")
        else:
            file_hash = sha256_file(path)
            if enforce_hash and file_hash != expected_hash:
                errors.append(f"SHA-256 mismatch: expected {expected_hash}, observed {file_hash}")
    elif enforce_hash:
        errors.append("source_path is required when enforce_hash=True")

    missing_columns = sorted(set(REQUIRED_INPUT_COLUMNS).difference(df.columns))
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")

    observed_rows = int(len(df))
    if observed_rows != int(expected_rows):
        errors.append(f"row count mismatch: expected {expected_rows}, observed {observed_rows}")

    if "eid" in df.columns:
        missing_eids = int(df["eid"].isna().sum())
        unique_eids = int(df["eid"].nunique(dropna=True))
        if missing_eids:
            errors.append(f"eid contains {missing_eids} missing values")
        if unique_eids != observed_rows:
            errors.append(f"eid is not unique: {unique_eids} unique values among {observed_rows} rows")
    else:
        unique_eids = 0

    observed_counts: dict[str, int] = {}
    if "noise_group" in df.columns:
        observed_counts = {str(k): int(v) for k, v in df["noise_group"].value_counts(dropna=False).items()}
        expected_counts = {str(k): int(v) for k, v in expected_group_counts.items()}
        clean_observed = {g: int((df["noise_group"] == g).sum()) for g in NOISE_GROUP_ORDER}
        unexpected = sorted(set(df["noise_group"].dropna().astype(str).unique()).difference(NOISE_GROUP_ORDER))
        missing_groups = int(df["noise_group"].isna().sum())
        if clean_observed != expected_counts:
            errors.append(f"noise-group counts mismatch: expected {expected_counts}, observed {clean_observed}")
        if unexpected:
            errors.append(f"unexpected noise_group values: {unexpected}")
        if missing_groups:
            errors.append(f"noise_group contains {missing_groups} missing values")

    if {"Lday", "Lnight", "noise_group"}.issubset(df.columns):
        expected_group = np.select(
            [
                (df["Lday"] < 60.0) & (df["Lnight"] < 60.0),
                (df["Lday"] >= 60.0) & (df["Lnight"] < 60.0),
                (df["Lday"] >= 60.0) & (df["Lnight"] >= 60.0),
            ],
            ["NC", "ND", "NN"],
            default="UNCLASSIFIED",
        )
        mismatch_n = int(np.sum(expected_group != df["noise_group"].astype(str).to_numpy()))
        if mismatch_n:
            errors.append(f"{mismatch_n} rows conflict with the frozen 60-dB NC/ND/NN rules")

    for column in AGEGAP_COLUMNS:
        if column in df.columns and not pd.api.types.is_numeric_dtype(df[column]):
            errors.append(f"{column} must be numeric")

    if "brain_imaging_sample" in df.columns:
        brain_values = set(pd.to_numeric(df["brain_imaging_sample"], errors="coerce").dropna().unique())
        if not brain_values.issubset({0, 1}):
            errors.append(f"brain_imaging_sample must contain only 0/1; observed {sorted(brain_values)}")
        if df["brain_imaging_sample"].isna().any():
            errors.append("brain_imaging_sample contains missing values")
        if NERVOUS_GAP_COLUMN in df.columns:
            invalid_nervous = df[NERVOUS_GAP_COLUMN].notna() & (df["brain_imaging_sample"] != 1)
            if invalid_nervous.any():
                errors.append(
                    f"{int(invalid_nervous.sum())} non-imaging rows contain a Nervous AgeGap value"
                )

    diagnostics: dict[str, Any] = {
        "contract_valid": not errors,
        "input_path": resolved_path,
        "input_sha256": file_hash,
        "expected_sha256": expected_hash,
        "n_rows": observed_rows,
        "n_unique_eid": unique_eids,
        "group_counts": {g: int((df.get("noise_group") == g).sum()) for g in NOISE_GROUP_ORDER}
        if "noise_group" in df.columns
        else {},
        "agegap_columns": list(AGEGAP_COLUMNS),
        "brain_marker_column": "brain_imaging_sample",
        "errors": errors,
    }
    if errors:
        raise DataContractError("Frozen legacy-AgeGap input contract failed:\n- " + "\n- ".join(errors))
    return diagnostics


def load_and_validate_input(
    path: str | Path,
    *,
    enforce_hash: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and validate the sole permitted legacy-AgeGap input CSV."""

    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    # Hash before parsing so a modified file can never silently enter analysis.
    observed_hash = sha256_file(input_path)
    if enforce_hash and observed_hash != EXPECTED_INPUT_SHA256:
        raise DataContractError(
            "Frozen legacy-AgeGap input contract failed:\n- "
            f"SHA-256 mismatch: expected {EXPECTED_INPUT_SHA256}, observed {observed_hash}"
        )
    df = pd.read_csv(input_path)
    diagnostics = validate_input_contract(df, input_path, enforce_hash=enforce_hash)
    return df, diagnostics


# ---------------------------------------------------------------------------
# Same-sample OLS + HC3 models
# ---------------------------------------------------------------------------


def _model_d_complete_cases(df: pd.DataFrame, gap_column: str) -> pd.DataFrame:
    _require_columns(df, ("eid", gap_column, *MODEL_D_REQUIRED_COLUMNS), "analysis data")
    columns = ["eid", gap_column, *MODEL_D_REQUIRED_COLUMNS]
    data = df.loc[:, columns].dropna().copy()
    data = data[data["noise_group"].isin(NOISE_GROUP_ORDER)].copy()
    data["noise_group"] = pd.Categorical(data["noise_group"], categories=NOISE_GROUP_ORDER, ordered=True)
    return data


def _fit_ols_hc3(formula: str, data: pd.DataFrame):
    return smf.ols(formula=formula, data=data).fit(cov_type="HC3")


def _extract_term(model: Any, term: str) -> tuple[float, float, float, float, float]:
    if term not in model.params.index:
        return (math.nan,) * 5
    estimate = float(model.params[term])
    se = float(model.bse[term])
    ci = model.conf_int().loc[term]
    return estimate, se, float(ci.iloc[0]), float(ci.iloc[1]), float(model.pvalues[term])


def fit_same_sample_pollutant_models(
    df: pd.DataFrame,
    gap_columns: Sequence[str] = AGEGAP_COLUMNS,
    *,
    bonferroni_family_size: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit four nested pollutant-adjustment models on identical outcome samples.

    Each outcome's complete-case sample is determined once using all Model D
    variables; Models A--D then use that exact participant set.
    """

    rows: list[dict[str, Any]] = []
    sample_diagnostics: list[dict[str, Any]] = []

    for gap_column in gap_columns:
        data = _model_d_complete_cases(df, gap_column)
        if data.empty:
            raise ValueError(f"No complete Model D cases for {gap_column}")
        counts = data["noise_group"].value_counts().reindex(NOISE_GROUP_ORDER, fill_value=0)
        if (counts == 0).any():
            raise ValueError(f"At least one noise group is absent for {gap_column}: {counts.to_dict()}")
        sample_hash = _eid_digest(data["eid"])
        outcome_sd = float(data[gap_column].std(ddof=1))
        sample_diagnostics.append(
            {
                "gap_column": gap_column,
                "model_n": int(len(data)),
                "nc_n": int(counts["NC"]),
                "nd_n": int(counts["ND"]),
                "nn_n": int(counts["NN"]),
                "sample_eid_sha256": sample_hash,
            }
        )

        for model_name, formula_template in SAME_SAMPLE_MODEL_FORMULAS.items():
            model = _fit_ols_hc3(formula_template.format(gap=gap_column), data)
            for comparison, term in GROUP_TERMS.items():
                beta, se, ci_low, ci_high, p_value = _extract_term(model, term)
                rows.append(
                    {
                        "system": _system_label(gap_column),
                        "gap_column": gap_column,
                        "model": model_name,
                        "comparison": comparison,
                        "beta": beta,
                        "se": se,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value": p_value,
                        "bonferroni_p": min(1.0, p_value * bonferroni_family_size)
                        if np.isfinite(p_value)
                        else math.nan,
                        "model_n": int(len(data)),
                        "nc_n": int(counts["NC"]),
                        "nd_n": int(counts["ND"]),
                        "nn_n": int(counts["NN"]),
                        "outcome_sd": outcome_sd,
                        "sample_eid_sha256": sample_hash,
                    }
                )

    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=SAME_SAMPLE_RESULT_COLUMNS)
    else:
        model_a = (
            result[result["model"] == "Model A"]
            .set_index(["gap_column", "comparison"])["beta"]
            .to_dict()
        )
        result["model_a_beta"] = [model_a.get((g, c), math.nan) for g, c in zip(result["gap_column"], result["comparison"])]
        result["beta_minus_model_a"] = result["beta"] - result["model_a_beta"]
        result["absolute_beta_difference_from_model_a"] = result["beta_minus_model_a"].abs()
        denominator = result["model_a_beta"].abs().replace(0.0, np.nan)
        result["percent_attenuation_from_model_a"] = (
            (result["model_a_beta"].abs() - result["beta"].abs()) / denominator * 100.0
        )
        result = result.loc[:, SAME_SAMPLE_RESULT_COLUMNS]

    diagnostics = {
        "analysis": "same_sample_pollutant_models",
        "independent_unit": "participant (eid)",
        "covariance": "HC3",
        "bonferroni_family_size": int(bonferroni_family_size),
        "samples": sample_diagnostics,
        "model_formulas": dict(SAME_SAMPLE_MODEL_FORMULAS),
    }
    return result, diagnostics


# ---------------------------------------------------------------------------
# Pollutant interactions and strictly in-support marginal effects
# ---------------------------------------------------------------------------


def _bh_adjust(values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    mask = values.notna() & np.isfinite(values)
    if mask.any():
        adjusted.loc[mask] = multipletests(values.loc[mask].to_numpy(), method="fdr_bh")[1]
    return adjusted


def _wald_joint_terms(model: Any, terms: Sequence[str]) -> tuple[float, int, float]:
    names = list(model.params.index)
    if any(term not in names for term in terms):
        return math.nan, 0, math.nan
    indices = [names.index(term) for term in terms]
    beta = np.asarray(model.params, dtype=float)[indices]
    covariance = np.asarray(model.cov_params(), dtype=float)[np.ix_(indices, indices)]
    df_test = int(np.linalg.matrix_rank(covariance))
    if df_test == 0:
        return math.nan, 0, math.nan
    statistic = float(beta.T @ np.linalg.pinv(covariance) @ beta)
    return statistic, df_test, float(chi2.sf(statistic, df_test))


def _common_support(
    data: pd.DataFrame,
    pollutant: str,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float, bool, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    lows: list[float] = []
    highs: list[float] = []
    for group in NOISE_GROUP_ORDER:
        values = data.loc[data["noise_group"] == group, pollutant].dropna()
        low = float(values.quantile(lower_quantile)) if not values.empty else math.nan
        high = float(values.quantile(upper_quantile)) if not values.empty else math.nan
        details.append(
            {
                "noise_group": group,
                "pollutant": pollutant,
                "n": int(len(values)),
                "group_q05": low,
                "group_q95": high,
            }
        )
        lows.append(low)
        highs.append(high)
    support_low = float(np.nanmax(lows)) if np.isfinite(lows).any() else math.nan
    support_high = float(np.nanmin(highs)) if np.isfinite(highs).any() else math.nan
    valid = bool(np.isfinite(support_low) and np.isfinite(support_high) and support_low < support_high)
    return support_low, support_high, valid, details


def _linear_contrast(
    model: Any,
    coefficients: Mapping[str, float],
) -> tuple[float, float, float, float, float]:
    names = list(model.params.index)
    contrast = np.zeros(len(names), dtype=float)
    for term, value in coefficients.items():
        if term not in names:
            return (math.nan,) * 5
        contrast[names.index(term)] = float(value)
    params = np.asarray(model.params, dtype=float)
    covariance = np.asarray(model.cov_params(), dtype=float)
    estimate = float(contrast @ params)
    variance = float(contrast @ covariance @ contrast)
    variance = max(variance, 0.0)
    se = math.sqrt(variance)
    z_critical = float(norm.ppf(0.975))
    if se > 0:
        p_value = float(2.0 * norm.sf(abs(estimate / se)))
    else:
        p_value = 0.0 if estimate != 0 else 1.0
    return estimate, se, estimate - z_critical * se, estimate + z_critical * se, p_value


def fit_pollutant_interactions(
    df: pd.DataFrame,
    gap_columns: Sequence[str] = AGEGAP_COLUMNS,
    *,
    marginal_quantiles: Sequence[float] = (0.25, 0.50, 0.75),
    support_quantiles: tuple[float, float] = (0.05, 0.95),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit NO2/PM2.5 interactions and calculate in-support marginal effects.

    Marginal quantiles are evaluated *after restricting pollutant observations to
    the three-group common-support interval*.  Rows outside that interval are
    never emitted, preventing accidental extrapolation by downstream figures.
    """

    marginal_quantiles = tuple(float(q) for q in marginal_quantiles)
    if not marginal_quantiles or any(q < 0 or q > 1 for q in marginal_quantiles):
        raise ValueError("marginal_quantiles must contain values in [0, 1]")
    q_low, q_high = map(float, support_quantiles)
    if not (0 <= q_low < q_high <= 1):
        raise ValueError("support_quantiles must satisfy 0 <= low < high <= 1")

    interaction_rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    support_diagnostics: list[dict[str, Any]] = []

    for gap_column in gap_columns:
        data = _model_d_complete_cases(df, gap_column)
        counts = data["noise_group"].value_counts().reindex(NOISE_GROUP_ORDER, fill_value=0)
        outcome_sd = float(data[gap_column].std(ddof=1))
        sample_hash = _eid_digest(data["eid"])
        for pollutant in POLLUTANT_COLUMNS:
            other_pollutant = next(p for p in POLLUTANT_COLUMNS if p != pollutant)
            pollutant_mean = float(data[pollutant].mean())
            pollutant_sd = float(data[pollutant].std(ddof=1))
            other_mean = float(data[other_pollutant].mean())
            other_sd = float(data[other_pollutant].std(ddof=1))
            if not np.isfinite(pollutant_sd) or pollutant_sd <= 0:
                raise ValueError(f"Cannot standardize constant pollutant {pollutant} for {gap_column}")
            if not np.isfinite(other_sd) or other_sd <= 0:
                raise ValueError(f"Cannot standardize constant pollutant {other_pollutant} for {gap_column}")

            pollutant_z = f"{pollutant}_z"
            other_z = f"{other_pollutant}_z"
            data[pollutant_z] = (data[pollutant] - pollutant_mean) / pollutant_sd
            data[other_z] = (data[other_pollutant] - other_mean) / other_sd
            formula = (
                f"{gap_column} ~ {GROUP_TERM_PREFIX} * {pollutant_z} + "
                f"{BASE_ADJUSTMENT_FORMULA} + {other_z}"
            )
            model = _fit_ols_hc3(formula, data)
            interaction_terms = {
                "ND": f"{GROUP_TERM_PREFIX}[T.ND]:{pollutant_z}",
                "NN": f"{GROUP_TERM_PREFIX}[T.NN]:{pollutant_z}",
            }
            nd_beta, nd_se, nd_low, nd_high, nd_p = _extract_term(model, interaction_terms["ND"])
            nn_beta, nn_se, nn_low, nn_high, nn_p = _extract_term(model, interaction_terms["NN"])
            global_stat, global_df, global_p = _wald_joint_terms(model, list(interaction_terms.values()))

            support_low, support_high, support_valid, support_details = _common_support(
                data, pollutant, q_low, q_high
            )
            for detail in support_details:
                detail.update(
                    {
                        "gap_column": gap_column,
                        "support_low": support_low,
                        "support_high": support_high,
                        "support_valid": support_valid,
                    }
                )
                support_diagnostics.append(detail)

            interaction_rows.append(
                {
                    "system": _system_label(gap_column),
                    "gap_column": gap_column,
                    "pollutant": pollutant,
                    "other_pollutant": other_pollutant,
                    "pollutant_mean": pollutant_mean,
                    "pollutant_sd": pollutant_sd,
                    "model_n": int(len(data)),
                    "nc_n": int(counts["NC"]),
                    "nd_n": int(counts["ND"]),
                    "nn_n": int(counts["NN"]),
                    "outcome_sd": outcome_sd,
                    "sample_eid_sha256": sample_hash,
                    "support_low": support_low,
                    "support_high": support_high,
                    "support_valid": support_valid,
                    "nd_beta": nd_beta,
                    "nd_se": nd_se,
                    "nd_ci_low": nd_low,
                    "nd_ci_high": nd_high,
                    "nd_p_value": nd_p,
                    "nn_beta": nn_beta,
                    "nn_se": nn_se,
                    "nn_ci_low": nn_low,
                    "nn_ci_high": nn_high,
                    "nn_p_value": nn_p,
                    "global_wald_chi2": global_stat,
                    "global_wald_df": global_df,
                    "global_p_value": global_p,
                }
            )

            if not support_valid:
                continue
            in_support = data.loc[data[pollutant].between(support_low, support_high), pollutant]
            if in_support.empty:
                continue
            point_values = in_support.quantile(list(marginal_quantiles)).to_numpy(dtype=float)
            for point_quantile, pollutant_value in zip(marginal_quantiles, point_values):
                # Quantile interpolation can differ at the last bit; force values
                # onto the closed observed support interval before testing.
                pollutant_value = float(np.clip(pollutant_value, support_low, support_high))
                z_value = (pollutant_value - pollutant_mean) / pollutant_sd
                for group, comparison in (("ND", "ND vs NC"), ("NN", "NN vs NC")):
                    group_term = GROUP_TERMS[comparison]
                    interaction_term = interaction_terms[group]
                    estimate, se, ci_low, ci_high, p_value = _linear_contrast(
                        model,
                        {group_term: 1.0, interaction_term: z_value},
                    )
                    within = bool(support_low <= pollutant_value <= support_high)
                    if not within:
                        # Defensive guard: downstream consumers never receive an
                        # extrapolated marginal-effect row.
                        continue
                    marginal_rows.append(
                        {
                            "system": _system_label(gap_column),
                            "gap_column": gap_column,
                            "pollutant": pollutant,
                            "comparison": comparison,
                            "point_label": f"CS-P{int(round(point_quantile * 100)):02d}",
                            "point_quantile": point_quantile,
                            "pollutant_value": pollutant_value,
                            "pollutant_z": z_value,
                            "estimate": estimate,
                            "se": se,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "p_value": p_value,
                            "model_n": int(len(data)),
                            "support_low": support_low,
                            "support_high": support_high,
                            "within_common_support": True,
                        }
                    )

    interactions = pd.DataFrame(interaction_rows)
    if interactions.empty:
        interactions = pd.DataFrame(columns=INTERACTION_RESULT_COLUMNS)
    else:
        # These are two explicitly separate families of 8 outcomes x 2 pollutants.
        interactions["nn_bh_fdr_p"] = _bh_adjust(interactions["nn_p_value"])
        interactions["global_bh_fdr_p"] = _bh_adjust(interactions["global_p_value"])
        interactions = interactions.loc[:, INTERACTION_RESULT_COLUMNS]

    marginals = pd.DataFrame(marginal_rows)
    if marginals.empty:
        marginals = pd.DataFrame(columns=MARGINAL_RESULT_COLUMNS)
    else:
        marginals = marginals.loc[:, MARGINAL_RESULT_COLUMNS]
        if not bool(marginals["within_common_support"].all()):
            raise AssertionError("Out-of-support marginal effect escaped the core guard")

    diagnostics = {
        "analysis": "pollutant_interactions",
        "independent_unit": "participant (eid)",
        "covariance": "HC3",
        "nn_interaction_family_size": int(len(interactions)),
        "global_interaction_family_size": int(len(interactions)),
        "support_quantiles": [q_low, q_high],
        "marginal_quantiles_within_support": list(marginal_quantiles),
        "common_support_by_group": support_diagnostics,
        "emitted_marginal_rows": int(len(marginals)),
        "extrapolated_rows_emitted": 0,
    }
    return interactions, marginals, diagnostics


# ---------------------------------------------------------------------------
# Overlap weighting with re-estimated propensity scores in every bootstrap
# ---------------------------------------------------------------------------


def _derive_urbanicity_binary(series: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    observed = series.notna()
    labels = series.astype("string")
    result.loc[observed & labels.str.contains("urban", case=False, na=False)] = "Urban"
    result.loc[observed & ~labels.str.contains("urban", case=False, na=False)] = "Non-urban"
    return result


def _propensity_formula(include_urban: bool, spline_df: int) -> str:
    continuous = " + ".join(
        f"cr({column}, df={int(spline_df)}, constraints='center')"
        for column in PROPENSITY_CONTINUOUS_COVARIATES
    )
    categorical = " + ".join(f"C({column})" for column in PROPENSITY_CATEGORICAL_COVARIATES)
    urban = " + C(urbanicity_binary)" if include_urban else ""
    return f"1 + {continuous} + {categorical}{urban}"


def _fit_propensity_array(
    design: np.ndarray,
    treatment: np.ndarray,
    *,
    maxiter: int = 100,
) -> tuple[np.ndarray | None, bool, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = sm.GLM(
                treatment,
                design,
                family=sm.families.Binomial(),
            ).fit(maxiter=maxiter, tol=1e-8, disp=0)
        converged = bool(getattr(fitted, "converged", True))
        iterations = int(getattr(fitted, "fit_history", {}).get("iteration", 0))
        propensity = np.asarray(fitted.predict(design), dtype=float)
        valid = converged and propensity.shape == treatment.shape and np.isfinite(propensity).all()
        valid = valid and bool(((propensity >= 0.0) & (propensity <= 1.0)).all())
        return (propensity if valid else None), bool(valid), iterations
    # A bootstrap replicate is allowed to fail cleanly (for example because of
    # complete separation or a singular resampled design).  The caller records
    # the failure and applies the prespecified 99% convergence threshold.
    except Exception:
        return None, False, 0


def _overlap_weights(treatment: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    return np.where(treatment == 1, 1.0 - propensity, propensity).astype(float, copy=False)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not mask.any():
        return math.nan
    weight_sum = float(np.sum(weights[mask], dtype=np.float64))
    if weight_sum <= np.finfo(float).tiny:
        return math.nan
    return float(np.sum(values[mask] * weights[mask], dtype=np.float64) / weight_sum)


def _weighted_difference(values: np.ndarray, treatment: np.ndarray, weights: np.ndarray) -> float:
    treated = treatment == 1
    control = treatment == 0
    treated_mean = _weighted_mean(values[treated], weights[treated])
    control_mean = _weighted_mean(values[control], weights[control])
    return float(treated_mean - control_mean) if np.isfinite([treated_mean, control_mean]).all() else math.nan


def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights >= 0)]
    if not len(weights):
        return math.nan
    numerator = float(np.sum(weights, dtype=np.float64)) ** 2
    denominator = float(np.sum(np.square(weights), dtype=np.float64))
    return numerator / denominator if denominator > 0 else math.nan


def _smd(values: np.ndarray, treatment: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    mask = np.isfinite(values)
    values = values[mask]
    treatment = treatment[mask]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[mask]
    v0 = values[treatment == 0]
    v1 = values[treatment == 1]
    if len(v0) < 2 or len(v1) < 2:
        return math.nan
    pooled_sd = math.sqrt((float(np.var(v0, ddof=1)) + float(np.var(v1, ddof=1))) / 2.0)
    if weights is None:
        mean0, mean1 = float(np.mean(v0)), float(np.mean(v1))
    else:
        mean0 = _weighted_mean(v0, weights[treatment == 0])
        mean1 = _weighted_mean(v1, weights[treatment == 1])
    difference = mean1 - mean0
    if pooled_sd <= np.finfo(float).eps:
        return 0.0 if abs(difference) <= np.finfo(float).eps else math.copysign(math.inf, difference)
    return float(difference / pooled_sd)


def _balance_features(
    data: pd.DataFrame,
    design: pd.DataFrame,
    *,
    include_urban: bool,
) -> list[tuple[str, str, str, np.ndarray]]:
    features: list[tuple[str, str, str, np.ndarray]] = []
    for covariate in PROPENSITY_CONTINUOUS_COVARIATES:
        features.append((covariate, covariate, "continuous_raw", data[covariate].to_numpy(dtype=float)))
        pattern = re.compile(rf"^cr\({re.escape(covariate)},")
        for term in design.columns:
            if pattern.search(term):
                features.append((covariate, term, "spline_basis", design[term].to_numpy(dtype=float)))

    categorical = list(PROPENSITY_CATEGORICAL_COVARIATES)
    if include_urban:
        categorical.append("urbanicity_binary")
    for covariate in categorical:
        levels = sorted(data[covariate].dropna().astype(str).unique())
        for level in levels:
            values = (data[covariate].astype(str) == level).astype(float).to_numpy()
            features.append((covariate, f"{covariate}={level}", "categorical_indicator", values))
    return features


def _balance_table(
    data: pd.DataFrame,
    design: pd.DataFrame,
    treatment: np.ndarray,
    propensity: np.ndarray,
    weights: np.ndarray,
    *,
    analysis_variant: str,
    analysis_set: str,
    include_urban: bool,
    balance_threshold: float,
) -> pd.DataFrame:
    nc = treatment == 0
    nn = treatment == 1
    nc_ess = _effective_sample_size(weights[nc])
    nn_ess = _effective_sample_size(weights[nn])
    total_ess = _effective_sample_size(weights)
    ps_quantiles = np.quantile(propensity, [0.0, 0.01, 0.50, 0.99, 1.0])
    weight_quantiles = np.quantile(weights, [0.0, 0.01, 0.50, 0.99, 1.0])
    shared = {
        "analysis_variant": analysis_variant,
        "analysis_set": analysis_set,
        "n": int(len(data)),
        "nc_n": int(nc.sum()),
        "nn_n": int(nn.sum()),
        "nc_ess": nc_ess,
        "nn_ess": nn_ess,
        "total_ess": total_ess,
        "ps_min": float(ps_quantiles[0]),
        "ps_p01": float(ps_quantiles[1]),
        "ps_median": float(ps_quantiles[2]),
        "ps_p99": float(ps_quantiles[3]),
        "ps_max": float(ps_quantiles[4]),
        "weight_min": float(weight_quantiles[0]),
        "weight_p01": float(weight_quantiles[1]),
        "weight_median": float(weight_quantiles[2]),
        "weight_p99": float(weight_quantiles[3]),
        "weight_max": float(weight_quantiles[4]),
    }
    rows: list[dict[str, Any]] = []
    for covariate, basis_term, variable_type, values in _balance_features(
        data, design, include_urban=include_urban
    ):
        unweighted = _smd(values, treatment)
        weighted = _smd(values, treatment, weights)
        abs_weighted = abs(weighted) if np.isfinite(weighted) else math.inf
        rows.append(
            {
                **shared,
                "covariate": covariate,
                "basis_term": basis_term,
                "variable_type": variable_type,
                "unweighted_smd": unweighted,
                "weighted_smd": weighted,
                "abs_weighted_smd": abs_weighted,
                "balance_pass": bool(abs_weighted < balance_threshold),
            }
        )
    return pd.DataFrame(rows).loc[:, OVERLAP_BALANCE_COLUMNS]


def _one_overlap_bootstrap(
    replicate_index: int,
    replicate_seed: int,
    design: np.ndarray,
    treatment: np.ndarray,
    outcomes: np.ndarray,
    maxiter: int,
) -> tuple[int, np.ndarray | None, int]:
    rng = np.random.default_rng(replicate_seed)
    control_indices = np.flatnonzero(treatment == 0)
    treated_indices = np.flatnonzero(treatment == 1)
    sampled = np.concatenate(
        [
            rng.choice(control_indices, size=len(control_indices), replace=True),
            rng.choice(treated_indices, size=len(treated_indices), replace=True),
        ]
    )
    xb = design[sampled]
    tb = treatment[sampled]
    propensity, converged, iterations = _fit_propensity_array(xb, tb, maxiter=maxiter)
    if not converged or propensity is None:
        return replicate_index, None, iterations
    weights = _overlap_weights(tb, propensity)
    estimates = np.asarray(
        [_weighted_difference(outcomes[sampled, column], tb, weights) for column in range(outcomes.shape[1])],
        dtype=float,
    )
    if not np.isfinite(estimates).all():
        return replicate_index, None, iterations
    return replicate_index, estimates, iterations


def _run_overlap_bootstrap(
    design: np.ndarray,
    treatment: np.ndarray,
    outcomes: np.ndarray,
    *,
    bootstrap_reps: int,
    seed: int,
    n_jobs: int,
    maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if bootstrap_reps < 0:
        raise ValueError("bootstrap_reps must be non-negative")
    output = np.full((bootstrap_reps, outcomes.shape[1]), np.nan, dtype=float)
    if bootstrap_reps == 0:
        return output, {"successful": 0, "failed": 0, "max_iterations": 0}

    seed_sequence = np.random.SeedSequence(int(seed))
    child_seeds = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in seed_sequence.spawn(bootstrap_reps)]

    def run(rep: int) -> tuple[int, np.ndarray | None, int]:
        return _one_overlap_bootstrap(
            rep,
            child_seeds[rep],
            design,
            treatment,
            outcomes,
            maxiter,
        )

    iteration_counts: list[int] = []
    if int(n_jobs) <= 1:
        completed = map(run, range(bootstrap_reps))
        for rep, estimates, iterations in completed:
            iteration_counts.append(iterations)
            if estimates is not None:
                output[rep] = estimates
    else:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
            futures = {executor.submit(run, rep): rep for rep in range(bootstrap_reps)}
            for future in as_completed(futures):
                rep, estimates, iterations = future.result()
                iteration_counts.append(iterations)
                if estimates is not None:
                    output[rep] = estimates

    successful = int(np.isfinite(output).all(axis=1).sum())
    return output, {
        "successful": successful,
        "failed": int(bootstrap_reps - successful),
        "max_iterations": int(max(iteration_counts, default=0)),
    }


def _prepare_overlap_dataset(
    df: pd.DataFrame,
    *,
    analysis_set: str,
    include_urban: bool,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    required = ["eid", "noise_group", *PROPENSITY_CONTINUOUS_COVARIATES, *PROPENSITY_CATEGORICAL_COVARIATES]
    if include_urban:
        required.append("urban_rural_group")
    _require_columns(df, required, "overlap-weighting data")
    data = df[df["noise_group"].isin(("NC", "NN"))].copy()
    if include_urban:
        data["urbanicity_binary"] = _derive_urbanicity_binary(data["urban_rural_group"])
        required.append("urbanicity_binary")

    if analysis_set == "nervous_brain_imaging":
        _require_columns(data, ("brain_imaging_sample", NERVOUS_GAP_COLUMN), "Nervous overlap-weighting data")
        data = data[(data["brain_imaging_sample"] == 1) & data[NERVOUS_GAP_COLUMN].notna()].copy()
        outcome_columns = (NERVOUS_GAP_COLUMN,)
    elif analysis_set == "non_nervous_shared":
        _require_columns(data, NON_NERVOUS_GAP_COLUMNS, "non-Nervous overlap-weighting data")
        outcome_columns = NON_NERVOUS_GAP_COLUMNS
    else:
        raise ValueError(f"Unknown overlap-weighting analysis_set: {analysis_set}")

    data = data.dropna(subset=required).copy()
    data["noise_group"] = pd.Categorical(data["noise_group"], categories=("NC", "NN"), ordered=True)
    if not ((data["noise_group"] == "NC").any() and (data["noise_group"] == "NN").any()):
        raise ValueError(f"NC or NN is absent from overlap-weighting set {analysis_set}")
    return data, outcome_columns


def fit_overlap_weighted_agegap(
    df: pd.DataFrame,
    *,
    bootstrap_reps: int = 2_000,
    seed: int = 20_260_905,
    n_jobs: int = 1,
    include_urban_sensitivity: bool = True,
    spline_df: int = DEFAULT_SPLINE_DF,
    balance_threshold: float = DEFAULT_BALANCE_THRESHOLD,
    success_threshold: float = DEFAULT_BOOTSTRAP_SUCCESS_THRESHOLD,
    propensity_maxiter: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Estimate NN-vs-NC overlap-weighted AgeGap differences.

    The main and urban/rural-sensitivity propensity models are fitted separately.
    Non-Nervous outcomes share one propensity model; Nervous AgeGap re-estimates
    propensity scores in its brain-imaging analysis set.  Bootstrap samples are
    stratified by NC/NN to preserve the rare-exposure group size, and each
    replicate re-estimates the complete propensity model.
    """

    if spline_df < 3:
        raise ValueError("spline_df must be at least 3 for a natural cubic spline")
    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be at least 1")
    if int(propensity_maxiter) < 1:
        raise ValueError("propensity_maxiter must be at least 1")
    if not (0 < success_threshold <= 1):
        raise ValueError("success_threshold must lie in (0, 1]")
    if balance_threshold <= 0:
        raise ValueError("balance_threshold must be positive")

    variants: list[tuple[str, bool]] = [("main", False)]
    if include_urban_sensitivity:
        _require_columns(df, ("urban_rural_group",), "urban/rural sensitivity data")
        variants.append(("urban_rural_sensitivity", True))
    analysis_sets = ("non_nervous_shared", "nervous_brain_imaging")

    effect_rows: list[dict[str, Any]] = []
    balance_frames: list[pd.DataFrame] = []
    analysis_diagnostics: list[dict[str, Any]] = []
    analysis_counter = 0

    for analysis_variant, include_urban in variants:
        for analysis_set in analysis_sets:
            data, outcome_columns = _prepare_overlap_dataset(
                df,
                analysis_set=analysis_set,
                include_urban=include_urban,
            )
            formula = _propensity_formula(include_urban, spline_df)
            design_frame = dmatrix(formula, data, return_type="dataframe", NA_action="raise")
            design = np.asarray(design_frame, dtype=float)
            treatment = (data["noise_group"].astype(str) == "NN").astype(int).to_numpy()
            propensity, converged, iterations = _fit_propensity_array(
                design, treatment, maxiter=propensity_maxiter
            )
            if not converged or propensity is None:
                raise RuntimeError(
                    f"Primary propensity model failed for {analysis_variant}/{analysis_set}"
                )
            weights = _overlap_weights(treatment, propensity)
            balance = _balance_table(
                data,
                design_frame,
                treatment,
                propensity,
                weights,
                analysis_variant=analysis_variant,
                analysis_set=analysis_set,
                include_urban=include_urban,
                balance_threshold=balance_threshold,
            )
            balance_frames.append(balance)
            max_abs_smd = float(balance["abs_weighted_smd"].max())
            balance_pass = bool((balance["abs_weighted_smd"] < balance_threshold).all())

            outcomes = data.loc[:, outcome_columns].to_numpy(dtype=float)
            point_estimates = np.asarray(
                [_weighted_difference(outcomes[:, column], treatment, weights) for column in range(outcomes.shape[1])],
                dtype=float,
            )
            analysis_seed = int(seed + analysis_counter * 1_000_003)
            bootstrap, bootstrap_diag = _run_overlap_bootstrap(
                design,
                treatment,
                outcomes,
                bootstrap_reps=int(bootstrap_reps),
                seed=analysis_seed,
                n_jobs=int(n_jobs),
                maxiter=int(propensity_maxiter),
            )
            analysis_counter += 1

            nc = treatment == 0
            nn = treatment == 1
            for column_index, gap_column in enumerate(outcome_columns):
                values = outcomes[:, column_index]
                outcome_mask = np.isfinite(values)
                valid_bootstrap = bootstrap[:, column_index]
                valid_bootstrap = valid_bootstrap[np.isfinite(valid_bootstrap)]
                successful = int(len(valid_bootstrap))
                success_rate = successful / bootstrap_reps if bootstrap_reps else 0.0
                inference_valid = bool(bootstrap_reps > 1 and success_rate >= success_threshold)
                if successful > 1:
                    se = float(np.std(valid_bootstrap, ddof=1))
                    ci_low, ci_high = np.quantile(valid_bootstrap, [0.025, 0.975]).astype(float)
                else:
                    se = ci_low = ci_high = math.nan
                estimate = float(point_estimates[column_index])
                if inference_valid and np.isfinite(se) and se > 0:
                    p_value = float(2.0 * norm.sf(abs(estimate / se)))
                elif inference_valid and se == 0:
                    p_value = 0.0 if estimate != 0 else 1.0
                else:
                    p_value = math.nan

                outcome_weights = weights[outcome_mask]
                outcome_treatment = treatment[outcome_mask]
                outcome_values = values[outcome_mask]
                outcome_nc = outcome_treatment == 0
                outcome_nn = outcome_treatment == 1
                outcome_sd = float(np.std(outcome_values, ddof=1)) if len(outcome_values) > 1 else math.nan
                effect_rows.append(
                    {
                        "system": _system_label(gap_column),
                        "gap_column": gap_column,
                        "analysis_variant": analysis_variant,
                        "analysis_set": analysis_set,
                        "comparison": "NN vs NC",
                        "estimate": estimate,
                        "se": se,
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "p_value": p_value,
                        "bootstrap_reps_requested": int(bootstrap_reps),
                        "bootstrap_reps_successful": successful,
                        "bootstrap_success_rate": success_rate,
                        "confirmatory_inference_valid": inference_valid,
                        "model_n": int(outcome_mask.sum()),
                        "nc_n": int(outcome_nc.sum()),
                        "nn_n": int(outcome_nn.sum()),
                        "nc_ess": _effective_sample_size(outcome_weights[outcome_nc]),
                        "nn_ess": _effective_sample_size(outcome_weights[outcome_nn]),
                        "total_ess": _effective_sample_size(outcome_weights),
                        "max_abs_weighted_smd": max_abs_smd,
                        "balance_pass": balance_pass,
                        "outcome_sd": outcome_sd,
                        "standardized_estimate": estimate / outcome_sd
                        if np.isfinite(outcome_sd) and outcome_sd > 0
                        else math.nan,
                        "seed": analysis_seed,
                    }
                )

            analysis_diagnostics.append(
                {
                    "analysis_variant": analysis_variant,
                    "analysis_set": analysis_set,
                    "propensity_formula": formula,
                    "propensity_converged": converged,
                    "propensity_iterations": iterations,
                    "propensity_model_n": int(len(data)),
                    "design_columns": list(design_frame.columns),
                    "bootstrap_reps_requested": int(bootstrap_reps),
                    "bootstrap_reps_successful": int(bootstrap_diag["successful"]),
                    "bootstrap_reps_failed": int(bootstrap_diag["failed"]),
                    "bootstrap_max_iterations": int(bootstrap_diag["max_iterations"]),
                    "bootstrap_seed": analysis_seed,
                    "bootstrap_sampling": "stratified participant-row resampling within NC and NN",
                    "max_abs_weighted_smd": max_abs_smd,
                    "balance_pass": balance_pass,
                }
            )

    effects = pd.DataFrame(effect_rows)
    if effects.empty:
        effects = pd.DataFrame(columns=OVERLAP_EFFECT_COLUMNS)
    else:
        effects = effects.loc[:, OVERLAP_EFFECT_COLUMNS]
    balances = (
        pd.concat(balance_frames, ignore_index=True).loc[:, OVERLAP_BALANCE_COLUMNS]
        if balance_frames
        else pd.DataFrame(columns=OVERLAP_BALANCE_COLUMNS)
    )
    diagnostics = {
        "analysis": "overlap_weighted_agegap",
        "estimand": "overlap-population weighted mean difference (NN minus NC)",
        "independent_unit": "participant (eid)",
        "spline_df": int(spline_df),
        "balance_threshold": float(balance_threshold),
        "bootstrap_success_threshold": float(success_threshold),
        "bootstrap_reps": int(bootstrap_reps),
        "seed": int(seed),
        "n_jobs": int(n_jobs),
        "analyses": analysis_diagnostics,
    }
    return effects, balances, diagnostics


# ---------------------------------------------------------------------------
# Conditional minimum detectable effects (design sensitivity)
# ---------------------------------------------------------------------------


def conditional_mde(standard_error: float, *, power: float, alpha: float) -> float:
    """Normal-approximation two-sided conditional MDE for a fixed standard error."""

    standard_error = float(standard_error)
    power = float(power)
    alpha = float(alpha)
    if not np.isfinite(standard_error) or standard_error < 0:
        return math.nan
    if not 0.5 < power < 1.0:
        raise ValueError("power must lie strictly between 0.5 and 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return float((norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)) * standard_error)


def build_main_effect_mde(
    same_sample_results: pd.DataFrame,
    *,
    powers: Sequence[float] = (0.80, 0.90),
    alpha_levels: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build conditional MDE rows for Model D NN-vs-NC effects."""

    _require_columns(
        same_sample_results,
        ("system", "gap_column", "model", "comparison", "se", "outcome_sd", "model_n"),
        "same-sample result table",
    )
    if alpha_levels is None:
        alpha_levels = {"nominal": 0.05, "bonferroni_8": 0.00625}
    selected = same_sample_results[
        (same_sample_results["model"] == "Model D")
        & (same_sample_results["comparison"] == "NN vs NC")
    ]
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        for power in powers:
            for correction, alpha in alpha_levels.items():
                mde = conditional_mde(row.se, power=float(power), alpha=float(alpha))
                rows.append(
                    {
                        "system": row.system,
                        "gap_column": row.gap_column,
                        "comparison": row.comparison,
                        "model": row.model,
                        "power": float(power),
                        "alpha": float(alpha),
                        "correction": str(correction),
                        "standard_error": float(row.se),
                        "outcome_sd": float(row.outcome_sd),
                        "mde_agegap_years": mde,
                        "mde_standardized_sd": mde / float(row.outcome_sd)
                        if np.isfinite(row.outcome_sd) and float(row.outcome_sd) > 0
                        else math.nan,
                        "model_n": int(row.model_n),
                    }
                )
    return pd.DataFrame(rows, columns=MAIN_MDE_COLUMNS)


def build_interaction_mde(
    interaction_results: pd.DataFrame,
    *,
    powers: Sequence[float] = (0.80, 0.90),
    alpha_levels: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build MDE rows for the 16 NN x pollutant interaction coefficients."""

    _require_columns(
        interaction_results,
        ("system", "gap_column", "pollutant", "nn_se", "pollutant_sd", "outcome_sd", "model_n"),
        "interaction result table",
    )
    if alpha_levels is None:
        alpha_levels = {"nominal": 0.05, "bonferroni_16": 0.05 / 16.0}
    rows: list[dict[str, Any]] = []
    for row in interaction_results.itertuples(index=False):
        for power in powers:
            for correction, alpha in alpha_levels.items():
                mde = conditional_mde(row.nn_se, power=float(power), alpha=float(alpha))
                rows.append(
                    {
                        "system": row.system,
                        "gap_column": row.gap_column,
                        "pollutant": row.pollutant,
                        "term": "NN x pollutant (per 1 SD pollutant)",
                        "power": float(power),
                        "alpha": float(alpha),
                        "correction": str(correction),
                        "standard_error": float(row.nn_se),
                        "pollutant_scale": f"1 SD = {float(row.pollutant_sd):.12g} raw units",
                        "outcome_sd": float(row.outcome_sd),
                        "mde_agegap_years_per_1sd_pollutant": mde,
                        "mde_standardized_outcome_sd_per_1sd_pollutant": mde / float(row.outcome_sd)
                        if np.isfinite(row.outcome_sd) and float(row.outcome_sd) > 0
                        else math.nan,
                        "model_n": int(row.model_n),
                    }
                )
    return pd.DataFrame(rows, columns=INTERACTION_MDE_COLUMNS)


__all__ = [
    "AGEGAP_COLUMNS",
    "BASE_CATEGORICAL_COVARIATES",
    "BASE_CONTINUOUS_COVARIATES",
    "DataContractError",
    "EXPECTED_GROUP_COUNTS",
    "EXPECTED_INPUT_ROWS",
    "EXPECTED_INPUT_SHA256",
    "INTERACTION_MDE_COLUMNS",
    "INTERACTION_RESULT_COLUMNS",
    "MAIN_MDE_COLUMNS",
    "MARGINAL_RESULT_COLUMNS",
    "NON_NERVOUS_GAP_COLUMNS",
    "OVERLAP_BALANCE_COLUMNS",
    "OVERLAP_EFFECT_COLUMNS",
    "POLLUTANT_COLUMNS",
    "SAME_SAMPLE_RESULT_COLUMNS",
    "build_interaction_mde",
    "build_main_effect_mde",
    "conditional_mde",
    "fit_overlap_weighted_agegap",
    "fit_pollutant_interactions",
    "fit_same_sample_pollutant_models",
    "load_and_validate_input",
    "sha256_file",
    "validate_input_contract",
]
