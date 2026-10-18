from __future__ import annotations
import math
import warnings
from typing import Iterable
import numpy as np
import pandas as pd
CONTINUOUS_COVARIATES = ["chronological_age", "bmi", "townsend", "no2_2010", "pm25_2010"]
CATEGORICAL_COVARIATES = ["sex_code", "ethnicity_group", "education_degree", "smoking_status", "alcohol_frequency"]
SYSTEMS = [(s + "_age_gap", s.title()) for s in ["overall", "cardiovascular", "hepatic", "pulmonary", "renal", "metabolic", "musculoskeletal", "nervous"]]

def bh_fdr(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    mask = values.notna()
    if not mask.any():
        return result
    pvals = values.loc[mask].astype(float).to_numpy()
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result.loc[mask] = restored
    return result

def normal_p_value(z_score: float) -> float:
    return math.erfc(abs(z_score) / math.sqrt(2.0))

def zscore_array(values: np.ndarray) -> np.ndarray | None:
    std = float(np.nanstd(values, ddof=1))
    if not np.isfinite(std) or std <= 0:
        return None
    return (values - float(np.nanmean(values))) / std

def standardize_numeric(values: pd.Series) -> pd.Series | None:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=1)
    if pd.isna(std) or std <= 0:
        return None
    return (numeric - numeric.mean()) / std

def select_proteins_by_missingness(
    merged: pd.DataFrame,
    protein_cols: Iterable[str],
    missing_threshold: float,
) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for protein in protein_cols:
        rows.append({"protein": protein, "missing_ratio": float(merged[protein].isna().mean())})
    summary = pd.DataFrame(rows)
    summary["keep"] = summary["missing_ratio"] <= missing_threshold
    summary = summary.sort_values(["keep", "missing_ratio", "protein"], ascending=[False, True, True])
    kept = summary.loc[summary["keep"], "protein"].tolist()
    return kept, summary

def build_adjustment_design(
    frame: pd.DataFrame,
    numeric_terms: Iterable[str],
    continuous_covariates: Iterable[str],
    categorical_covariates: Iterable[str],
) -> pd.DataFrame:
    design = pd.DataFrame(index=frame.index)
    for col in numeric_terms:
        if col in frame.columns:
            design[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in continuous_covariates:
        if col not in frame.columns:
            continue
        standardized = standardize_numeric(frame[col])
        if standardized is not None:
            design[col] = standardized
    for col in categorical_covariates:
        if col not in frame.columns:
            continue
        dummies = pd.get_dummies(frame[col], prefix=col, drop_first=True, dummy_na=False)
        for dummy_col in dummies.columns:
            values = dummies[dummy_col].astype(float)
            if values.std(ddof=1) > 0:
                design[dummy_col] = values
    return design.astype(float)

def fit_ols_terms(y: np.ndarray, design: pd.DataFrame, terms: list[str]) -> dict[str, dict[str, float]] | None:
    design_values = design.to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(design_values).all(axis=1)
    if int(valid.sum()) <= 2:
        return None
    y_valid = y[valid]
    design_valid = design.loc[valid].astype(float)
    x = np.column_stack([np.ones(len(design_valid)), design_values[valid]])
    names = ["Intercept", *design_valid.columns.tolist()]
    rank = int(np.linalg.matrix_rank(x))
    df_resid = len(y_valid) - rank
    if df_resid <= 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            xtx = x.T @ x
    if not np.isfinite(xtx).all():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            xtx_inv = np.linalg.pinv(xtx)
            beta = xtx_inv @ x.T @ y_valid
    if not np.isfinite(beta).all():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            resid = y_valid - x @ beta
    if not np.isfinite(resid).all():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            sigma2 = float((resid @ resid) / df_resid)
    if not np.isfinite(sigma2):
        return None
    covariance = xtx_inv * sigma2
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    output: dict[str, dict[str, float]] = {}
    for term in terms:
        if term not in names:
            continue
        idx = names.index(term)
        if not np.isfinite(se[idx]) or se[idx] <= 0:
            continue
        z_value = float(beta[idx] / se[idx])
        ci_delta = 1.959963984540054 * float(se[idx])
        output[term] = {
            "n": int(len(y_valid)),
            "beta": float(beta[idx]),
            "se": float(se[idx]),
            "ci_low": float(beta[idx] - ci_delta),
            "ci_high": float(beta[idx] + ci_delta),
            "z_value": z_value,
            "p_value": normal_p_value(z_value),
        }
    return output

def screen_exposure_to_protein(
    agegap_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    exposure_terms: list[str],
    protein_cols: list[str],
    base_covariates: list[str] | None = None,
    categorical_covariates: list[str] | None = None,
    min_n: int = 500,
) -> pd.DataFrame:
    base_covariates = base_covariates or CONTINUOUS_COVARIATES
    categorical_covariates = categorical_covariates or CATEGORICAL_COVARIATES
    merged = agegap_df.merge(protein_df[["eid", *protein_cols]], on="eid", how="inner")
    design = build_adjustment_design(merged, exposure_terms, base_covariates, categorical_covariates)
    rows: list[dict[str, object]] = []
    for protein in protein_cols:
        protein_values = pd.to_numeric(merged[protein], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(protein_values) & design.notna().all(axis=1).to_numpy()
        if int(valid.sum()) < min_n:
            continue
        y = np.full(len(protein_values), np.nan, dtype=float)
        protein_z = zscore_array(protein_values[valid])
        if protein_z is None:
            continue
        y[valid] = protein_z
        fitted = fit_ols_terms(y, design, exposure_terms)
        if fitted is None:
            continue
        for contrast, stats in fitted.items():
            rows.append({"protein": protein, "contrast": contrast, **stats})
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["fdr_p"] = output.groupby("contrast", group_keys=False)["p_value"].apply(bh_fdr)
    output["abs_beta"] = output["beta"].abs()
    output["neg_log10_p"] = -np.log10(output["p_value"].clip(lower=1e-300))
    return output.sort_values(["contrast", "p_value", "abs_beta", "protein"], ascending=[True, True, False, True])

def screen_protein_to_agegap(
    agegap_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    exposure_terms: list[str],
    protein_cols: list[str],
    systems: list[tuple[str, str]] | None = None,
    base_covariates: list[str] | None = None,
    categorical_covariates: list[str] | None = None,
    min_n: int = 500,
) -> pd.DataFrame:
    systems = systems or SYSTEMS
    base_covariates = base_covariates or CONTINUOUS_COVARIATES
    categorical_covariates = categorical_covariates or CATEGORICAL_COVARIATES
    merged = agegap_df.merge(protein_df[["eid", *protein_cols]], on="eid", how="inner")
    base_design = build_adjustment_design(merged, exposure_terms, base_covariates, categorical_covariates)
    rows: list[dict[str, object]] = []
    for system_col, system_label in systems:
        if system_col not in merged.columns:
            continue
        y = pd.to_numeric(merged[system_col], errors="coerce").to_numpy(dtype=float)
        for protein in protein_cols:
            protein_values = pd.to_numeric(merged[protein], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y) & np.isfinite(protein_values) & base_design.notna().all(axis=1).to_numpy()
            if int(valid.sum()) < min_n:
                continue
            protein_z = zscore_array(protein_values[valid])
            if protein_z is None:
                continue
            design = base_design.copy()
            design["protein_z"] = np.nan
            design.loc[valid, "protein_z"] = protein_z
            fitted = fit_ols_terms(y, design, ["protein_z"])
            if fitted is None or "protein_z" not in fitted:
                continue
            rows.append(
                {
                    "protein": protein,
                    "system_col": system_col,
                    "system": system_label,
                    **fitted["protein_z"],
                }
            )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["fdr_p"] = output.groupby("system_col", group_keys=False)["p_value"].apply(bh_fdr)
    output["abs_beta"] = output["beta"].abs()
    output["neg_log10_p"] = -np.log10(output["p_value"].clip(lower=1e-300))
    return output.sort_values(["system_col", "p_value", "abs_beta", "protein"], ascending=[True, True, False, True])

def screen_protein_to_agegap_by_stratum(
    agegap_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    stratum_col: str,
    strata: list[str],
    protein_cols: list[str],
    systems: list[tuple[str, str]] | None = None,
    base_covariates: list[str] | None = None,
    categorical_covariates: list[str] | None = None,
    min_n: int = 500,
) -> pd.DataFrame:
    rows = []
    for stratum in strata:
        sub_agegap = agegap_df[agegap_df[stratum_col] == stratum].copy()
        if sub_agegap.empty:
            continue
        screen = screen_protein_to_agegap(
            sub_agegap,
            protein_df,
            exposure_terms=[],
            protein_cols=protein_cols,
            systems=systems,
            base_covariates=base_covariates,
            categorical_covariates=categorical_covariates,
            min_n=min_n,
        )
        if screen.empty:
            continue
        screen.insert(0, "stratum_col", stratum_col)
        screen.insert(1, "stratum", stratum)
        screen["stratum_n"] = int(len(sub_agegap))
        rows.append(screen)
    if not rows:
        return pd.DataFrame()
    output = pd.concat(rows, ignore_index=True)
    return output.sort_values(
        ["stratum", "system_col", "p_value", "abs_beta", "protein"],
        ascending=[True, True, True, False, True],
    )
