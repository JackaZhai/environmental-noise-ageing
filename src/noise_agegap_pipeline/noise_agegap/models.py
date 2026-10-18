from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from ..common import csv, read_table, gap_columns, covariates, rhs, adjust, require
from ..data_preparation.cohorts import define_exposures


def fit_contrasts(df, cfg):
    rows = []
    for col in gap_columns(cfg):
        for model in cfg["models"]:
            num, cat = covariates(cfg, model)
            for group, ref, contrasts in [
                ("noise_group", "NC", ["ND", "NN"]),
                ("lowhigh_group", "Low", ["High"]),
            ]:
                require(df, [col, group] + num + cat)
                block = df.dropna(subset=[col, group] + num + cat)
                term = f'C({group}, Treatment(reference="{ref}"))'
                present = set(block[group])
                fit = None
                if (
                    ref in present
                    and len(block) > len(num) + len(cat) + 5
                    and len(present) > 1
                ):
                    fit = smf.ols(f"{col} ~ {term} + {rhs(cfg,model)}", block).fit(
                        cov_type="HC3"
                    )
                for contrast in contrasts:
                    key = f"{term}[T.{contrast}]"
                    row = {
                        "gap_column": col,
                        "model": model,
                        "comparison": f"{contrast} vs {ref}",
                        "n": len(block),
                        "reference_n": (block[group] == ref).sum(),
                        "comparison_n": (block[group] == contrast).sum(),
                        "status": "not_estimated",
                        "beta": np.nan,
                        "se": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": np.nan,
                    }
                    if fit is not None and key in fit.params:
                        low, high = fit.conf_int().loc[key]
                        row.update(
                            status="ok",
                            beta=fit.params[key],
                            se=fit.bse[key],
                            ci_low=low,
                            ci_high=high,
                            p_value=fit.pvalues[key],
                        )
                    rows.append(row)
    result = pd.DataFrame(rows)
    # The denominator stays eight even if a system cannot be estimated.
    result["bonferroni_p"] = np.minimum(result.p_value * len(cfg["systems"]), 1)
    return result


def fit_splines(df, cfg):
    rows = []
    curves = []
    for col in gap_columns(cfg):
        num, cat = covariates(cfg, "Expanded")
        require(df, [col, "Lden"] + num + cat)
        block = df.dropna(subset=[col, "Lden"] + num + cat)
        if len(block) < 30 or block.Lden.nunique() < 6:
            rows.append(
                {
                    "gap_column": col,
                    "n": len(block),
                    "status": "not_estimated",
                    "overall_p": np.nan,
                    "nonlinear_p": np.nan,
                }
            )
            continue
        cov = rhs(cfg, "Expanded")
        null = smf.ols(f"{col} ~ {cov}", block).fit()
        linear = smf.ols(f"{col} ~ Lden + {cov}", block).fit()
        formula = (
            f'{col} ~ cr(Lden, df={cfg["spline_df"]}, constraints="center") + {cov}'
        )
        ordinary = smf.ols(formula, block).fit()
        robust = smf.ols(formula, block).fit(cov_type="HC3")
        overall = ordinary.compare_f_test(null)
        nonlinear = ordinary.compare_f_test(linear)
        low, high = block.Lden.quantile([0.01, 0.99])
        grid = pd.DataFrame({"Lden": np.linspace(low, high, 101)})
        for c in num:
            grid[c] = block[c].median()
        for c in cat:
            grid[c] = block[c].mode().iloc[0]
        # Curves are contrasts relative to median exposure; covariance of the difference is retained.
        reference = grid.iloc[[0]].copy()
        reference["Lden"] = block.Lden.median()
        design = robust.model.data.design_info
        x = np.asarray(build_design_matrices([design], grid)[0])
        x0 = np.asarray(build_design_matrices([design], reference)[0])
        delta = x - x0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mean = delta @ np.asarray(robust.params)
            se = np.sqrt(
                np.clip(
                    np.einsum(
                        "ij,jk,ik->i", delta, np.asarray(robust.cov_params()), delta
                    ),
                    0,
                    None,
                )
            )
        if not np.isfinite(mean).all() or not np.isfinite(se).all():
            raise FloatingPointError("Nonfinite spline contrasts")
        curves.append(
            pd.DataFrame(
                {
                    "gap_column": col,
                    "Lden": grid.Lden,
                    "beta": mean,
                    "ci_low": mean - 1.959964 * se,
                    "ci_high": mean + 1.959964 * se,
                    "reference_Lden": block.Lden.median(),
                }
            )
        )
        rows.append(
            {
                "gap_column": col,
                "n": len(block),
                "status": "ok",
                "overall_p": overall[1],
                "nonlinear_p": nonlinear[1],
                "p01": low,
                "p99": high,
                "method": "nested OLS F tests; HC3 contrast intervals",
            }
        )
    return pd.DataFrame(rows), (
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    )


def run(paths, cfg, fields):
    from ..common import load_analysis

    df = load_analysis(paths, cfg, fields)
    result = fit_contrasts(df, cfg)
    summary, curves = fit_splines(df, cfg)
    outputs = []
    for name, frame in [
        ("contrasts.csv", result),
        ("spline_tests.csv", summary),
        ("spline_curves.csv", curves),
    ]:
        p = paths["output_dir"] / "noise_agegap" / name
        csv(frame, p)
        outputs.append(p)
    return outputs
