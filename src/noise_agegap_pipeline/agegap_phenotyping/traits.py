from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from ..common import require, rhs, covariates, csv


def evaluate_traits(data, specs, cfg):
    require(specs, ["target", "marker", "kind"])
    rows = []
    for row in specs.to_dict("records"):
        target, marker, kind = row["target"], row["marker"], row["kind"]
        num, cat = covariates(cfg, "Primary")
        num = [c for c in num if c != marker]
        cat = [c for c in cat if c != marker]
        columns = [target, marker] + num + cat
        require(data, columns, "trait models")
        block = data.dropna(subset=columns).copy()
        metadata = {
            "target": target,
            "marker": marker,
            "kind": kind,
            "n": len(block),
            "status": "not_estimated",
            "beta": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        }
        if len(block) < 30 or block[marker].nunique() < 2:
            rows.append(metadata)
            continue
        if kind == "continuous":
            block["trait"] = (block[marker] - block[marker].mean()) / block[marker].std(
                ddof=1
            )
        elif kind == "binary":
            if not set(block[marker].unique()) <= set([0, 1]):
                raise ValueError("Binary trait must be explicitly encoded as 0/1")
            block["trait"] = block[marker]
        else:
            raise ValueError("Trait kind must be continuous or binary")
        cov = " + ".join(num + [f"C({c})" for c in cat])
        fit = smf.ols(f"{target} ~ trait + {cov}", block).fit(cov_type="HC3")
        low, high = fit.conf_int().loc["trait"]
        rows.append(
            {
                **metadata,
                "status": "ok",
                "beta": fit.params["trait"],
                "se": fit.bse["trait"],
                "ci_low": low,
                "ci_high": high,
                "p_value": fit.pvalues["trait"],
                "scale": (
                    "AgeGap years per 1 SD trait"
                    if kind == "continuous"
                    else "AgeGap years: category 1 vs 0"
                ),
                "adjustment": cov,
            }
        )
    return pd.DataFrame(rows)
