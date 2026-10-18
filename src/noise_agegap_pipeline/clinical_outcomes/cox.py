from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.duration.hazard_regression import PHReg
from ..common import covariates, require


def fit_cox(block, predictor, cfg, model, standardize=False):
    num, cat = covariates(cfg, model)
    require(block, [predictor, "time_years", "event"] + num + cat)
    data = block.dropna(subset=[predictor, "time_years", "event"] + num + cat).copy()
    row = {
        "model": model,
        "n": len(data),
        "events": int(data.event.sum()),
        "status": "not_estimated",
        "beta": np.nan,
        "se": np.nan,
        "hr": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
    }
    if len(data) < 20 or data.event.sum() < cfg["clinical"]["min_events"]:
        row["status"] = "not_estimated_low_events"
        return row, []
    if data[predictor].nunique() < 2:
        row["status"] = "not_estimated_no_variation"
        return row, []
    sd = data[predictor].std(ddof=1)
    if standardize:
        data[predictor] = (data[predictor] - data[predictor].mean()) / sd
    x = pd.concat(
        [
            data[[predictor] + num].astype(float),
            pd.get_dummies(data[cat], drop_first=True, dtype=float),
        ],
        axis=1,
    )
    x = x.loc[:, x.nunique() > 1]
    if np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
        row["status"] = "not_estimated_singular_design"
        return row, []
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fit = PHReg(
                data.time_years, x, status=data.event, ties=cfg["clinical"]["ties"]
            ).fit(maxiter=100)
        idx = x.columns.get_loc(predictor)
        beta = fit.params[idx]
        se = fit.bse[idx]
        if not np.isfinite(beta) or not np.isfinite(se) or se <= 0:
            row["status"] = "not_estimated_nonfinite"
            return row, []
        row.update(
            status="warning" if captured else "ok",
            beta=beta,
            se=se,
            hr=np.exp(beta),
            ci_low=np.exp(beta - 1.959964 * se),
            ci_high=np.exp(beta + 1.959964 * se),
            p_value=2 * norm.sf(abs(beta / se)),
        )
        if standardize:
            row["agegap_sd_years"] = sd
        diagnostics = (
            r_schoenfeld(x, data, cfg)
            if cfg["clinical"].get("ph_backend") == "r_survival"
            else schoenfeld_screen(fit, x.columns, data)
        )
        return row, diagnostics
    except (ValueError, np.linalg.LinAlgError) as exc:
        row["status"] = "fit_failed_" + type(exc).__name__
        return row, []


def schoenfeld_screen(fit, names, data):
    # Transparent screening test, distinct from R survival::cox.zph.
    # Correlation with ranked event times is descriptive; exact cox.zph replay is provided in R.
    from scipy.stats import spearmanr

    residuals = np.asarray(fit.schoenfeld_residuals)
    rows = []
    for i, name in enumerate(names):
        valid = np.isfinite(residuals[:, i]) & (data.event.to_numpy() == 1)
        if valid.sum() > 3 and np.std(residuals[valid, i]) > 0:
            test = spearmanr(residuals[valid, i], data.time_years.to_numpy()[valid])
            rows.append(
                {
                    "term": name,
                    "rho": test.statistic,
                    "p_value": test.pvalue,
                    "event_n": valid.sum(),
                    "method": "Schoenfeld residual / event-time Spearman screen; not cox.zph",
                }
            )
    return rows


def r_schoenfeld(design, data, cfg):
    import subprocess
    import tempfile
    from pathlib import Path

    # Temporary risk sets remain local and are deleted after R returns.
    with tempfile.TemporaryDirectory(prefix="noise_cox_ph_") as folder:
        root = Path(folder)
        frame = design.copy()
        original = list(frame.columns)
        frame.columns = ["x" + str(i) for i in range(len(original))]
        frame["time_years"] = data.time_years.values
        frame["event"] = data.event.values
        path = root / "riskset.csv"
        frame.to_csv(path, index=False)
        script = Path(__file__).resolve().parents[3] / "scripts/cox_ph.R"
        formula = "Surv(time_years,event) ~ " + " + ".join(frame.columns[:-2])
        prefix = root / "diagnostic"
        subprocess.run(
            [
                "Rscript",
                str(script),
                str(path),
                formula,
                str(prefix),
                cfg["clinical"].get("ph_transform", "rank"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = pd.read_csv(str(prefix) + "_cox_zph.csv", index_col=0)
        rows = []
        for term, row in result.iterrows():
            mapped = (
                original[int(term[1:])]
                if term.startswith("x") and term[1:].isdigit()
                else term
            )
            rows.append(
                {
                    "term": mapped,
                    "chi_square": row["chisq"],
                    "df": row["df"],
                    "p_value": row["p"],
                    "method": "R survival::cox.zph ("
                    + cfg["clinical"].get("ph_transform", "rank")
                    + ")",
                }
            )
        return rows
