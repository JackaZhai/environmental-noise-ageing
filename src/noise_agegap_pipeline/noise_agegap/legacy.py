from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
PRIMARY_NUM = ["chronological_age", "bmi"]
PRIMARY_CAT = ["sex_code", "education_degree", "smoking_status", "alcohol_frequency"]
EXPANDED_NUM = PRIMARY_NUM + ["townsend", "no2_2010", "pm25_2010"]
EXPANDED_CAT = PRIMARY_CAT
MODEL_DEFINITIONS = {"Primary": "age + sex + BMI + education + smoking + alcohol", "Expanded": "Primary + Townsend + NO2 + PM2.5"}
SYSTEMS = [(s.title(), s + "_age_gap", s.title(), None, "#707070") for s in ["nervous", "cardiovascular", "pulmonary", "hepatic", "renal", "metabolic", "musculoskeletal", "overall"]]
SYSTEM_ORDER = {col: i for i, (_, col, _, _, _) in enumerate(SYSTEMS)}
SYSTEM_META = {col: (system, label, icon, color) for system, col, label, icon, color in SYSTEMS}

def bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    ok = values.notna()
    if ok.any():
        result.loc[ok] = multipletests(values.loc[ok].to_numpy(), method="fdr_bh")[1]
    return result

def complete_case(df: pd.DataFrame, outcome: str, group_col: str, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    cols = ["eid", outcome, group_col] + numeric + categorical
    out = df[cols].dropna().copy()
    return out

def fit_one(data: pd.DataFrame, outcome: str, group_col: str, reference: str, comparisons: list[str], model: str) -> list[dict]:
    numeric = PRIMARY_NUM if model == "Primary" else EXPANDED_NUM
    categorical = PRIMARY_CAT if model == "Primary" else EXPANDED_CAT
    terms = [f"C({group_col}, Treatment(reference='{reference}'))"] + numeric + [f"C({x})" for x in categorical]
    formula = f"{outcome} ~ " + " + ".join(terms)
    rows: list[dict] = []
    counts = data[group_col].value_counts().to_dict()
    fit = smf.ols(formula=formula, data=data).fit(cov_type="HC3")
    for comparison in comparisons:
        term = f"C({group_col}, Treatment(reference='{reference}'))[T.{comparison}]"
        ci = fit.conf_int().loc[term]
        system, label, icon, color = SYSTEM_META[outcome]
        rows.append(
            {
                "system": system,
                "system_label": label,
                "gap_column": outcome,
                "model": model,
                "model_definition": MODEL_DEFINITIONS[model],
                "grouping": "NC/ND/NN" if group_col == "noise_group" else "L24h <60 / >=60 dB",
                "comparison": f"{comparison} vs {reference}",
                "reference_group": reference,
                "comparison_group": comparison,
                "beta": float(fit.params[term]),
                "se_hc3": float(fit.bse[term]),
                "ci_low": float(ci.iloc[0]),
                "ci_high": float(ci.iloc[1]),
                "p_value": float(fit.pvalues[term]),
                "model_n": int(len(data)),
                "reference_n": int(counts.get(reference, 0)),
                "comparison_n": int(counts.get(comparison, 0)),
                "sample_rule": "Nervous: brain-imaging AgeGap available; other systems: outcome-specific complete cases",
                "missing_data_handling": "complete-case within each AgeGap outcome and model",
                "method": "OLS with HC3 robust standard errors",
                "organ_icon": icon,
                "organ_color": color,
                "status": "ok",
            }
        )
    return rows

def run_models(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    df = df.copy()
    df["lowhigh_group"] = np.where(df["L24h"] >= 60.0, "High", "Low")
    for _, outcome, _, _, _ in SYSTEMS:
        for model in ["Primary", "Expanded"]:
            numeric = PRIMARY_NUM if model == "Primary" else EXPANDED_NUM
            categorical = PRIMARY_CAT if model == "Primary" else EXPANDED_CAT
            group_data = complete_case(df, outcome, "noise_group", numeric, categorical)
            rows.extend(fit_one(group_data, outcome, "noise_group", "NC", ["ND", "NN"], model))
            lowhigh_data = complete_case(df, outcome, "lowhigh_group", numeric, categorical)
            rows.extend(fit_one(lowhigh_data, outcome, "lowhigh_group", "Low", ["High"], model))
    result = pd.DataFrame(rows)
    result["fdr_p"] = result.groupby(["model", "grouping", "comparison"], group_keys=False)["p_value"].transform(bh)
    result["bonferroni_p"] = np.minimum(result["p_value"] * 8.0, 1.0)
    result["is_fdr_significant"] = result["fdr_p"] < 0.05
    result["is_bonferroni_significant"] = result["bonferroni_p"] < 0.05
    result["positive_nominal_display"] = (result["beta"] > 0) & (result["p_value"] < 0.05)
    result["system_order"] = result["gap_column"].map(SYSTEM_ORDER)
    return result.sort_values(["model", "grouping", "system_order", "comparison"]).reset_index(drop=True)
