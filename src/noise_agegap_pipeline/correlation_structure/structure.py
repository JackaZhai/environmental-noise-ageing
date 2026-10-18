from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
from patsy import dmatrix
from ..common import csv, read_table, gap_columns, covariates, rhs, adjust, require
from ..data_preparation.cohorts import define_exposures


def correlation_difference(reference, comparison, columns, reps=5000, seed=20260905):
    rng = np.random.default_rng(seed)
    rows = []
    # One participant row index is reused for every edge in each resample.
    pairs = list(combinations(columns, 2))
    samples = {pair: [] for pair in pairs}
    for _ in range(reps):
        a = reference.iloc[rng.integers(0, len(reference), len(reference))]
        b = comparison.iloc[rng.integers(0, len(comparison), len(comparison))]
        ca = a[columns].corr()
        cb = b[columns].corr()
        for x, y in pairs:
            delta = cb.loc[x, y] - ca.loc[x, y]
            if np.isfinite(delta):
                samples[(x, y)].append(delta)
    for x, y in pairs:
        a = reference[[x, y]].dropna()
        b = comparison[[x, y]].dropna()
        row = {
            "system_a": x,
            "system_b": y,
            "reference_n": len(a),
            "comparison_n": len(b),
            "status": "not_estimated",
            "reference_r": np.nan,
            "comparison_r": np.nan,
            "delta_r": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "successful_bootstraps": len(samples[(x, y)]),
        }
        if (
            min(len(a), len(b)) > 3
            and a[x].std() > 0
            and a[y].std() > 0
            and b[x].std() > 0
            and b[y].std() > 0
        ):
            ra = a.corr().iloc[0, 1]
            rb = b.corr().iloc[0, 1]
            z = (
                np.arctanh(np.clip(rb, -1 + 1e-12, 1 - 1e-12))
                - np.arctanh(np.clip(ra, -1 + 1e-12, 1 - 1e-12))
            ) / np.sqrt(1 / (len(a) - 3) + 1 / (len(b) - 3))
            row.update(
                status="ok",
                reference_r=ra,
                comparison_r=rb,
                delta_r=rb - ra,
                p_value=2 * norm.sf(abs(z)),
            )
            if samples[(x, y)]:
                row["ci_low"], row["ci_high"] = np.quantile(
                    samples[(x, y)], [0.025, 0.975]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def pca(df, columns):
    data = df.dropna(subset=columns).copy()
    x = data[columns].to_numpy()
    scale = x.std(axis=0, ddof=1)
    if len(data) < 3 or np.any(scale == 0):
        raise ValueError("PCA requires complete cases and nonconstant columns")
    standardized = (x - x.mean(axis=0)) / scale
    u, s, vt = np.linalg.svd(standardized, full_matrices=False)
    scores = pd.DataFrame(u * s, columns=["PC" + str(i + 1) for i in range(len(s))])
    scores.insert(0, "eid", data.eid.values)
    scores["noise_group"] = data.noise_group.values
    variance = pd.DataFrame(
        {"component": scores.columns[1:-1], "variance_fraction": s * s / (s * s).sum()}
    )
    loadings = (
        pd.DataFrame(
            vt.T, index=columns, columns=["PC" + str(i + 1) for i in range(len(s))]
        )
        .rename_axis("system")
        .reset_index()
    )
    return scores, variance, loadings


def partial_edges(df, columns, cfg):
    num, cat = covariates(cfg, "Expanded")
    block = df.dropna(subset=columns + num + cat)
    design = np.asarray(dmatrix(rhs(cfg, "Expanded"), block))
    if len(block) < design.shape[1] + 4:
        return pd.DataFrame(
            columns=[
                "system_a",
                "system_b",
                "partial_r",
                "p_value",
                "fdr_p",
                "retained",
            ]
        )
    values = block[columns].to_numpy()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        resid = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    if not np.isfinite(resid).all():
        raise FloatingPointError("Nonfinite partial correlation residuals")
    rows = []
    rank = np.linalg.matrix_rank(design)
    from scipy.stats import t

    for i, j in combinations(range(len(columns)), 2):
        correlation = np.corrcoef(resid[:, i], resid[:, j])[0, 1]
        dof = len(block) - rank - 1
        stat = correlation * np.sqrt(dof / max(1 - correlation**2, 1e-12))
        rows.append(
            {
                "system_a": columns[i],
                "system_b": columns[j],
                "partial_r": correlation,
                "n": len(block),
                "df": dof,
                "p_value": 2 * t.sf(abs(stat), dof),
                "family": "partial_network",
            }
        )
    result = adjust(pd.DataFrame(rows), ["family"])
    result["retained"] = (result.fdr_p < 0.05) & (
        result.partial_r.abs() >= cfg["partial_correlation_threshold"]
    )
    return result


def run(paths, cfg, fields):
    from ..common import load_analysis

    df = load_analysis(paths, cfg, fields)
    columns = [c for c in gap_columns(cfg) if c != "overall_age_gap"]
    require(df, columns)
    out = paths["output_dir"] / "correlation_structure"
    outputs = []

    def save(frame, name):
        p = out / (name + ".csv")
        csv(frame, p)
        outputs.append(p)

    save(df[columns].corr().rename_axis("system").reset_index(), "correlations")
    edges = []
    for group, ref, comparisons in [
        ("noise_group", "NC", ["ND", "NN"]),
        ("lowhigh_group", "Low", ["High"]),
    ]:
        for comparison in comparisons:
            a = df[df[group] == ref]
            b = df[df[group] == comparison]
            if a.empty or b.empty:
                raise ValueError(f"Empty network group: {comparison} / {ref}")
            result = correlation_difference(
                a, b, columns, cfg["network_bootstrap"], cfg["seed"]
            )
            result["comparison"] = f"{comparison} vs {ref}"
            edges.append(result)
    edges = adjust(pd.concat(edges, ignore_index=True), ["comparison"])
    edges["display_edge"] = edges.delta_r.abs() >= cfg["network_display_delta"]
    save(edges, "delta_correlations")
    scores, variance, loadings = pca(df, columns)
    save(scores, "pca_scores")
    save(variance, "pca_variance")
    save(loadings, "pca_loadings")
    partial = partial_edges(df, columns, cfg)
    save(partial, "partial_correlations")
    stats = []
    for group, block in df.groupby("noise_group"):
        corr = block[columns].corr().to_numpy()
        upper = corr[np.triu_indices(len(columns), 1)]
        stats.append(
            {
                "group": group,
                "n": len(block),
                "mean_absolute_r": np.nanmean(abs(upper)),
                "edge_n_abs_r_ge_threshold": np.sum(
                    abs(upper) >= cfg["partial_correlation_threshold"]
                ),
            }
        )
    save(pd.DataFrame(stats), "network_summary")
    degree = []
    for col in columns:
        links = partial[(partial.system_a == col) | (partial.system_b == col)]
        degree.append(
            {
                "system": col,
                "weighted_degree": links.loc[links.retained, "partial_r"].abs().sum(),
                "retained_edges": links.retained.sum(),
            }
        )
    save(pd.DataFrame(degree), "partial_weighted_degree")
    return outputs
