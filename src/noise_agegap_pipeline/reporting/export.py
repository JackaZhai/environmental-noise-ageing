from __future__ import annotations
import shutil
import numpy as np
import pandas as pd
from ..common import read_table, csv, digest, json_write

# Only aggregate tables are eligible for Source Data export. Participant tables are excluded.
SOURCE_TABLES = {
    "data_preparation/baseline.csv": "Table1_baseline.csv",
    "data_preparation/cohort_flow.csv": "cohort_flow.csv",
    "agegap_phenotyping/trait_associations.csv": "Supplementary_trait_associations.csv",
    "agegap_phenotyping/phenotype_summary.csv": "Figure2_phenotype_summary.csv",
    "noise_agegap/contrasts.csv": "Figure3_agegap_contrasts.csv",
    "noise_agegap/spline_curves.csv": "Supplementary_spline_curves.csv",
    "noise_agegap/spline_tests.csv": "Supplementary_spline_tests.csv",
    "robustness/overlap_effects.csv": "Supplementary_overlap_effects.csv",
    "robustness/overlap_balance.csv": "Supplementary_overlap_balance.csv",
    "robustness/pollutant_interactions.csv": "Supplementary_interactions.csv",
    "robustness/main_effect_mde.csv": "Supplementary_MDE.csv",
    "correlation_structure/delta_correlations.csv": "Figure3_delta_correlations.csv",
    "correlation_structure/partial_correlations.csv": "Supplementary_partial_correlations.csv",
    "correlation_structure/pca_variance.csv": "Supplementary_PCA_variance.csv",
    "clinical_outcomes/noise_cox.csv": "Figure4_noise_cox.csv",
    "clinical_outcomes/agegap_cox.csv": "Supplementary_agegap_cox.csv",
    "proteomics/protein_agegap.csv": "Figure5_protein_agegap.csv",
    "proteomics/noise_protein.csv": "Figure5_noise_protein.csv",
    "proteomics/stratified_protein_agegap.csv": "Supplementary_stratified_proteomics.csv",
    "bridge_enrichment/bridge_candidates.csv": "Figure5_bridge_candidates.csv",
    "bridge_enrichment/enrichment_storey.csv": "Figure5_enrichment_storey.csv",
}


def plot_contrasts(table, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = list(table.groupby(["model", "comparison"], sort=False))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for ax, ((model, contrast), block) in zip(axes.flat, panels):
        block = block.sort_values("gap_column")
        ok = block.beta.notna()
        y = np.arange(len(block))
        ax.axvline(0, color="0.7", lw=0.8)
        ax.errorbar(
            block.loc[ok, "beta"],
            y[ok],
            xerr=[
                block.loc[ok, "beta"] - block.loc[ok, "ci_low"],
                block.loc[ok, "ci_high"] - block.loc[ok, "beta"],
            ],
            fmt="o",
            color="#2F6B9A",
            capsize=2,
        )
        ax.set_yticks(y, block.gap_column.str.replace("_age_gap", "", regex=False))
        ax.set_title(f"{model}: {contrast}")
        ax.set_xlabel("Adjusted AgeGap difference (years)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(path.with_suffix("." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [path.with_suffix("." + s) for s in ["png", "pdf", "svg"]]


def run(paths, cfg, fields):
    source = paths.get("existing_results") or paths["output_dir"]
    out = paths["output_dir"] / "reporting"
    exports = []
    index = []
    missing = []
    for relative, name in SOURCE_TABLES.items():
        src = source / relative
        if not src.exists():
            missing.append(relative)
            continue
        table = read_table(src)
        if "eid" in table:
            raise ValueError(
                f"Participant-level table cannot be exported as Source Data: {src}"
            )
        target = out / "source_data" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != target.resolve():
            shutil.copyfile(src, target)
        if digest(src) != digest(target):
            raise AssertionError("Source Data differs from statistical results")
        exports.append(target)
        index.append(
            {
                "result_table": relative,
                "source_data": name,
                "rows": len(table),
                "sha256": digest(target),
                "identity_verified": True,
            }
        )
    if not exports:
        raise ValueError("No statistical results available for reporting")
    p = out / "source_data_manifest.csv"
    csv(pd.DataFrame(index), p)
    exports.append(p)
    p = out / "reporting_status.json"
    json_write(
        {
            "missing_tables": missing,
            "figure_parity": "Aggregate previews only; exact manuscript panel mapping is documented in docs/script_inventory.csv",
            "participant_data_exported": False,
        },
        p,
    )
    exports.append(p)
    contrast = source / "noise_agegap/contrasts.csv"
    if contrast.exists():
        exports.extend(
            plot_contrasts(read_table(contrast), out / "figures/agegap_contrasts")
        )
    return exports
