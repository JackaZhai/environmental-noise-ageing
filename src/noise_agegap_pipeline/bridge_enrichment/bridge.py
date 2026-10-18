from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from ..common import read_table, require, csv, adjust, json_write


def select_candidates(noise, age, spec):
    require(noise, ["protein", "contrast", "beta", "p_value"])
    require(age, ["protein", "system_col", "beta", "fdr_p"])
    overall = age[age.system_col == "overall_age_gap"][
        ["protein", "beta", "fdr_p"]
    ].rename(columns={"beta": "b_beta", "fdr_p": "b_fdr"})
    if overall.protein.duplicated().any():
        raise ValueError("Duplicate Overall protein associations")
    detail = noise.rename(columns={"beta": "a_beta", "p_value": "a_p"}).merge(
        overall, on="protein", how="inner", validate="many_to_one"
    )
    detail["bridge_score"] = detail.a_beta * detail.b_beta
    detail["eligible"] = (detail.a_p < spec["noise_p"]) & (
        detail.b_fdr < spec["agegap_fdr"]
    )
    detail["direction"] = np.where(
        detail.bridge_score > 0,
        "same_sign",
        np.where(detail.bridge_score < 0, "opposite_sign", "zero"),
    )
    return detail.sort_values(["contrast", "bridge_score"], ascending=[True, False])


def load_gmt(path):
    result = []
    for line in path.read_text().splitlines():
        if line.strip():
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError("GMT must contain term, description, genes")
            result.append((fields[0], fields[1], set(fields[2:])))
    return result


def run(paths, cfg, fields):
    noise = read_table(paths["output_dir"] / "proteomics/noise_protein.csv")
    age = read_table(paths["output_dir"] / "proteomics/protein_agegap.csv")
    detail = select_candidates(noise, age, cfg["bridge"])
    mapping = read_table(paths["protein_mapping"])
    require(mapping, ["protein", "gene"])
    mapping = mapping.dropna(subset=["protein", "gene"])
    if mapping.protein.duplicated().any():
        raise ValueError(
            "Protein-to-gene mapping must explicitly resolve multiple identifiers"
        )
    mapped = detail.merge(mapping, on="protein", how="left", validate="many_to_one")
    gmt = load_gmt(paths["gene_sets"])
    rows = []
    universe_rows = []
    candidate_rows = []
    # Universe uses proteins actually tested in BOTH screens for this contrast, mapped to unique genes.
    for contrast, block in mapped.groupby("contrast"):
        universe = set(block.gene.dropna())
        universe_rows.extend(
            {"contrast": contrast, "gene": gene} for gene in sorted(universe)
        )
        candidates = set(
            block.loc[
                block.eligible & (block.direction == "same_sign"), "gene"
            ].dropna()
        )
        candidate_rows.extend(
            {"contrast": contrast, "gene": gene} for gene in sorted(candidates)
        )
        for term, description, genes in gmt:
            genes = genes & universe
            if (
                not cfg["bridge"]["min_genes"]
                <= len(genes)
                <= cfg["bridge"]["max_genes"]
            ):
                continue
            hit = genes & candidates
            p = (
                hypergeom.sf(len(hit) - 1, len(universe), len(genes), len(candidates))
                if candidates
                else 1.0
            )
            rows.append(
                {
                    "contrast": contrast,
                    "term": term,
                    "description": description,
                    "universe_n": len(universe),
                    "candidate_n": len(candidates),
                    "term_n": len(genes),
                    "overlap_n": len(hit),
                    "genes": ";".join(sorted(hit)),
                    "p_value": p,
                }
            )
    ora = adjust(
        pd.DataFrame(
            rows,
            columns=[
                "contrast",
                "term",
                "description",
                "universe_n",
                "candidate_n",
                "term_n",
                "overlap_n",
                "genes",
                "p_value",
            ],
        ),
        ["contrast"],
        output="bh_fdr_audit",
    )
    # Never label BH values as Storey q-values.
    ora["storey_q"] = np.nan
    ora["enrichment_status"] = "Storey q pending R clusterProfiler/qvalue run"
    out = paths["output_dir"] / "bridge_enrichment"
    outputs = []
    for name, frame in [
        ("bridge_candidates", mapped),
        ("ora_audit", ora),
        (
            "enrichment_universe",
            pd.DataFrame(universe_rows, columns=["contrast", "gene"]),
        ),
        (
            "enrichment_candidates",
            pd.DataFrame(candidate_rows, columns=["contrast", "gene"]),
        ),
    ]:
        p = out / (name + ".csv")
        csv(frame, p)
        outputs.append(p)
    p = out / "enrichment_status.json"
    json_write(
        {
            "candidate_analysis": "exploratory coefficient-product prioritization; no formal mediation fit",
            "storey_q": "Requires scripts/enrich_storey.R; BH audit is not manuscript Storey q",
            "gene_set_source_sha256": __import__(
                "noise_agegap_pipeline.common", fromlist=["digest"]
            ).digest(paths["gene_sets"]),
        },
        p,
    )
    outputs.append(p)
    return outputs
