from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from .common import read_config, resolve_paths, record_run

MODULES = {
    "data_preparation": "data_preparation.cohorts",
    "age_inference": "age_inference.inference",
    "agegap_phenotyping": "agegap_phenotyping.phenotypes",
    "noise_agegap": "noise_agegap.models",
    "robustness": "robustness.run",
    "correlation_structure": "correlation_structure.structure",
    "clinical_outcomes": "clinical_outcomes.run",
    "proteomics": "proteomics.run",
    "bridge_enrichment": "bridge_enrichment.bridge",
    "reporting": "reporting.export",
}
ORDER = list(MODULES)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Noise/AgeGap modular pipeline; no LLM call without --invoke-api"
    )
    parser.add_argument("command", choices=["list", "run", "check"])
    parser.add_argument("--paths", default="config/paths.example.yaml")
    parser.add_argument("--analysis", default="config/analysis.yaml")
    parser.add_argument("--fields", default="config/fields.yaml")
    parser.add_argument("--module", action="append", choices=ORDER)
    parser.add_argument("--from-module", choices=ORDER)
    parser.add_argument("--to-module", choices=ORDER)
    parser.add_argument("--invoke-api", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="19 bootstrap replicates in a separate smoke output directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        for i, name in enumerate(ORDER, 1):
            print(f"{i:2} {name}")
        return 0
    paths = resolve_paths(args.paths)
    cfg = read_config(args.analysis)
    fields = read_config(args.fields)
    selected = [m for m in ORDER if m != "age_inference"]
    if args.module:
        selected = [m for m in ORDER if m in args.module]
    else:
        start = ORDER.index(args.from_module) if args.from_module else 0
        end = ORDER.index(args.to_module) + 1 if args.to_module else len(ORDER)
        selected = [
            m for m in ORDER[start:end] if m != "age_inference" or args.invoke_api
        ]
    if args.invoke_api and "age_inference" not in selected:
        parser.error("--invoke-api requires age_inference in the selected range")
    if args.smoke:
        cfg["ow_bootstrap"] = 19
        cfg["network_bootstrap"] = 19
        cfg["proteomics"]["min_n"] = 30
        cfg["proteomics"]["stratum_min_n"] = 20
        paths["output_dir"] = paths["output_dir"] / "smoke"
    cfg["run_mode"] = "smoke" if args.smoke else "full"
    if len(cfg["systems"]) != 8 or len(set(cfg["systems"])) != 8:
        raise ValueError("Expected eight distinct AgeGap systems")
    if cfg.get("model_profile") == "legacy_20260812":
        for spec in cfg["models"].values():
            spec["categorical"] = [
                x for x in spec["categorical"] if x != "ethnicity_group"
            ]
    elif cfg.get("model_profile") != "manuscript":
        raise ValueError("Unknown model_profile")
    print(
        json.dumps(
            {
                "modules": selected,
                "output_dir": str(paths["output_dir"]),
                "invoke_api": args.invoke_api,
                "mode": cfg["run_mode"],
            },
            indent=2,
        )
    )
    needed = {
        "data_preparation": ["analysis_input"],
        "age_inference": ["profiles", "prompt"],
        "agegap_phenotyping": ["analysis_input"],
        "noise_agegap": ["analysis_input"],
        "robustness": ["analysis_input"],
        "correlation_structure": ["analysis_input"],
        "clinical_outcomes": [
            "clinical_noise_input",
            "clinical_agegap_input",
            "endpoints",
        ],
        "proteomics": ["analysis_input", "proteins"],
        "bridge_enrichment": ["protein_mapping", "gene_sets"],
        "reporting": [],
    }
    errors = []
    for m in selected:
        for key in needed[m]:
            if not paths.get(key) or not paths[key].is_file():
                errors.append(f"{m}: missing {key}")
    if (
        "agegap_phenotyping" in selected
        and cfg["input_agegap_state"] == "predicted"
        and "age_inference" not in selected
    ):
        if not paths.get("predicted_ages") or not paths["predicted_ages"].is_file():
            errors.append("missing predicted_ages")
    if errors:
        raise FileNotFoundError("; ".join(sorted(set(errors))))
    if args.command == "check" or args.dry_run:
        return 0
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "noise_agegap_mpl")
    )
    os.environ.setdefault("MPLBACKEND", "Agg")
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    outputs = []
    try:
        for module in selected:
            print(f"Running {module}", flush=True)
            implementation = importlib.import_module(
                "noise_agegap_pipeline." + MODULES[module]
            )
            kwargs = (
                {"invoke_api": args.invoke_api} if module == "age_inference" else {}
            )
            outputs.extend(implementation.run(paths, cfg, fields, **kwargs))
    except Exception as exc:
        record_run(
            paths,
            cfg,
            selected,
            outputs,
            "failed",
            type(exc).__name__ + ": " + str(exc),
        )
        raise
    record_run(paths, cfg, selected, outputs)
    print(f'Complete: {len(outputs)} output files. Results: {paths["output_dir"]}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
