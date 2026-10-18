# Usage guide

Run all commands from the repository root after installation.

## Analysis configuration

- `config/analysis.yaml`: eight AgeGap domains, covariate sets, noise threshold, bootstrap counts, clinical censoring, protein filtering, and inference settings.
- `config/fields.yaml`: UK Biobank field mappings and fields excluded from inference profiles.
- `config/paths.example.yaml`: paths to synthetic inputs, resolved relative to the configuration file.

For controlled inputs, copy the example path file and edit the private copy:

```bash
cp config/paths.example.yaml config/paths.local.yaml
python scripts/run_pipeline.py check --paths config/paths.local.yaml
```

`config/paths.local.yaml` is ignored by Git. Keep authorized participant data outside this repository. Use a fresh output directory for each independent analysis so downstream modules read the intended upstream results.

## Selecting modules

```bash
# Inspect module names and their execution order.
python scripts/run_pipeline.py list

# Run the main association module using synthetic AgeGap inputs.
python scripts/run_pipeline.py run --paths config/paths.example.yaml --module noise_agegap --smoke

# Run a contiguous range of modules.
python scripts/run_pipeline.py run --paths config/paths.example.yaml --from-module data_preparation --to-module noise_agegap --smoke
```

Downstream modules can require files produced by upstream modules. The default `input_agegap_state` is `residualized`. To start from saved biological-age predictions, set `input_agegap_state` to `predicted` in a separate analysis configuration and pass it with `--analysis`.

## Live biological-age inference

Set `LLM_API_KEY` and `LLM_BASE_URL` securely in the local environment, and configure `profiles` and `prompt` in the private path file. The endpoint must support the configured model. Use profiles approved for transmission to that endpoint.

```bash
python scripts/run_pipeline.py run --paths config/paths.local.yaml --module age_inference --invoke-api
```

This writes inference outputs; run the downstream modules separately with matching paths and analysis settings. The default pipeline omits this API step.

## R diagnostics and enrichment

The configured clinical workflow invokes `scripts/cox_ph.R` through `Rscript` for `survival::coxph` and `survival::cox.zph` diagnostics. Ensure `Rscript` is on the executable search path and `survival` is installed before running `clinical_outcomes`.

After the Python workflow has exported bridge-enrichment inputs, the separate R command is:

```bash
Rscript scripts/enrich_storey.R results/smoke tests/fixtures/gene_sets.gmt
```

For controlled-data runs, replace these arguments with the corresponding results directory and gene-set file. This step requires `clusterProfiler` and `qvalue`. The Python bridge module writes a BH-adjusted audit and a pending enrichment-status file; the R helper writes `enrichment_storey.csv` and `R_sessionInfo.txt`. Storey estimates can be unavailable for small gene-set collections.

## Outputs and provenance

Module outputs are stored in named subdirectories under the configured output directory. `--smoke` appends `smoke/` and reduces the bootstrap counts to 19. Run manifests record the modules, settings, completion status, input/output SHA-256 hashes, and core Python-library versions.

`--dry-run` checks module input paths without fitting models. It does not validate every input column or the R environment.

The optional `scripts/check_real_legacy_parity.py` compares an authorized input/reference pair for the legacy association models. It is a separate local audit; its default report path is `docs/real_legacy_parity.json`. Keep that locally generated report outside the distributed package.

## Script and export inventory

See `docs/script_inventory.csv` for module paths, helper scripts, and configured aggregate exports. Figure numbers retained in export filenames should be checked against the final manuscript panels.
