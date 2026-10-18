# Environmental noise and biological ageing

Code accompanying the manuscript **“Residential road traffic noise is associated with selective patterns of multi-organ biological ageing.”**

This repository contains the code package prepared for confidential editorial assessment and peer review. See [REVIEW_ONLY_NOTICE.txt](REVIEW_ONLY_NOTICE.txt) for the current terms of use.

## Abstract

Road traffic noise is associated with chronic disease, but whether its relationship with biological ageing differs across physiological systems remains unclear. In 50,627 UK Biobank participants, we examined residential road traffic noise in relation to overall and seven system-specific biological-age gaps (AgeGaps) derived using a large language model framework. Associations were system selective rather than uniform. Cardiovascular AgeGap was higher among participants with daytime and nighttime noise levels ≥60 dB than among those with both levels <60 dB, persisting after adjustment for neighbourhood deprivation and air pollution (β = 0.367 years, 95% CI 0.141–0.593; Bonferroni-adjusted *P* = 0.012), whereas broader associations were attenuated. Noise profiles were also associated with selected incident diseases. Proteomics revealed system-specific AgeGap correlates but no significant direct noise signature. These findings identify cardiovascular biological-age variation as the most robust system-specific correlate of residential road traffic noise, highlighting heterogeneous physiological ageing in relation to environmental exposure.

## Analysis workflow

The Python package, `noise_agegap_pipeline`, provides ten modules:

| Module | Purpose |
| --- | --- |
| `data_preparation` | Cohort assembly, covariate recoding, and noise-profile definitions |
| `age_inference` | Biological-age inference using the saved prompt and an OpenAI-compatible API |
| `agegap_phenotyping` | AgeGap construction and phenotype evaluation |
| `noise_agegap` | Noise–AgeGap association models |
| `robustness` | Robustness and sensitivity analyses |
| `correlation_structure` | Inter-system AgeGap correlation and structure analyses |
| `clinical_outcomes` | Incident-disease risk sets and Cox models |
| `proteomics` | Noise–protein and protein–AgeGap association analyses |
| `bridge_enrichment` | Exploratory bridge-protein prioritization and enrichment audit |
| `reporting` | Analysis-table and figure-source-data export |

The primary noise profiles use a 60 dB threshold for daytime and nighttime exposure:

| Profile | Daytime noise | Nighttime noise |
| --- | --- | --- |
| NC | <60 dB | <60 dB |
| ND | ≥60 dB | <60 dB |
| NN | ≥60 dB | ≥60 dB |

The separate Low/High contrast uses `Lden <60 dB` versus `Lden ≥60 dB`; the historical `L24h` alias is recorded in `config/fields.yaml`.

## Repository structure

```text
config/                     Analysis settings, UK Biobank field mappings,
                            saved inference prompt, and example input paths
docs/                       Usage, example documentation, and package provenance
examples/example_01/        Fictional input.json / output.json pair
examples/example_02/        Fictional input.json / output.json pair
examples/example_03/        Fictional input.json / output.json pair
scripts/                    Pipeline entry point, synthetic-data generator,
                            legacy-comparison tool, and R helpers
src/noise_agegap_pipeline/  Modular Python implementation
tests/                      Data-contract and statistical-consistency tests
requirements.txt            Supported Python dependency ranges
requirements-tested.txt     Verified direct dependency versions
pyproject.toml              Package metadata and command-line entry point
FILE_MANIFEST_SHA256.txt     SHA-256 checksums for the distributed files
```

## Installation

Python 3.9 or later is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The installation above uses the dependency versions verified for this package. `requirements.txt` and `pyproject.toml` record the supported version ranges.

R with `survival` is required for the configured Cox proportional-hazards diagnostics. The separate enrichment step uses R with `clusterProfiler` and `qvalue`.

## Quick start with synthetic data

Run these commands from the repository root:

```bash
# Generate synthetic inputs.
python scripts/make_synthetic_test_data.py --output-dir tests/fixtures

# Inspect the available modules and check the input paths.
python scripts/run_pipeline.py list
python scripts/run_pipeline.py check --paths config/paths.example.yaml

# Execute the reduced workflow (19 bootstrap replicates).
python scripts/run_pipeline.py run --paths config/paths.example.yaml --smoke

# Run the automated tests.
python -m pytest -q
```

The smoke workflow writes to `results/smoke/`, including timestamped run manifests with input/output hashes and software versions. It excludes live LLM inference. Generated inputs, results, and the local environment are ignored by Git.

For individual modules, controlled-data configuration, LLM inference, and the separate R enrichment command, see [the usage guide](docs/usage.md).

## Data and inference examples

UK Biobank participant-level data are accessed separately through the [UK Biobank access process](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access). Study-result reproduction requires authorized participant data and the study-specific inputs described in the manuscript Methods.

The three input/output pairs in `examples/` are hand-authored fictional schema examples. Generated test fixtures are also synthetic. See [the inference-example documentation](docs/inference_output_examples.md) for the complete field schema.

## Data protection and release checks

The distribution contains code, field mappings, documentation, and three fictional schema examples. Participant-level UK Biobank data, individual AgeGaps, clinical-event dates, protein measurements, inference responses, and local run manifests must remain in the authorized research environment. UK Biobank field identifiers in the configuration describe variables; the configuration supplies no participant measurements.

Use the reviewed distribution manifest to check or rebuild the release:

```bash
python scripts/audit_distribution.py
python scripts/audit_distribution.py --build ../environmental-noise-ageing_reviewed.zip
```

The release tool accepts only the files and SHA-256 hashes recorded in `FILE_MANIFEST_SHA256.txt`, checks their allowed paths and formats, and excludes all other local files. Review any content change before updating the manifest. Git ignore rules provide an additional barrier for local data and outputs. Upload the reviewed release contents rather than the wider research workspace.

Live LLM inference transmits the structured health profile to the configured API endpoint. Use authorized saved age estimates for controlled-data analyses unless the endpoint and transmission are covered by the applicable UK Biobank agreement and institutional approvals. The noise-field exclusion check is an exposure-blinding safeguard; privacy review covers all participant fields.

See [the data-safety audit](docs/ukb_data_safety.md) for the checked contents and data-flow findings.

## LLM configuration

The saved prompt is in `config/prompt.txt`. The configuration specifies `DeepSeek-V3-0324`, temperature `0.1`, and a maximum output length of `2,048` tokens; seed and top-p are unspecified. Residential noise fields are excluded from the inference profiles. API credentials are read from `LLM_API_KEY` and `LLM_BASE_URL` in the local environment.

Live inference requires explicit selection of `age_inference` together with `--invoke-api`. Default runs use saved age estimates or AgeGaps according to `config/analysis.yaml`.

## Code availability and contact

This version is supplied for editorial assessment and peer review under the accompanying review-only notice. Questions about the study or access to study-specific materials should be directed to the corresponding authors listed in the manuscript.
