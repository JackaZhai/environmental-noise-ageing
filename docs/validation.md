# Package validation

Validation date: 2026-10-18.

## Environment

- Platform: macOS arm64
- Python: 3.11.15
- R: 4.6.0; `survival` available
- Verified direct Python dependencies: `requirements-tested.txt`

## Executed checks

| Check | Result |
| --- | --- |
| Original source, configuration, prompt, tests, and example content comparison | Identical to the supplied package |
| Synthetic-input generation | Passed |
| Configured input-path check | Passed; exit code 0 |
| Automated test suite | 25 passed; exit code 0 |
| Default synthetic smoke workflow | Completed 9 modules; exit code 0 |
| Smoke output integrity | 65 recorded output files; all SHA-256 values matched the final successful run manifest |
| README relative links | Resolved |
| Distribution manifest and ZIP | All packaged files checked against SHA-256 manifest |

The smoke workflow used 19 bootstrap replicates and executed the configured R survival diagnostics. Test fixtures and generated outputs are local artifacts excluded from the ZIP.

## Separate validation items

The Storey enrichment R helper was not run because `clusterProfiler` and `qvalue` are absent from the local R environment. Its expected output, `bridge_enrichment/enrichment_storey.csv`, is consequently listed as missing by the reporting module. The execution instructions are in `docs/usage.md`.

Live LLM calls and analyses using controlled UK Biobank data were not run during package preparation. The provided Abstract is the author-supplied study summary. Code validation used synthetic inputs.

`docs/script_inventory.csv` lists modules, helper scripts, and configured aggregate-export paths. Legacy figure numbers in the export filenames require author review against the final manuscript panels.

## Dependency compatibility

The original unconstrained installation selected pandas 3.0.6 and statsmodels 0.15.0; the spline test failed because `PandasData.design_info` was unavailable. The final package selects pandas 2.x and statsmodels 0.14.x and supplies verified direct dependency pins. The statistical source code and analysis settings were preserved.

## Data-safety release checks

The revised release adds manifest-controlled packaging and a dedicated data-safety audit (`docs/ukb_data_safety.md`). Release checks verify all listed file hashes and archive contents. An injected unlisted data file and a changed reviewed file are rejected; an unlisted local file is excluded from archive creation. The original 25 statistical/contract tests passed before these packaging-only changes.
