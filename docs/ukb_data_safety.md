# UK Biobank data-safety audit

Audit date: 2026-10-18. Scope: the prepared code distribution and the code paths that read, export, or transmit participant information.

## Distribution contents

All 61 files in the previous prepared ZIP were inspected by file inventory, UTF-8 content scan, and source/data-flow review. The archive was 84,846 bytes (SHA-256 `7a9d01fb602538e85df4db3c05dad8b41efdca8af2617d8eed9d66c546244b3c`). This report and the release-audit helper are additions to the revised distribution.

| Item | Finding |
| --- | --- |
| Participant-level UK Biobank datasets | Absent from the distribution |
| Derived individual AgeGaps, clinical dates, protein matrices, images, and fitted model objects | Absent from the distribution |
| JSON data | Three paired fictional input/output examples, with `SYNTHETIC_EXAMPLE_01` to `SYNTHETIC_EXAMPLE_03` identifiers and fictional output explanations |
| CSV data | `docs/script_inventory.csv` only; it lists code modules and export filenames |
| UK Biobank field identifiers | Present as variable/schema metadata in configuration and code |
| Cohort information | Abstract-level study summary and aggregate legacy cohort/group counts in the robustness code |
| Legacy input SHA-256 constant | Present in robustness source as a dataset fingerprint; no participant records accompany it |
| Embedded credentials, private keys, personal email addresses, and absolute home-directory paths | No matches in the inspected archive |
| Test fixtures | Generated from fixed-seed distributions and hand-written toy arrays; generated fixture files excluded |
| Git history, caches, environments, logs, results, and nested archives | Excluded |

The examples are described by the supplied package as hand-authored fictional schema pairs. This audit checked their contents and identifiers; their original authorship was not independently reconstructed from research records.

## Data flows requiring controlled handling

1. `data_preparation` and `agegap_phenotyping` write participant-level datasets. `correlation_structure` can write individual PCA scores. On controlled inputs, keep these outputs inside the authorized research environment.
2. `age_inference` can send each structured profile to the configured OpenAI-compatible endpoint when explicitly invoked with `--invoke-api`. The outer `eid` field is used locally, while the complete `profile` mapping becomes the request content. A nested identifier would travel with that mapping. Verify the applicable UK Biobank agreement, institutional approval, endpoint processing location, retention, and permitted recipients before transmitting controlled profiles.
3. Inference response files, age estimates, failures, and local run manifests can contain participant identifiers, derived information, input fingerprints, and filesystem paths. Exclude these files from the repository and review package.
4. `reporting` uses an aggregate-export allowlist and rejects tables containing a column named `eid`. Review any future real-data aggregate releases for other identifiers, rare combinations, and small cells under the applicable release rules.

The default pipeline and the validation performed during preparation used synthetic inputs and made no LLM requests.

## Release controls added

- Expanded `.gitignore` for input/output directories, common tabular/imaging/genetic formats, logs, local path configurations, and response files.
- Added `scripts/audit_distribution.py`: an explicit manifest, path, hash, format, JSON-example, CSV-schema, and credential check, with optional archive creation or verification.
- Restricted packaging to the audited manifest. Local generated files are omitted even when present in the working directory.
- Preserved the original statistical source, prompt, configurations, tests, and examples.

## Release conclusion

The reviewed code distribution contains no identified real participant records. It is prepared for code-only upload using the manifest-controlled archive. The wider research workspace and real-data runtime outputs remain outside this release.
