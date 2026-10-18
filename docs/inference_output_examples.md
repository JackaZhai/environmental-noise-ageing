# Complete inference input/output examples

The examples directory contains three subdirectories, each holding input.json and output.json only. Each input includes a synthetic identifier, chronological age, demographic and lifestyle fields, and all eight physiological-domain sections. Each paired output follows the complete fixed-prompt response schema:

| Domain | Numeric field | Explanation field |
|---|---|---|
| Overall | Overall | inference process - Overall |
| Cardiovascular | Cardiovascular | inference process - Cardiovascular |
| Hepatic | Hepatic | inference process - Hepatic |
| Pulmonary | Pulmonary | inference process - Pulmonary |
| Renal | Renal | inference process - Renal |
| Metabolic | Metabolic | inference process - Metabolic |
| Musculoskeletal | Musculoskeletal | inference process - Musculoskeletal |
| Nervous | Nervous | inference process - Nervous |

All eight ages are integers, and all eight explanation fields contain complete English text. The output files have no omitted systems, ellipses, placeholder values, or API request metadata. The input files are complete structured synthetic records; measurements include units in their field names. The paired outputs provide illustrative summaries consistent with the patterns in the corresponding fictional profiles.

These are hand-authored, fictional input/output schema pairs. They are not actual DeepSeek responses, research observations, validated age estimates, or reproduced manuscript findings. Explanatory text describes hypothetical indicators solely to illustrate a complete output format. The use of the historical field name "inference process" preserves the response schema; it does not establish access to any model's internal reasoning.

Tests validate all 16 fields and confirm that the age-inference parser accepts each object. Synthetic data needed by downstream statistical tests are created outside examples in temporary pytest directories or ignored tests/fixtures files.

The profile loader accepts the example input JSON directly and also preserves JSONL batch support. Input and output completeness, profile age consistency, noise-field exclusion, prompt expansion, and output parsing are tested. Matching directory names define the pairing. A live model run may produce different values and explanations.
