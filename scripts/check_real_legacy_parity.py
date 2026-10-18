#!/usr/bin/env python3
# Local audit only. No participant-level data are written to the package.
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from noise_agegap_pipeline.noise_agegap.legacy import run_models
from noise_agegap_pipeline.common import digest, csv, json_write
import argparse

p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--output", type=Path, default=root / "docs/real_legacy_parity.json")
a = p.parse_args()
df = pd.read_csv(a.input)
actual = run_models(df)
expected = pd.read_csv(a.reference)
keys = ["gap_column", "model", "comparison"]
joined = actual.merge(
    expected, on=keys, suffixes=("_new", "_saved"), validate="one_to_one"
)
if len(actual) != len(expected) or len(joined) != len(actual):
    raise AssertionError("Result row mismatch")
metrics = [
    "beta",
    "se_hc3",
    "ci_low",
    "ci_high",
    "p_value",
    "bonferroni_p",
    "model_n",
    "reference_n",
    "comparison_n",
]
checks = {}
for metric in metrics:
    x = joined[metric + "_new"].to_numpy(float)
    y = joined[metric + "_saved"].to_numpy(float)
    checks[metric] = {
        "max_absolute_difference": float(np.nanmax(abs(x - y))),
        "matches": bool(np.allclose(x, y, rtol=1e-8, atol=1e-10, equal_nan=True)),
    }
report = {
    "rows": len(joined),
    "input_sha256": digest(a.input),
    "reference_sha256": digest(a.reference),
    "checks": checks,
    "all_match": all(v["matches"] for v in checks.values()),
    "scope": "Unchanged legacy Primary/Expanded statistical extraction only; does not validate new manuscript-profile models, Cox, enrichment or all figures.",
}
json_write(report, a.output)
print(json.dumps(report, indent=2))
if not report["all_match"]:
    raise SystemExit(1)
