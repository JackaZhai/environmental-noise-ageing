from __future__ import annotations
import json
import hashlib
import os
import platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def read_config(path):
    path = Path(path).expanduser().resolve()
    text = path.read_text()
    try:
        result = json.loads(text)  # Shipped configs use the JSON subset of YAML.
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "Install PyYAML to read non-JSON YAML configurations"
            ) from exc
        result = yaml.safe_load(text)
    if not isinstance(result, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return result


def resolve_paths(path):
    path = Path(path).expanduser().resolve()
    paths = read_config(path)
    for key, value in paths.items():
        if value is not None:
            value = os.path.expandvars(value)
            if "$" in value:
                raise ValueError(f"Unresolved environment variable in {key}")
            paths[key] = (path.parent / Path(value).expanduser()).resolve()
    return paths


def read_table(path):
    import gzip

    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic.startswith(b"PK") or magic.startswith(b"\xd0\xcf"):
        raise ValueError("Binary Excel input requires explicit conversion to CSV/TSV")
    compressed = magic.startswith(b"\x1f\x8b")
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        header = handle.readline()
    delimiter = "\t" if "\t" in header else ","
    return pd.read_csv(path, sep=delimiter, compression="gzip" if compressed else None)


def require(df, columns, label="input"):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def unique_eid(df, label="input"):
    require(df, ["eid"], label)
    if df.eid.isna().any() or df.eid.duplicated().any():
        raise ValueError(f"{label} requires unique nonmissing eid")


def csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def json_write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def encode(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)

    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def adjust(df, group_columns, method="fdr_bh", p="p_value", output="fdr_p"):
    df = df.copy()
    df[output] = np.nan
    if df.empty:
        return df
    for _, ids in df.groupby(group_columns, dropna=False).groups.items():
        good = df.loc[ids, p].dropna()
        if len(good):
            df.loc[good.index, output] = multipletests(good, method=method)[1]
    return df


def gap_columns(cfg):
    return [s + "_age_gap" for s in cfg["systems"]]


def covariates(cfg, model):
    spec = cfg["models"][model]
    return spec["numeric"], spec["categorical"]


def rhs(cfg, model):
    num, cat = covariates(cfg, model)
    return " + ".join(num + [f"C({c})" for c in cat])


def record_run(paths, cfg, modules, outputs, status="complete", error=None):
    import scipy, statsmodels

    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "error": error,
        "modules": modules,
        "analysis": cfg,
        "inputs": {
            k: {"path": str(p), "sha256": digest(p)}
            for k, p in paths.items()
            if k not in ["output_dir", "existing_results"]
            and p is not None
            and p.is_file()
        },
        "outputs": {
            str(p.relative_to(paths["output_dir"])): digest(p)
            for p in outputs
            if p.is_file()
        },
        "python": platform.python_version(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "warning": "Modular implementation; manuscript numerical parity requires documented validation. Synthetic examples do not reproduce manuscript findings.",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    json_write(manifest, paths["output_dir"] / "run_manifests" / (stamp + ".json"))
    return manifest


def load_analysis(paths, cfg, fields):
    from .data_preparation.cohorts import define_exposures, recode_fields

    p = paths["output_dir"] / "agegap_phenotyping/phenotypes.csv"
    if not p.exists() and cfg["input_agegap_state"] == "predicted":
        raise ValueError(
            "Predicted-age input requires agegap_phenotyping before downstream models"
        )
    data = read_table(p if p.exists() else paths["analysis_input"])
    unique_eid(data)
    data = define_exposures(recode_fields(data, fields), cfg["threshold_db"])
    require(data, gap_columns(cfg))
    if "brain_imaging_sample" in data:
        data.loc[data.brain_imaging_sample != 1, "nervous_age_gap"] = np.nan
    return data
