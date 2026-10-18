from __future__ import annotations
import json
import os
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from ..common import csv, digest, unique_eid, json_write


def parse_response(text, systems):
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    text = match.group(1) if match else text
    obj = json.loads(text)
    ages = {}
    for system in systems:
        value = obj.get(system.title())
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"Missing or nonnumeric {system} age")
        ages[system + "_biological_age"] = float(value)
    return ages


def validate_profile(profile, excluded):
    # Recursively inspect structured profile keys. Free-text profiles cannot be audited safely.
    if not isinstance(profile, dict):
        raise ValueError("Profile must be a structured mapping, not free text")
    bad = []

    def inspect(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k)
                field = key.split("-")[0]
                if key in excluded or field in excluded or "noise" in key.lower():
                    bad.append(key)
                inspect(v)
        elif isinstance(obj, list):
            for v in obj:
                inspect(v)

    inspect(profile)
    if bad:
        raise ValueError(f"Noise leakage in profile fields: {sorted(set(bad))}")


def load_profiles(path):
    """Read one JSON profile, a JSON list, or a JSONL batch."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        records = payload if isinstance(payload, list) else [payload]
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError("At least one profile record is required")
    for record in records:
        if not isinstance(record, dict) or not {
            "eid",
            "chronological_age",
            "profile",
        } <= set(record):
            raise ValueError(
                "Each profile record requires eid, chronological_age, and profile"
            )
        age = record["chronological_age"]
        if (
            isinstance(age, bool)
            or not isinstance(age, (int, float))
            or not np.isfinite(age)
        ):
            raise ValueError("Chronological age must be a finite numeric value")
        if not isinstance(record["profile"], dict):
            raise ValueError("Each profile must be a structured mapping")
        if record["profile"].get("chronological_age") != age:
            raise ValueError("Record and profile chronological ages must agree")
    return records


def run(paths, cfg, fields, invoke_api=False):
    records = load_profiles(paths["profiles"])
    unique_eid(pd.DataFrame({"eid": [x["eid"] for x in records]}), "profiles")
    out = paths["output_dir"] / "age_inference"
    out.mkdir(parents=True, exist_ok=True)
    settings = cfg["inference"]
    prompt = paths["prompt"].read_text()
    raw_path = out / "responses.jsonl"
    rows = []
    failures = []
    if invoke_api:
        from openai import OpenAI

        key = os.environ.get(settings["api_key_env"])
        base = os.environ.get(settings["base_url_env"])
        if not key or not base:
            raise ValueError(
                "Inference requires configured API key and base URL environment variables"
            )
        client = OpenAI(api_key=key, base_url=base)
        # Existing successful raw records are reusable only when settings + prompt match.
        expected = {"settings": settings, "prompt_sha256": digest(paths["prompt"])}
        metadata = out / "inference_metadata.json"
        if raw_path.exists() and (
            not metadata.exists() or json.loads(metadata.read_text()) != expected
        ):
            raise ValueError(
                "Existing responses belong to a different inference configuration; use a new output directory"
            )
        json_write(expected, metadata)
        completed = {}
        if raw_path.exists():
            for line in raw_path.read_text().splitlines():
                record = json.loads(line)
                completed[record["eid"]] = record
        with raw_path.open("a") as stream:
            for case in records:
                validate_profile(
                    case["profile"], fields["noise_profile_excluded_fields"]
                )
                case_hash = (
                    __import__("hashlib")
                    .sha256(json.dumps(case["profile"], sort_keys=True).encode())
                    .hexdigest()
                )
                if case["eid"] in completed:
                    if completed[case["eid"]]["profile_sha256"] != case_hash:
                        raise ValueError("Changed profile with reused eid")
                    continue
                request = dict(
                    model=settings["model"],
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                    messages=[
                        {
                            "role": "user",
                            "content": prompt.replace(
                                "{input}",
                                json.dumps(case["profile"], ensure_ascii=False),
                            ),
                        }
                    ],
                )
                for key in ["seed", "top_p"]:
                    if settings.get(key) is not None:
                        request[key] = settings[key]
                try:
                    response = client.chat.completions.create(**request)
                    record = {
                        "eid": case["eid"],
                        "chronological_age": case["chronological_age"],
                        "response": response.choices[0].message.content,
                        "profile_sha256": case_hash,
                        "actual_model": response.model,
                        "response_id": response.id,
                        "utc": datetime.now(timezone.utc).isoformat(),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                except Exception as exc:
                    failures.append(
                        {
                            "eid": case["eid"],
                            "status": "api_failed",
                            "error_type": type(exc).__name__,
                        }
                    )
    if not raw_path.exists():
        raise FileNotFoundError(
            "Saved responses missing; use predicted_ages for downstream runs or explicitly invoke the API"
        )
    for line in raw_path.read_text().splitlines():
        record = json.loads(line)
        try:
            rows.append(
                {
                    "eid": record["eid"],
                    "chronological_age": record["chronological_age"],
                    **parse_response(record["response"], cfg["systems"]),
                }
            )
        except (ValueError, TypeError) as exc:
            failures.append(
                {
                    "eid": record["eid"],
                    "status": "parse_failed",
                    "error_type": type(exc).__name__,
                }
            )
    ages = out / "ages.csv"
    failed = out / "failures.csv"
    csv(pd.DataFrame(rows), ages)
    csv(pd.DataFrame(failures, columns=["eid", "status", "error_type"]), failed)
    return [ages, failed, raw_path]
