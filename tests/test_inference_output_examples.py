import json
from pathlib import Path
import pytest
from noise_agegap_pipeline.age_inference.inference import (
    parse_response,
    validate_profile,
    load_profiles,
)
from noise_agegap_pipeline.common import read_config

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = [
    "overall",
    "cardiovascular",
    "hepatic",
    "pulmonary",
    "renal",
    "metabolic",
    "musculoskeletal",
    "nervous",
]


def test_examples_contain_only_three_input_output_pairs():
    folders = sorted(p.name for p in (ROOT / "examples").iterdir())
    assert folders == ["example_01", "example_02", "example_03"]
    for name in folders:
        assert sorted(p.name for p in (ROOT / "examples" / name).iterdir()) == [
            "input.json",
            "output.json",
        ]


@pytest.mark.parametrize("number", [1, 2, 3])
def test_inference_output_is_complete(number):
    path = ROOT / "examples" / f"example_{number:02d}" / "output.json"
    text = path.read_text(encoding="utf-8")
    output = json.loads(text)
    required = {
        field
        for system in SYSTEMS
        for field in [system.title(), "inference process - " + system.title()]
    }
    assert set(output) == required and len(output) == 16
    for system in SYSTEMS:
        assert type(output[system.title()]) is int
        rationale = output["inference process - " + system.title()]
        assert isinstance(rationale, str) and len(rationale) >= 60
        assert "fictional" in rationale.lower()
        assert "..." not in rationale and "TODO" not in rationale
    assert len(parse_response(text, SYSTEMS)) == 8


@pytest.mark.parametrize("number", [1, 2, 3])
def test_inference_input_is_complete_and_compatible(number):
    path = ROOT / "examples" / f"example_{number:02d}" / "input.json"
    records = load_profiles(path)
    assert len(records) == 1
    record = records[0]
    assert record["eid"] == f"SYNTHETIC_EXAMPLE_{number:02d}"
    assert record["chronological_age"] == record["profile"]["chronological_age"]
    assert set(SYSTEMS) <= set(record["profile"])
    assert {"demographics", "lifestyle"} <= set(record["profile"])
    excluded = read_config(ROOT / "config/fields.yaml")["noise_profile_excluded_fields"]
    validate_profile(record["profile"], excluded)
    prompt = (ROOT / "config/prompt.txt").read_text()
    expanded = prompt.replace("{input}", json.dumps(record["profile"]))
    assert "{input}" not in expanded


def test_profile_loader_preserves_jsonl_batch_support(tmp_path):
    source = ROOT / "examples/example_01/input.json"
    record = json.loads(source.read_text())
    path = tmp_path / "profiles.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert load_profiles(path) == [record]
