from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("noise_agegap_test_data")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_synthetic_test_data.py"), "--output-dir", str(directory)],
        check=True, capture_output=True, text=True,
    )
    return directory
