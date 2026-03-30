"""Tests for WFDB handler."""

from pathlib import Path
import pytest

from croissant_baker.handlers.wfdb_handler import WFDBHandler


@pytest.fixture
def wfdb_sample_path():
    """Path to a WFDB record for unit testing."""
    path = (
        Path(__file__).parent
        / "data"
        / "input"
        / "mitdb_wfdb"
        / "physionet.org"
        / "files"
        / "mitdb"
        / "1.0.0"
        / "100.hea"
    )
    if not path.exists():
        pytest.skip(f"WFDB sample data not found at {path}")
    return path


def test_can_handle_hea_file(wfdb_sample_path):
    handler = WFDBHandler()
    assert handler.can_handle(wfdb_sample_path)


def test_cannot_handle_non_hea_file(tmp_path):
    handler = WFDBHandler()
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("data")
    assert not handler.can_handle(csv_file)


def test_extract_metadata(wfdb_sample_path):
    handler = WFDBHandler()
    metadata = handler.extract_metadata(wfdb_sample_path)

    assert metadata["record_name"] == "100"
    assert metadata["encoding_format"] == "application/x-wfdb-header"
    assert metadata["num_signals"] == 2
    assert metadata["sampling_frequency"] == 360
    assert metadata["num_samples"] == 650000
    assert "MLII" in metadata["signal_names"]
    assert "V5" in metadata["signal_names"]
    # We expect .dat and .atr files to be found alongside .hea
    assert len(metadata["related_files"]) == 2
    assert metadata["signal_types"]["MLII"] == "sc:Float"
    assert metadata["signal_types"]["V5"] == "sc:Float"


def test_missing_dat_file_raises_error(tmp_path):
    handler = WFDBHandler()
    hea_file = tmp_path / "test.hea"
    hea_file.write_text("test 1 360 1000")

    with pytest.raises(ValueError, match="WFDB data file missing"):
        handler.extract_metadata(hea_file)
