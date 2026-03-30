"""Tests for file discovery utilities."""

from pathlib import Path
import pytest
from croissant_baker.files import discover_files


def test_discover_files(tmp_path: Path) -> None:
    """Test discover_files finds files recursively and returns relative paths."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "file1.txt").write_text("test")
    (tmp_path / "sub/file2.txt").write_text("test")

    files = discover_files(str(tmp_path))
    expected = {Path("file1.txt"), Path("sub/file2.txt")}
    assert set(files) == expected


def test_discover_files_empty_directory(tmp_path: Path) -> None:
    """Test discover_files returns empty list for empty directory."""
    files = discover_files(str(tmp_path))
    assert files == []


def test_discover_files_nonexistent_directory() -> None:
    """Test discover_files raises FileNotFoundError for nonexistent directory."""
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        discover_files("/nonexistent/path")


def test_discover_files_not_a_directory(tmp_path: Path) -> None:
    """Test discover_files raises FileNotFoundError for non-directory path."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")
    with pytest.raises(FileNotFoundError, match="is not a directory"):
        discover_files(str(file_path))
