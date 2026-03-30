"""Tests for CSV handler."""

from pathlib import Path
import pytest
from croissant_baker.handlers.csv_handler import CSVHandler


def test_csv_handler_can_handle() -> None:
    """Test CSV handler file type detection."""
    handler = CSVHandler()

    assert handler.can_handle(Path("test.csv"))
    assert handler.can_handle(Path("data.CSV"))
    assert handler.can_handle(Path("data.csv.gz"))
    assert not handler.can_handle(Path("test.txt"))


def test_csv_handler_extract_metadata(tmp_path: Path) -> None:
    """Test CSV metadata extraction (default: no row counting)."""
    csv_content = "id,name,age\n1,Alice,25\n2,Bob,30"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    handler = CSVHandler()
    metadata = handler.extract_metadata(csv_file)

    assert metadata["encoding_format"] == "text/csv"
    assert metadata["file_name"] == "test.csv"
    assert metadata["num_rows"] is None
    assert metadata["num_columns"] == 3
    assert metadata["columns"] == ["id", "name", "age"]

    column_types = metadata["column_types"]
    assert column_types["id"] == "cr:Int64"
    assert column_types["name"] == "sc:Text"
    assert column_types["age"] == "cr:Int64"


def test_csv_handler_count_rows(tmp_path: Path) -> None:
    """Test CSV metadata extraction with explicit row counting."""
    csv_content = "id,name,age\n1,Alice,25\n2,Bob,30"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    handler = CSVHandler()
    metadata = handler.extract_metadata(csv_file, count_rows=True)

    assert metadata["num_rows"] == 2


def test_csv_handler_empty_file(tmp_path: Path) -> None:
    """Test empty CSV file handling."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")

    handler = CSVHandler()
    with pytest.raises(ValueError):
        handler.extract_metadata(empty_csv)


def test_csv_handler_data_types(tmp_path: Path) -> None:
    """Test data type inference."""
    csv_content = "bool_col,float_col,text_col\ntrue,3.14,hello\nfalse,2.71,world"
    csv_file = tmp_path / "types.csv"
    csv_file.write_text(csv_content)

    handler = CSVHandler()
    metadata = handler.extract_metadata(csv_file)

    column_types = metadata["column_types"]
    assert column_types["bool_col"] == "sc:Boolean"
    assert column_types["float_col"] == "cr:Float64"
    assert column_types["text_col"] == "sc:Text"
