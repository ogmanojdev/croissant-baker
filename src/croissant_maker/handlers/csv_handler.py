"""CSV file handler for tabular data processing."""

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pa_csv

from croissant_maker.handlers.base_handler import FileTypeHandler
from croissant_maker.handlers.utils import (
    compute_file_hash,
    infer_column_types_from_arrow_schema,
)


class CSVHandler(FileTypeHandler):
    """
    Handler for CSV and compressed CSV files with automatic type inference.

    Supports:
    - Standard CSV files (.csv)
    - Gzip-compressed CSV files (.csv.gz)
    - Bzip2-compressed CSV files (.csv.bz2)
    - XZ-compressed CSV files (.csv.xz)
    - Automatic column type detection using PyArrow
    - SHA256 hash computation for file integrity

    Uses PyArrow's CSV reader which:
    - Auto-detects compressed formats from filename extension
    - Infers precise types (timestamp[s], date32, int64, float64, etc.)
    - Reads multi-threaded by default for performance
    """

    # Common timestamp formats for medical/clinical data beyond ISO-8601.
    # PyArrow uses ISO8601 by default; these cover additional patterns found
    # in datasets like MIMIC, eICU, and OMOP.
    _TIMESTAMP_PARSERS = [
        pa_csv.ISO8601,
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]

    def can_handle(self, file_path: Path) -> bool:
        """
        Check if the file is a CSV or compressed CSV file.

        Args:
            file_path: Path to check

        Returns:
            True if file has supported CSV extension
        """
        name_lower = file_path.name.lower()
        return (
            file_path.suffix.lower() == ".csv"
            or name_lower.endswith(".csv.gz")
            or name_lower.endswith(".csv.bz2")
            or name_lower.endswith(".csv.xz")
        )

    def extract_metadata(self, file_path: Path) -> dict:
        """
        Extract comprehensive metadata from a CSV file.

        Uses PyArrow to read the CSV with automatic type inference,
        including timestamp detection and precise numeric types.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary containing:
            - Basic file info (path, name, size, hash)
            - Format information (encoding)
            - Data structure (columns, types, row count)

        Raises:
            ValueError: If the CSV file cannot be read or processed
            FileNotFoundError: If the file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Parse CSV — only this call needs error translation
        try:
            convert_options = pa_csv.ConvertOptions(
                timestamp_parsers=self._TIMESTAMP_PARSERS,
            )
            table = pa_csv.read_csv(str(file_path), convert_options=convert_options)
        except pa.lib.ArrowInvalid as e:
            raise ValueError(f"Failed to parse CSV file {file_path}: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Encoding error in CSV file {file_path}: {e}")

        if table.num_rows == 0:
            raise ValueError(f"CSV file is empty: {file_path}")

        # Infer types from the Arrow schema (shared with Parquet handler)
        column_types = infer_column_types_from_arrow_schema(table.schema)

        # Extract file properties
        file_size = file_path.stat().st_size
        sha256_hash = compute_file_hash(file_path)

        # Determine encoding format based on file extension
        name_lower = file_path.name.lower()
        if name_lower.endswith(".csv.gz"):
            encoding_format = "application/gzip"
        elif name_lower.endswith(".csv.bz2"):
            encoding_format = "application/x-bzip2"
        elif name_lower.endswith(".csv.xz"):
            encoding_format = "application/x-xz"
        else:
            encoding_format = "text/csv"

        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_size,
            "sha256": sha256_hash,
            "encoding_format": encoding_format,
            "column_types": column_types,
            "num_rows": table.num_rows,
            "num_columns": table.num_columns,
            "columns": table.column_names,
        }
