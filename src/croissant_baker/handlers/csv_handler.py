"""CSV file handler for tabular data processing."""

import logging
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pa_csv

from croissant_baker.handlers.base_handler import FileTypeHandler
from croissant_baker.handlers.utils import (
    compute_file_hash,
    infer_column_types_from_arrow_schema,
)

logger = logging.getLogger(__name__)

# Pattern for extracting the column index and inferred type from ArrowInvalid.
# PyArrow does not expose this via API; format may change in future versions.
_ARROW_COL_RE = re.compile(r"In CSV column #(\d+): CSV conversion error to (\w+)")

# Max promotions before falling back to all-string types. One retry per conflicting
# column; beyond this we read everything as strings to bound I/O.
_MAX_TYPE_CONFLICT_RETRIES = 50


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

    Uses PyArrow's streaming CSV reader (open_csv) which:
    - Auto-detects compressed formats from filename extension
    - Infers precise types (timestamp[s], date32, int64, float64, etc.)
    - Streams data for constant memory usage regardless of file size

    Type inference works in two stages:
    1. PyArrow infers column types from the first block of data.
    2. If a later block contains values incompatible with the inferred type
       (e.g. a float in an integer column), the affected column is promoted
       to a wider type and the file is re-read. Integer columns are first
       widened to float64; any remaining conflicts fall back to string.
       Only the conflicting column is overridden -- all others keep their
       inferred types.
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

    # ------------------------------------------------------------------
    # Streaming with per-column type promotion
    # ------------------------------------------------------------------

    def _stream_csv(self, file_path: Path, count_rows: bool = False):
        """Return (column_types, columns, num_rows) by streaming the CSV."""
        overrides = {}
        col_names = None

        for _ in range(_MAX_TYPE_CONFLICT_RETRIES):
            opts = pa_csv.ConvertOptions(
                timestamp_parsers=self._TIMESTAMP_PARSERS,
                column_types=overrides or None,
            )
            try:
                result = self._read_streaming(file_path, opts, count_rows=count_rows)
                if overrides:
                    logger.info(
                        "%s: promoted %d column(s) due to type conflicts",
                        file_path.name,
                        len(overrides),
                    )
                return result
            except pa.lib.ArrowInvalid as exc:
                idx, inferred = self._parse_conflict(str(exc))
                if idx is None:
                    break

                if col_names is None:
                    col_names = self._header(file_path)
                if idx >= len(col_names):
                    break

                name = col_names[idx]
                # int/uint → float64 preserves numeric; other types → string
                if inferred.startswith(("int", "uint")) and name not in overrides:
                    overrides[name] = pa.float64()
                else:
                    overrides[name] = pa.string()

                logger.debug(
                    "%s: promoted column '%s' to %s",
                    file_path.name,
                    name,
                    overrides[name],
                )
                continue

        # Last resort: read everything as strings.
        if col_names is None:
            col_names = self._header(file_path)
        if len(overrides) >= _MAX_TYPE_CONFLICT_RETRIES:
            logger.warning(
                "%s: hit type conflict limit (%d), falling back to all-string types",
                file_path.name,
                _MAX_TYPE_CONFLICT_RETRIES,
            )
        else:
            logger.warning(
                "%s: falling back to all-string types (could not parse type conflict)",
                file_path.name,
            )
        opts = pa_csv.ConvertOptions(
            column_types={n: pa.string() for n in col_names},
        )
        return self._read_streaming(file_path, opts, count_rows=count_rows)

    @staticmethod
    def _read_streaming(file_path: Path, convert_options, count_rows: bool = False):
        """Open a streaming CSV reader, extract schema, and optionally count rows."""
        try:
            reader = pa_csv.open_csv(
                str(file_path),
                convert_options=convert_options,
            )
        except UnicodeDecodeError as exc:
            raise ValueError(f"Encoding error in {file_path}: {exc}")

        schema = reader.schema
        column_types = infer_column_types_from_arrow_schema(schema)
        columns = schema.names

        if count_rows:
            num_rows = 0
            for batch in reader:
                num_rows += batch.num_rows
        else:
            num_rows = None

        return column_types, columns, num_rows

    @staticmethod
    def _parse_conflict(msg):
        m = _ARROW_COL_RE.search(msg)
        return (int(m.group(1)), m.group(2)) if m else (None, None)

    @staticmethod
    def _header(file_path: Path):
        reader = pa_csv.open_csv(str(file_path))
        names = reader.schema.names
        del reader
        return names

    # ------------------------------------------------------------------
    # FileTypeHandler interface
    # ------------------------------------------------------------------

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

    def extract_metadata(
        self, file_path: Path, count_rows: bool = False, **kwargs
    ) -> dict:
        """
        Extract comprehensive metadata from a CSV file.

        Uses PyArrow to read the CSV with automatic type inference,
        including timestamp detection and precise numeric types.

        Args:
            file_path: Path to the CSV file
            count_rows: If True, scan entire file for exact row count.
                        Defaults to False for performance (returns num_rows=None).

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

        column_types, columns, num_rows = self._stream_csv(
            file_path, count_rows=count_rows
        )

        if count_rows and num_rows == 0:
            raise ValueError(f"CSV file is empty: {file_path}")

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
            "num_rows": num_rows,
            "num_columns": len(columns),
            "columns": columns,
        }
