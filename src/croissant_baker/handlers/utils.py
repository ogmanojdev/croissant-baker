"""Shared utilities for file handlers."""

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, Union

import pyarrow as pa
import pyarrow.types as patypes

logger = logging.getLogger(__name__)

# Characters that are invalid in Croissant @id values.
# mlcroissant rejects whitespace and URI-unsafe characters like >, (, ), %.
_INVALID_ID_CHARS = re.compile(r"[^A-Za-z0-9_.\-]")

# Read files in 64 KB chunks for hashing — power-of-2 aligns with OS page cache.
_HASH_CHUNK_SIZE = 64 * 1024


def sanitize_id(raw: str) -> str:
    """Replace characters that mlcroissant rejects in @id values.

    Column names like 'Image Name' or 'Age>30(%)' contain spaces or
    URI-unsafe characters that cause mlcroissant validation errors.
    This replaces anything outside [A-Za-z0-9_.-] with underscores.
    """
    return _INVALID_ID_CHARS.sub("_", raw)


def map_arrow_type(arrow_type: pa.DataType) -> str:
    """
    Map a PyArrow data type to the corresponding Croissant type string.

    Uses precise Croissant types where available (cr:Int64, cr:Float32, etc.)
    and falls back to schema.org types for dates, text, and booleans.

    This is the single source of truth for type mapping across all handlers
    (CSV, Parquet, and future formats like JSON, O RC, Feather).

    Args:
        arrow_type: A PyArrow DataType from a table or file schema.

    Returns:
        Croissant-compatible type string (e.g. "sc:DateTime", "cr:Int64").
    """
    try:
        # Timestamps (with or without timezone) → sc:DateTime
        if patypes.is_timestamp(arrow_type):
            return "sc:DateTime"

        # Date-only (no time component) → sc:Date
        if patypes.is_date(arrow_type):
            return "sc:Date"

        # Time-only → sc:Time
        if patypes.is_time(arrow_type):
            return "sc:Time"

        # Integers — use precise Croissant types with bit-width
        if patypes.is_integer(arrow_type):
            prefix = "cr:UInt" if patypes.is_unsigned_integer(arrow_type) else "cr:Int"
            return f"{prefix}{arrow_type.bit_width}"

        # Floats — use precise Croissant types with bit-width
        # Croissant spec only defines cr:Float16, cr:Float32, cr:Float64.
        # For smaller widths (e.g. float8) fall back to generic sc:Float,
        # matching HuggingFace's behavior.
        if patypes.is_floating(arrow_type):
            bw = arrow_type.bit_width
            if bw in (16, 32, 64):
                return f"cr:Float{bw}"
            return "sc:Float"

        # Decimals → cr:Float64 (best general approximation)
        if patypes.is_decimal(arrow_type):
            return "cr:Float64"

        # Booleans
        if patypes.is_boolean(arrow_type):
            return "sc:Boolean"

        # Strings
        if patypes.is_string(arrow_type) or patypes.is_large_string(arrow_type):
            return "sc:Text"

        # Binary data
        if patypes.is_binary(arrow_type) or patypes.is_large_binary(arrow_type):
            return "sc:Text"

        # Null type (all values null) → safe fallback
        if patypes.is_null(arrow_type):
            return "sc:Text"

        # List/large-list: caller sets is_array=True; return the inner element type.
        if patypes.is_list(arrow_type) or patypes.is_large_list(arrow_type):
            return map_arrow_type(arrow_type.value_type)

    except Exception:
        pass

    # Fallback for any unrecognized or exotic types (including struct — callers
    # that want nested sub_fields should detect is_struct before calling this).
    return "sc:Text"


def is_arrow_list(arrow_type: pa.DataType) -> bool:
    """Return True if the Arrow type is a list or large-list."""
    return patypes.is_list(arrow_type) or patypes.is_large_list(arrow_type)


def infer_column_types_from_arrow_schema(schema: pa.Schema) -> Dict[str, str]:
    """
    Infer Croissant types for all columns in a PyArrow schema.

    This is the shared entry point used by both CSV and Parquet handlers.

    Args:
        schema: A PyArrow Schema (from a Table, ParquetFile, etc.)

    Returns:
        Dictionary mapping column names to Croissant type strings.
    """
    return {field.name: map_arrow_type(field.type) for field in schema}


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """
    Compute SHA256 hash of a file for Croissant integrity verification.

    Reads the file as-is on disk (compressed bytes included) rather than
    decompressing first. This matches what users download and verify.

    Args:
        file_path: Path to the file (str or Path object)

    Returns:
        Hexadecimal SHA256 hash string

    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file cannot be read
    """
    # Convert to Path only if needed
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        sha256_hash = hashlib.sha256()
        # Hash the file as-is on disk (compressed bytes). This matches what users
        # download and verify, and avoids decompressing gigabytes just for hashing.
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    except (IOError, OSError) as e:
        raise PermissionError(f"Cannot read file {file_path}: {e}")


def get_clean_record_name(file_name: str) -> str:
    """
    Generate a clean record set name from a file name.

    Removes common file extensions in a generic way, not hardcoded to any format.

    Args:
        file_name: Original file name

    Returns:
        Clean name suitable for record set naming. Returns original name if
        cleaning would result in empty string.
    """
    if not file_name or not isinstance(file_name, str):
        logger.warning(f"Invalid file_name provided: {repr(file_name)}")
        return str(file_name) if file_name else "unknown"

    name = file_name.strip()

    # Remove common compression extensions first
    if name.endswith(".gz"):
        name = name[:-3]
    elif name.endswith(".bz2"):
        name = name[:-4]
    elif name.endswith(".xz"):
        name = name[:-3]
    elif name.endswith(".zip"):
        name = name[:-4]

    # Remove common data file extensions
    extensions = [".csv", ".tsv", ".json", ".parquet", ".txt", ".dat"]
    for ext in extensions:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break

    # Ensure we return something valid
    return name if name else file_name
