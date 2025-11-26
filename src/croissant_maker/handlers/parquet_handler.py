"""Parquet file handler for tabular event streams (e.g., MEDS)."""

from pathlib import Path
from typing import Dict

from croissant_maker.handlers.base_handler import FileTypeHandler
from croissant_maker.handlers.utils import compute_file_hash
from pyarrow.parquet import ParquetFile
import pyarrow.types as patypes


class ParquetHandler(FileTypeHandler):
    """
    Handler for Parquet files (.parquet) with schema-based type inference.

    - Uses pyarrow to read schema and row count without loading full data
    - Emits Croissant-compatible column types
    - Computes SHA256 for reproducibility
    - Keeps memory usage minimal (schema-only)
    """

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".parquet"

    def extract_metadata(self, file_path: Path) -> dict:
        """Extract metadata from a Parquet file via pyarrow schema inspection."""
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        try:
            pq = ParquetFile(str(file_path))
            schema = pq.schema_arrow
            num_rows = pq.metadata.num_rows if pq.metadata is not None else 0

            column_types: Dict[str, str] = {}
            columns = []
            for field in schema:
                columns.append(field.name)
                column_types[field.name] = self._map_arrow_type_to_croissant(
                    field.type, patypes
                )

            file_size = file_path.stat().st_size
            sha256_hash = compute_file_hash(file_path)

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_size,
                "sha256": sha256_hash,
                "encoding_format": "application/x-parquet",
                "column_types": column_types,
                "num_rows": num_rows,
                "num_columns": len(columns),
                "columns": columns,
            }
        except Exception as e:
            raise ValueError(f"Failed to process Parquet file {file_path}: {e}") from e

    @staticmethod
    def _map_arrow_type_to_croissant(arrow_type, patypes) -> str:
        """Map pyarrow types to Croissant schema.org data types."""
        try:
            if patypes.is_integer(arrow_type):
                return "sc:Integer"
            if patypes.is_floating(arrow_type) or patypes.is_decimal(arrow_type):
                return "sc:Float"
            if patypes.is_boolean(arrow_type):
                return "sc:Boolean"
            if patypes.is_timestamp(arrow_type):
                return "sc:Date"
            if patypes.is_date(arrow_type):
                return "sc:Date"
            if patypes.is_string(arrow_type) or patypes.is_large_string(arrow_type):
                return "sc:Text"
            if patypes.is_binary(arrow_type) or patypes.is_large_binary(arrow_type):
                return "sc:Text"
        except Exception:
            # Fallback to text for any exotic or extension types
            pass
        return "sc:Text"
