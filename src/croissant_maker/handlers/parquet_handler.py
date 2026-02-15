"""Parquet file handler for tabular event streams (e.g., MEDS)."""

from pathlib import Path

from pyarrow.parquet import ParquetFile

from croissant_maker.handlers.base_handler import FileTypeHandler
from croissant_maker.handlers.utils import (
    compute_file_hash,
    infer_column_types_from_arrow_schema,
)


class ParquetHandler(FileTypeHandler):
    """
    Handler for Parquet files (.parquet) with schema-based type inference.

    - Uses pyarrow to read schema and row count without loading full data
    - Emits Croissant-compatible column types via shared map_arrow_type()
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

            # Use the shared Arrow type mapper (same as CSV handler)
            column_types = infer_column_types_from_arrow_schema(schema)
            columns = [field.name for field in schema]

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
