"""Abstract base class for file type handlers."""

from abc import ABC, abstractmethod
from pathlib import Path


class FileTypeHandler(ABC):
    """
    Abstract base class for file type handlers.

    Each handler is responsible for:
    - Detecting if it can process a specific file type
    - Extracting comprehensive metadata from files it handles

    This design allows easy extension to new file formats (Parquet, JSON, etc.)
    without modifying existing code.
    """

    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """
        Check if the handler can process the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this handler can process the file, False otherwise
        """
        pass

    @abstractmethod
    def extract_metadata(self, file_path: Path, **kwargs) -> dict:
        """
        Extract comprehensive metadata from a single file.

        Should return a dictionary containing file information, structure,
        types, and any format-specific metadata needed for Croissant generation.

        Subclasses may declare additional named parameters before **kwargs
        to support handler-specific options (e.g. count_rows for CSV).

        Args:
            file_path: Path to the file to process
            **kwargs: Handler-specific options forwarded from MetadataGenerator

        Returns:
            Dictionary containing extracted metadata. For tabular data, should include:
            - column_types: Dict mapping column names to Croissant types
            - Basic file info (path, name, size, hash, encoding_format)

        Raises:
            Exception: If the file cannot be processed
        """
        pass
