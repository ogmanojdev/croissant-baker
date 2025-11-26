"""Handler registry for managing file type handlers."""

from pathlib import Path
from typing import Optional, List
from croissant_maker.handlers.base_handler import FileTypeHandler


# Global registry for file handlers
_registry: List[FileTypeHandler] = []


def register_handler(handler: FileTypeHandler) -> None:
    """
    Register a file type handler in the global registry.

    Handlers are checked in registration order when processing files.
    Duplicate handlers are automatically ignored.

    Args:
        handler: Handler instance to register
    """
    if handler not in _registry:
        _registry.append(handler)


def find_handler(file_path: Path) -> Optional[FileTypeHandler]:
    """
    Find the first registered handler that can process the given file.

    Args:
        file_path: Path to the file needing processing

    Returns:
        Handler instance that can process the file, or None if no handler found
    """
    for handler in _registry:
        if handler.can_handle(file_path):
            return handler
    return None


def get_registered_handlers() -> List[FileTypeHandler]:
    """
    Get a list of all registered handlers.

    Returns:
        List of registered handler instances
    """
    return _registry.copy()


def register_all_handlers() -> None:
    """
    Register all available handlers.

    This is the SINGLE place where all handler registration happens.
    Call this once to set up all handlers.
    """
    # Import and register all handlers here
    from croissant_maker.handlers.csv_handler import CSVHandler
    from croissant_maker.handlers.wfdb_handler import WFDBHandler
    from croissant_maker.handlers.parquet_handler import ParquetHandler

    register_handler(CSVHandler())
    register_handler(WFDBHandler())
    register_handler(ParquetHandler())

    # Future handlers go here. Example:
    # from croissant_maker.handlers.parquet_handler import ParquetHandler
    # register_handler(ParquetHandler())
