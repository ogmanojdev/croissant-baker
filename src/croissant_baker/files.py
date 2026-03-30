"""File discovery utilities for Croissant Maker."""

from pathlib import Path
from typing import List


def discover_files(dir_path: str) -> List[Path]:
    """
    Recursively discover all files in a directory and return their relative paths.

    Args:
        dir_path: Path to the directory to scan.

    Returns:
        List of relative file paths found in the directory.

    Raises:
        FileNotFoundError: If the directory does not exist or is not a directory.
        PermissionError: If the directory cannot be accessed.
    """
    try:
        directory = Path(dir_path).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"{dir_path} is not a directory")

        return [
            file.relative_to(directory)
            for file in directory.rglob("*")
            if file.is_file()
        ]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Directory not found: {e}")
    except PermissionError as e:
        raise PermissionError(f"Permission denied: {e}")
