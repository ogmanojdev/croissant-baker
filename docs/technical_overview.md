# 🥐 Croissant Baker Technical Overview

This document provides a high-level technical overview of the `croissant-baker` codebase to facilitate onboarding for developers and contributors.

## Project Goal

`croissant-baker` is a Python tool designed to automate the generation of [Croissant](https://mlcommons.org/en/news/croissant-format-for-ml-datasets/) metadata for datasets. It achieves this by:
1.  Discovering files in a dataset directory.
2.  Inspecting the content of these files using specialized handlers.
3.  Inferring schema and structural metadata (e.g., column types for tabular data, signal properties for physiological data).
4.  Generating a valid `metadata.jsonld` file following the Croissant specification.

---

## 🏛️ Core Architecture

The architecture is built on a "plug-and-play" model where different file formats are supported through modular **handlers**.

### 1. The Orchestration Layer: `MetadataGenerator`
Located in `src/croissant_baker/metadata_generator.py`.
-   This is the entry point for metadata generation.
-   It iterates through discovered files and delegates metadata extraction to the appropriate handler.
-   It uses the `mlcroissant` library to build the final Croissant metadata objects (`Metadata`, `FileObject`, `RecordSet`, `Field`, etc.) and serializes them to JSON-LD.

### 2. The Abstraction: `FileTypeHandler`
Located in `src/croissant_baker/handlers/base_handler.py`.
-   An abstract base class (ABC) that every file handler must implement.
-   Interfaces:
    -   `can_handle(file_path: Path) -> bool`: Determines if the handler can process a given file (e.g., by extension or magic number).
    -   `extract_metadata(file_path: Path, **kwargs) -> dict`: Extracts format-specific metadata.

### 3. The Registry Mechanism
Located in `src/croissant_baker/handlers/registry.py`.
-   Manages a global registry of available handlers.
-   `find_handler()` searches the registry for the first handler that can process a given file.
-   `register_all_handlers()` is called during initialization to load all standard handlers.

---

## 📦 Supported File Handlers

### CSVHandler (`csv_handler.py`)
Handles tabular data in CSV format, including compressed versions (.gz, .bz2, .xz).
- **Technology**: Uses PyArrow's streaming CSV reader for constant memory usage.
- **Type Inference**: Implements a two-stage inference process. It auto-detects types and performs "per-column promotion" (e.g., widening an integer column to float64 or string) when it encounters incompatible values later in the file.
- **Features**: Supports custom timestamp parsing for medical datasets and optional exact row counting.

### WFDBHandler (`wfdb_handler.py`)
Specifically designed for physiological waveform data (common on PhysioNet).
- **Multi-file Logic**: Implements the "related files" pattern where a single logical record consists of multiple physical files (`.hea` header, `.dat` binary signals, and `.atr` annotations).
- **Metadata**: Extracts signal names, sampling frequencies, units, and durations. All signals are mapped to `sc:Float` in the Croissant RecordSet.

### ParquetHandler (`parquet_handler.py`)
Handles Apache Parquet files, often used for large event streams.
- **Efficiency**: Performs schema-only inspection using PyArrow, avoiding the need to load the actual data into memory.
- **Metadata**: Extracts precise Arrow-based column types and row counts from the Parquet metadata footer.

### ImageHandler (`image_handler.py`)
Processes image datasets, supporting standard formats (JPEG, PNG, etc.) and scientific TIFFs.
- **Engine**: Uses `Pillow` for standard images and falls back to `tifffile` for multi-band/scientific imagery (e.g., Sentinel-2 12-band TIFFs).
- **Grouped Metadata**: Unlike tabular files, images are often grouped into a single `RecordSet`. This handler provides a `collect_image_summary` utility to aggregate dimension ranges and format counts across all images in a dataset.

---

## 🛠️ Key Components & Directory Structure

```text
src/croissant_baker/
├── __main__.py             # CLI entry point (Typer-based)
├── metadata_generator.py   # Core orchestration logic
├── files.py               # File discovery utilities
└── handlers/              # Format-specific extraction logic
    ├── base_handler.py    # ABC for all handlers
    ├── registry.py        # Handler management and discovery
    ├── csv_handler.py     # Tabular (CSV) metadata extraction
    ├── image_handler.py   # Image format (JPG, PNG, TIFF) extraction
    ├── parquet_handler.py # Parquet metadata extraction
    ├── wfdb_handler.py    # Physiological signal (WFDB) extraction
    └── utils.py           # Shared utilities (SHA256 hashing, type mapping, ID sanitization)
```

---

## ➕ Adding a New Handler

To add support for a new file format (e.g., JSON, Excel):

1.  **Create a new handler class**: In `src/croissant_baker/handlers/`, create `your_format_handler.py`.
2.  **Inherit from `FileTypeHandler`**: Implement `can_handle` and `extract_metadata`.
3.  **Return standard metadata**: Ensure `extract_metadata` returns a dictionary including `encoding_format`, and for structures, a `column_types` (or equivalent) mapping to Croissant types (e.g., `sc:Number`, `sc:Text`).
4.  **Register the handler**: Import and add your handler to `register_all_handlers()` in `src/croissant_baker/handlers/registry.py`.
5.  **Add tests**: Create a corresponding test file in `tests/` and add sample data to `tests/data/`.

---

## 🧪 Testing and Quality Assurance

### Testing Strategy
The project uses `pytest` for testing, covering multiple levels:
-   **Unit Tests**: Files like `tests/test_<handler>.py` (e.g., `tests/test_csv_handler.py`), focusing on individual handlers.
-   **End-to-End Tests**: `tests/test_end_to_end.py` runs the full `MetadataGenerator` on various test datasets and asserts on the structure and key top-level fields of the generated JSON-LD output.
-   **CLI Tests**: `tests/test_cli.py` ensures the command-line interface handles arguments correctly.

### Code Quality
-   **Linting & Formatting**: `Ruff` is used for linting and formatting.
-   **Pre-commit Hooks**: Enforces code style, trailing newlines, and basic YAML/JSON validation.
-   **Type Hinting**: Extensive use of Python type hints to improve developer experience and catch bugs early.

---

## 🚀 Key Libraries and Dependencies

-   [**mlcroissant**](https://mlcroissant.readthedocs.io/): The official Python library for Croissant, used for building and validating metadata objects.
-   [**Typer**](https://typer.tiangolo.com/): CLI builder based on Python type hints.
-   [**wfdb**](https://github.com/MIT-LCP/wfdb-python): Library for physiological signal data (important for PhysioNet integration).
-   [**pyarrow**](https://arrow.apache.org/docs/python/): Used for Parquet file inspection.
-   [**Pillow**](https://python-pillow.org/): For image metadata extraction.
-   [**tifffile**](https://github.com/cgohlke/tifffile): For multi-band scientific image support.

---

## 📍 Future Directions / TODOs

-   **Automatic Relationship Detection**: Detect foreign keys between files to add `cr:references`.
-   **Categorical Detection**: Identify columns that should be `sc:Enumeration`.
-   **Enhanced Validation**: More granular validation for community-standard vocabularies.
