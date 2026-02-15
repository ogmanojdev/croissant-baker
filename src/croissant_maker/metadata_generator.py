"""Croissant metadata generator for datasets."""

import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import mlcroissant as mlc

from croissant_maker.files import discover_files
from croissant_maker.handlers.registry import find_handler, register_all_handlers
from croissant_maker.handlers.utils import get_clean_record_name, sanitize_id

# Register all handlers
register_all_handlers()


def serialize_datetime(obj):
    """Convert datetime objects to ISO format strings for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class MetadataGenerator:
    """
    Generates Croissant metadata for datasets with automatic type inference.

    This class discovers files in a dataset directory, processes them using
    registered file handlers, and generates rich Croissant JSON-LD metadata
    that describes the dataset structure and types.
    """

    def __init__(
        self,
        dataset_path: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        license: Optional[str] = None,
        citation: Optional[str] = None,
        version: Optional[str] = None,
        date_published: Optional[str] = None,
        creators: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Initialize the metadata generator for a dataset.

        Args:
            dataset_path: Path to the directory containing dataset files
            name: Dataset name (defaults to directory name)
            description: Dataset description
            url: Dataset URL
            license: License URL or SPDX identifier
            citation: Citation text (preferably BibTeX format)
            version: Dataset version
            date_published: Publication date (e.g., "2023-12-15" or "2023-12-15T10:30:00")
            creators: List of creator dictionaries with name, email, url fields

        Raises:
            ValueError: If the dataset path is not a directory
        """
        self.dataset_path = Path(dataset_path).resolve()
        if not self.dataset_path.is_dir():
            raise ValueError(f"Dataset path {dataset_path} is not a directory")

        # Store metadata overrides
        self.name = name
        self.description = description
        self.url = url
        self.license = license
        self.citation = citation
        self.version = version
        self.date_published = date_published
        self.creators = creators

    def generate_metadata(self) -> dict:
        """
        Generate complete Croissant metadata for the dataset.

        Discovers all files in the dataset, processes them with appropriate
        handlers, and creates a comprehensive metadata structure following
        the Croissant specification.

        Returns:
            Dictionary containing the generated Croissant metadata

        Raises:
            ValueError: If no supported files are found in the dataset
        """
        # Discover and process files
        files = discover_files(str(self.dataset_path))
        file_metadata = []

        for file_path in files:
            full_path = self.dataset_path / file_path
            handler = find_handler(full_path)
            if handler:
                try:
                    metadata = handler.extract_metadata(full_path)
                    metadata["relative_path"] = str(file_path)
                    file_metadata.append(metadata)
                except Exception as e:
                    print(f"Warning: Failed to process {file_path}: {e}")

        if not file_metadata:
            raise ValueError("No supported files found in the dataset")

        # Create Croissant metadata structure with user overrides or defaults
        dataset_name = self.name or self.dataset_path.name

        # Generate dataset description - prioritize user override, then try to be descriptive
        if self.description:
            description = self.description
        else:
            file_types = set(
                meta.get("encoding_format", "unknown") for meta in file_metadata
            )
            file_types_str = ", ".join(sorted(file_types))
            description = f"Dataset containing {len(file_metadata)} files ({file_types_str}) with automatically inferred types and structure"

        # Generate dataset URL - prioritize user override, then use file path as fallback
        dataset_url = self.url or f"file://{self.dataset_path}"

        # Handle license - support both SPDX identifiers and URLs following Croissant spec
        # See: https://github.com/mlcommons/croissant/blob/main/docs/croissant-spec.md#license
        # Croissant license should be a single string (URL)
        if self.license:
            if self.license.startswith(("http://", "https://")):
                license_value = self.license  # Already a URL
            else:
                # Convert SPDX identifier to URL (common licenses)
                # Based on official Croissant examples from HuggingFace and Kaggle
                spdx_to_url = {
                    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
                    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
                    "CC-BY-NC-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
                    "CC-BY-ND-4.0": "https://creativecommons.org/licenses/by-nd/4.0/",
                    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "MIT": "https://opensource.org/licenses/MIT",
                    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
                    "GPL-3.0": "https://www.gnu.org/licenses/gpl-3.0.html",
                    "BSD-3-Clause": "https://opensource.org/licenses/BSD-3-Clause",
                }
                license_value = spdx_to_url.get(self.license, self.license)
        else:
            license_value = (
                "https://creativecommons.org/licenses/by/4.0/"  # Default to CC-BY-4.0
            )

        # Handle creators - convert to schema.org Person/Organization objects following Croissant spec
        # Croissant spec: creator is REQUIRED with cardinality MANY (supports multiple creators)
        # See: https://docs.mlcommons.org/croissant/docs/croissant-spec.html#required
        # Real examples: https://huggingface.co/api/datasets/ibm/duorc/croissant
        if self.creators:
            creator_objects = []
            for creator_dict in self.creators:
                person_kwargs = {}
                # Add available properties - Croissant/schema.org Person supports name, email, url
                if "name" in creator_dict:
                    person_kwargs["name"] = creator_dict["name"]
                if "email" in creator_dict:
                    person_kwargs["email"] = creator_dict["email"]
                if "url" in creator_dict:
                    person_kwargs["url"] = creator_dict["url"]

                # Create Person object with available properties
                creator_objects.append(mlc.Person(**person_kwargs))
        else:
            # Default creator - could be improved by parsing CITATION or README files
            creator_objects = [
                mlc.Person(name="Dataset Creator", email="creator@example.com")
            ]

        # Handle citation - prioritize user override, then generate basic citation
        if self.citation:
            cite_as = self.citation
        else:
            current_year = datetime.now().year
            cite_as = f"Dataset Creator. ({current_year}). {dataset_name} Dataset. Generated with automated type inference."

        # Handle date_published - prioritize user override, then default to current time
        if self.date_published:
            try:
                # Parse user-provided date - supports ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
                publication_date = datetime.fromisoformat(self.date_published)
            except ValueError as e:
                raise ValueError(
                    f"Invalid date format for --date-published: '{self.date_published}'. "
                    f"Expected ISO format like '2023-12-15' or '2023-12-15T10:30:00'. Error: {e}"
                )
        else:
            publication_date = datetime.now()

        # Handle version
        dataset_version = self.version or "1.0.0"

        metadata = mlc.Metadata(
            name=dataset_name,
            description=description,
            url=dataset_url,
            license=license_value,
            creators=creator_objects,
            date_published=publication_date,
            version=dataset_version,
            cite_as=cite_as,
        )

        file_objects = []
        record_sets = []
        # Use a counter instead of enumerate(i) to ensure unique FileObject IDs.
        # Some formats (e.g., WFDB) create multiple FileObjects per iteration via
        # related_files, so enumerate(i) would cause ID conflicts. The counter
        # increments for every FileObject created, not just per iteration.
        file_counter = 0
        recordset_counter = 0

        # Collect image file IDs so we can build one FileSet + summary
        # RecordSet after the per-file loop (images are grouped, not per-file).
        image_file_ids = []
        image_metas = []

        for file_meta in file_metadata:
            file_id = f"file_{file_counter}"
            file_counter += 1

            # Create FileObject for each file.
            # Correctly creates a cr:FileObject per discovered file with
            # relative contentUrl, SHA256 checksum, size, and encoding format.
            file_obj = mlc.FileObject(
                id=file_id,
                name=file_meta["file_name"],
                content_url=file_meta["relative_path"],
                encoding_formats=[file_meta["encoding_format"]],
                content_size=str(file_meta["file_size"]),
                sha256=file_meta["sha256"],
            )
            file_objects.append(file_obj)

            # Handle multi-file records (e.g., WFDB: .hea + .dat + .atr)
            # Some formats like WFDB have multiple physical files per logical record
            related_file_ids = []
            if "related_files" in file_meta:
                for related in file_meta["related_files"]:
                    related_id = f"file_{file_counter}"
                    file_counter += 1
                    related_file_ids.append(related_id)

                    rel_path = Path(related["path"])
                    relative_path = str(rel_path.relative_to(self.dataset_path))

                    related_obj = mlc.FileObject(
                        id=related_id,
                        name=related["name"],
                        content_url=relative_path,
                        encoding_formats=[related["encoding"]],
                        content_size=str(related["size"]),
                        sha256=related["sha256"],
                    )
                    file_objects.append(related_obj)

            # --- RecordSet creation per file type ---

            # Tabular data (CSV, Parquet): one RecordSet per file with column fields
            #
            # TODO: Add support for more advanced Croissant features.
            # - references: Detect and add `references` for foreign key relationships
            #   (e.g., subject_id, hadm_id). This is high-impact.
            # - enumerations: For categorical columns, generate sc:Enumeration RecordSets.
            if "column_types" in file_meta:
                fields = []
                for col_name, col_type in file_meta["column_types"].items():
                    safe_name = sanitize_id(col_name)
                    field = mlc.Field(
                        id=f"{file_id}_{safe_name}",
                        name=col_name,
                        description=f"Column '{col_name}' from {file_meta['file_name']}",
                        data_types=[col_type],
                        source=mlc.Source(
                            id=f"{file_id}_source_{safe_name}",
                            file_object=file_id,
                            extract=mlc.Extract(column=col_name),
                        ),
                    )
                    fields.append(field)

                record_set = mlc.RecordSet(
                    id=f"recordset_{recordset_counter}",
                    name=get_clean_record_name(file_meta["file_name"]),
                    description=f"Records from {file_meta['file_name']} ({file_meta.get('num_rows', 'unknown')} rows)",
                    fields=fields,
                )
                record_sets.append(record_set)
                recordset_counter += 1

            # Signal data (e.g., WFDB physiological waveforms)
            elif "signal_types" in file_meta:
                fields = []
                for signal_name, signal_type in file_meta["signal_types"].items():
                    safe_name = sanitize_id(signal_name)
                    field = mlc.Field(
                        id=f"{file_id}_{safe_name}",
                        name=signal_name,
                        description=f"Signal '{signal_name}' from {file_meta['record_name']}",
                        data_types=[signal_type],
                        source=mlc.Source(
                            id=f"{file_id}_source_{safe_name}",
                            file_object=file_id,
                        ),
                    )
                    fields.append(field)

                duration = file_meta.get("duration_seconds", 0)
                num_samples = file_meta.get("num_samples", 0)
                sampling_freq = file_meta.get("sampling_frequency", 0)

                record_set = mlc.RecordSet(
                    id=f"recordset_{recordset_counter}",
                    name=file_meta["record_name"],
                    description=f"WFDB record {file_meta['record_name']}: {file_meta.get('num_signals', 0)} signals at {sampling_freq} Hz, {num_samples} samples ({duration:.2f} seconds)",
                    fields=fields,
                )
                record_sets.append(record_set)
                recordset_counter += 1

            # Image data: defer to summary RecordSet after the loop
            elif "image_properties" in file_meta:
                image_file_ids.append(file_id)
                image_metas.append(file_meta)

        # Build a FileSet + summary RecordSet for all images in the dataset.
        # Each image already has its own cr:FileObject (with SHA256 and size).
        # The FileSet groups them by glob pattern so the RecordSet source can
        # reference all images at once, following the Croissant spec PASS example.
        if image_file_ids:
            from croissant_maker.handlers.image_handler import collect_image_summary

            summary = collect_image_summary(image_metas)
            w_lo, w_hi = summary["width_range"]
            h_lo, h_hi = summary["height_range"]
            b_lo, b_hi = summary["num_bands_range"]
            formats_str = ", ".join(
                f"{fmt} ({cnt})" for fmt, cnt in summary["format_counts"].items()
            )

            # Human-readable dimension string for the RecordSet description.
            # Uniform size → "1920x1080", mixed → "640-1920x480-1080".
            if w_lo == w_hi and h_lo == h_hi:
                dims = f"{w_lo}x{h_lo}"
            else:
                dims = f"{w_lo}-{w_hi}x{h_lo}-{h_hi}"

            # Only mention band count when images are multi-band (>4),
            # i.e. scientific imagery like Sentinel-2. Standard RGB/RGBA
            # images (1-4 bands) don't need the note.
            bands_note = f", {b_lo}-{b_hi} bands" if b_hi > 4 else ""

            # Determine unique extensions and MIME types across all images.
            extensions = set()
            mime_types = set()
            for meta in image_metas:
                ext = Path(meta["file_name"]).suffix.lower()
                extensions.add(ext)
                mime_types.add(meta["encoding_format"])

            # Build glob patterns -- use **/ prefix to match subdirectories.
            includes = [f"**/*{ext}" for ext in sorted(extensions)]

            fileset_id = "image-files"
            image_fileset = mlc.FileSet(
                id=fileset_id,
                name="Image files",
                description=f"{summary['num_images']} image files ({formats_str})",
                encoding_formats=sorted(mime_types),
                includes=includes,
            )
            file_objects.append(image_fileset)

            image_fields = [
                mlc.Field(
                    id="images/image_content",
                    name="image",
                    description=f"Image content ({summary['num_images']} files, {formats_str})",
                    data_types=["sc:ImageObject"],
                    source=mlc.Source(
                        id="images/image_content/source",
                        file_set=fileset_id,
                        extract=mlc.Extract(file_property="content"),
                    ),
                ),
            ]

            image_record_set = mlc.RecordSet(
                id=f"recordset_{recordset_counter}",
                name="images",
                description=f"{summary['num_images']} images ({dims}{bands_note}): {formats_str}",
                fields=image_fields,
            )
            record_sets.append(image_record_set)
            recordset_counter += 1

        metadata.distribution = file_objects
        metadata.record_sets = record_sets

        return metadata.to_json()

    def save_metadata(self, output_path: str, validate: bool = True) -> None:
        """
        Generate and save Croissant metadata to a file.

        Args:
            output_path: Path where the metadata file should be saved
            validate: Whether to validate the metadata before saving

        Raises:
            ValueError: If validation fails or file cannot be saved
        """
        metadata_dict = self.generate_metadata()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if validate:
            # Validate using temporary file before saving to final location
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonld", delete=False
            ) as tmp_file:
                json.dump(
                    metadata_dict,
                    tmp_file,
                    indent=2,
                    ensure_ascii=False,
                    default=serialize_datetime,
                )
                tmp_path = tmp_file.name

            try:
                # Validate by attempting to load with mlcroissant
                mlc.Dataset(tmp_path)
                self._save_to_file(metadata_dict, output_file)
            except Exception as e:
                raise ValueError(f"Validation failed: {e}")
            finally:
                # Clean up the temporary file
                Path(tmp_path).unlink(missing_ok=True)
        else:
            self._save_to_file(metadata_dict, output_file)

    def _save_to_file(self, metadata_dict: dict, output_file: Path) -> None:
        """
        Save metadata dictionary to a JSON-LD file.

        Args:
            metadata_dict: The metadata to save
            output_file: Path where the file should be saved
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                metadata_dict,
                f,
                indent=2,
                ensure_ascii=False,
                default=serialize_datetime,
            )
            # Ensure newline at end of file to avoid pre-commit edits
            f.write("\n")
