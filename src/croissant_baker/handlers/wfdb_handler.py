"""WFDB file handler for physiological waveform data."""

from pathlib import Path
import wfdb

from croissant_baker.handlers.base_handler import FileTypeHandler
from croissant_baker.handlers.utils import compute_file_hash


class WFDBHandler(FileTypeHandler):
    """
    Handler for WFDB format files (PhysioNet waveform databases).

    WFDB records consist of multiple related files:
    - .hea: Header file (text, contains metadata)
    - .dat: Data file (binary, contains signals)
    - .atr: Annotation file (optional, contains beat/event annotations)

    The 'related_files' pattern in extract_metadata() allows handlers to indicate
    that multiple physical files form a single logical record. This is a general
    capability that other multi-file formats (DICOM series, HDF5 with external
    links, Parquet partitions) can also use. Each related file becomes a separate
    cr:FileObject in the Croissant metadata, but they all describe one RecordSet.
    """

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".hea"

    def extract_metadata(self, file_path: Path, **kwargs) -> dict:
        if not file_path.exists():
            raise FileNotFoundError(f"WFDB header file not found: {file_path}")

        record_path = file_path.with_suffix("")
        record = wfdb.rdheader(str(record_path))

        dat_file = file_path.with_suffix(".dat")
        if not dat_file.exists():
            raise ValueError(f"WFDB data file missing for header: {file_path}")

        # Build list of related files that form this logical WFDB record.
        # Each becomes a separate cr:FileObject in Croissant metadata.
        related_files = [
            {
                "path": str(dat_file),
                "name": dat_file.name,
                "type": "data",
                "encoding": "application/x-wfdb-data",
                "size": dat_file.stat().st_size,
                "sha256": compute_file_hash(dat_file),
            }
        ]

        atr_file = file_path.with_suffix(".atr")
        if atr_file.exists():
            related_files.append(
                {
                    "path": str(atr_file),
                    "name": atr_file.name,
                    "type": "annotation",
                    "encoding": "application/x-wfdb-annotation",
                    "size": atr_file.stat().st_size,
                    "sha256": compute_file_hash(atr_file),
                }
            )

        signal_types = {sig: "sc:Float" for sig in record.sig_name}
        duration_seconds = record.sig_len / record.fs if record.fs > 0 else 0

        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "sha256": compute_file_hash(file_path),
            "encoding_format": "application/x-wfdb-header",
            "record_name": record.record_name,
            "related_files": related_files,
            "signal_names": record.sig_name,
            "signal_types": signal_types,
            "units": record.units,
            "sampling_frequency": record.fs,
            "num_samples": record.sig_len,
            "num_signals": record.n_sig,
            "duration_seconds": duration_seconds,
            "comments": record.comments if record.comments else [],
        }

        if hasattr(record, "base_datetime") and record.base_datetime:
            metadata["base_datetime"] = record.base_datetime.isoformat()
        if hasattr(record, "base_date") and record.base_date:
            metadata["base_date"] = str(record.base_date)
        if hasattr(record, "base_time") and record.base_time:
            metadata["base_time"] = str(record.base_time)
        if hasattr(record, "adc_gain") and record.adc_gain:
            metadata["adc_gain"] = record.adc_gain
        if hasattr(record, "baseline") and record.baseline:
            metadata["baseline"] = record.baseline
        if hasattr(record, "init_value") and record.init_value:
            metadata["init_value"] = record.init_value
        if hasattr(record, "checksum") and record.checksum:
            metadata["checksum"] = record.checksum
        if hasattr(record, "fmt") and record.fmt:
            metadata["fmt"] = record.fmt

        return metadata
