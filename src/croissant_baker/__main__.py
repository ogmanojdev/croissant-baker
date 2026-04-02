"""Command-line interface for Croissant Baker."""

import csv
import typer
from pathlib import Path
import importlib.metadata
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional, List

from croissant_baker.metadata_generator import MetadataGenerator
from croissant_baker.files import discover_files

# Create the Typer application instance
app = typer.Typer(
    name="croissant-baker",
    help="🥐 Generate Croissant metadata for datasets with automatic type inference",
    add_completion=False,
    rich_markup_mode="markdown",
)


def _get_version() -> str:
    """Get version from package metadata."""
    try:
        return importlib.metadata.version("croissant-baker")
    except importlib.metadata.PackageNotFoundError:
        return "unknown (not installed as package)"


def _get_default_output_name(input_path: str) -> str:
    """Generate default output filename based on input path."""
    dataset_name = Path(input_path).name
    return f"{dataset_name}-croissant.jsonld"


# Croissant spec: these CLI flags map to fields that *must* be specified for every dataset.
# https://docs.mlcommons.org/croissant/docs/croissant-spec.html
# The tool generates defaults for all of them, but warns when user hasn't explicitly set them.
_SPEC_REQUIRED_FLAGS = {
    "creator": "--creator",
    "description": "--description",
    "url": "--url",
    "license": "--license",
    "date_published": "--date-published",
}


def _warn_missing_spec_fields(**provided: object) -> None:
    """Warn about spec-required fields that were not explicitly provided."""
    missing = [
        flag for key, flag in _SPEC_REQUIRED_FLAGS.items() if not provided.get(key)
    ]
    if missing:
        typer.echo(
            f"\nWarning: {', '.join(missing)} are required by the Croissant spec but were not provided.\n"
            "  The tool used defaults — review them before publishing.\n"
            "  See: https://docs.mlcommons.org/croissant/docs/croissant-spec.html",
            err=True,
        )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input: str = typer.Option(
        None, "--input", "-i", help="Directory containing dataset files"
    ),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate metadata before saving"
    ),
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
    # Metadata override options
    name: Optional[str] = typer.Option(
        None, "--name", help="Dataset name (defaults to directory name)"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", help="Dataset description"
    ),
    url: Optional[str] = typer.Option(
        None, "--url", help="Dataset URL (e.g., https://example.com/dataset)"
    ),
    license: Optional[str] = typer.Option(
        None, "--license", help="License URL or SPDX identifier (e.g., CC-BY-4.0)"
    ),
    citation: Optional[str] = typer.Option(
        None, "--citation", help="Citation text (preferably BibTeX format)"
    ),
    dataset_version: Optional[str] = typer.Option(
        None, "--dataset-version", help="Dataset version (e.g., 1.0.0)"
    ),
    date_published: Optional[str] = typer.Option(
        None,
        "--date-published",
        help="Publication date (e.g., 2023-12-15 or 2023-12-15T10:30:00)",
    ),
    # Creator information following mlcroissant specification
    # Spec: creator is REQUIRED with cardinality MANY (supports multiple creators)
    # ExpectedType: Organization OR Person with flexible properties (name, email, url)
    # Examples: --creator "John Doe" --creator "Jane Smith,jane@example.com,https://jane.com"
    creator: Optional[List[str]] = typer.Option(
        None,
        "--creator",
        help="Creator information. Format: 'Name[,Email[,URL]]'. Use multiple times for multiple creators. Examples: --creator 'John Doe' --creator 'Jane Smith,jane@example.com,https://jane.com'",
    ),
    count_csv_rows: bool = typer.Option(
        False,
        "--count-csv-rows",
        help="Count exact row numbers for CSV files (slow for large datasets)",
    ),
) -> None:
    """🥐 **Croissant Baker** - Generate rich metadata for your datasets"""

    if version:
        typer.echo(f"🥐 croissant-baker {_get_version()}")
        return

    if ctx.invoked_subcommand is not None:
        return

    if not input:
        typer.echo("croissant-baker: try 'croissant-baker --help' for more information")
        typer.echo("")
        typer.echo("Usage: croissant-baker --input <dataset-path> [--output <file>]")
        typer.echo("       croissant-baker validate <file>")
        typer.echo("       croissant-baker --version")
        typer.echo("       croissant-baker --help")
        return

    if not output:
        output = _get_default_output_name(input)
        typer.echo(f"Auto-generated output filename: {output}")

    # Validate required fields and input path
    # 1. Dataset path must exist and be a directory
    dataset_path_obj = Path(input)
    if not dataset_path_obj.is_dir():
        typer.echo(f"Error: Dataset path '{input}' is not a directory", err=True)
        raise typer.Exit(code=1)

    # 2. At least one creator required by the Croissant spec (cardinality MANY)
    if not creator:
        typer.echo(
            "Error: At least one '--creator' option is required "
            "to comply with the Croissant specification.",
            err=True,
        )
        typer.echo(
            "Example: --creator 'John Doe,john@example.com' or --creator 'Jane Smith'",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            # Initialize generator with metadata overrides
            metadata_progress = progress.add_task("Analyzing dataset...", total=None)

            # Parse creators following mlcroissant specification
            # Allows flexible Person/Organization objects with optional properties
            parsed_creators = []
            if creator:
                for creator_info in creator:
                    creator_info = creator_info.strip()

                    # Preferred: semicolon
                    if ";" in creator_info:
                        creator_parts = [p.strip() for p in creator_info.split(";")]

                    else:
                        # Use CSV parsing for comma cases (handles quotes properly)
                        creator_parts = next(csv.reader([creator_info]))
                        creator_parts = [p.strip() for p in creator_parts]

                    if not creator_parts or not creator_parts[0]:
                        continue

                    creator_obj = {"name": creator_parts[0]}

                    if len(creator_parts) > 1 and creator_parts[1]:
                        creator_obj["email"] = creator_parts[1]

                    if len(creator_parts) > 2 and creator_parts[2]:
                        creator_obj["url"] = creator_parts[2]

                    parsed_creators.append(creator_obj)

            # Warn early if --count-csv-rows is set but dataset has no CSV files
            if count_csv_rows:
                csv_extensions = {".csv", ".csv.gz", ".csv.bz2", ".csv.xz"}
                all_files = discover_files(input)
                has_csv = any(
                    any(str(f).endswith(ext) for ext in csv_extensions)
                    for f in all_files
                )
                if not has_csv:
                    typer.echo(
                        "Warning: --count-csv-rows has no effect: no CSV files found in dataset",
                        err=True,
                    )

            generator = MetadataGenerator(
                dataset_path=input,
                name=name,
                description=description,
                url=url,
                license=license,
                citation=citation,
                version=dataset_version,
                date_published=date_published,
                creators=parsed_creators if parsed_creators else None,
                count_csv_rows=count_csv_rows,
            )

            # Generate metadata
            progress.update(metadata_progress, description="Generating metadata...")
            metadata_dict = generator.generate_metadata()

            # Save and optionally validate
            output_file = Path(output)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if validate:
                progress.update(
                    metadata_progress, description="Validating and saving..."
                )
                generator.save_metadata(output, validate=True)
                progress.update(metadata_progress, description="Validation completed!")
            else:
                progress.update(metadata_progress, description="Saving metadata...")
                generator.save_metadata(output, validate=False)
                progress.update(metadata_progress, description="Save completed!")

        # Show results
        file_count = len(metadata_dict.get("distribution", []))
        record_count = len(metadata_dict.get("recordSet", []))

        typer.echo(
            f"Success! Generated {'validated ' if validate else ''}Croissant metadata"
        )
        typer.echo(f"Files: {file_count}")
        typer.echo(f"Record sets: {record_count}")
        typer.echo(f"Saved to: {output}")

        if not validate:
            typer.echo(
                f"Tip: Run `croissant-baker validate {output}` to validate later"
            )

        _warn_missing_spec_fields(
            creator=creator,
            description=description,
            url=url,
            license=license,
            date_published=date_published,
        )

    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def validate(
    file_path: str = typer.Argument(..., help="Path to Croissant metadata file"),
) -> None:
    """Validate a Croissant metadata file."""
    try:
        typer.echo(f"Validating: {file_path}")
        import mlcroissant as mlc

        dataset = mlc.Dataset(file_path)
        typer.echo("Valid! Croissant file passed validation")
        typer.echo(f"Dataset: {dataset.metadata.name}")
        typer.echo(f"Description: {dataset.metadata.description}")

        if hasattr(dataset.metadata, "distribution"):
            file_count = (
                len(dataset.metadata.distribution)
                if dataset.metadata.distribution
                else 0
            )
            typer.echo(f"Files: {file_count}")

        if hasattr(dataset.metadata, "record_sets"):
            record_count = (
                len(dataset.metadata.record_sets) if dataset.metadata.record_sets else 0
            )
            typer.echo(f"Record sets: {record_count}")

    except ImportError:
        typer.echo("Error: mlcroissant is required for validation", err=True)
        typer.echo("Fix: pip install mlcroissant", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Validation failed: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
