# 🥐 Croissant Maker

A tool to automatically generate [Croissant](https://mlcommons.org/en/news/croissant-format-for-ml-datasets/) metadata for datasets, starting with those hosted on [PhysioNet](https://physionet.org/).

*Status: Alpha - Development*

## Installation (Development)

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MIT-LCP/croissant-maker.git
    cd croissant-maker
    ```

2.  **Install dependencies:**
    ```bash
    uv sync --group dev
    ```
    This creates a `.venv`, installs the package in editable mode, and includes all development and test dependencies.

3.  **Install the pre-commit hook** (run once; fires automatically on every `git commit` thereafter):
    ```bash
    uv run pre-commit install
    ```

## Usage

After `uv sync`, run the CLI directly by activating the venv first, or prefix with `uv run`:

```bash
source .venv/bin/activate   # once per terminal session — then just use `croissant-maker` directly
# or without activating:
uv run croissant-maker --help
```

### Quick start

Try it out on one of the bundled test datasets:

```bash
croissant-maker --input tests/data/input/mimiciv_demo_meds --creator "Jane Doe" --output example-metadata.jsonld
```

This scans the dataset, extracts metadata from the Parquet files it finds, and writes a Croissant JSON-LD file to `example-metadata.jsonld`. You can inspect the result or validate it:

```bash
croissant-maker validate example-metadata.jsonld
```

### Generate Croissant Metadata

Point the tool at your own dataset directory:

```bash
croissant-maker --input /path/to/dataset --creator "Your Name" --output my-metadata.jsonld
```

### Metadata Override Options

You can override default metadata fields:

```bash
croissant-maker --input /path/to/dataset \
  --name "My Dataset" \
  --description "A machine learning dataset" \
  --creator "John Doe,john@example.com,https://john.com" \
  --creator "Jane Smith,jane@example.com" \
  --license "MIT" \
  --citation "Doe et al. (2024). My Dataset."
```

| Flag | Description | Example | Required |
|------|-------------|---------|----------|
| `--input, -i` | Dataset directory | `--input /data/my-dataset` | Yes |
| `--creator` | Creator info (repeat for multiple) | `--creator "Name,email,url"` | Yes |
| `--output, -o` | Output file | `--output metadata.jsonld` | |
| `--name` | Dataset name | `--name "MIMIC-IV Demo"` | |
| `--description` | Dataset description | `--description "Medical records"` | |
| `--license` | License (SPDX ID or URL) | `--license "MIT"` | |
| `--citation` | Citation text | `--citation "Author (2024)..."` | |
| `--url` | Dataset homepage | `--url "https://example.com"` | |
| `--dataset-version` | Version | `--dataset-version "1.0.0"` | |
| `--date-published` | Publication date | `--date-published "2023-12-15"` | |
| `--no-validate` | Skip validation | `--no-validate` | |
| `--count-csv-rows` | Count exact row numbers for CSV files (slow for large datasets) | `--count-csv-rows` | |

### Validate a Croissant Metadata File

Validation checks that the file can be loaded by `mlcroissant` and conforms to the basic structure of the specification.

```bash
croissant-maker validate my-metadata.jsonld
```

## Testing

```bash
# Run all tests
uv run pytest -v

# Run a single test
uv run pytest tests/test_cli.py::test_creator_formats -v
```

## Pre-Commit Hooks & Code Quality

This project uses `pre-commit` with [Ruff](https://docs.astral.sh/ruff/) to automatically lint and format Python code before commits. Basic configuration file checks (TOML, YAML) are also included.

After running `uv run pre-commit install` once, the hook fires automatically on every `git commit`. To run it manually across all files:

```bash
uv run pre-commit run --all-files
```

## Releases

This project uses [release-please](https://github.com/googleapis/release-please) to automate versioning and GitHub Releases. You do not need to manually edit the version number or write a changelog.

### How it works

Every time a PR is merged into `main`, the release-please GitHub Action opens or updates a **Release PR** — a bot-authored pull request that:

- bumps the version in `pyproject.toml`
- generates a `CHANGELOG.md` entry listing every merged PR since the last release

The Release PR stays open and accumulates entries as more PRs are merged. When the team is ready to cut a release, **merge the Release PR**. That merge:

1. creates a git tag (e.g. `v0.1.0`)
2. creates a GitHub Release with the generated notes
3. triggers the `publish` job, which builds the package with `uv build`

Nothing else is required — no manual tagging, no direct commits to `main`.

### Commit message conventions

release-please reads commit messages (or PR titles, which become the squash-merge commit) to decide what kind of version bump to apply:

| Prefix | Effect | Example |
|--------|--------|---------|
| `feat:` | minor bump (`0.1.0 → 0.2.0`) | `feat: add Parquet handler` |
| `fix:` | patch bump (`0.1.0 → 0.1.1`) | `fix: handle empty CSV files` |
| `perf:` | patch bump | `perf: stream CSV in chunks` |
| `feat!:` or `BREAKING CHANGE:` in body | major bump (`0.1.0 → 1.0.0`) | `feat!: remove legacy API` |
| `docs:`, `build:`, `chore:` | no bump (appear in changelog or hidden) | `docs: update README` |

If a PR contains only `chore:` or `build:` commits it will not trigger a version bump on its own, but those commits will appear in the Release PR's diff so you can decide to merge it when combined with a real bump.

### One-time repository setup

The workflow needs permission to open PRs and create tags. In the GitHub repository settings:

**Settings → Actions → General → Workflow permissions → select "Read and write permissions"**

Without this the bot will fail silently and no Release PR will appear.

### Publishing to PyPI

The `publish` job in `.github/workflows/release-please.yaml` already runs `uv build` on every release. To enable PyPI upload:

1. Add a PyPI API token as a repository secret named `PYPI_API_TOKEN`
2. Uncomment the publish step in the workflow file

## License

MIT License - see [LICENSE](LICENSE) file.
