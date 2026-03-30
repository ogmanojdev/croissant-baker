# Contributing to Croissant Baker

Thank you for your interest in contributing to Croissant Baker! This guide will help you get started.

## Ways to contribute

- **Report bugs** by opening a [GitHub issue](https://github.com/MIT-LCP/croissant-baker/issues).
- **Suggest features** or improvements via issues or discussions.
- **Submit pull requests** for bug fixes, new features, documentation, or new file handlers.
- **Improve documentation** in `docs/` or the README.

## Getting started

See the [README](README.md) for development setup instructions, including how to install dependencies with `uv` and configure pre-commit hooks.

## Development workflow

1. Create a branch from `main` for your changes.
2. Make your changes, ensuring tests pass and pre-commit hooks are satisfied (see [Testing](README.md#testing) and [Pre-Commit Hooks & Code Quality](README.md#pre-commit-hooks--code-quality) in the README).
3. Commit with a message following the [commit message conventions](README.md#commit-message-conventions).
4. Open a pull request against `main`.

## Adding a new file handler

One of the most impactful contributions is adding support for new file formats. The handler system is designed to be extensible. See the "Adding a New Handler" section in [docs/technical_overview.md](docs/technical_overview.md) for a step-by-step guide. The [technical overview](docs/technical_overview.md) also covers the project architecture and directory structure.

## Testing guidelines

- Add tests for any new functionality or bug fixes.
- Place test data in `tests/data/input/` and expected output in `tests/data/output/`.
- Integration tests are in `tests/test_end_to_end.py`; unit tests are in handler-specific files (e.g. `tests/test_csv_handler.py`).
- CI runs the test suite on Python 3.10 and 3.12.

## Pull request process

1. Ensure all tests pass locally (`uv run pytest -v`).
2. Ensure pre-commit hooks pass (`uv run pre-commit run --all-files`).
3. Write a clear PR description explaining what changed and why.
4. CI will automatically run tests (Python 3.10 and 3.12) and pre-commit checks on your PR.
5. A maintainer will review your PR and may request changes.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
