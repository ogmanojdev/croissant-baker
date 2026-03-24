# PR Report: Modern Python tooling

**Base:** `MIT-LCP/croissant-maker @ main`
**Head:** `slolab/croissant-maker @ chore/modern-python-tooling`
**Commits:** 6

This PR modernises the developer tooling and CI setup. It contains no functional changes to the metadata generation code.

> **Note:** A second PR (`feat/open-targets-partitioned-parquet`) depends on this one and should be merged afterwards.

---

## 1. Build system & packaging

**Files:** `pyproject.toml`

**What changed:**
- Build backend switched from `setuptools` to `hatchling`.
- Dependency groups migrated from `[project.optional-dependencies]` (PEP 508 extras) to `[dependency-groups]` (PEP 735). The `[dev]` and `[test]` groups no longer propagate to downstream consumers when the package is installed as a library — they are development-only concerns.
- All dependency names normalised to lowercase PEP 508 canonical form (e.g. `PyArrow` → `pyarrow`).
- Added `[tool.hatch.build.targets.wheel]` with `packages = ["src/croissant_maker"]` to explicitly declare the src layout to hatchling.

**Discussion point:** This is a breaking change for anyone currently using `pip install -e '.[test]'`. The new equivalent is `uv sync --group test`. If upstream prefers to keep setuptools, the packaging changes can be held back independently of the functional PRs.

---

## 2. CI / developer tooling

**Files:** `.github/workflows/test.yaml`, `.github/workflows/pre-commit.yaml`, `.github/dependabot.yml`, `uv.lock`, `.gitignore`, `README.md`

**What changed:**
- Both CI workflows (`test.yaml`, `pre-commit.yaml`) replace `actions/setup-python` + `pip install` with `astral-sh/setup-uv@v5` + `uv sync --group ...`. The test matrix (Python 3.10, 3.12) is unchanged.
- `uv.lock` added — pinned lockfile for reproducible installs.
- `dependabot.yml` added — weekly automated PRs to bump GitHub Actions versions and Python dependencies in `uv.lock`, preventing silent staleness.
- `.gitignore` extended: `.DS_Store`, `CLAUDE.md` (AI assistant context), and a large reference JSON that does not belong in version control.
- README updated throughout to use `uv run` prefix for all CLI and test commands.

**Discussion point:** `uv.lock` is large (~2700 lines). Some projects prefer not to commit lockfiles for libraries (only for applications). This could be dropped if upstream prefers, though committing it means Dependabot can open automated update PRs.

---

## 3. Automated releases

**Files:** `.github/workflows/release-please.yaml`, `release-please-config.json`, `.release-please-manifest.json`

**What added:**
- `release-please-action@v4` workflow: on every push to `main`, opens or updates a Release PR that bumps the version in `pyproject.toml` and generates a `CHANGELOG.md` from conventional commit messages.
- A `publish` job runs `uv build` when a release is created (PyPI upload step is present but commented out, pending a `PYPI_API_TOKEN` secret).
- Commit convention table documented in README: `feat:` → minor bump, `fix:`/`perf:` → patch, `feat!:` → major, `docs:`/`chore:` → no bump.

**Discussion point:** Requires a one-time repo setting to work: **Settings → Actions → General → Workflow permissions → Read and write permissions**. Without this the bot fails silently and no Release PR appears.
