# Croissant Maker

A tool to automatically generate [Croissant](https://mlcommons.org/en/news/croissant-format-for-ml-datasets/) metadata for datasets, starting with those hosted on [PhysioNet](https://physionet.org/).

*Status: Alpha - Initial Setup*

## Installation (Development)

It is highly recommended to use a virtual environment to manage dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MIT-LCP/croissant-maker.git
    cd croissant-maker
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create a venv
    python3 -m venv .venv

    # Activate the venv 
    source .venv/bin/activate
    ```

3.  **Install dependencies:** (Make sure the venv is active)
    ```bash
    pip install -e '.[test]'
    ```
    This installs the package in editable mode along with testing requirements *inside* your virtual environment.

## Basic Usage (Placeholder)

Make sure your virtual environment is activated (`source .venv/bin/activate`).

Check the command-line help:

```bash
python -m croissant_maker --help