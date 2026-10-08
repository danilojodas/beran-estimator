# Project

## Installation

This project requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/) to manage the environment.

1. Create the virtual environment with Python 3.12:

   ```bash
   uv venv --python 3.12
   ```

2. Install the dependencies:

   ```bash
   uv pip install -r requirements.txt --index-strategy unsafe-best-match
   ```

   The `--index-strategy unsafe-best-match` flag is required. It lets uv take PyTorch from the CUDA 11.8 index and every other package from PyPI.
