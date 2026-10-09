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

## Running the examples

The `examples/` folder contains runnable scripts that demonstrate the estimator on sample datasets. For instance, `ex_pumps.py` loads `examples/pumps.csv`, normalizes the `pressure_in_bar` column and plots the estimated survival function.

1. Activate the virtual environment created during installation:

   ```bash
   source .venv/bin/activate
   ```

2. Run the example from inside the `examples/` folder, with the project root on `PYTHONPATH` so the `beran` package can be imported:

   ```bash
   python -m examples.ex_pumps
   ```

   A plot window with the estimated survival curve should appear once the script finishes.

## Disclaimer

The `OpfKnnKernel` relies on the Optimum-Path Forest (OPF) classifier, which is retrained for every time point evaluated by the Beran estimator. Depending on the size of the dataset and the number of covariates, this can make the examples and experiments considerably slow to run. If execution takes too long, consider using a smaller dataset, a subset of the data, or one of the other available kernels (e.g. `GaussianKernel` or `PlKnnKernel`).

## Citations

If you are interested in using this repo, please cite the following work:

```
@inproceedings{10.1007/978-3-032-04555-3_9,
author = {Jodas, Danilo Samuel and Barry, Christian Laurence Almeida and Martins, Guilherme Brand{\~a}o and Santana, Marcos Cleison and Abrego, Andre Luis Severino and Colombo, Danilo and Papa, Jo{\~a}o Paulo},
title = {Learning a&nbsp;Kernel-Based Beran Estimator Using Nearest-Neighbours and&nbsp;Its Application to&nbsp;Reliability Analysis},
year = {2025},
isbn = {978-3-032-04554-6},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-032-04555-3_9},
doi = {10.1007/978-3-032-04555-3_9},
booktitle = {Artificial Neural Networks and Machine Learning – ICANN 2025: 34th International Conference on Artificial Neural Networks, Kaunas, Lithuania, September 9–12, 2025, Proceedings, Part IV},
pages = {102–114},
numpages = {13},
}
```
