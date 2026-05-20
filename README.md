# EM Surrogate Inv3D

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Julia](https://img.shields.io/badge/Julia-1.11-purple)


## Overview

An example of 3D stochastic electromagnetic inversion using a surrogate model to accelerate forward modeling.

This repository consists of three main modules:

1. Prior falsification based on field data to be solved (Python)
2. Training of EM surrogate model (Python)
3. Stochastic inversion with McMC sampling
   - Functions for McMC sampling and Probabilistic Perturbation Method (PPM) (Julia)
   - Main McMC inversion code (Julia)
   - Functions for plotting MCMC results (Julia)
   - Scripts for visualization and result analysis (Julia)

## Authors

- Zhuo Liu — zliu93@stanford.edu, liuzhuolz@outlook.com
- Jonas Kloeckner — jkloeckn@stanford.edu

## Data Availability

The pre-generated prior datasets (~1.29 GB) are hosted on Hugging Face Datasets due to GitHub's file size limits.

You can download the dataset in two ways:

### Method 1: Manual Download (Web UI)
1. Visit the [Hugging Face Dataset](https://huggingface.co/datasets/ZLiu93/Generated_prior).
2. Go to the **Files and versions** tab.
3. Download the `.npy` files and place them under `Prior falsification/Generated prior/`.

### Method 2: Python Script (Automated)
You can also download it programmatically using the `huggingface_hub` library:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ZLiu93/Generated_prior", 
    repo_type="dataset", 
    local_dir="Prior falsification/Generated prior"
)

## License

This project is licensed under the MIT License.
