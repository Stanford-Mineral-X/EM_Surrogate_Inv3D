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

## Data Availability (Large Files)

Due to GitHub's file size limitations, the pre-generated prior datasets (~1.29 GB) are hosted externally on Hugging Face Datasets. 

To run the notebooks successfully, you need to download these files and place them into the correct directory.

- Step 1: Download the Data
  Visit the [Hugging Face Dataset](https://huggingface.co/datasets/ZLiu93/Generated_prior) to download the following files:
  * `EMsigma_padded.npy`
  * `EMsigma_core.npy`
  * `dpred.npy`
  * `Hyper_Param.npy`

- Step 2: Place Files in the Correct Directory
  Create a folder named `Generated prior` inside the `Prior falsification/` directory, and move the downloaded files there.

  Your local repository structure **must** look like this:
  ```text
   EM_Surrogate_Inv3D/                  <-- Your GitHub Root
   ├── Surrogate model training/
   ├── Stochastic inversion with McMC/
   └── Prior falsification/             <-- The second folder
       ├── 3D EM prior falsification.ipynb
       ├── juliapkg.json
       ├── utils/
       ├── Test data/
       └── Generated prior/             <-- ⚠️ PUT YOUR DOWNLOADED FILES HERE
           ├── EMsigma_padded.npy
           ├── EMsigma_core.npy
           ├── dpred.npy
           └── Hyper_Param.npy

## License

This project is licensed under the MIT License.
