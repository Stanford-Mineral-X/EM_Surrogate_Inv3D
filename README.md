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

## License

This project is licensed under the MIT License.
