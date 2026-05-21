# EM Surrogate Inv3D
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Julia](https://img.shields.io/badge/Julia-1.11-purple)

An open-source framework for **3D stochastic electromagnetic (EM) inversion** leveraging a deep learning surrogate model to accelerate 3D forward modeling.

---

## 📌 Overview
This repository providing a complete pipeline from prior validation to accelerated MCMC inversion. It consists of three core modules:

1. **Prior Falsification**: Validates whether the geostatistical prior matches the field data to be solved (Python/Jupyter).
2. **Surrogate Model Training**: Code and architectures for training the 3D EM forward surrogate model (Python).
3. **Stochastic Inversion with MCMC**:
   - Core functions for **Probabilistic Perturbation Method (PPM)** and MCMC sampling (Julia).
   - Main 3D geophysics inversion execution scripts (Julia).
   - Post-processing, visualization, and result analysis scripts (Julia).

---

## Authors
- **Zhuo Liu** — zliu93@stanford.edu, liuzhuolz@outlook.com
- **Jonas Kloeckner** — jkloeckn@stanford.edu

---

## 📦 Repository Structure
Below is the complete project layout. Please ensure your local file structure matches this perfectly before running the scripts:

   ```text
   EM_Surrogate_Inv3D/                      <-- GitHub Root
   ├── Prior falsification/                  <-- Module 1: Prior Validation
   │   ├── 3D EM prior falsification.ipynb  <-- Run this for prior falsification
   │   ├── juliapkg.json
   │   ├── Test data/                       <-- Observation data & true hyper-params
   │   ├── utils/                           <-- DGSA and FFT prior generators
   │   └── Generated prior/                 <-- ⚠️ [LARGE FILES PLACEHOLDER]
   │       ├── EMsigma_padded.npy
   │       ├── EMsigma_core.npy
   │       ├── dpred.npy
   │       └── Hyper_Param.npy
   ├── Stochastic inversion with McMC/      <-- Module 3: MCMC Inversion (Julia)
   │   ├── 3D_EM_Stochastic_Inv.jl          <-- 🚀 MAIN INVERSION RUN SCRIPT
   │   ├── PPM MCMC utils.jl                <-- Inversion helper functions
   │   ├── Plotting utils.jl                <-- Plotting & metrics
   │   ├── Results analysis.jl              <-- Post-inversion analysis
   │   ├── Project.toml / Manifest.toml     <-- Julia environment
   │   ├── Outputs/                         <-- Saved inversion results
   │   └── em_surrogate_pack/               <-- Pre-trained U-Net surrogate model
   │       ├── EM 3D Surrogate_Unet.pth     <-- PyTorch Model Weights
   │       └── EM_Unet_surrogate_prediction.py
   ├── Surrogate model training/            <-- Module 2: NN Training scripts
   ├── LICENSE
   └── README.md

---

## 💾 Data Availability (Large Files)

Due to GitHub's file size limitations, the pre-generated prior datasets (~1.29 GB) are hosted externally on Hugging Face Datasets. 

To run the notebooks successfully, you need to download these files and place them into the correct directory.

- Step 1: Download the Data
  Visit the [Hugging Face Dataset](https://huggingface.co/datasets/ZLiu93/Generated_prior) to download the following files:
  * `EMsigma_padded.npy`
  * `EMsigma_core.npy`
  * `dpred.npy`
  * `Hyper_Param.npy`

- Step 2: Place Files in the Correct Directory
  Create a folder named Generated prior inside the Prior falsification/ directory, and move the four downloaded files into it (as shown in the repository structure above).

---

** 🚀 Quick Start
1. Environment Setup
   Python: Make sure you have PyTorch installed to load the .pth surrogate weights.
   Julia: Navigate to the MCMC directory, activate and instantiate the package environment:
      Bash
      cd "Stochastic inversion with McMC"
      julia --project=. -e 'using Pkg; Pkg.instantiate()'
2. Running Inversion
   To start the 3D stochastic EM inversion using PPM MCMC, simply execute the main script:
      Bash
      julia --project=. 3D_EM_Stochastic_Inv.jl
   Note: Inversion results and chains will be automatically saved into the Outputs/ folder or serialized as .jld2 files.



## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
