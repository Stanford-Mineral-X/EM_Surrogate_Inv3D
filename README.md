# EM Surrogate Inv3D
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Julia](https://img.shields.io/badge/Julia-1.12-purple)

An open-source framework for **3D stochastic electromagnetic (EM) inversion** leveraging a deep learning surrogate model to accelerate 3D forward modeling.


## 📌 Overview
This repository providing a complete pipeline from prior validation to accelerated MCMC inversion. It consists of three core modules:

1. **Prior Falsification**: Validates whether the geostatistical prior matches the field data to be solved (Python/Jupyter).
2. **Surrogate Model Training**: Code and architectures for training the 3D EM forward surrogate model (Python/Jupyter).
3. **Stochastic Inversion with McMC**:
   - Core functions for **Probabilistic Perturbation Method (PPM)** and McMC sampling with delayed rejection (Julia).
   - Main 3D geophysics inversion execution scripts (Julia).
   - Post-processing, visualization, and result analysis scripts (Julia).


## Authors
- **Zhuo Liu** — zliu93@stanford.edu, liuzhuolz@outlook.com  
- **Jonas Kloeckner** — jkloeckn@stanford.edu  


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
   │
   ├── Surrogate model training/            <-- Module 2: NN Training scripts
   │
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
   │  
   ├── LICENSE
   └── README.md
   ```


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
  Create a folder named `Generated prior` inside the `Prior falsification/` directory, and move the four downloaded files into it (as shown in the repository structure above).  


## 🚀 Quick Start

### 1. Requirements & Prerequisites
* For Python Jupyter Notebook:
  **Python (v3.11)**:
   - Ensure you have `PyTorch` installed, so that the surrogate model can load the weights from `.pth` file.
   - For falsification and training code, **_there is no need to run Python code unless if you want to play with them._** You can jump to Julia code for inversion.
* For MCMC stochastic inversion code in Julia:
  **Julia (v1.12.6)**:
   - * **VS Code Configuration**:
       1. Install Julia extension
       2. Find your `JULIA_PATH`, run `which julia` (or `which Julia`) in your terminal. For example:  
          ```bash
          which julia
          # Output example: /Users/YOUR_NAME/.juliaup/bin/julia
          ```
       3. Config path for VS Code: open **Settings**, search for `Julia: Executable Path`, and paste your `JULIA_PATH`.  
      -> e.g., in VS Code:
            Seetings>Julia: Executable Path>/JULIA_PATH.
            ![VS Code Settings Screenshot](https://github.com/user-attachments/assets/ecb7da41-8944-4b3c-b710-5ad01ffb3753)  
       4. Make sure both Julia Path and LanguageServer are correctly directed in user's setting JSON file (Preference:Open User Settings (JSON) in VSCode). Normally, this should be done automatically once you complete previous step, but you can never be too careful... There should be two line as such:
       ```
       ...
       "julia.languageServerExecutablePath": "/Users/YOUR_NAME/.juliaup/bin/julia",  
       "julia.executablePath": "/Users/YOUR_NAME/.juliaup/bin/julia",
       ...
       ```

### 2. Running the Inversion (Recommended Workflow)
This repository consists of detailed research scripts rather than a packaged command-line app. **It is designed to be run entirely within an IDE (Visual Studio Code is highly recommended)** so you can easily adjust parameters, run code blocks interactively, and inspect the inversion results.

Note: Inversion results and chains will be automatically saved into the Outputs/ folder or serialized as `.jld2` files.


## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
         
