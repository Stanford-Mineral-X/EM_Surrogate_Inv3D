# Activate environment and solve dependencies
using Pkg
Pkg.precompile();    # Precompile packages for faster loading
current_dir = (@__DIR__);    # Set up current directory
Pkg.activate((current_dir));    # Activate the current folder as the environment
Pkg.instantiate();    # Install dependencies from Project.toml

# --- Imports packages used in this script---
using GeoStats, GeoStatsBase, GeoStatsProcesses
using Meshes
using CSV, DataFrames
using Random, Distributions, Statistics
using JLD2: jldsave

# ============================================
# --- Load files ---
# ============================================
# Load Rx locations
Rx_loc = CSV.read(
    joinpath(current_dir, "../Prior falsification/Test data/Rx_loc.csv"), 
    DataFrame; 
    header=false
) |> Matrix;

# Load true EM data;
EMdata_true = CSV.read(
    joinpath(current_dir, "../Prior falsification/Test data/EMdata.csv"), 
    DataFrame; 
    header=false
)|> Matrix;
EMdata_true = Float32.(vec(EMdata_true'));

# Load noise file
EMnoise = CSV.read(
    joinpath(current_dir, "../Prior falsification/Test data/EMnoise.csv"), 
    DataFrame; 
    header=false
)|> Matrix;
EMnoise = Float32.(vec(EMnoise));

# Get key dimensions of the data
N_obs = size(Rx_loc)[1];
N_time = length(EMdata_true) / N_obs |> Int;

# ============================================
# --- Main test ---
# ============================================
# Load all top-level functions from ppm module, from same folder
include("PPM MCMC utils.jl");

# Import surrogate model from EM forward modeling
Surrogate_dir = joinpath(current_dir, "em_surrogate_pack");
pushfirst!(pyimport("sys")["path"], Surrogate_dir);    # Add directory containing mymodule.py
Surrogate_weights_dir = joinpath(Surrogate_dir, "EM 3D Surrogate_Unet.pth");
surro_pred = pyimport("EM_Unet_surrogate_prediction");

# Define modeling space
nx, ny, nz = 45, 20, 45;
dx, dy, dz = 4000/nx, 2000/ny, 4000/nz; # in meters
grid = CartesianGrid((0, 0, 0), (4000, 2000, 4000), dims=(nx, ny, nz));

# Set number of total realizations
Max_iter = 100;

# Inversion using PPM with EM surrogate model
param_ranges = (
    (1600.0, 2400.0), 
    (0.5, 1.0), 
    (0.3, 1.0), 
    (5.0, 175.0), 
    (-85.0, 90.0), 
    (-85.0, 85.0), 
    (-2.5, 0.5),
    (1.2, 3.5)
);

# Run inversion for multiple realizations
seed = rand(1:1e6);
opt = optimize_ppm_outer_vranges(
    EMdata_true,
    EMnoise,
    param_ranges,
    grid,
    Rx_loc,
    N_time,
    N_obs,
    surro_pred,
    Surrogate_weights_dir,
    Max_iter,
    MersenneTwister(Int(seed));
    print_log=false, 
    print_time=true
);

# Save dictionary
jldsave(joinpath(current_dir, "Outputs/mcmc_full_chain.jld2"); opt)    # Save the entire McMC chain to the specified directory


