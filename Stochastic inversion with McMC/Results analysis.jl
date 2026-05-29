# Activate environment and solve dependencies
using Pkg
Pkg.precompile();    # Precompile packages for faster loading
current_dir = (@__DIR__);    # Set up current directory
Pkg.activate((current_dir));    # Activate the current folder as the environment
Pkg.instantiate();    # Install dependencies from Project.toml

# --- Imports packages used in this script---
using Statistics
using CSV
using DataFrames
using LinearAlgebra
using JLD2
using Plots

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
) |> Matrix;
EMdata_true = Float32.(vec(EMdata_true'));

# Get key dimensions of the data
N_obs = size(Rx_loc)[1];
N_time = length(EMdata_true) / N_obs |> Int;

# # Load true conductivity model, if you running synthetic test
# sigma_true = CSV.read(joinpath(current_dir, "Prior falsification/Test data/sigma_true.csv"), DataFrame; header=false) |> Matrix;

# # Load true Gaussian Process hyper-parameters
# GP_hyper_true = CSV.read(joinpath(current_dir, "Test data/GP_hyper_true.csv"), DataFrame; header=false) |> Matrix;
# GP_hyper_true = vec(GP_hyper_true');

# Load handy functions
include("Plotting utils.jl");

# Define the grid for reconstruction
nx, ny, nz = 45, 20, 45;
dx, dy, dz = 4000/nx, 2000/ny, 4000/nz; # in meters
grid = CartesianGrid((0, 0, 0), (4000, 2000, 4000), dims=(nx, ny, nz));

# Examine truth conductivity model if available
if @isdefined(sigma_true)
    sigma_true = reshape(sigma_true, (nx, ny, nz));
    plot_sigma(
        sigma_true;
        title_inverted="True Conductivity Model",
        slice_x=0 , slice_y=16 , slice_z=45,
        x_length=4000, y_length=2000, z_length=4000, origin=(-7224.47916667, -4717.1875    , -5758.33333333), 
        view_azimuth=-120, view_elevation=-5,
        scale="ln",
        colorrange=(-3, 1)
    );
end;

# Plot true data
plot_signal_comparison(
    EMdata_true, 
    [];
    signal_error_std=nothing,
    compared_label="True Data",
    N_time=N_time,
    N_obs=N_obs,
    Rx_loc=Rx_loc, 
);

# ============================================
# --- Examine results ---
# ============================================
# Load files of posterior 
opt = load(joinpath(current_dir, "Outputs/mcmc_full_chain.jld2"))["opt"];

# Set names of Gaussian-process parameters
param_names = [
    "Major Range", "Ratio M-I", "Ratio I-M", 
    "Yaw", "Pitch", "Roll", 
    "Sigma Mean", "Sigma Std"
];

# Check temperature, acceptance rate and loss history to see if MCMC has good convergence
 plot_mcmc_analysis(
    opt;
    target_acc_rate = 0.20
);

# Check how r-parameter in PPM evolves to check annealing
plot_r_evolution(
    opt.prop_logs
);

# Check autocorrelation of MCMC chain to see if it has good mixing and convergence
plot_acf(
    opt,
    param_names; 
    burn_in_ratio=0.2,
    anneal_ratio=0.3,
    max_lag=200
);

# Check ESS of each GP-process params
print_ess_report(
    opt,
    param_names;
    burn_in_ratio=0.2,
    anneal_ratio=0.3,
    max_lag=200
);

# Extract accepted samples from MCMC chain for posterior analysis and visualization
N_exam = 50;
acc_samples = extract_accepted_samples(
    opt; 
    N_needed=N_exam, 
    thinning=5,
    burn_in_ratio=0.2,
    anneal_ratio=0.3,
    sort_by_loss=true
);

# Get predicted data and conductivity models from accepted samples
dpred = [s.signal for s in acc_samples];
dpred_mean = vec(mean(hcat(dpred...), dims=2));

# Plot mean data
plot_signal_comparison(
    EMdata_true, 
    [dpred_mean];
    signal_error_std=nothing,
    compared_label="Mean of Predicted Data",
    N_time=N_time,
    N_obs=N_obs,
    Rx_loc=Rx_loc,
);

# Reconstruct conductivity fields from accepted samples
w_pred = [s.w for s in acc_samples];
GP_param_spatial_pred = [s.GP_param_spatial for s in acc_samples];
GP_param_sigma_mean_pred = [s.GP_param_sigma_mean for s in acc_samples];
GP_param_sigma_std_pred = [s.GP_param_sigma_std for s in acc_samples];

fields_pred = reconstruction_fields(
    w_pred,
    GP_param_spatial_pred,
    GP_param_sigma_mean_pred,
    GP_param_sigma_std_pred,
    grid
);

# Compute RMSE for each accepted sample
misfit_list = zeros(N_exam);
for i in 1:N_exam
    misfit_list[i] = sqrt(mean((dpred[i] .- EMdata_true).^2))
end;

# Plot histogram of misfit distribution
histogram(
    misfit_list, bins=30, 
    title="Posterior Misfit Distribution", 
    xlabel="RMSE", ylabel="Count",
    color=:dodgerblue, alpha=0.5, 
    strokewidth=1, strokecolor=:black
)

# Extract posterior data of the first 30% best-fitting samples (with lowest misfit)
threshold = 30.0;    # User-defined based on the histogram
filtered_idx = findall(x -> x <= threshold, misfit_list);
println("Number of filtered models: ", length(filtered_idx), " / ", length(misfit_list));

# Extract sample after filtering
acc_samples_filtered = acc_samples[filtered_idx];
dpred_filtered = [s.signal for s in acc_samples_filtered];
dpred_mean_filtered = vec(mean(hcat(dpred_filtered...), dims=2));

# Plot mean data
plot_signal_comparison(
    EMdata_true, 
    [dpred_mean_filtered];
    signal_error_std=nothing,
    compared_label="Mean of Predicted Data",
    N_time=N_time,
    N_obs=N_obs,
    Rx_loc=Rx_loc,
);

# Check data fit of the filtered samples
plot_accepted_fits(
    acc_samples_filtered,
    EMdata_true;
    signal_error_std=nothing,
    N_time=12,
    Rx_loc=Rx_loc,
);

# Calculate mean field based on the selection
field_filtered = fields_pred[filtered_idx];
fields_filtered_stack = cat(field_filtered..., dims=4);
field_filtered_mean = dropdims(mean(fields_filtered_stack, dims=4), dims=4);
field_filtered_std = dropdims(std(fields_filtered_stack, dims=4), dims=4);

# Plot mean conductivity model of the filtered samples
plot_sigma(
    field_filtered_mean, 
    title_inverted="Mean of Posterior Conductivity Model",
    slice_x=0 , slice_y=16 , slice_z=45,
    view_azimuth=-120, view_elevation=-5,
    scale="ln",
    colorrange=(-5, 2)
);

# Plot Std of conductivity
plot_sigma(
    field_filtered_std, 
    title_inverted="Standard Deviation of Posterior Conductivity Model",
    slice_x=0 , slice_y=16 , slice_z=45,
    view_azimuth=-120, view_elevation=-5,
    scale="ln"
);

# Plot a few posterior realization
plot_index = 1;
plot_sigma(
    fields_filtered_stack[:, :, :, plot_index];
    title_inverted="Posterior Conductivity Model No. $plot_index",
    slice_x=0 , slice_y=16 , slice_z=45,
    view_azimuth=-120, view_elevation=-5, 
    scale="ln",
    colorrange=(-5, 2)
);

# Find out best-fitting model compared to true conductivity model based on L2 distance, if true model is available
if @isdefined(sigma_true)
    N_filtered = length(filtered_idx);
    L2_dist = zeros(N_filtered);
    for i in 1:N_filtered
        # Calculate the L2 distance between two 3D conductivity models
        L2_dist[i] = norm(fields_filtered_stack[:, :, :, i] .- sigma_true)
    end;

    best_l2_idx = argmin(L2_dist);
    println("Best Model Index: ", best_l2_idx)
    println("Minimum L2 Distance: ", L2_dist[best_l2_idx])

    # Plot the best-fitting model based on L2 distance
    plot_sigma(
        fields_filtered_stack[:, :, :, best_l2_idx], 
        title_inverted="L2-distance of Best of Posterior Conductivity Model",
        slice_x=0 , slice_y=16 , slice_z=45,
        view_azimuth=-120, view_elevation=-5,
        scale="ln",
        colorrange=(-5, 2)
    );

    # Plot corresponding predicted data of the best-fitting model
    plot_signal_comparison(
        EMdata_true, 
        [acc_samples_filtered[best_l2_idx].signal];
        signal_error_std=nothing,
        compared_label="L2-optimal Predicted Data",
        N_time=N_time,
        N_obs=N_obs,
        Rx_loc=Rx_loc,
    );
end

# Pick 3 clostest-to-mean and furthest-from-mean models based on L2 distance
mean_vec = vec(field_filtered_mean);
dist_to_mean = zeros(N_filtered);
for i in 1:N_filtered
    sample_vec = vec(fields_filtered_stack[:, :, :, i])
    dist_to_mean[i] = norm(sample_vec .- mean_vec)
end

sorted_idx_by_dist = sortperm(dist_to_mean);
top3_closest = sorted_idx_by_dist[1:3];
bottom3_farthest = sorted_idx_by_dist[end-2:end];
println("Top 3 Closest indices in Filtered: ", top3_closest);
println("Top 3 Furthest indices in Filtered: ", bottom3_farthest);

# Plot the 3 best-fitting models based on L2 distance
top_prefixes = ["Closest to Mean", "Second Closest to Mean", "Third Closest to Mean"];
for i in 1:3
    plot_sigma(
        fields_filtered_stack[:, :, :, top3_closest[i]], 
        title_inverted = "$(top_prefixes[i]) of Posterior Conductivity Model",
        slice_x=0, slice_y=16, slice_z=45,
        view_azimuth=-120, view_elevation=-5,
        scale="ln",
        colorrange=(-5, 2)
    )

    # Plot corresponding predicted data of the best-fitting model
    plot_signal_comparison(
        EMdata_true, 
        [acc_samples_filtered[top3_closest[i]].signal];
        signal_error_std=nothing,
        compared_label="$(top_prefixes[i]) of Predicted Data",
        N_time=N_time,
        N_obs=N_obs,
        Rx_loc=Rx_loc,
    );
end;

# Plot the 3 worst-fitting models based on L2 distance
bottom_prefixes = ["Furthest from Mean", "Second Furthest from Mean", "Third Furthest from Mean"];
for i in 1:3
    plot_sigma(
        fields_filtered_stack[:, :, :, bottom3_farthest[i]], 
        title_inverted = "$(bottom_prefixes[i]) of Posterior Conductivity Model",
        slice_x=0, slice_y=16, slice_z=45,
        view_azimuth=-120, view_elevation=-5,
        scale="ln",
        colorrange=(-5, 2)
    )

    # Plot corresponding predicted data of the best-fitting model
    plot_signal_comparison(
        EMdata_true, 
        [acc_samples_filtered[bottom3_farthest[i]].signal];
        signal_error_std=nothing,
        compared_label="$(bottom_prefixes[i]) of Predicted Data",
        N_time=N_time,
        N_obs=N_obs,
        Rx_loc=Rx_loc,
    );
end;

# ==================================================================
# ---For synthetic test with simple geometry only ---
# --- Check topologic relationship to mean model based on area near anomaly, not whole space ---
# ==================================================================
# Get the indices of the area near the anomaly based on the true hyper-parameters
if @isdefined(GP_hyper_true)
    roi_x, roi_y, roi_z = get_roi_indices(GP_hyper_true);
    println("Detected ROI: X:$roi_x, Y:$roi_y, Z:$roi_z");

    # Calculate the distanct to mean model only based on the area near the anomaly
    mean_roi_vec = vec(field_filtered_mean[roi_x, roi_y, roi_z]);
    dist_roi = zeros(N_filtered);

    for i in 1:N_filtered
        sample_roi_vec = vec(fields_filtered_stack[roi_x, roi_y, roi_z, i])
        dist_roi[i] = norm(sample_roi_vec .- mean_roi_vec)
    end;

    # Find out the top 3 closest and furthest models based on the distance to mean model in the area near the anomaly
    sorted_roi_idx = sortperm(dist_roi);
    top3_closest_roi = sorted_roi_idx[1:3];   
    bottom3_farthest_roi = sorted_roi_idx[end-2:end];

    # Plot the 3 best-fitting models based on L2 distance
    top_prefixes = ["Closest to Mean", "Second Closest to Mean", "Third Closest to Mean"];
    for i in 1:3
        plot_sigma(
            fields_filtered_stack[:, :, :, top3_closest_roi[i]], 
            title_inverted = "$(top_prefixes[i]) of Posterior Conductivity Model Based on ROI",
            slice_x=0, slice_y=16, slice_z=45,
            view_azimuth=-120, view_elevation=-5,
            scale="ln",
            colorrange=(-5, 2)
        )

        # Plot corresponding predicted data of the best-fitting model
        plot_signal_comparison(
            EMdata_true, 
            [acc_samples_filtered[top3_closest_roi[i]].signal];
            signal_error_std=nothing,
            compared_label="$(top_prefixes[i]) of Predicted Data",
            N_time=N_time,
            N_obs=N_obs,
            Rx_loc=Rx_loc,
        );
    end;

    # Plot the 3 worst-fitting models based on L2 distance
    bottom_prefixes = ["Furthest from Mean", "Second Furthest from Mean", "Third Furthest from Mean"];
    for i in 1:3
        plot_sigma(
            fields_filtered_stack[:, :, :, bottom3_farthest_roi[i]], 
            title_inverted = "$(bottom_prefixes[i]) of Posterior Conductivity Model Based on ROI",
            slice_x=0, slice_y=16, slice_z=45,
            view_azimuth=-120, view_elevation=-5,
            scale="ln",
            colorrange=(-5, 2)
        )

        # Plot corresponding predicted data of the best-fitting model
        plot_signal_comparison(
            EMdata_true, 
            [acc_samples_filtered[bottom3_farthest_roi[i]].signal];
            signal_error_std=nothing,
            compared_label="$(bottom_prefixes[i]) of Predicted Data",
            N_time=N_time,
            N_obs=N_obs,
            Rx_loc=Rx_loc,
        );
    end;
end;
