# --- PACKAGE IMPORTS ---
using CairoMakie
const CMke = CairoMakie
using KernelDensity
using CSV, DataFrames
using Statistics, Distributions
using GeoStats, GeoStatsBase
using FFTW
using StatsBase
using Printf


"""
A struct to record the proposal details in each iteration.
"""
struct ProposalLog
    iter::Int
    loss::Float64
    ΔL::Float64
    α::Float64        # acceptance probability
    u::Float64        # random number for acceptance decision
    accepted::Bool    # whether the proposal for spatial parameters is accepted
    r_vals::NTuple{4, Float64} # the r values used for PPM in this proposal
    T::Float64        # the temperature used for acceptance probability in this proposal
    stage_indicator::Int    # 1 for stage 1 acceptance, 2 for stage 2 acceptance
end


"""
A struct to hold the state of MCMC optimization.
"""
struct MCMCState
    loss::Float64
    channel_MSE::Vector{Float64}
    signal::Vector{Float64}
    w::Array{Float64,3}
    GP_param_spatial::Vector{Float64}
    GP_param_sigma_mean::Vector{Float64}
    GP_param_sigma_std::Vector{Float64}
    ΔL::Float64
    α::Float64
    u::Float64
    accepted::Bool
end


function randsingle_controlled(
    rng::AbstractRNG, 
    process::GaussianProcess, 
    method::FFTSIM, 
    domain, 
    data, 
    preproc, 
    w::AbstractArray
    )
    """
    A modified GeoStats `randsingle` function that accepts a pre-generated random grid `w`.
    """
    
    (; var, F, z̄, dinds) = preproc; 
    f = process.func; 
    μ = process.mean; 
    sdom = domain; 
    grid = parent(sdom); 
    inds = parentindices(sdom)
    P = F .* exp.(im .* angle.(fft(w))); 
    Z = real(ifft(P)) * unit(μ); 
    σ² = Statistics.var(Z, mean=zero(eltype(Z)))
    Z .= √(sill(f) / σ²) .* Z .+ μ; zᵤ = Z[inds]
    z = isnothing(data) ? zᵤ : begin; ktab = (; var => view(zᵤ, dinds)); 
    kdom = view(sdom, dinds); kdata = georef(ktab, kdom)
    (; minneighbors, maxneighbors, neighborhood, distance) = method; 
    krigᵤ = GeoStatsModels.fitpredict(Kriging(f, μ), kdata, sdom; minneighbors, maxneighbors, neighborhood, distance)
    z̄ᵤ = krigᵤ[:, var]; z̄ .+ (zᵤ .- z̄ᵤ); end; return (; var => z)
end


function process_from_GP_params(
    major::Real, interm::Real, minor::Real,
    yaw::Real, pitch::Real, roll::Real,
    sigma_mean::Real, sigma_std::Real
    )
    """
    Function to generate Gaussian random field with parameters
    """

    γ = SphericalVariogram(
        ranges=(major, interm, minor), 
        sill=sigma_std^2, 
        rotation=RotZYX(yaw, pitch, roll)
    )
    return GaussianProcess(γ, sigma_mean)
end


function reconstruction_fields(
    w_list,
    GP_param_spatial_list,
    GP_param_sigma_mean_list,
    GP_param_sigma_std_list,
    grid;
    rng=MersenneTwister(2025)
    )
    """
    Function to reconstruct 3D fields based on posterior
    """

    Nx, Ny, Nz = size(grid)

    realization_pred = []
    for i in 1:length(w_list)
        # Update Gaussian process (variogram) with new hyper-parameters for optimal realization
        process = process_from_GP_params(
            GP_param_spatial_list[i][1],    # major
            GP_param_spatial_list[i][1] * GP_param_spatial_list[i][2],    # major * ratio_major_interm
            GP_param_spatial_list[i][1] * GP_param_spatial_list[i][2] * GP_param_spatial_list[i][3],    # major * ratio_major_interm * ratio_interm_minor
            GP_param_spatial_list[i][4],    # yaw
            GP_param_spatial_list[i][5],    # pitch
            GP_param_spatial_list[i][6],    # roll
            GP_param_sigma_mean_list[i][1],    # sigma_mean
            GP_param_sigma_std_list[i][1]     # sigma_std
        );

        # Create initial preprocessor with initial Gaussian process for optimal realization
        preproc = GeoStatsProcesses.preprocess(rng, process, FFTSIM(), nothing, grid, nothing)

        # Reconstruct inverted optimal 3D field
        field3d_named = randsingle_controlled(
            rng,
            process,
            FFTSIM(),
            grid,           
            nothing,            
            preproc, 
            w_list[i], 
        )
        var_name = first(keys(field3d_named))
        z_vec = field3d_named[var_name]
        push!(realization_pred, reshape(z_vec, Nx, Ny, Nz));
    end

    return realization_pred
end


function get_roi_indices(
    params; nx=45, ny=20, nz=45, padding=0,
    # Mesh world coordinates from your mesh_info
    x_min=-656.77,  x_max=3343.23,
    y_min=-1400.00, y_max=600.00,
    z_min=-3733.33, z_max=266.67,
    # Curved ellipsoid flag
    is_curved=false
    )
    """
    A function to extract region of interest among spatial model.
    Used for synthetic test with conductivyt of a simple geometry
    like cube or ellipsoid.(Not for Gaussian field)
    """

    # Extract parameters 
    a        = params[1]
    center_x = params[7]
    center_y = params[8]
    center_z = params[9]

    # For curved ellipsoid, amplitude expands the Z bounding box
    amplitude = is_curved ? params[10] : 0.0   # params[10] = amplitude if curved

    # Build cell centers in world coordinates
    dh_x = (x_max - x_min) / nx
    dh_y = (y_max - y_min) / ny
    dh_z = (z_max - z_min) / nz

    cc_x = collect(x_min + dh_x/2 : dh_x : x_max - dh_x/2)  # length nx
    cc_y = collect(y_min + dh_y/2 : dh_y : y_max - dh_y/2)  # length ny
    cc_z = collect(z_min + dh_z/2 : dh_z : z_max - dh_z/2)  # length nz

    #  Find index ranges
    # X and Y: use major axis a as bounding radius (same as before)
    x_idxs = findall(x -> (center_x - a) <= x <= (center_x + a), cc_x)
    y_idxs = findall(y -> (center_y - a) <= y <= (center_y + a), cc_y)

    # Z: expand by amplitude to account for sinusoidal deformation
    z_lo = center_z - a - amplitude   # <-- key change for curved case
    z_hi = center_z + a + amplitude
    z_idxs = findall(z -> z_lo <= z <= z_hi, cc_z)

    # Safe range with padding
    function get_range(idxs, max_dim)
        if isempty(idxs)
            mid = div(max_dim, 2)
            return mid:mid
        end
        low  = clamp(minimum(idxs) - padding, 1, max_dim)
        high = clamp(maximum(idxs) + padding, 1, max_dim)
        return low:high
    end

    roi_x = get_range(x_idxs, nx)
    roi_y = get_range(y_idxs, ny)
    roi_z = get_range(z_idxs, nz)

    return roi_x, roi_y, roi_z
end


# --- VISUALIZATION FUNCTIONS ---
function plot_mcmc_analysis(
    opt;
    target_acc_rate=0.23,
    window = 100
    )
    """
    Function to plot MCMC diagnostics: loss curve, acceptance rate, temperature schedule
    """
    
    logs = opt.prop_logs
    iters = [l.iter for l in logs]
    losses = [l.loss for l in logs]
    temps = [l.T for l in logs]
    acc_status = [l.accepted for l in logs]
    stages = [l.stage_indicator for l in logs]
    acc_stage1 = [l.accepted && l.stage_indicator == 1 for l in logs]
    acc_stage2 = [l.accepted && l.stage_indicator == 2 for l in logs]
    moving_ar = [mean(acc_status[max(1, i-window+1):i]) for i in 1:length(acc_status)]
    moving_ar_s1 = [mean(acc_stage1[max(1, i-window+1):i]) for i in 1:length(acc_stage1)]
    moving_ar_s2 = [mean(acc_stage2[max(1, i-window+1):i]) for i in 1:length(acc_stage2)]

    fig = CMke.Figure(size = (1200, 950), fontsize = 16)

    # --- Top Panel ---
    ax_loss = CMke.Axis(fig[1, 1], 
        title = "MCMC Convergence (Delayed Rejection)",
        xlabel = "Iteration", ylabel = "Weighted MSE Loss",
        yscale = log10)
    CMke.lines!(ax_loss, iters, losses, color = (:grey, 0.4), linewidth = 1)
    CMke.scatter!(ax_loss, iters[acc_stage1], losses[acc_stage1], 
             color = :forestgreen, markersize = 7, label = "Stage 1 Accepted", marker=:circle)
    CMke.scatter!(ax_loss, iters[acc_stage2], losses[acc_stage2], 
             color = :darkorchid, markersize = 5, label = "Stage 2 Accepted", marker=:diamond)

    # --- Middle Panel ---
    ax_t = CMke.Axis(fig[2, 1], 
        title = "Adaptive Temperature (T) & Step Size Control",
        xlabel = "Iteration", ylabel = "T",
        yscale = log10)
    CMke.lines!(ax_t, iters, temps, color = :crimson, linewidth = 2.5)

    # --- Bottom Panel ---
    ax_ar = CMke.Axis(fig[3, 1], 
        title = "Moving Average Acceptance Rate",
        xlabel = "Iteration", ylabel = "Rate")
    CMke.lines!(ax_ar, iters, moving_ar, color = :black, linewidth = 2.5, label = "Overall AR")
    CMke.lines!(ax_ar, iters, moving_ar_s1, color = :forestgreen, linewidth = 1.5, label = "Stage 1 AR", linestyle=:dot)
    CMke.lines!(ax_ar, iters, moving_ar_s2, color = :darkorchid, linewidth = 1.5, label = "Stage 2 AR", linestyle=:dot)
    CMke.hlines!(ax_ar, [target_acc_rate], color = :red, linestyle = :dash, linewidth = 1.5, label = "Target")

    CMke.Legend(fig[1, 2], ax_loss, framevisible = true)
    CMke.Legend(fig[3, 2], ax_ar, framevisible = true)
    CMke.colsize!(fig.layout, 2, CMke.Fixed(180))
    CMke.colgap!(fig.layout, 20)

    display(fig)
    return fig
end


function plot_acf(
    opt_result,
    param_names; 
    burn_in_ratio=0.2,
    anneal_ratio=0.3,
    max_lag=200
    )
    """
    Function to calculate and plot autocorrelation for each parameter
    """
    
    # Sanity check
    full_chain = opt_result.chain
    N_total = length(full_chain)    # Get length of the MCMC chain

    # Hard constraint 1: Total ratio cannot exceed 100%
    @assert burn_in_ratio + anneal_ratio <= 1.0 "The sum of burn_in_ratio ($burn_in_ratio) and anneal_ratio ($anneal_ratio) cannot be greater than 1.0!"

    # Soft suggestion: Warn if the optimization/warmup phase takes up too much of the chain
    if burn_in_ratio + anneal_ratio > 0.5
        @warn "The combined burn-in and annealing ratio is high ($(burn_in_ratio + anneal_ratio) > 0.5). More than half of the samples will be discarded as transient phases!"
    end

    # Discard burn-in and annealing periods
    burn_in_length = max(1, Int(floor(N_total * burn_in_ratio)))    
    anneal_length = max(1, Int(floor(N_total * anneal_ratio)))
    start_idx = Int(burn_in_length + anneal_length) + 1   # Find the index of the first valid sample
    chain = full_chain[start_idx:end]
    N_iter = length(chain)
    
    if N_iter <= max_lag
        @warn "Chain length ($N_iter) is shorter than max_lag ($max_lag). Adjusting max_lag."
        max_lag = max(1, N_iter - 1)
    end
    
    fig = CMke.Figure(size=(1200, 800))
    for p in 1:8
        # Calculate index of subplot
        row_idx = (p - 1) ÷ 4 + 1
        col_idx = (p - 1) % 4 + 1
        
        # Extract Gaussian-process parameters
        series = map(chain) do state
            if p <= 6
                return state.GP_param_spatial[p]
            elseif p == 7
                return state.GP_param_sigma_mean[1]
            else
                return state.GP_param_sigma_std[1]
            end
        end
        
        # Calculate ACF, with sanity check
        if all(x -> x ≈ series[1], series) || std(series) < 1e-9
            vals = zeros(max_lag + 1)
            vals[1] = 1.0  
        else
            vals = autocor(series, 0:max_lag)
        end
        
        # Check and replace NaN if any
        replace!(vals, NaN => 0.0)

        # Make figures
        ax = CMke.Axis(fig[row_idx, col_idx], 
            title="$(param_names[p])", 
            xlabel="Lag", 
            ylabel="Correlation",
            titlesize = 14
        )
        CMke.ylims!(ax, -0.2, 1.1)
        CMke.stem!(ax, 0:max_lag, vals, markersize=4, color=:dodgerblue)
        CMke.hlines!(ax, [0.0], color=:black, linewidth=0.8)
        conf_limit = 1.96 / sqrt(N_iter)
        CMke.hlines!(ax, [conf_limit, -conf_limit], color=:red, linestyle=:dash, linewidth=1)
    end
    
    CMke.Label(fig[0, :], "Post-Burn-in Parameter Autocorrelation (Delayed Rejection MCMC)", 
             fontsize=20, font=:bold)
    
    display(fig)
    return fig
end


function print_ess_report(
    opt_result,
    param_names;
    burn_in_ratio=0.2,
    anneal_ratio=0.3,
    max_lag::Int=200
    )
    """
    Function to calculate ESS based on ACF
    """

    full_chain = opt_result.chain
    n_full = length(full_chain)    # Get length of the MCMC chain

    # Sanity check
    # Hard constraint 1: Total ratio cannot exceed 100%
    @assert burn_in_ratio + anneal_ratio <= 1.0 "The sum of burn_in_ratio ($burn_in_ratio) and anneal_ratio ($anneal_ratio) cannot be greater than 1.0!"

    # Soft suggestion: Warn if the optimization/warmup phase takes up too much of the chain
    if burn_in_ratio + anneal_ratio > 0.5
        @warn "The combined burn-in and annealing ratio is high ($(burn_in_ratio + anneal_ratio) > 0.5). More than half of the samples will be discarded as transient phases!"
    end
    
    # Discard burn-in period
    burn_in_length = max(1, Int(floor(n_full * burn_in_ratio)))
    anneal_length = max(1, Int(floor(n_full * anneal_ratio)))
    start_idx = Int(burn_in_length + anneal_length) + 1
    chain = full_chain[start_idx:end]
    n_total = length(chain)

    # Print table header
    println("\n" * "="^60)
    println("      MCMC Efficiency Report (ESS Analysis - Post Burn-in)")
    @printf("      Chain Length: %d | Burn-in Ratio: %.1f\n", n_total, burn_in_ratio)
    println("-"^60)
    @printf("%-15s | %-10s | %-12s | %-10s\n", "Parameter", "ESS", "Efficiency", "Total Samples")
    println("-"^60)

    # Extract Gaussian-process parameter
    for i in 1:8
        if i <= 6
            values = map(s -> s.GP_param_spatial[i], chain)
        elseif i == 7
            values = map(s -> s.GP_param_sigma_mean[1], chain)
        else
            values = map(s -> s.GP_param_sigma_std[1], chain)
        end

        # Sanity check
        if all(x -> x ≈ values[1], values)
            @printf("%-15s | %-10.1f | %-10.2f%% | %-10d (Stuck?)\n", 
                param_names[i], 1.0, (1.0/n_total)*100, n_total)
            continue
        end

        # Calcualte ACF
        actual_max_lag = min(max_lag, n_total - 1)
        lags = 0:actual_max_lag
        rho = autocor(values, lags)

        # 3. Improved Effective Sample Size (ESS) Calculation (Geyer's Initial Positive Sequence)
        sum_rho = 0.0
        for k in 2:length(rho)
            # Stopping criterion: Terminate summation when autocorrelation is negligible 
            # or starts fluctuating, minimizing noise interference.
            if rho[k] < 0.01 
                break
            end
            # Ensure positive contribution to avoid overly optimistic ESS estimates.
            sum_rho += max(0.0, rho[k])
            
            # Additional robustness check: Persistent high autocorrelation indicates poor mixing.
            if k == actual_max_lag && rho[k] > 0.5
            end
        end

        # 4. Calculate ESS and efficiency
        ess = n_total / (1.0 + 2.0 * sum_rho)
        eff = (ess / n_total) * 100

        @printf("%-15s | %-10.1f | %-10.2f%% | %-10d\n", 
            param_names[i], ess, eff, n_total)
    end
    println("="^60 * "\n")
end


function plot_r_evolution(
    prop_logs::Vector{ProposalLog}
    )
    """
    A function to print trace plots of r values across iterations
    """

    # Extract recorded r-values used in PPM
    iters = [log.iter for log in prop_logs]
    stages = [log.stage_indicator for log in prop_logs]
    accepts = [log.accepted for log in prop_logs]
    
    r_spatial = [log.r_vals[1] for log in prop_logs]
    r_mean    = [log.r_vals[2] for log in prop_logs]
    r_std     = [log.r_vals[3] for log in prop_logs]
    r_field   = [log.r_vals[4] for log in prop_logs]
    
    acc_s1_idx = findall(i -> stages[i] == 1 && accepts[i], 1:length(accepts))
    acc_s2_idx = findall(i -> stages[i] == 2 && accepts[i], 1:length(accepts))
    rej_idx = findall(.!accepts)

    # Setup canvas
    fig = CMke.Figure(size=(1400, 800))
    titles = ["r_spatial", "r_sigma_mean", "r_sigma_std", "r_field"]
    r_data = [r_spatial, r_mean, r_std, r_field]
    
    local last_ax    # Used to store the last coordinate axis for use by Legend

    for i in 1:4
        row = (i - 1) ÷ 2 + 1
        col = (i - 1) % 2 + 1
        
        ax = CMke.Axis(fig[row, col], 
            title=titles[i], 
            xlabel="Iteration", 
            ylabel="r value",
            limits=(nothing, (0, 1.1))
        )
        last_ax = ax
        
        CMke.scatter!(ax, iters[rej_idx], r_data[i][rej_idx], 
                     color=(:grey, 0.2), markersize=3, label="Rejected")
        
        if !isempty(acc_s1_idx)
            CMke.scatter!(ax, iters[acc_s1_idx], r_data[i][acc_s1_idx], 
                         color=:forestgreen, markersize=6, label="Stage 1 Acc.")
        end
        
        if !isempty(acc_s2_idx)
            CMke.scatter!(ax, iters[acc_s2_idx], r_data[i][acc_s2_idx], 
                         color=:darkorchid, markersize=6, label="Stage 2 Acc.")
        end
    end

    # Setup legend
    CMke.Legend(fig[1:2, 3], last_ax, "Status", framevisible = true, halign = :left)
    
    # Config gap
    CMke.colsize!(fig.layout, 3, CMke.Fixed(150))
    CMke.colgap!(fig.layout, 20)

    CMke.Label(fig[0, 1:2], "Evolution of PPM r-values (Delayed Rejection Strategy)", 
             font=:bold, fontsize=20)
    
    display(fig)
    return fig
end


function plot_all_hyper_params_history(
    opt_list,
    param_names,
    param_ranges
    )
    """
    A function to plot the evolution of all hyper-parameters across iterations
    """
    
    opts = opt_list isa Vector ? opt_list : [opt_list]
    N_real = length(opts)
    N_iter = length(opts[1].chain)
    N_params = Int(size(param_names)[1])
    
    fig = CMke.Figure(size=(1600, 700))
    for p in 1:N_params
        row = (p - 1) ÷ 4 + 1   
        col = (p - 1) % 4 + 1
        
        p_min, p_max = param_ranges[p]

        ax = CMke.Axis(
            fig[row, col],
            title = param_names[p],
            xlabel = row == 2 ? "Iterations" : "",
            ylabel = col == 1 ? "Value" : "",
            limits = (nothing, (p_min - 0.05*(p_max-p_min), p_max + 0.05*(p_max-p_min)))
        )

        # Plot trace for each realization in the list, with different colors if multiple
        for i in 1:N_real
            param_series = map(opts[i].chain) do state
                if p <= 6
                    return state.GP_param_spatial[p]
                elseif p == 7
                    return state.GP_param_sigma_mean[1] 
                else
                    return state.GP_param_sigma_std[1] 
                end
            end

            CMke.lines!(
                ax, 1:N_iter, 
                param_series, 
                color = N_real > 1 ? (CMke.ColorSchemes[:tab10][i], 0.6) : :black,
                linewidth = 1.5
            )
        end

        # Plot horizontal lines for parameter bounds based on prior
        CMke.hlines!(ax, [p_min, p_max], color = :red, linestyle = :dash, linewidth = 1.2)
    end

    CMke.Label(fig[0, :], "Hyper-parameter Trace of All Proposals", 
             fontsize=22, font=:bold)

    display(fig)
    return fig
end


function extract_accepted_samples(
    opt; 
    N_needed=50, 
    thinning=1,
    burn_in_ratio=0.2,
    anneal_ratio=0.3,
    loss_threshold=nothing,
    sort_by_loss=true
    )
    """
    Function to extract accepted samples after burn-in and annealing of step length, with optional filtering and sorting
    """

    # Sanity check
    total_iters = length(opt.prop_logs)    # Get length of the MCMC chain

    # Hard constraint 1: Total ratio cannot exceed 100%
    @assert burn_in_ratio + anneal_ratio <= 1.0 "The sum of burn_in_ratio ($burn_in_ratio) and anneal_ratio ($anneal_ratio) cannot be greater than 1.0!"

    # Hard constraint 2: Number of valid samples must be sufficient
    @assert total_iters * (1 - (burn_in_ratio + anneal_ratio)) > N_needed * thinning "Not enough valid samples after burn-in and annealing to meet the N_needed with thinning. Consider reducing burn_in_ratio, anneal_ratio, or thinning."
    
    # Soft suggestion: Warn if the optimization/warmup phase takes up too much of the chain
    if burn_in_ratio + anneal_ratio > 0.5
        @warn "The combined burn-in and annealing ratio is high ($(burn_in_ratio + anneal_ratio) > 0.5). More than half of the samples will be discarded as transient phases!"
    end
    
    # Determine the starting index of valid sample after burn-in and step-length annealing
    burn_in_length = Int(ceil(burn_in_ratio * total_iters))
    anneal_length = Int(ceil(anneal_ratio * total_iters))
    start_idx = Int(burn_in_length + anneal_length) + 1
    
    # Extract axxepted samples after burn-in and annealing
    logs = opt.prop_logs
    accepted_after_burnin = findall(i -> i >= start_idx && logs[i].accepted, 1:total_iters)
    
    if isempty(accepted_after_burnin)
        @warn "After burn-in and annealing, no accepted samples were found. Chain might be stuck."
        return [opt.chain[end]] 
    end

    # Extract corresponding MCMCState samples
    candidate_samples = opt.chain[accepted_after_burnin]
    
    # Filter samples based on loss threshold if provided
    if loss_threshold !== nothing
        candidate_samples = filter(s -> s.loss < loss_threshold, candidate_samples)
    end

    # Sorting or thinning
    if sort_by_loss
        sort!(candidate_samples, by = s -> s.loss)    # Sort by ascending loss for accuracy
    else
        candidate_samples = candidate_samples[1:thinning:end]    # No sorting, just thinning for posterior diversity
    end

    # Limit the number of samples to N_needed
    num_to_get = min(N_needed, length(candidate_samples))
    final_samples = candidate_samples[1:num_to_get]

    @info "Extracted $(length(final_samples)) samples from $(length(accepted_after_burnin)) accepted candidates."
    
    return final_samples
end


function plot_accepted_fits(
    accepted_samples,
    target_signal;
    signal_error_std=nothing,
    N_time=12,
    Rx_loc=nothing
    )
    """
    Function to plot the accepted samples
    """

    N_samples = length(accepted_samples)
    if N_samples == 0
        @warn "No accepted samples provided for plotting."
        return nothing
    end


    N_total_points = length(target_signal)
    N_obs = N_total_points ÷ N_time
    @assert Rx_loc !== nothing "Receiver locations (Rx_loc) must be provided."
    x_coords = Rx_loc[1:N_obs, 1]

    # Get layout for subplots based on number of time channels
    cols = min(4, N_time)
    rows = ceil(Int, N_time / cols)
    
    # Config canvas
    fig = CMke.Figure(size=(500 * cols, 380 * rows + 80))

    # Define local variables to hold references for legend elements
    local line_ref, scatter_ref, ensemble_ref

    for j in 1:N_time
        row_idx = (j - 1) ÷ cols + 1
        col_idx = (j - 1) % cols + 1
        
        ax = CMke.Axis(
            fig[row_idx, col_idx],
            title = "Time Channel $(j)",
            xlabel = "X-offset (m)",
            ylabel = "Response (unit)",
            titlesize = 14
        )

        # Plot accepted samples as ensemble
        for i in 1:N_samples
            sig_segment = accepted_samples[i].signal[(j - 1) * N_obs + 1 : j * N_obs]
            ensemble_ref = CMke.lines!(
                ax, x_coords, sig_segment, 
                color=(:black, 0.12), 
                linewidth=0.8
                )
        end

        # 2. Plot observed data
        target_segment = target_signal[(j - 1) * N_obs + 1 : j * N_obs]
        
        # Plot line for true data
        line_ref = CMke.lines!(
            ax, x_coords, target_segment,
            color = :red,
            linewidth = 2.5
        )
        
        # Plot scatter points for observed data
        scatter_ref = CMke.scatter!(
            ax, x_coords, target_segment,
            color = :red,
            markersize = 10,
            strokewidth = 0.5,
            strokecolor = :white
        )

        # Plot error bars for observed data if provided
        if signal_error_std !== nothing
            err_segment = signal_error_std[(j - 1) * N_obs + 1 : j * N_obs]
            CMke.errorbars!(
                ax, x_coords, target_segment, err_segment,
                color = (:red, 0.5),
                whiskerwidth = 10,
                linewidth = 1.5
            )
        end
    end

    # Gather lagend elements
    group_target = [
        CMke.LineElement(color = :red, linewidth = 2.5),
        CMke.MarkerElement(color = :red, marker = :circle, markersize = 10, strokecolor = :white)
    ]
    
    group_ensemble = [
        CMke.LineElement(color = (:black, 0.5), linewidth = 1.5)
    ]

    # Put legend at the bottom center, spanning all columns
    CMke.Legend(fig[rows + 1, :], 
        [group_target, group_ensemble], 
        ["Target Data (with Error Bars)", "Posterior Ensemble"],
        orientation = :horizontal,  # 水平排列
        framevisible = false,
        labelsize = 16,
        nbanks = 1,                
        tellheight = true,   
        tellwidth = false
    )

    CMke.Label(fig[0, :], "Data Fit: Posterior Ensemble vs Observed Signal", 
             fontsize = 22, font = :bold)
    CMke.rowgap!(fig.layout, 30)
    CMke.colgap!(fig.layout, 20)

    display(fig)
    return fig
end


function plot_signal_comparison(
    target_signal, 
    best_signal; 
    signal_error_std=nothing,
    compared_label="Predicted EM Data from PPM Inversion",
    N_time=1,
    N_obs=nothing,
    Rx_loc=nothing,
    )
    """
    Quick diagnostic plot: target vs best signal. 
    Layout updated to match plot_accepted_fits style.
    """

    if N_obs === nothing
        N_obs = length(target_signal) ÷ N_time
    end

    @assert Rx_loc !== nothing "Receiver locations (Rx_loc) must be provided."
    
    # Get x-coordinate
    x_coords = Rx_loc[1:N_obs, 1]
    N_real = length(best_signal)

    # Update layout logic
    cols = min(4, N_time)
    rows = ceil(Int, N_time / cols)
    
    # Setup canvas size
    fig = CMke.Figure(size=(450 * cols, 350 * rows + 120), backgroundcolor=:white)

    for j in 1:N_time
        row_idx = (j - 1) ÷ cols + 1
        col_idx = (j - 1) % cols + 1
        
        ax = CMke.Axis(
            fig[row_idx, col_idx],
            title = "Time Channel $(j)",
            xlabel = "X-offset (m)",
            ylabel = "Bz Data (nT)",
            titlesize = 14,
            
            # --- Label font size & bold ---
            xlabelsize = 18,
            ylabelsize = 18,
            xlabelfont = :bold,
            ylabelfont = :bold,

            # --- Tick font size & bold ---
            xticklabelsize = 14,
            yticklabelsize = 14,
            xticklabelfont = :bold,
            yticklabelfont = :bold,
        )

        # Compare predicted signal
        for i in 1:N_real
            sig_segment = best_signal[i][(j - 1) * N_obs + 1 : j * N_obs]
            CMke.lines!(
                ax, x_coords, sig_segment; 
                color = N_real > 1 ? (:black, 0.15) : (:black, 1.0),
                linewidth = 1.2
            )
        end

        # Plot target signal
        target_segment = target_signal[(j - 1) * N_obs + 1 : j * N_obs]
        
        # Red line
        CMke.lines!(ax, x_coords, target_segment, color = :red, linewidth = 2.5)
        
        # Red dots
        CMke.scatter!(
            ax, x_coords, target_segment, 
            color = :red, 
            markersize = 10,
            strokewidth = 0.5,
            strokecolor = :white
        )

        # Plot error bars
        if signal_error_std !== nothing
            err_segment = signal_error_std[(j - 1) * N_obs + 1 : j * N_obs]
            CMke.errorbars!(
                ax, x_coords, target_segment, err_segment,
                color = (:red, 0.5),
                whiskerwidth = 10,
                linewidth = 1.5
            )
        end
    end

    # Add legend and Title
    group_target = [
        CMke.LineElement(color = :red, linewidth = 2.5),
        CMke.MarkerElement(color = :red, marker = :circle, markersize = 10, strokecolor = :white)
    ]
    
    group_compared = [
        CMke.LineElement(color = (:black, 0.6), linewidth = 1.5)
    ]

    if best_signal != []
        CMke.Legend(fig[rows + 1, :], 
            [group_target, group_compared], 
            [signal_error_std !== nothing ? "True EM Data (with Error Bars)" : "True EM Data", compared_label],
            orientation = :horizontal,
            framevisible = false,
            labelsize = 16,
            tellheight = true,   
            tellwidth = false
        )
    else
        CMke.Legend(fig[rows + 1, :], 
            [group_target], 
            [signal_error_std !== nothing ? "True EM Data (with Error Bars)" : "True EM Data"],
            orientation = :horizontal,
            framevisible = false,
            labelsize = 16,
            tellheight = true,   
            tellwidth = false
        )
    end

    # Setup figure title
    if best_signal != []
        CMke.Label(fig[0, :], "Data Comparison: Target vs $compared_label", 
                fontsize = 22, font = :bold)
    else
        CMke.Label(fig[0, :], compared_label, 
                fontsize = 22, font = :bold)
    end

    # Adjust the spacing to prevent the axis labels from overlapping.
    CMke.rowgap!(fig.layout, 30)
    CMke.colgap!(fig.layout, 20)

    display(fig)
    return fig
end


function plot_post_hyper_param_hist(
    samples,
    param_names;             
    param_ranges=nothing,
    true_params=nothing,
    title="Posterior Distribution of Hyper-parameters"
    )
    
    # Sanity check
    N_samples = length(samples)
    if N_samples == 0
        @warn "No samples provided for histogram plotting."
        return nothing
    end
    
    # Get number of Gaussian-process parameters
    N_params = int(size(param_names)[1])

    # Initialization and obtain Gaussian-process parameters
    post_mat = zeros(N_samples, N_params)
    for i in 1:N_samples
        post_mat[i, 1:6] = samples[i].GP_param_spatial
        post_mat[i, 7]   = samples[i].GP_param_sigma_mean[1]
        post_mat[i, 8]   = samples[i].GP_param_sigma_std[1]
    end

    # Make Figure
    fig = CMke.Figure(size=(1450, 750))
    CMke.Label(fig[0, 1:4], text=title, fontsize=24, font=:bold)

    # Plot for each parameter
    for j in 1:N_params
        row = (j - 1) ÷ 4 + 1   
        col = (j - 1) % 4 + 1   

        ax = CMke.Axis(
            fig[row, col],
            title = param_names[j],
            xlabel = "Parameter Value",
            ylabel = "Frequency"
        )

        # Plot histogram
        CMke.hist!(ax, post_mat[:, j], bins=25, color=(:dodgerblue, 0.5), strokewidth=1, strokecolor=:black)

        # Plot range of prior
        if param_ranges !== nothing   
            p_min, p_max = param_ranges[j]
            CMke.vlines!(ax, [p_min, p_max], color=:grey40, linestyle=:dash, linewidth=2)
            CMke.xlims!(ax, p_min - 0.1*(p_max-p_min), p_max + 0.1*(p_max-p_min))
        end

        # plot true parameter is available
        if true_params !== nothing
            CMke.vlines!(ax, [true_params[j]], color=:red, linewidth=3)
        end
    end

    # Add legend
    labels = ["Posterior", "Prior Range"]
    elements = [
        CMke.PolyElement(color=(:dodgerblue, 0.5), strokecolor=:black, strokewidth=1),
        CMke.LineElement(color=:grey40, linestyle=:dash, linewidth=2)
    ]
    
    if true_params !== nothing
        push!(labels, "True Value")
        push!(elements, CMke.LineElement(color=:red, linewidth=3))
    end

    # Position legend
    CMke.Legend(fig[1:2, 5], elements, labels, framevisible=false, halign=:left)

    CMke.rowgap!(fig.layout, 40)
    CMke.colgap!(fig.layout, 30)

    display(fig)
    return fig
end


function plot_acc_hyper_params_history(
    samples::Vector{MCMCState},    # only accepted proposals
    param_ranges,
    param_names
    )

    # Sanity check
    N_acc = length(samples)
    if N_acc == 0
        @warn "No accepted samples provided for trace plotting."
        return nothing
    end

    # Get number of Gaussian-process parameters
    N_params = int(size(param_names)[1])

    # Make a figure
    fig = CMke.Figure(size=(1600, 800), fontsize=18)
    CMke.Label(fig[0, :], "Trace Plot of Accepted Hyper-parameters", fontsize=24, font=:bold)

    for p in 1:N_params
        row = (p - 1) ÷ 4 + 1   
        col = (p - 1) % 4 + 1
        p_min, p_max = param_ranges[p]
        
        ax = CMke.Axis(
            fig[row, col],
            title = param_names[p],
            xlabel = "Accepted Index",
            ylabel = "Value",
            limits = (nothing, (p_min - 0.1*(p_max-p_min), p_max + 0.1*(p_max-p_min)))
        )

        # Hide y-label if not the left-most column
        if col > 1
            ax.ylabelvisible = false
        end
        # Hide x-label if not the bottom row
        if row < 2
            ax.xlabelvisible = false
        end

        # Extract parameters
        param_series = Vector{Float64}(undef, N_acc)
        for k in 1:N_acc
            state = samples[k]
            if p <= 6
                param_series[k] = state.GP_param_spatial[p]
            elseif p == 7
                param_series[k] = state.GP_param_sigma_mean[1]
            else
                param_series[k] = state.GP_param_sigma_std[1]
            end
        end

        # Plot
        CMke.lines!(ax, 1:N_acc, param_series, color = (:black, 0.6), linewidth = 1.2)
        CMke.scatter!(ax, 1:N_acc, param_series, color = :black, markersize = 5, marker = :circle)
        CMke.hlines!(ax, [p_min, p_max], color = :red, linestyle = :dash, linewidth = 1.5)
    end

    # Force all axes (subplots) to have same size
    for i in 1:2, j in 1:4
        CMke.colsize!(fig.layout, j, CMke.Aspect(i, 1.0)) # Equal width
    end

    CMke.rowgap!(fig.layout, 40)
    CMke.colgap!(fig.layout, 30)

    display(fig)
    return fig
end


function kde1d(x; npts=300)
    k = kde(x)
    xs = range(minimum(k.x), maximum(k.x), length=npts)
    ys = pdf(k, collect(xs))
    return collect(xs), ys
end


function kde2d_contour(x, y)
    k = kde((x, y))
    return k
end


function get_contour_levels(
    k2d, fracs=(0.68, 0.95)
    )
    """
    A function to calculate the contour density levels corresponding to 
    specific enclosed probability fractions (e.g., 68% and 95%) for a 
    2D probability density function.
    """
    
    z     = k2d.density
    zsort = sort(vec(z), rev=true)
    cumz  = cumsum(zsort)
    cumz ./= cumz[end]
    levels = Float64[]
    for f in fracs
        idx = findfirst(>=(f), cumz)
        push!(levels, zsort[min(idx, length(zsort))])
    end
    return reverse(levels)
end


function plot_hyper_params_cross(
    acc_samples,
    param_names 
    )
    """
    A function to plot crosscorrelation results of each parameter pair
    """

    # Extract and reorg parameters
    data = hcat(
        [s.GP_param_spatial[1]    for s in acc_samples],
        [s.GP_param_spatial[2]    for s in acc_samples],
        [s.GP_param_spatial[3]    for s in acc_samples],
        [s.GP_param_spatial[4]    for s in acc_samples],
        [s.GP_param_spatial[5]    for s in acc_samples],
        [s.GP_param_spatial[6]    for s in acc_samples],
        [s.GP_param_sigma_mean[1] for s in acc_samples],
        [s.GP_param_sigma_std[1]  for s in acc_samples],
    )

    nsamples, np = size(data)
    cell = 150

    # make plot
    fig = CMke.Figure(
        size            = (np * cell + 120, np * cell + 120),
        backgroundcolor = :white,
    )

    CMke.Label(fig[1, 1], "Cross-Correlation of Posterior Hyper-parameters";
              fontsize  = 35,
              font      = :bold,
              tellwidth = false,
              padding   = (0, 0, 8, 0))

    gl = fig[2, 1] = CMke.GridLayout()
    CMke.rowgap!(gl, 4)
    CMke.colgap!(gl, 4)

    scatter_color = (:royalblue, 0.25)
    kde_fill      = (:royalblue, 0.20)
    kde_line      = (:royalblue, 0.90)

    axs = Matrix{Any}(nothing, np, np)

    # Build all axes first, THEN set sizes
    axis_style = (
        xticklabelsize     = 15,
        yticklabelsize     = 15,
        xlabelsize         = 20,
        ylabelsize         = 20,
        xticklabelfont     = :bold,
        yticklabelfont     = :bold,
        xlabelfont         = :bold,
        ylabelfont         = :bold,
        xgridvisible       = false,
        ygridvisible       = false,
        topspinevisible    = false,
        rightspinevisible  = false,
        spinewidth         = 0.8,
        xticks             = CMke.LinearTicks(3),
        xticklabelrotation = π/4,
    )

    for row in 1:np
        for col in 1:row

            if row == col
                ax = CMke.Axis(gl[row, col];
                    axis_style...,
                    yticksvisible      = false,
                    yticklabelsvisible = false,
                )
                axs[row, col] = ax

                xs, ys = kde1d(data[:, col])
                CMke.band!(ax, xs, zeros(length(xs)), ys; color = kde_fill)
                CMke.lines!(ax, xs, ys; color = kde_line, linewidth = 1.8)

                if row < np
                    CMke.hidexdecorations!(ax; ticklabels=true, label=true,
                                              ticks=false, grid=true)
                else
                    ax.xlabel = param_names[col]
                end

            else
                ax = CMke.Axis(gl[row, col];
                    axis_style...,
                    yticksvisible      = false,
                    yticklabelsvisible = false,
                )
                axs[row, col] = ax

                xdata = data[:, col]
                ydata = data[:, row]

                step = max(1, nsamples ÷ 1500)
                CMke.scatter!(ax, xdata[1:step:end], ydata[1:step:end];
                             color       = scatter_color,
                             markersize  = 5,
                             strokewidth = 0)

                try
                    k2d    = kde2d_contour(xdata, ydata)
                    levels = get_contour_levels(k2d, (0.68, 0.95))
                    xs2d   = collect(k2d.x)
                    ys2d   = collect(k2d.y)
                    zs2d   = k2d.density

                    CMke.contour!(ax, xs2d, ys2d, zs2d;
                                 levels    = levels,
                                 color     = :royalblue,
                                 linewidth = 1.2,
                                 alpha     = 0.85)
                catch
                end

                if col == 1
                    ax.ylabel = param_names[row]
                else
                    CMke.hideydecorations!(ax; ticklabels=true, label=true,
                                              ticks=false, grid=true)
                end

                if row < np
                    CMke.hidexdecorations!(ax; ticklabels=true, label=true,
                                              ticks=false, grid=true)
                else
                    ax.xlabel = param_names[col]
                end
            end
        end
    end

    # NOW set sizes — grid is populated so columns/rows exist
    for i in 1:np
        CMke.colsize!(gl, i, CMke.Fixed(cell))
        CMke.rowsize!(gl, i, CMke.Fixed(cell))
    end

    CMke.resize_to_layout!(fig)
    display(fig)
    return fig
end


function plot_sigma(
    inverted_3d_model;
    title_inverted="Inverted Conductivity Model",
    slice_x=nothing, 
    slice_y=nothing, 
    slice_z=nothing,
    x_length=nothing,
    y_length=nothing,
    z_length=nothing,
    origin=nothing,
    Rx_loc=nothing,
    view_azimuth=45,
    view_elevation=30,
    scale="ln",
    colorrange=nothing  # 1. Added optional keyword argument
    )
    """ 
    A function to plot a conductivity model
    """
    
    # Make grid
    nx, ny, nz = size(inverted_3d_model)
    if x_length !== nothing && y_length !== nothing && z_length !== nothing
        x0, y0, z0 = origin !== nothing ? origin : (0.0, 0.0, 0.0)
        x = range(x0, x0 + x_length, length=nx)
        y = range(y0, y0 + y_length, length=ny)
        z = range(z0, z0 + z_length, length=nz)
    else
        x, y, z = 1:nx, 1:ny, 1:nz   # fallback to cell indices
    end

    # Prepare slice indices for 3D plotting if not specified
    if slice_x === nothing
        slice_x = Int(round(nx / 2))
    end

    if slice_y === nothing
        slice_y = Int(round(ny / 2))
    end

    if slice_z === nothing
        slice_z = Int(round(nz / 2))
    end

    # Determine limits of color scale if not provided by the user
    if colorrange === nothing
        vmin = scale=="ln" ? minimum(inverted_3d_model) : minimum(exp.(inverted_3d_model))
        vmax = scale=="ln" ? maximum(inverted_3d_model) : maximum(exp.(inverted_3d_model))
        colorrange = (vmin, vmax)
    end

    # Plot 3D slices for two models
    fig = CMke.Figure(size=(600, 800))
    ax = CMke.Axis3(
        fig[1,1], 
        perspectiveness = 0.75,
        azimuth = deg2rad(view_azimuth), 
        elevation = deg2rad(view_elevation),
        xlabel = "X (Northing)", 
        ylabel = "Y (Easting)", 
        zlabel = "Z (Depth)",
        title = title_inverted,

        # --- Label font size & bold ---
        xlabelsize = 18,
        ylabelsize = 18,
        zlabelsize = 18,
        xlabelfont = :bold,
        ylabelfont = :bold,
        zlabelfont = :bold,

        # --- Tick font size & bold ---
        xticklabelsize = 14,
        yticklabelsize = 14,
        zticklabelsize = 14,
        xticklabelfont = :bold,
        yticklabelfont = :bold,
        zticklabelfont = :bold,

        # --- Tick padding (space between tick marks and tick labels) ---
        #xticklabelpad = 5,
        #yticklabelpad = 10,
        #zticklabelpad = 10,

        # --- Tick label padding (space between tick marks and tick labels) ---
        xlabeloffset = 50,
        ylabeloffset = 60,
        zlabeloffset = 60,
    )

    # --- Z-slice (horizontal plane at depth z[zi]) ---
    if slice_z !== 0
        X_z_slice = repeat(reshape(x, nx, 1), 1, ny)
        Y_z_slice = repeat(reshape(y, 1, ny), nx, 1)
        Z_z_slice = fill(z[slice_z], nx, ny)
        z_slice = scale=="ln" ? inverted_3d_model[:, :, slice_z] : exp.(inverted_3d_model[:, :, slice_z])
        CMke.surface!(
            ax, 
            X_z_slice, Y_z_slice, Z_z_slice, 
            color = z_slice; 
            colormap = :turbo, 
            colorrange = colorrange # 3. Updated to use the variable
        )
    end

    # --- X-slice (vertical plane at X = x[xi]) : Y-Z plane ---
    if slice_x !== 0
        Y_x_slice = repeat(reshape(y, ny, 1), 1, nz)      # ny × nz
        Z_x_slice = repeat(reshape(z, 1, nz), ny, 1)      # ny × nz
        X_x_slice = fill(x[slice_x], ny, nz)
        x_slice = scale=="ln" ? inverted_3d_model[slice_x, :, :] : exp.(inverted_3d_model[slice_x, :, :])
        CMke.surface!(
            ax, 
            X_x_slice,  Y_x_slice, Z_x_slice, 
            color = x_slice; 
            colormap = :turbo, 
            colorrange = colorrange # 3. Updated to use the variable
        )
    end

    # --- Y-slice (vertical plane at Y = y[yi]) : X-Z plane ---
    if slice_y !== 0
        X_y_slice = repeat(reshape(x, nx, 1), 1, nz)      # nx × nz
        Z_y_slice = repeat(reshape(z, 1, nz), nx, 1)      # nx × nz
        Y_y_slice = fill(y[slice_y], nx, nz)
        y_slice = scale=="ln" ? inverted_3d_model[:, slice_y, :] : exp.(inverted_3d_model[:, slice_y, :])
        CMke.surface!(
            ax, 
            X_y_slice, Y_y_slice, Z_y_slice, 
            color = y_slice; 
            colormap = :turbo, 
            colorrange = colorrange # 3. Updated to use the variable
        )
    end

    cbar = CMke.Colorbar(
        fig[2, 1],
        colormap=:turbo,
        colorrange = colorrange, # 3. Updated to use the variable
        vertical=false,
        label=scale=="ln" ? "Inverted Conductivity (ln(S/m))" : "Inverted Conductivity (S/m)",
        labelrotation=0,  
        halign=:center,   
        valign=:bottom,      
        width=Relative(0.65),
        height=Relative(0.4),
        flipaxis=true  ,
        labelsize = 20,
        labelfont = :bold,   
        ticklabelsize = 14,  
        ticklabelfont = :bold,   
    )

    if Rx_loc !== nothing
        n_rx = size(Rx_loc, 1)
        rx_x = Rx_loc[:, 1]
        rx_y = Rx_loc[:, 2]
        rx_z = size(Rx_loc, 2) >= 3 ? Rx_loc[:, 3] : fill(z[slice_z], n_rx)
        CMke.scatter!(
            ax,
            rx_x,
            rx_y,
            rx_z;
            color=:white,
            markersize=8,
            marker=:diamond,
        )
    end

    CMke.rowsize!(fig.layout, 1, Relative(0.85))  
    CMke.rowsize!(fig.layout, 2, Relative(0.15))  

    display(fig)
    
    return fig
end


