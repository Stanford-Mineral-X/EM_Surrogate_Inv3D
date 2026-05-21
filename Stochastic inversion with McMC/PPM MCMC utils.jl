# --- 0. PACKAGE IMPORTS ---
using StatsBase
using GeoStats, GeoStatsProcesses, GeoStatsBase
using GeoStatsProcesses: preprocess, randinit
using Meshes, DataFrames, Unitful, Tables
using Random, Distributions, Statistics
using ProgressMeter, Printf
using FFTW
using CSV
using Base.Math: rad2deg
using PyCall
using KernelDensity

# --- 1. HELPER FUNCTIONS ---
function dynamic_weighted_mse(
    true_signal::AbstractVector,
    predicted_signal::AbstractVector,
    N_time::Integer,
    N_obs::Integer;
    noise::AbstractVector,
    eps::Real = 1e-12,
    )
    """
    Mean squared error where each time-channel (block of length `N_obs`) is weighted
    by the inverse of the dynamic range of the true data in that channel.
    """
    channel_weighted_mses = zeros(N_time)
    dr_weights = zeros(N_time)
    channel_MSE = zeros(N_time)    
    @views for i in 1:N_time
        idx = (i - 1) * N_obs + 1 : i * N_obs
        t_sec = true_signal[idx]
        p_sec = predicted_signal[idx]
        n_sec = noise[idx]
        t_min, t_max = minimum(t_sec), maximum(t_sec)
        dr_weights[i] = 1 / (t_max - t_min + eps)
        diff = (t_sec .- p_sec) ./ n_sec
        channel_MSE[i] = sum(diff.^2) / N_obs
        channel_weighted_mses[i] = channel_MSE[i] * dr_weights[i]^2
    end

    total_loss = mean(channel_weighted_mses)
    return total_loss, channel_weighted_mses
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


function clamp_unit_interval(u; ε=1e-12) 
    return clamp(u, ε, 1 - ε)
end


function ppm_sampler_uniform_deterministic(u_old::Real, r::Real, u_perturb::Real)
    lower_bound = r * u_old
    upper_bound = 1 - r + lower_bound
    if lower_bound < u_perturb < upper_bound
        return u_old
    elseif u_perturb <= lower_bound
        return clamp(u_perturb / r, 0.0, 1.0)
    else
        return clamp((u_perturb + r - 1) / r, 0.0, 1.0)
    end
end


function ppm_apply_uniform(u_old::AbstractArray{T}, r::Real, U0::AbstractArray{T}) where {T<:Real}
    @assert size(u_old) == size(U0) "ppm_apply_uniform: u_old and U0 must have same size"
    u_next = similar(u_old)
    @inbounds for idx in eachindex(u_old)
        u_next[idx] = ppm_sampler_uniform_deterministic(u_old[idx], r, U0[idx])
    end
    return u_next
end


# --- 2. FORWARD MODEL AND INVERSION COMPONENTS ---
function process_from_GP_params(
    major::Real, interm::Real, minor::Real,
    yaw::Real, pitch::Real, roll::Real,
    sigma_mean::Real, sigma_std::Real
    )
    γ = SphericalVariogram(
        ranges=(major, interm, minor), 
        sill=sigma_std^2, 
        rotation=RotZYX(yaw, pitch, roll)
    )
    return GaussianProcess(γ, sigma_mean)
end


function realization_from_w!(
    out_namedtuple::Base.RefValue, rng::AbstractRNG, 
    process::GaussianProcess, method::FFTSIM, grid, preproc, w::AbstractArray
    )
    res = randsingle_controlled(rng, process, method, grid, nothing, preproc, w)
    out_namedtuple[] = res
    return nothing
end


function eval_em_forward(
    rng, grid, target_signal, signal_error_std, param_ranges,
    U1_field, U1_param_spatial, U1_param_sigma_mean, U1_param_sigma_std,
    w_base, GP_param_spatial_base, GP_param_sigma_mean_base, GP_param_sigma_std_base,
    Rx_loc, N_time, N_obs, surro_pred, Surrogate_weights_dir;
    r_vals::NTuple{4,Float64}=(0.2, 0.2, 0.2, 0.2)
    )

    Nx, Ny, Nz = size(grid);

    # Spatial pattern deterministic PPM
    gp_param_spatial_base = [(v .- vmin) ./ (vmax .- vmin) for (v, (vmin, vmax)) in zip(GP_param_spatial_base, param_ranges[1:6])];    
    gp_param_spatial_new = ppm_apply_uniform(gp_param_spatial_base, r_vals[1], U1_param_spatial);        
    gp_param_spatial_new .= clamp_unit_interval.(gp_param_spatial_new);    
    GP_param_spatial_new = [v_scaled .* (vmax .- vmin) .+ vmin for (v_scaled, (vmin, vmax)) in zip(gp_param_spatial_new, param_ranges[1:6])];    

    # Physical property deterministic PPM
    gp_param_sigma_mean_base = [(v .- vmin) ./ (vmax .- vmin) for (v, (vmin, vmax)) in zip(GP_param_sigma_mean_base, param_ranges[7:7])];    
    gp_param_sigma_mean_new = ppm_apply_uniform(gp_param_sigma_mean_base, r_vals[2], U1_param_sigma_mean)        
    gp_param_sigma_mean_new .= clamp_unit_interval.(gp_param_sigma_mean_new);    
    GP_param_sigma_mean_new = [v_scaled .* (vmax .- vmin) .+ vmin for (v_scaled, (vmin, vmax)) in zip(gp_param_sigma_mean_new, param_ranges[7:7])];    
    
    gp_param_sigma_std_base = [(v .- vmin) ./ (vmax .- vmin) for (v, (vmin, vmax)) in zip(GP_param_sigma_std_base, param_ranges[8:8])];    
    gp_param_sigma_std_new = ppm_apply_uniform(gp_param_sigma_std_base, r_vals[3], U1_param_sigma_std)        
    gp_param_sigma_std_new .= clamp_unit_interval.(gp_param_sigma_std_new);    
    GP_param_sigma_std_new = [v_scaled .* (vmax .- vmin) .+ vmin for (v_scaled, (vmin, vmax)) in zip(gp_param_sigma_std_new, param_ranges[8:8])];    

    # GP Field deterministic PPM
    u_base = cdf.(Normal(0, 1), w_base);    
    u_new = ppm_apply_uniform(u_base, r_vals[4], U1_field);    
    u_new .= clamp_unit_interval.(u_new);    
    w_new = quantile.(Normal(0, 1), u_new);    

    # Update GP 
    process = process_from_GP_params(
        GP_param_spatial_new[1], 
        GP_param_spatial_new[1] * GP_param_spatial_new[2],
        GP_param_spatial_new[1] * GP_param_spatial_new[2] * GP_param_spatial_new[3], 
        GP_param_spatial_new[4],
        GP_param_spatial_new[5], 
        GP_param_spatial_new[6],
        GP_param_sigma_mean_new[1],    
        GP_param_sigma_std_new[1]    
    )
    preproc = GeoStatsProcesses.preprocess(rng, process, FFTSIM(), nothing, grid, nothing)

    # Get 3d fields and directly shape to 4D (1, Nx, Ny, Nz) for PyCall
    res_named = randsingle_controlled(rng, process, FFTSIM(), grid, nothing, preproc, w_new)
    var_name = first(keys(res_named))
    z_vec = res_named[var_name]
    field3d = reshape(collect(z_vec), 1, Nx, Ny, Nz)

    # Call Python Surrogate Network
    signal = surro_pred.em_surrogate_prediction(
        sigma_models=Float32.(field3d),
        param_path=Surrogate_weights_dir,
        Rx_loc=Float32.(Rx_loc)
    )
    signal = Float64.(vec(signal))

    @assert length(signal) == length(target_signal)    
    loss, channel_MSE = dynamic_weighted_mse(target_signal, signal, N_time, N_obs, noise=signal_error_std)   

    return (; loss, channel_MSE, signal, w_new, GP_param_spatial_new, GP_param_sigma_mean_new, GP_param_sigma_std_new)
end


struct ProposalLog
    iter::Int
    loss::Float64
    ΔL::Float64
    α::Float64        
    u::Float64        
    accepted::Bool    
    r_vals::NTuple{4, Float64} 
    T::Float64        
    stage_indicator::Int    
end


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


function metropolis_step_delay_reject(
    param_ranges::NTuple{8, Tuple{Float64, Float64}},
    target_signal::AbstractVector,
    signal_error_std::AbstractVector,
    grid,
    Rx_loc,  
    N_time, 
    N_obs, 
    surro_pred,
    Surrogate_weights_dir,
    rng::AbstractRNG,
    state::MCMCState;
    T=100.0::Float64,
    r_stage1::NTuple{4, Float64}=(0.5, 0.5, 0.5, 0.5),
    r_stage2::NTuple{4, Float64}=(0.05, 0.05, 0.05, 0.05)
    )

    # --- Phase 1: Aggressive Big Jump ---
    U1_field = rand(rng, Float64, size(grid))
    U1_param_spatial = [rand(rng) for _ in 1:6]
    U1_param_sigma_mean = [rand(rng) for _ in 1:1]
    U1_param_sigma_std = [rand(rng) for _ in 1:1]

    prop1 = eval_em_forward(
        rng, grid, target_signal, signal_error_std, param_ranges,
        U1_field, U1_param_spatial, U1_param_sigma_mean, U1_param_sigma_std,
        state.w, state.GP_param_spatial, state.GP_param_sigma_mean, state.GP_param_sigma_std,
        Rx_loc, N_time, N_obs, surro_pred, Surrogate_weights_dir;
        r_vals = r_stage1
    )

    L_old = state.loss
    L_new1 = prop1.loss
    ΔL1 = L_new1 - L_old
    α1 = ΔL1 <= 0 ? 1.0 : exp(-ΔL1/T)

    u1 = rand(rng)
    if u1 < α1
        accepted1 = true
        stage_indicator = 1
        prop1_state = MCMCState(
            prop1.loss, prop1.channel_MSE, prop1.signal,
            prop1.w_new, prop1.GP_param_spatial_new,
            prop1.GP_param_sigma_mean_new, prop1.GP_param_sigma_std_new,
            ΔL1, α1, u1, true
        )
        return prop1_state, accepted1, stage_indicator
    else
        stage_indicator = 2

        # --- Phase 2: Conservative Small Jump ---
        U2_field = rand(rng, Float64, size(grid))
        U2_param_spatial = [rand(rng) for _ in 1:6]
        U2_param_sigma_mean = [rand(rng) for _ in 1:1]
        U2_param_sigma_std = [rand(rng) for _ in 1:1]

        prop2 = eval_em_forward(
            rng, grid, target_signal, signal_error_std, param_ranges,
            U2_field, U2_param_spatial, U2_param_sigma_mean, U2_param_sigma_std,
            state.w, state.GP_param_spatial, state.GP_param_sigma_mean, state.GP_param_sigma_std,
            Rx_loc, N_time, N_obs, surro_pred, Surrogate_weights_dir;
            r_vals = r_stage2
        )

        L_new2 = prop2.loss
        ΔL2 = L_new2 - L_old

        # Delayed Rejection Correction
        ΔL2_1 = L_new1 - L_new2
        α2_1 = ΔL2_1 <= 0 ? 1.0 : exp(-ΔL2_1 / T)

        post_ratio = exp(-ΔL2 / T)
        num = 1.0 - α2_1
        den = 1.0 - α1
        correction = den > 1e-6 ? num / den : 0.0
        α2 = min(1.0, post_ratio * correction)

        u2 = rand(rng)
        accepted2 = u2 < α2

        prop2_state = MCMCState(
            prop2.loss, prop2.channel_MSE, prop2.signal,
            prop2.w_new, prop2.GP_param_spatial_new,
            prop2.GP_param_sigma_mean_new, prop2.GP_param_sigma_std_new,
            ΔL2, α2, u2, accepted2
        )
    end

    return prop2_state, accepted2, stage_indicator
end


# --- 3. MAIN MCMC SAMPLER ---
function optimize_ppm_outer_vranges(
    target_signal,
    signal_error_std,
    param_ranges::NTuple{8, Tuple{Float64, Float64}},
    grid,
    Rx_loc,
    N_time,
    N_obs,
    surro_pred,
    Surrogate_weights_dir,
    max_iter,
    rng=MersenneTwister(2025);
    print_log=false,
    print_time=false
    )
    
    start_time = time()
    dims3 = size(grid)
    Nx, Ny, Nz = dims3[1], dims3[2], dims3[3]

    current_w = randn(rng, Float64, dims3) 
    current_GP_param = collect(ntuple(i -> rand(rng, Uniform(param_ranges[i]...)), length(param_ranges)));
    current_GP_param_spatial = current_GP_param[1:6];
    current_GP_param_sigma_mean = current_GP_param[7:7];
    current_GP_param_sigma_std = current_GP_param[8:8];

    if print_log
        println("\nInitial GP-parameters: ", join([@sprintf("%.3f", param) for param in current_GP_param], ", "))
    end

    process = process_from_GP_params(
        current_GP_param_spatial[1],
        current_GP_param_spatial[1] * current_GP_param_spatial[2],
        current_GP_param_spatial[1] * current_GP_param_spatial[2] * current_GP_param_spatial[3],
        current_GP_param_spatial[4],
        current_GP_param_spatial[5],
        current_GP_param_spatial[6],
        current_GP_param_sigma_mean[1], 
        current_GP_param_sigma_std[1] 
    )
    preproc = GeoStatsProcesses.preprocess(rng, process, FFTSIM(), nothing, grid, nothing)

    res_named = randsingle_controlled(rng, process, FFTSIM(), grid, nothing, preproc, current_w)
    var_name = first(keys(res_named))
    z_vec = res_named[var_name]
    field3d = reshape(collect(z_vec), 1, Nx, Ny, Nz)

    signal = surro_pred.em_surrogate_prediction(
        sigma_models=Float32.(field3d),
        param_path=Surrogate_weights_dir,
        Rx_loc=Float32.(Rx_loc)
    )
    signal = Float64.(vec(signal))
    @assert length(signal) == length(target_signal)

    loss, channel_MSE = dynamic_weighted_mse(target_signal, signal, N_time, N_obs, noise=signal_error_std)

    if print_log
        println("Initial loss = ", @sprintf("%.3f", loss))
    end

    current_state = MCMCState(
        loss, channel_MSE, signal, current_w,
        current_GP_param_spatial, current_GP_param_sigma_mean, current_GP_param_sigma_std,
        loss, 1, 0, true
    )

    chain = Vector{MCMCState}(undef, max_iter)
    prop_logs = Vector{ProposalLog}(undef, max_iter)
    accepts = BitVector(undef, max_iter)
    best_state = current_state

    iter = 1
    acc_rate = 0
    current_T = 10 * loss
    target_AR = 0.23
    adaptation_speed = 0.1
    max_r = 1.0
    fine_r = 0.1
    burn_in_ratio = 0.2
    anneal_start = burn_in_ratio * max_iter
    
    while iter <= max_iter && best_state.loss >= 1e-4
    
        if print_log && iter % 10 == 0
            println("\nIteration $iter / $max_iter")
        end

        # Annealing schedules
        anneal_period = 0.3 * max_iter 
        anneal_end = anneal_start + anneal_period
        if iter <= anneal_start
            annealed_r = (max_r, max_r, max_r, max_r)    
        elseif iter < anneal_end
            # Normalize iteration to [0, 1] for decay function
            t = (iter - anneal_start) / anneal_period
            
            # Define a decay function for smooth annealing
            decay(fine, max, decay_rate) = fine + (max - fine) * exp(-decay_rate * t)

            # Apply decay to each r value
            annealed_r = (
                decay(fine_r, max_r, 4.0),     
                decay(fine_r, max_r, 4.0),     
                decay(fine_r, max_r, 4.0),     
                decay(3*fine_r, max_r, 3.0)    
            )
        else
            # After annealing period, fix r to fine values
            annealed_r = (fine_r, fine_r, fine_r, 3*fine_r)
        end

        r_stage1 = ntuple(i -> 0.8 + 0.2 * rand(), 4) 
        # Fixed potential type-mismatch bug by leveraging clean ntuple broadcast
        r_stage2 = ntuple(i -> rand() * annealed_r[i], 4)

        prop_state, accepted, stage_id = metropolis_step_delay_reject(
            param_ranges, target_signal, signal_error_std,
            grid, Rx_loc, N_time, N_obs, surro_pred, Surrogate_weights_dir,
            rng, current_state,
            T = current_T, r_stage1 = r_stage1, r_stage2 = r_stage2
        )

        current_r = stage_id == 1 ? r_stage1 : r_stage2 
        prop_logs[iter] = ProposalLog(
            iter, prop_state.loss, prop_state.ΔL, prop_state.α, prop_state.u,
            accepted, current_r, current_T, stage_id
        )

        if accepted
            current_state = prop_state
        end

        chain[iter] = current_state
        accepts[iter] = accepted

        window = 100
        low_bound = max(1, iter - window + 1)
        acc_rate = mean(@view accepts[low_bound:iter])

        # Robbins-Monro Adaptive T Adjustment
        if iter % 50 == 0
            current_speed = iter < burn_in_ratio * max_iter ? adaptation_speed : adaptation_speed * (1.0 - (iter-burn_in_ratio*max_iter)/(max_iter-burn_in_ratio*max_iter)) 
            new_log_T = log(current_T) + current_speed * (target_AR - acc_rate) 
            current_T = exp(new_log_T)

            if print_log
                @printf(" [Adaptive FeedBack] AR=%.2f, T=%.1f\n", acc_rate, current_T)
            end
        end

        if prop_state.loss < best_state.loss
            best_state = prop_state
        end

        if print_log && iter % 10 == 0
            best_GP_params = vcat(best_state.GP_param_spatial, best_state.GP_param_sigma_mean, best_state.GP_param_sigma_std)
            println(@sprintf("Current_Loss = %.3f Best_Loss = %.3f acc_rate = %.2f", current_state.loss, best_state.loss, acc_rate))
            println(" Post GP-parameters: ", join([@sprintf("%.3f", param) for param in best_GP_params], ", "), ".")
            println(" Proposal r values: ", join([@sprintf("%.3f", r) for r in current_r], ", "), ".")
            println(" Proposal ΔL = ", @sprintf("%.3f", prop_state.ΔL), " α = ", @sprintf("%.3f", prop_state.α), " accepted = ", prop_state.accepted, " T = ", @sprintf("%.3f", prop_logs[iter].T))
        end

        iter += 1
    end

    if print_time
        total_time = time() - start_time
        println(@sprintf("Completed %d attempts in %.3f s", iter-1, total_time))
    end

    return (; chain, prop_logs, best_state)
end

