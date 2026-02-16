using CairoMakie
using Statistics
using Printf

# ==============================================================================
# 1. HELPERS
# ==============================================================================

"""
Calculates the indices in the data array corresponding to the requested time range.
"""
function get_time_indices(config::MarketConfig, time_range::Tuple{Real, Real})
    t_start, t_end = time_range
    dt = config.dt

    idx_start = floor(Int, t_start / dt) + 1
    idx_end   = floor(Int, t_end / dt) + 1

    idx_start = max(1, idx_start)
    idx_end   = min(config.M + 1, idx_end)

    return idx_start:idx_end
end

"""
Calculates the single index corresponding to a specific time.
"""
function get_time_index(config::MarketConfig, t::Real)
    idx = floor(Int, t / config.dt) + 1
    return clamp(idx, 1, config.M + 1)
end

# ==============================================================================
# 2. CORE PLOTTING KERNELS
# ==============================================================================

function plot_paths_kernel!(ax, times, data::AbstractMatrix; count::Int=20, color=(:blue, 0.1))
    sim_count = size(data, 1)
    n_plot = min(sim_count, count)
    path_indices = rand(1:sim_count, n_plot)

    for i in path_indices
        lines!(ax, times, data[i, :], color=color)
    end
end

function plot_mean_ci_kernel!(ax, times, data::AbstractMatrix; mean_color=:black, ci_color=(:black, 0.2))
    mean_path = vec(mean(data, dims=1))
    q_low     = [quantile(col, 0.05) for col in eachcol(data)]
    q_high    = [quantile(col, 0.95) for col in eachcol(data)]

    band!(ax, times, q_low, q_high, color=ci_color, label="90% C.I.")
    lines!(ax, times, mean_path, color=mean_color, linewidth=2, label="Mean")
end

# ==============================================================================
# 3. HIGH-LEVEL FUNCTIONS
# ==============================================================================

"""
    plot_simulation(world; ...)

Plots simulation results with flexible options.

# Arguments
- `vars`: List of symbols to plot (e.g. `[:r, :S]`). Defaults to all.
- `time_range`: Tuple (start, end) for the trajectory x-axis.
- `output_dir`: Folder to save plots.
- `plot_traj`: Bool, show trajectories (default true).
- `plot_dist`: Bool, show distribution histogram (default true).
- `dist_time`: Real, specific time for the histogram. Defaults to `time_range[2]`.
"""
function plot_simulation(world::SimulationWorld;
                         vars::Vector{Symbol}=Symbol[],
                         time_range::Tuple{Real, Real}=(0.0, world.config.T),
                         output_dir::String="plots",
                         samples::Int=50,
                         # New Control Arguments
                         plot_traj::Bool=true,
                         plot_dist::Bool=true,
                         dist_time::Union{Real, Nothing}=nothing)

    if !isdir(output_dir)
        mkdir(output_dir)
    end

    if isempty(vars)
        vars = [k for k in keys(world.paths) if isa(getproperty(world.paths, k), AbstractMatrix)]
    end

    # 1. Prepare Time Slices for Trajectories
    idxs_traj = get_time_indices(world.config, time_range)
    full_time_grid = range(0, world.config.T, length=world.config.M+1)
    t_slice = full_time_grid[idxs_traj]

    # 2. Prepare Index for Distribution
    # If dist_time is not provided, default to the end of the trajectory window
    t_dist_target = isnothing(dist_time) ? time_range[2] : dist_time
    idx_dist = get_time_index(world.config, t_dist_target)

    # Snap actual time for the title
    t_dist_actual = full_time_grid[idx_dist]

    println("Generating plots...")
    println("  > Trajectory Range: $time_range")
    println("  > Distribution Time: $t_dist_actual")

    for name in vars
        data_full = getproperty(world.paths, name)

        # Configure Layout based on what we are plotting
        if plot_traj && plot_dist
            fig = Figure(size = (1000, 500))
            ax_traj = CairoMakie.Axis(fig[1, 1], title="$name Trajectories", xlabel="Time", ylabel=String(name))
            ax_dist = CairoMakie.Axis(fig[1, 2], title="Distribution at t=$(round(t_dist_actual, digits=2))", xlabel=String(name))
        elseif plot_traj
            fig = Figure(size = (800, 600))
            ax_traj = CairoMakie.Axis(fig[1, 1], title="$name Trajectories", xlabel="Time", ylabel=String(name))
        elseif plot_dist
            fig = Figure(size = (600, 400))
            ax_dist = CairoMakie.Axis(fig[1, 1], title="$name Dist (t=$(round(t_dist_actual, digits=2)))", xlabel=String(name))
        else
            error("Nothing to plot! Set plot_traj or plot_dist to true.")
        end

        # --- Draw Trajectories ---
        if plot_traj
            data_slice = data_full[:, idxs_traj]
            plot_paths_kernel!(ax_traj, t_slice, data_slice; count=samples, color=(:blue, 0.15))
            plot_mean_ci_kernel!(ax_traj, t_slice, data_slice)
        end

        # --- Draw Distribution ---
        if plot_dist
            # We take the column corresponding to dist_time
            dist_data = data_full[:, idx_dist]
            hist!(ax_dist, dist_data, bins=50, color=:cornflowerblue, strokewidth=1, strokecolor=:black)
        end

        fname = joinpath(output_dir, "$(name)_plot.png")
        save(fname, fig)
        println("  Saved $fname")
    end
end