# plotting.jl

using GLMakie
using Statistics

# ==============================================================================
# 1. LOW-LEVEL PLOTTING KERNELS (Generic)
# ==============================================================================

function _core_plot_paths!(ax, t, data::AbstractMatrix; count::Int=20, color=(:blue, 0.1))
    sim_count = size(data, 1)
    n_plot = min(sim_count, count)
    path_indices = rand(1:sim_count, n_plot)

    @views for i in path_indices
        lines!(ax, t, data[i, :], color=color)
    end
end

function _core_plot_mean_ci!(ax, t, data::AbstractMatrix; show_ci=true, mean_color=:black, ci_color=(:black, 0.2))
    mean_path = vec(mean(data, dims=1))
    if show_ci
        q_low = [quantile(col, 0.05) for col in eachcol(data)]
        q_high = [quantile(col, 0.95) for col in eachcol(data)]
        band!(ax, t, q_low, q_high, color=ci_color, label="90% C.I.")
    end
    lines!(ax, t, mean_path, color=mean_color, linewidth=2, label="Mean")
end


# ==============================================================================
# 2. INTERFACE A: SIMULATION WORLD
# ==============================================================================

function plot_makie_paths!(
    fig::Figure, ax::Axis,
    world::SimulationWorld, name::Symbol;
    count::Int=20, color=(:blue, 0.1)
)
    if !haskey(world.paths, name)
        error("Path ':$name' not found in SimulationWorld.")
    end

    data = world.paths[name]
    t = 0:world.δt:world.sim_params.T

    _core_plot_paths!(ax, t, data; count=count, color=color)
    return fig, ax
end

function plot_makie_mean_path!(
    fig::Figure, ax::Axis,
    world::SimulationWorld, name::Symbol;
    show_ci::Bool=true, mean_color=:black, ci_color=(:black, 0.2)
)
    data = world.paths[name]
    t = 0:world.δt:world.sim_params.T

    _core_plot_mean_ci!(ax, t, data; show_ci=show_ci, mean_color=mean_color, ci_color=ci_color)
    return fig, ax
end


# ==============================================================================
# 3. HIGH-LEVEL ORCHESTRATION
# ==============================================================================

function plot_world_paths(world::SimulationWorld)
    println("Generating Market Asset plots...")

    paths_to_plot = [
        (:P_N, "Nominal Bond Price P_N(t, T)", "Price", "nominal_bond_plot.png", (:green, 0.05)),
        (:S, "Stock Price S(t)", "Price", "stock_price_plot.png", (:blue, 0.05)),
        (:r, "Short Rate r(t)", "Short Rate", "short_rate_plot.png", (:cyan, 0.05)),
        (:Re_Stock, "Excess Simple Return Stock", "Excess Return", "excess_return_stock_plot.png", (:magenta, 0.05)),
        (:Re_NominalBond, "Excess Simple Return Bond", "Excess Return", "excess_return_bond_plot.png", (:brown, 0.05)),
        (:Re_InflationBond, "Excess Simple Return Inflation Bond", "Excess Return", "excess_return_inflation_bond_plot.png", (:brown, 0.05)),
        (:R_Stock, "Simple return Stock", "Return", "simple_return_stock_plot.png", (:blue, 0.05)),
        (:R_NominalBond, "Simple return Bond", "Return", "simple_return_bond_plot.png", (:green, 0.05)),
        (:R_InflationBond, "Simple return Inflation Bond", "Return", "simple_return_inflation_bond_plot.png", (:green, 0.05)),
    ]

    for (path_name, title_str, ylabel_str, filename, color) in paths_to_plot
        if !haskey(world.paths, path_name)
            continue
        end

        fig = Figure(size = (800, 800))

        ax_paths = Axis(fig[1, 1], title = "$title_str (T=$(world.sim_params.T))", ylabel = ylabel_str)
        plot_makie_paths!(fig, ax_paths, world, path_name, count=100, color=color)
        plot_makie_mean_path!(fig, ax_paths, world, path_name, mean_color=:black)

        ax_hist = Axis(fig[2, 1], title = "Distribution at T - δt", xlabel = ylabel_str)
        end_values = world.paths[path_name][:, end - 1]
        hist!(ax_hist, end_values, bins=50, color=color[1], strokewidth=1, strokecolor=:black)

        save(filename, fig)
    end
end

function plot_policy(world::SimulationWorld, solver_params, ω_l)

    t_list = 1:(world.sim_params.M)
    for t_to_plot in t_list
        println("Generating final policy plot for t=$t_to_plot...")

        policy_at_t = ω_l[t_to_plot] # Get the (sim,) vector of functions

        # Evaluate each function on the W_grid
        sim = world.sim_params.sim
        N_assets = length(solver_params.asset_names)
        W_grid = solver_params.W_grid
        policy_on_grid = Matrix{SVector{N_assets, Float64}}(undef, sim, length(W_grid))

        for i in 1:sim, (j, W) in enumerate(W_grid)
            policy_on_grid[i, j] = policy_at_t[i](W)
        end

        # Calculate the mean policy at each grid point
        mean_policy_per_W = [mean(policy_on_grid[:, j]) for j in 1:length(W_grid)]

        # Create the final plot
        fig_policy = Figure(size = (800, 600))
        ax_policy = Axis(fig_policy[1, 1],
            title = "Mean Optimal Policy ω(W) at t=$t_to_plot",
            xlabel = "Wealth (W)",
            ylabel = "Policy Weight"
        )

        # --- DYNAMIC ASSET LOOP ---
        for k in 1:N_assets
            # Extract the k-th weight component for every grid point
            mean_pol_asset_k = [v[k] for v in mean_policy_per_W]

            # Plot with automatic color cycling
            lines!(ax_policy, W_grid, mean_pol_asset_k,
                   label=String(solver_params.asset_names[k]),
                   linewidth=2)
        end
        # ---------------------------

        axislegend(ax_policy, position=:rt)

        save("optimal_policy_t$t_to_plot.png", fig_policy)

    end



end

function save_value_and_CE_to_csv(i, world, solver_params, ω_l, my_utility)
    println("Calculating value function and certainty equivalent wealth...")
    open("value_function_set_$i.csv", "w") do io
            println(io, "==================================================")
            println(io, "METRICS REPORT: $i")
            println(io, "==================================================")
            println(io, "")
            println(io, "Wealth,J_star,CE_star")
            for W_1 in solver_params.W_grid
                J_W1, CE_1 = calculate_expected_utility(world, solver_params, ω_l, 1, W_1, nothing, my_utility)
                println(io, "$W_1,$J_W1,$CE_1")
            end
        end
    println("All tasks complete. Final policy plot saved.")
end