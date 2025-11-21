"""
    plot_makie_paths!(
        fig::Figure,
        ax::Axis,
        world::SimulationWorld,
        name::Symbol;
        count::Int = 20,
        color = (:blue, 0.1) # Tuple for color and alpha
    )

Plots a sample of simulated paths to a Makie Axis.
"""
function plot_makie_paths!(
    fig::Figure,
    ax::Axis,
    world::SimulationWorld,
    name::Symbol;
    count::Int = 20,
    color = (:blue, 0.1) # Makie likes (color, alpha) tuples
)
    # 1. Get the data
    if !haskey(world.paths, name)
        error("Path ':$name' not found in SimulationWorld.")
    end
    data = world.paths[name]
    sim_count = size(data, 1)

    # 2. Create the time vector
    t = 0:world.δt:world.sim_params.T

    # 3. Determine which paths to plot (random sample)
    path_indices = rand(1:sim_count, min(sim_count, count))

    # 4. Add the paths
    @views for i in path_indices
        # `lines!` adds a line plot to the axis
        lines!(ax, t, data[i, :], color=color)
    end

    return fig, ax
end

"""
    plot_makie_mean_path!(
        fig::Figure,
        ax::Axis,
        world::SimulationWorld,
        name::Symbol;
        show_ci::Bool = true,
        mean_color = :black,
        ci_color = (:black, 0.2)
    )

Plots the mean path and (optionally) a 90% confidence interval to a Makie Axis.
"""
function plot_makie_mean_path!(
    fig::Figure,
    ax::Axis,
    world::SimulationWorld,
    name::Symbol;
    show_ci::Bool = true,
    mean_color = :black,
    ci_color = (:black, 0.2)
)
    # 1. Get the data
    if !haskey(world.paths, name)
        error("Path ':$name' not found in SimulationWorld.")
    end
    data = world.paths[name]

    # 2. Create the time vector
    t = 0:world.δt:world.sim_params.T

    # 3. Calculate statistics
    mean_path = mean(data, dims=1)[1, :]

    # 4. Add confidence interval
    if show_ci
        q_low = [quantile(col, 0.05) for col in eachcol(data)]
        q_high = [quantile(col, 0.95) for col in eachcol(data)]

        # `band!` is Makie's version of `ribbon`
        band!(ax, t, q_low, q_high, color=ci_color, label="90% C.I.")
    end

    # 5. Add the mean path (plotted on top)
    lines!(ax, t, mean_path, color=mean_color, linewidth=2, label="Mean")

    return fig, ax
end


function plot_world_paths(world::SimulationWorld)

    println("Generating plots...")
    # 1. Create a Figure with a 2-row, 1-column layout
    fig = Figure(size = (800, 800))

    # 2. Create the top axis for the paths
    ax_paths = Axis(fig[1, 1],
        title = "Nominal Bond Price P_N(t, T) (T=$(world.sim_params.T))",
        ylabel = "Price"
    )

    # 3. Create the bottom axis for the histogram
    ax_hist = Axis(fig[2, 1],
        title = "Distribution of P_N(T - δt)",
        xlabel = "Price"
    )

    # 4. Plot paths on the top axis
    plot_makie_paths!(fig, ax_paths, world, :P_N, count=200, color=(:green, 0.05))
    plot_makie_mean_path!(fig, ax_paths, world, :P_N, mean_color=:black)
    axislegend(ax_paths, position=:rb) # bottom-right

    # 5. Plot a histogram on the bottom axis
    # Get prices just before maturity (at T they are all exactly 1.0)
    prices_before_maturity = world.paths[:P_N][:, end-1]
    hist!(ax_hist, prices_before_maturity, bins=100, color=:green, strokewidth=1, strokecolor=:black)

    # 6. Save and display the figure
    save("nominal_bond_plot.png", fig)

    # Now make same for inflation bond, but with window with CPI added
    fig2 = Figure(size = (800, 800))
    ax2_paths = Axis(fig2[1, 1],
        title = "Inflation Bond Price P_I(t, T) (T=$(world.sim_params.T))",
        ylabel = "Price"
    )
    ax2_hist = Axis(fig2[2, 1],
        title = "Distribution of P_I(T - δt)",
        xlabel = "Price"
    )
    plot_makie_paths!(fig2, ax2_paths, world, :P_I, count=200, color=(:orange, 0.05))
    plot_makie_mean_path!(fig2, ax2_paths, world, :P_I, mean_color=:black)
    axislegend(ax2_paths, position=:rb)
    prices_I_before_maturity = world.paths[:P_I][:, end-1]
    hist!(ax2_hist, prices_I_before_maturity, bins=100, color=:orange, strokewidth=1, strokecolor=:black)

    # Save and display the second figure
    save("inflation_bond_plot.png", fig2)

    # Now make the same for stock prices
    fig3 = Figure(size = (800, 800))
    ax3_paths = Axis(fig3[1, 1],
        title = "Stock Price S(t) (T=$(world.sim_params.T))",
        ylabel = "Price"
    )
    ax3_hist = Axis(fig3[2, 1],
        title = "Distribution of S(T - δt)",
        xlabel = "Price"
    )
    plot_makie_paths!(fig3, ax3_paths, world, :S, count=200, color=(:blue, 0.05))
    plot_makie_mean_path!(fig3, ax3_paths, world, :S, mean_color=:black)
    axislegend(ax3_paths, position=:rb)
    prices_stock_before_maturity = world.paths[:S][:, end-1]
    hist!(ax3_hist, prices_stock_before_maturity, bins=100, color=:blue, strokewidth=1, strokecolor=:black)

    # Save and display the third figure
    save("stock_price_plot.png", fig3)

    # Plot also the CPI path
    fig4 = Figure(size = (800, 600))
    ax4_paths = Axis(fig4[1, 1],
        title = "Consumer Price Index Π(t) (T=$(world.sim_params.T))",
        ylabel = "CPI"
    )
    plot_makie_paths!(fig4, ax4_paths, world, :Π, count=200, color=(:purple, 0.05))
    plot_makie_mean_path!(fig4, ax4_paths, world, :Π, mean_color=:black)
    axislegend(ax4_paths, position=:rb)
    save("CPI_plot.png", fig4)

    # Plot the inflation path
    fig5 = Figure(size = (800, 600))
    ax5_paths = Axis(fig5[1, 1],
        title = "Inflation Rate π(t) (T=$(world.sim_params.T))",
        ylabel = "Inflation Rate"
    )
    plot_makie_paths!(fig5, ax5_paths, world, :π, count=200, color=(:red, 0.05))
    plot_makie_mean_path!(fig5, ax5_paths, world, :π, mean_color=:black)
    axislegend(ax5_paths, position=:rb) # bottom-right
    save("inflation_rate_plot.png", fig5)

    # Plot the interest rate path
    fig6 = Figure(size = (800, 600))
    ax6_paths = Axis(fig6[1, 1],
        title = "Short Rate r(t) (T=$(world.sim_params.T))",
        ylabel = "Short Rate"
    )
    plot_makie_paths!(fig6, ax6_paths, world, :r, count=200, color=(:cyan, 0.05))
    plot_makie_mean_path!(fig6, ax6_paths, world, :r, mean_color=:black)
    axislegend(ax6_paths, position=:rb) # bottom-right
    save("short_rate_plot.png", fig6)

    # Plot the excess return of the stock
    fig7 = Figure(size = (800, 600))
    ax7_paths = Axis(fig7[1, 1],
        title = "Excess Simple Return of Stock Re_Stock(t) (T=$(world.sim_params.T))",
        ylabel = "Excess Simple Return"
    )
    plot_makie_paths!(fig7, ax7_paths, world, :Re_Stock, count=200, color=(:magenta, 0.05))
    plot_makie_mean_path!(fig7, ax7_paths, world, :Re_Stock, mean_color=:black)
    axislegend(ax7_paths, position=:rb) # bottom-right
    save("excess_return_stock_plot.png", fig7)

    # Plot the excess return of the nominal bond
    fig8 = Figure(size = (800, 600))
    ax8_paths = Axis(fig8[1, 1],
        title = "Excess Simple Return of Nominal Bond Re_NominalBond(t) (T=$(world.sim_params.T))",
        ylabel = "Excess Simple Return"
    )
    plot_makie_paths!(fig8, ax8_paths, world, :Re_NominalBond, count=200, color=(:brown, 0.05))
    plot_makie_mean_path!(fig8, ax8_paths, world, :Re_NominalBond, mean_color=:black)
    axislegend(ax8_paths, position=:rb) # bottom-right
    save("excess_return_nominal_bond_plot.png", fig8)

    # Plot the excess return of the inflation bond
    fig9 = Figure(size = (800, 600))
    ax9_paths = Axis(fig9[1, 1],
        title = "Excess Simple Return of Inflation Bond Re_InflBond(t) (T=$(world.sim_params.T))",
        ylabel = "Excess Simple Return"
    )
    plot_makie_paths!(fig9, ax9_paths, world, :Re_InflBond, count=200, color=(:teal, 0.05))
    plot_makie_mean_path!(fig9, ax9_paths, world, :Re_InflBond, mean_color=:black)
    axislegend(ax9_paths, position=:rb) # bottom-right
    save("excess_return_inflation_bond_plot.png", fig9)
end

function plot_policy(world::SimulationWorld, solver_params, ω_l)


    t_list = [1, 2]
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

        # Separate components for plotting
        mean_pol_asset1 = [v[1] for v in mean_policy_per_W]
        mean_pol_asset2 = [v[2] for v in mean_policy_per_W]
        mean_pol_asset3 = [v[3] for v in mean_policy_per_W]

        # Create the final plot
        fig_policy = Figure(size = (800, 600))
        ax_policy = Axis(fig_policy[1, 1],
            title = "Mean Optimal Policy ω(W) at t=$t_to_plot",
            xlabel = "Wealth (W)",
            ylabel = "Policy Weight"
        )

        lines!(ax_policy, W_grid, mean_pol_asset1, label=String(solver_params.asset_names[1]), linewidth=2)
        lines!(ax_policy, W_grid, mean_pol_asset2, label=String(solver_params.asset_names[2]), linewidth=2)
        lines!(ax_policy, W_grid, mean_pol_asset3, label=String(solver_params.asset_names[3]), linewidth=2)
        axislegend(ax_policy, position=:rt)

        save("optimal_policy_t$t_to_plot.png", fig_policy)

    end
    println("All tasks complete. Final policy plot saved.")
end