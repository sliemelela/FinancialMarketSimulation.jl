
@testset "Plotting Integration" begin
    println("Testing Plotting Functionality...")

    # 1. Setup Simulation (Physics)
    r_model = VasicekProcess(:r, 0.3, 0.05, 0.02, 0.05, 1)

    drift_S(t, S) = 0.05 * S
    diff_S(t, S)  = 0.20 * S
    stock = GenericSDEProcess(:S, drift_S, diff_S, 100.0, [2])

    # We define risk factors for 2 shocks (Rate and Stock)
    mprs = [-0.1, -0.2]
    bond = NominalBondProcess(:P_N, r_model, mprs)

    config = MarketConfig(
        sims=50, T=1.0, dt=0.01, M=100,
        processes=[r_model, stock, bond]
    )

    world = build_world(config)

    # 2. Test Plotting Logic
    # mktempdir creates a temporary directory that is deleted after the block finishes
    mktempdir() do tmp_dir

        # A. Test Default Plot (All variables, Full time range)
        println("  > Generating default plots...")
        plot_simulation(world, output_dir=tmp_dir)

        # Assert files were created
        @test isfile(joinpath(tmp_dir, "r_plot.png"))
        @test isfile(joinpath(tmp_dir, "S_plot.png"))
        @test isfile(joinpath(tmp_dir, "P_N_plot.png"))

        # B. Test Time Slice (Zooming in)
        println("  > Generating time-sliced plots...")
        slice_dir = joinpath(tmp_dir, "slice")
        plot_simulation(world, time_range=(0.5, 1.0), output_dir=slice_dir)

        @test isfile(joinpath(slice_dir, "r_plot.png"))
        # We can't easily check the content of the PNG, but checking creation is usually sufficient for CI.

        # C. Test Specific Variable Selection
        println("  > Generating specific variable plots...")
        vars_dir = joinpath(tmp_dir, "vars")
        plot_simulation(world, vars=[:r], output_dir=vars_dir)

        @test isfile(joinpath(vars_dir, "r_plot.png"))
        @test !isfile(joinpath(vars_dir, "S_plot.png")) # Should NOT exist
    end

    println("Plotting tests passed.")
end