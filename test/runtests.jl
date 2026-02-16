using FinancialMarketSimulation
using Statistics
using Test

@testset "Correlated & Multi-Shock Simulation" begin

    # 1. Define Physics
    # We want 2 shocks with correlation -0.8 (e.g., Stock vs Volatility)
    ρ = [ 1.0 -0.8;
         -0.8  1.0]

    # 2. Define Processes
    # Stock uses Shock 1: dS = μS dt + σS dW_1
    drift_S(t, x) = 0.05 * x
    diff_S(t, x)  = 0.20 * x
    stock = GenericSDEProcess(:S, drift_S, diff_S, 100.0, [1])

    # "Noise Tracer" - A dummy process to just record Shock 1 accumulation
    # dN = 0 dt + 1 dW_1 (Just creates a Brownian Motion path)
    noise_drift(t, x) = 0.0
    noise_diff(t, x)  = 1.0
    shock_path = GenericSDEProcess(:W1, noise_drift, noise_diff, 0.0, [1])

    # 3. Build World
    config = MarketConfig(
        sims = 5000, T = 1.0, dt = 0.01, M = 100,
        processes = [stock, shock_path],
        correlations = ρ
    )

    world = build_world(config)

    # 4. Accessing Data
    # A. Via the computed path (User-friendly)
    println("Final value of recorded Brownian Motion: ", mean(world.paths.W1[:, end]))

    # B. Via the raw shocks (Low-level)
    # world.brownian_shocks is (sims, steps, shocks)
    raw_dW1 = world.brownian_shocks[:, :, 1]

    @test size(raw_dW1) == (5000, 101)
end

@testset "Mixed Process Simulation" begin

    # 1. Define a Generic Process (Stock)
    # dS = 0.05 S dt + 0.2 S dW_1
    drift_S(t, S) = 0.05 * S
    diff_S(t, S)  = 0.20 * S
    stock = GenericSDEProcess(:S, drift_S, diff_S, 100.0, [1])

    # 2. Define a Vasicek Process (Rate)
    rate = VasicekProcess(
        :r,    # name
        0.3,   # κ (speed)
        0.05,  # θ (mean)
        0.02,  # σ (vol)
        0.05,  # initial
        2      # shock_idx (uses 2nd column of dW)
    )

    # 3. Put them together
    config = MarketConfig(
        sims = 1000, T = 1.0, dt = 0.01, M = 100,
        processes = [stock, rate] # <--- Mixed types in one vector!
    )

    world = build_world(config)

    # 4. It just works
    println("Mean Stock: ", mean(world.paths.S[:, end]))
    println("Mean Rate:  ", mean(world.paths.r[:, end]))
end

@testset "Brownian Motion Correlation Diagnostics" begin

    # 1. Define the Physics (Target Correlations)
    rho = [ 1.0  0.8 -0.5;
            0.8  1.0  0.0;
           -0.5  0.0  1.0]

    # 2. Define "Dummy" Processes
    null_drift(t, x) = 0.0
    null_diff(t, x)  = 0.0

    p1 = GenericSDEProcess(:W1, null_drift, null_diff, 0.0, [1])
    p2 = GenericSDEProcess(:W2, null_drift, null_diff, 0.0, [2])
    p3 = GenericSDEProcess(:W3, null_drift, null_diff, 0.0, [3])

    # 3. Configure the Simulation
    # We use a high number of simulations (50k) to ensure statistical significance.
    config = MarketConfig(
        sims = 50_000,
        T = 1.0,
        dt = 1.0, # One big step is enough to check correlation
        M = 1,
        processes = [p1, p2, p3],
        correlations = rho
    )

    # 4. Build World (This runs the Cholesky generation)
    world = build_world(config)

    # 5. Extract the Generated Shocks
    # Structure is (sims, steps, shocks)
    # We inspect the data at step index 2 (corresponds to the first simulated step n=1 in the loop)
    # Dimensions: (50000, 3)
    generated_data = world.brownian_shocks[:, 2, :]

    println("Generated Data Size: ", size(generated_data))

    # 6. Calculate Empirical Correlations
    empirical_corr = cor(generated_data)
    println("Target Correlation:\n$rho")
    println("Empirical Correlation:\n$empirical_corr")

    # 7. Assertions
    tol = 0.02 # Tolerance for Monte Carlo noise

    # Check off-diagonals
    @test isapprox(empirical_corr[1, 2], rho[1, 2], atol=tol) # 0.8
    @test isapprox(empirical_corr[1, 3], rho[1, 3], atol=tol) # -0.5
    @test isapprox(empirical_corr[2, 3], rho[2, 3], atol=tol) # 0.0

    # Check diagonal (should be exactly 1.0)
    @test isapprox(empirical_corr[1, 1], 1.0, atol=1e-10)
    @test isapprox(empirical_corr[2, 2], 1.0, atol=1e-10)
    @test isapprox(empirical_corr[3, 3], 1.0, atol=1e-10)
end


using FinancialMarketSimulation
using Statistics
using Test

@testset "Vectorized Risk Factors" begin
    # Physics (Rates)
    r_model = VasicekProcess(:r, 0.3, 0.05, 0.02, 0.05, 1)
    pi_model = VasicekProcess(:pi, 0.1, 0.02, 0.01, 0.02, 2)

    cpi_dummy = GenericSDEProcess(:CPI, (t,x)->0.0, (t,x)->0.0, 1.0, [2])

    # Risk Factors (Vector)
    # Index 1: Risk premium for Rate Shock
    # Index 2: Risk premium for Inflation Shock
    # Index 3: Risk premium for Stock Shock (if we had one)
    mprs = [-0.1, -0.1, -0.2]

    # 3. Derivatives (Bonds)
    nom_bond = NominalBondProcess(:P_N, r_model, mprs)
    infl_bond = InflationBondProcess(:P_I, r_model, pi_model, :CPI, mprs   )

    # 4. Simulation
    config = MarketConfig(
        sims=100, T=1.0, dt=0.01, M=100,
        processes=[r_model, pi_model, cpi_dummy, nom_bond, infl_bond],
    )

    world = build_world(config)

    # 5. Verify
    @test isapprox(mean(world.paths.P_N[:, end]), 1.0, atol=1e-8)
end

@testset "Generic Process with Dependencies" begin
    # 1. Physics (Inflation Rate)
    # Vasicek: dr = ...
    pi_model = VasicekProcess(:pi, 0.1, 0.02, 0.01, 0.02, 1)

    # 2. CPI Index (Generic, Dependent)
    # Formula: dΠ = Π * π * dt  (No noise, purely driven by π)

    # Signature: (t, Current_State, Dependency_1)
    cpi_drift(t, Pi, pi_val) = Pi * pi_val
    cpi_diff(t, Pi, pi_val)  = 0.0

    cpi_model = GenericSDEProcess(
        :CPI,        # Name
        cpi_drift,   # Drift function
        cpi_diff,    # Diffusion function
        100.0,       # Initial Value
        Int[],       # No intrinsic shocks (it's an ODE driven by pi)
        [:pi]        # <--- DEPENDENCY: Reads the :pi path
    )

    # 3. Run
    # Order matters: :pi must be simulated before :CPI
    config = MarketConfig(
        sims=100, T=1.0, dt=0.01, M=100,
        processes=[pi_model, cpi_model]
    )

    world = build_world(config)

    # 4. Verify
    # Approx check: CPI ≈ 100 * exp(avg_pi * T)
    final_cpi = mean(world.paths.CPI[:, end])
    avg_pi    = mean(world.paths.pi)
    expected_cpi = 100.0 * exp(avg_pi * config.T)

    println("Avg Inflation: $avg_pi")
    println("Final CPI:     $final_cpi")
    println("Expected CPI:  $expected_cpi")

    # Should definitely be higher than 100 if inflation is positive
    @test final_cpi > 100.0
    @test isapprox(final_cpi, expected_cpi, atol=0.01) # Allow some tolerance due to discretization
end

@testset "Returns Calculation" begin
    # 1. Physics
    r_model = VasicekProcess(:r, 0.05, 0.05, 0.0, 0.05, 1) # Constant rate r=5%

    # Stock with 10% drift
    drift_S(t, S) = 0.10 * S
    diff_S(t, S)  = 0.20 * S
    stock = GenericSDEProcess(:S, drift_S, diff_S, 100.0, [2])

    # 2. Derived Return Processes
    # Simple Return of Stock
    ret_S = SimpleReturnProcess(:R_S, :S)

    # Excess Return of Stock over Rate
    ex_ret_S = ExcessReturnProcess(:Re_S, :S, :r)

    # 3. Simulation
    config = MarketConfig(
        sims=1000, T=1.0, dt=0.01, M=100,
        processes=[r_model, stock, ret_S, ex_ret_S] # Note: Order matters for dependencies
    )

    world = build_world(config)

    # 4. Verify
    # Expected Simple Return ≈ drift * dt = 0.10 * 0.01 = 0.001
    avg_R = mean(world.paths.R_S)
    println("Avg Simple Return (dt=0.01): $avg_R")
    @test isapprox(avg_R, 0.001, atol=0.0005)

    # Expected Excess Return ≈ (drift - r) * dt = (0.10 - 0.05) * 0.01 = 0.0005
    avg_Re = mean(world.paths.Re_S)
    println("Avg Excess Return (dt=0.01): $avg_Re")
    @test isapprox(avg_Re, 0.0005, atol=0.0005)
end

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