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

