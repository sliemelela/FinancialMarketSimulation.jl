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

