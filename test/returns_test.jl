
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

    # Gross return of Stock
    gross_ret_S = GrossReturnProcess(:G_S, :S)

    # 3. Simulation
    config = MarketConfig(
        sims=1000, T=1.0, dt=0.01, M=100,
        processes=[r_model, stock, ret_S, ex_ret_S, gross_ret_S] # Note: Order matters for dependencies
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

    # Expected Gross Return ≈ 1 + drift * dt = 1 + 0.001 = 1.001
    avg_G = mean(world.paths.G_S)
    println("Avg Gross Return (dt=0.01): $avg_G")
    @test isapprox(avg_G, 1.001, atol=0.0005)

    # Check that Gross Return is exactly 1 + Simple Return (since it's a direct transformation)
    @test isapprox(avg_G, 1.0 + avg_R, atol=1e-10)
end