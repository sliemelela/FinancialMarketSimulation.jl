
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
    infl_bond = InflationBondProcess(:P_I, r_model, pi_model, :CPI, mprs)

    # 4. Simulation
    config = MarketConfig(
        sims=100, T=1.0, dt=0.01, M=100,
        processes=[r_model, pi_model, cpi_dummy, nom_bond, infl_bond],
    )

    world = build_world(config)

    # 5. Verify
    @test isapprox(mean(world.paths.P_N[:, end]), 1.0, atol=1e-8)
end