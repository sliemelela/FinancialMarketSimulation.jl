using Test
using Statistics

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
    T_mat = 1.0
    nom_bond = NominalBondProcess(:P_N, r_model, T_mat, mprs)
    infl_bond = InflationBondProcess(:P_I, r_model, pi_model, :CPI, T_mat, mprs)

    # 4. Simulation
    config = MarketConfig(
        sims=100, T=1.0, dt=0.01, M=100,
        processes=[r_model, pi_model, cpi_dummy, nom_bond, infl_bond],
    )

    world = build_world(config)

    # 5. Verify
    @test isapprox(mean(world.paths.P_N[:, end]), 1.0, atol=1e-8)
end

@testset "Bond Post-Maturity Behavior" begin
    # 1. Physics
    r_model = VasicekProcess(:r, 0.3, 0.05, 0.02, 0.05, 1)
    pi_model = VasicekProcess(:pi, 0.1, 0.02, 0.01, 0.02, 2)

    cpi_drift(t, Pi, pi_val) = Pi * pi_val
    cpi_diff(t, Pi, pi_val)  = 0.0
    cpi_model = GenericSDEProcess(:CPI, cpi_drift, cpi_diff, 100.0, Int[], [:pi])

    mprs = [-0.1, -0.1]

    # 2. Bonds that mature HALFWAY through the simulation
    T_mat = 0.5
    nom_bond = NominalBondProcess(:P_N, r_model, T_mat, mprs)
    infl_bond = InflationBondProcess(:P_I, r_model, pi_model, :CPI, T_mat, mprs)

    # 3. Simulation runs for T=1.0 (outlasting the bonds)
    config = MarketConfig(
        sims=100, T=1.0, dt=0.01, M=100,
        processes=[r_model, pi_model, cpi_model, nom_bond, infl_bond],
    )
    world = build_world(config)

    # 4. Find the maturity index
    mat_idx = floor(Int, T_mat / config.dt) + 1

    # --- Nominal Bond Test ---
    # At maturity, the price should be exactly 1.0.
    # It should remain 1.0 for the rest of the simulation.
    @test all(isapprox.(world.paths.P_N[:, mat_idx:end], 1.0, atol=1e-8))

    # --- Inflation Bond Test ---
    # At maturity, the price should equal the CPI at that exact moment.
    # It should remain frozen at that CPI value for the rest of the simulation.
    cpi_at_maturity = world.paths.CPI[:, mat_idx]

    for i in mat_idx:size(world.paths.P_I, 2)
        @test all(isapprox.(world.paths.P_I[:, i], cpi_at_maturity, atol=1e-8))
    end
end

@testset "Bond Returns" begin

    # Market parameters
    sims = 1000
    T = 10.0
    dt = 0.01

    # Creation of interest rate and inflation processes
    κ_r, σ_r, θ_r, r_0 = 0.3, 0.05, 0.02, 0.05
    κ_π, σ_π, θ_π, π_0 = 0.1, 0.02, 0.01, 0.02
    idx_r_shock = 1
    idx_pi_shock = 2
    r_model = VasicekProcess(:r, κ_r, θ_r, σ_r, r_0, idx_r_shock)
    pi_model = VasicekProcess(:pi, κ_π, θ_π, σ_π,π_0, idx_pi_shock)

    # Creation of the CPI process (driven by inflation)
    cpi_drift(t, Pi, pi_val) = Pi * pi_val
    cpi_diff(t, Pi, pi_val)  = 0.0
    cpi_model = GenericSDEProcess(
        :CPI, cpi_drift, cpi_diff, 100.0, Int[], [:pi]
    )

    # Risk premium vector and correlations
    ϕ_r, ϕ_π = -0.1, -0.1
    mprs = [ϕ_r, ϕ_π]
    ρ_rπ = 0.5
    ρ = [ 1.0  ρ_rπ;
          ρ_rπ  1.0 ]

    # Market price of risk calculation
    λ_r = -ϕ_r - ϕ_π * ρ_rπ
    λ_π = -ϕ_π - ϕ_r * ρ_rπ

    # Bond processes
    T_mat = 10.0
    nom_bond = NominalBondProcess(:P_N, r_model, T_mat, mprs)
    infl_bond = InflationBondProcess(:P_I, r_model, pi_model, :CPI, T_mat, mprs)

    # Nominal Bond process as generic process
    B_r(h) = (1 - exp(-κ_r * h)) / κ_r
    B_π(h) = (1 - exp(-κ_π * h)) / κ_π

    nom_drift(t, P_N, r_val) = (r_val - λ_r * σ_r * B_r(T_mat - t)) * P_N
    nom_diff(t, P_N, r_val) = -B_r(T_mat - t) * σ_r * P_N
    nom_bond_as_generic = GenericSDEProcess(
        :P_N_generic,
        nom_drift,
        nom_diff,
        1.0,
        [idx_r_shock],
        [:r]
    )

    # Inflation Bond process as generic process
    infl_drift(t, P_I, r_val, pi_val) = (r_val - λ_r * σ_r * B_r(T - t) + pi_val - λ_π * σ_π * B_π(T - t)) * P_I
    infl_diff(t, P_I, r_val, pi_val) = (-B_r(T - t) * σ_r, B_π(T - t) * σ_π) .* P_I
    infl_bond_as_generic = GenericSDEProcess(
        :P_I_generic,
        infl_drift,
        infl_diff,
        100.0,
        [idx_r_shock, idx_pi_shock],
        [:r, :pi]
    )

    # Return processes
    ret_nom = GrossReturnProcess(:R_N, :P_N)
    ret_infl = GrossReturnProcess(:R_I, :P_I)
    ret_nom_generic = GrossReturnProcess(:R_N_generic, :P_N_generic)
    ret_infl_generic = GrossReturnProcess(:R_I_generic, :P_I_generic)

    # Simulation configuration
    config = MarketConfig(
        sims=sims, T=T, dt=dt,
        processes=[r_model, pi_model, cpi_model, nom_bond, infl_bond, nom_bond_as_generic, infl_bond_as_generic, ret_nom, ret_infl, ret_nom_generic, ret_infl_generic],
        correlations=ρ
    )
    world = build_world(config)

    # Uncomment if you would like a visual check of the bond price and return paths
    # plot_simulation(world, vars=[:P_N, :P_I])
    # plot_simulation(world, vars=[:P_N_generic, :P_I_generic])
    # plot_simulation(world, vars=[:R_N, :R_I])
    # plot_simulation(world, vars=[:R_N_generic, :R_I_generic])
    # plot_simulation(world, vars=[:R_N, :R_N_generic])
    # plot_simulation(world, vars=[:R_I, :R_I_generic])

    # Verify that the returns from the direct bond processes match those from the generic processes
    @test isapprox(mean(world.paths.R_N), mean(world.paths.R_N_generic), atol=1e-4)
    @test isapprox(mean(world.paths.R_I), mean(world.paths.R_I_generic), atol=1e-4)
end