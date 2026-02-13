"""
    simulate_correlated_shocks(sims, steps, dt, corr_matrix)

Generates correlated Brownian motions dW using Cholesky decomposition.
Dimensions: (sims, steps, n_shocks)
"""
function simulate_correlated_shocks(sims::Int, steps::Int, dt::Float64, ρ::AbstractMatrix)
    n_shocks = size(ρ, 1)
    sqrt_dt = sqrt(dt)

    # 1. Decompose Correlation Matrix (ρ = L * L')
    # We use Cholesky. If ρ is not pos-def, this throws an error (good).
    L = cholesky(Symmetric(ρ)).L

    # 2. Allocate Output
    dW = Array{Float64}(undef, sims, steps, n_shocks)

    # 3. Generate time steps
    # We do this step-by-step to keep memory usage for the intermediate 'Z' low
    for n in 1:steps
        Z = randn(sims, n_shocks) # Independent N(0,1)

        # Correlate: W = Z * L' (or L * Z' depending on layout)
        # We want (sims x shocks) * (shocks x shocks) -> (sims x shocks)
        # Since L is LowerTriangular, we use Right-Multiplication by L' (Upper)
        dW[:, n, :] .= (Z * L') .* sqrt_dt
    end

    return dW
end

function build_world(config::MarketConfig)

    # Preallocation
    N_sims = config.sims
    N_steps = config.M + 1

    data_pairs = (
        p.name => zeros(Float64, N_sims, N_steps)
        for p in config.processes
    )
    paths = ComponentArray(; data_pairs...)

    # --- 2. GENERATE SHOCKS ---
    # Determine required number of shocks
    max_idx = maximum(maximum(shock_indices(p)) for p in config.processes)

    # Handle Correlation Matrix
    ρ = config.correlations
    if isempty(ρ)
        # Default: Identity matrix of sufficient size
        ρ = Matrix{Float64}(I, max_idx, max_idx)
    elseif size(ρ, 1) < max_idx
        error("Correlation matrix size ($(size(ρ))) is smaller than required shocks ($max_idx).")
    end

    # Generate the 3D Cube of Shocks
    brownian_shocks = simulate_correlated_shocks(N_sims, N_steps, config.dt, ρ)

    world = SimulationWorld(paths, brownian_shocks, config)


    # --- 3. RUN SIMULATION ---
    for p in config.processes
        target_path = @view world.paths[p.name]
        simulate!(target_path, p, world)
    end

    return world
end