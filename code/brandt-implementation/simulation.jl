using StaticArrays
using LinearAlgebra

## -- COOKBOOK (The Registry) --
# This maps strings in your YAML to Julia functions
# This is how you implement `drift_rule: "stock_drift_1"`
if !isdefined(Main, :DRIFT_RULES)
    const DRIFT_RULES = Dict(
        "stock_drift_1" => function(w::SimulationWorld, p::NamedTuple, n::Int)
            # p is the "Stock_S" block from the pantry
            r_prev = @view w.paths[:r][:, n - 1]
            return (p.a .* r_prev .+ p.λ_S * p.σ_S .- 0.5 * p.σ_S^2)
        end
    )
end

if !isdefined(Main, :DIFFUSION_RULES)
    const DIFFUSION_RULES = Dict(
        "stock_diffusion_1" => function(w::SimulationWorld, p::NamedTuple, shock_idxs::Vector{Int}, n::Int)
            # p is the "Stock_S" block from the pantry
            dW_S_n = @view w.brown_shocks[:, n, shock_idxs]
            return p.σ_S .* dW_S_n
        end
    )
end

# --- MATH HELPER ---
"""
Calculates the B(h) term from the Vasicek model.
h = time to maturity (s - t)
"""
function B_vasicek(κ::Float64, h::Float64)
    # Numerically stable version for κ ≈ 0
    if abs(κ) < 1e-8
        return h
    end
    return (1.0 - exp(-κ * h)) / κ
end

"""
Calculates the A(h) term for the Nominal Bond Price, ``A^N(h)``.
"""
function calculate_A_N(h::Float64, vp_r::NamedTuple, mp::NamedTuple,
    ρ_matrix::AbstractMatrix{<:Real}, shock_map::Dict{Symbol, Int})

    # Unpack parameters
    κ_r, r_bar, σ_r = vp_r.κ, vp_r.θ, vp_r.σ
    ρ_rπ = ρ_matrix[shock_map[:r], shock_map[:π]]
    ρ_rS = ρ_matrix[shock_map[:r], shock_map[:S]]
    ϕ_r, ϕ_π, ϕ_S = mp.ϕ_r, mp.ϕ_π, mp.ϕ_S

    # Calculate shorthand market price of risk from your formula
    λ_r = -ϕ_r - ρ_rπ * ϕ_π - ρ_rS * ϕ_S

    # B_r(h)
    B_r_h = B_vasicek(κ_r, h)

    # --- Implement the A^N(h) formula from the lemma ---

    # Grouping [B_r(h) - h] terms:
    # (r_bar - λ_r * σ_r / κ_r) * [B_r(h) - h]
    term1_drift = (r_bar - λ_r * σ_r / κ_r) * (B_r_h - h)

    # Volatility terms:
    # (σ_r² / 2κ_r³) * (2e^(-κ_r h) - 0.5e^(-2κ_r h) - 1.5)
    exp_kh = exp(-κ_r * h)
    exp_2kh = exp(-2.0 * κ_r * h)
    term2_vol_adj = (σ_r^2 / (2.0 * κ_r^3)) * (2.0 * exp_kh - 0.5 * exp_2kh - 1.5)

    # Ito/h term: (σ_r² / 2κ_r²) * h
    term3_ito = (σ_r^2 / (2.0 * κ_r^2)) * h

    return term1_drift + term2_vol_adj + term3_ito
end

"""
Calculates the A(h) term for the Inflation Bond Price, ``A^I(h)``.
"""
function calculate_A_I(h::Float64, vp_r::NamedTuple, vp_π::NamedTuple, mp::NamedTuple,
    ρ_matrix::AbstractMatrix{<:Real}, shock_map::Dict{Symbol, Int})

    # Unpack all parameters
    κ_r, σ_r, r_bar = vp_r.κ, vp_r.σ, vp_r.θ
    κ_π, σ_π, π_bar = vp_π.κ, vp_π.σ, vp_π.θ
    ρ_rπ = ρ_matrix[shock_map[:r], shock_map[:π]]
    ρ_rS = ρ_matrix[shock_map[:r], shock_map[:S]]
    ρ_πS = ρ_matrix[shock_map[:π], shock_map[:S]]
    ϕ_r, ϕ_π, ϕ_S = mp.ϕ_r, mp.ϕ_π, mp.ϕ_S

    # Calculate shorthand market prices of risk
    λ_r = -ϕ_r - ρ_rπ * ϕ_π - ρ_rS * ϕ_S
    λ_π = -ϕ_π - ρ_rπ * ϕ_r - ρ_πS * ϕ_S

    B_r_h = B_vasicek(κ_r, h)
    B_pi_h = B_vasicek(κ_π, h)

    # --- Implement the A^I(h) formula from the lemma ---
    # Term 1: Mean-reversion drift terms
    term1_drift = r_bar * (B_r_h - h) - π_bar * (B_pi_h - h)

    # Term 2: Volatility adjustment for r
    exp_kh_r = exp(-κ_r * h)
    exp_2kh_r = exp(-2.0 * κ_r * h)
    term2_vol_r = (σ_r^2 / (2.0 * κ_r^3)) * (2.0 * exp_kh_r - 0.5 * exp_2kh_r - 1.5)

    # Term 3: Volatility adjustment for π
    exp_kh_pi = exp(-κ_π * h)
    exp_2kh_pi = exp(-2.0 * κ_π * h)
    term3_vol_pi = (σ_π^2 / (2.0 * κ_π^3)) * (2.0 * exp_kh_pi - 0.5 * exp_2kh_pi - 1.5)

    # Term 4: Ito/h terms
    term4_ito = h * (σ_r^2 / (2.0 * κ_r^2) + σ_π^2 / (2.0 * κ_π^2))

    # Term 5: Market Price of Risk terms
    term5_mpr = -λ_r * (σ_r / κ_r) * (B_r_h - h) + λ_π * (σ_π / κ_π) * (B_pi_h - h)

    # Term 6: Correlation term
    κ_sum = κ_r + κ_π
    B_sum_h = (abs(κ_sum) < 1e-8) ? h : (1.0 - exp(-κ_sum * h)) / κ_sum
    term6_corr = (ρ_rπ * σ_r * σ_π / (κ_r * κ_π)) * (B_r_h + B_pi_h - B_sum_h - h)

    return term1_drift + term2_vol_r + term3_vol_pi + term4_ito + term5_mpr + term6_corr
end

# --- INTERNAL HELPER ---
function simulate_brownian_shocks_per_time(ρ::AbstractMatrix{<:Real}, sim::Int64)
    N_shocks     = size(ρ, 1)                # Number of brownian motions
    shocks       = randn(sim, N_shocks)      # Simulate N(0,1) independent shocks
    R            = cholesky(Symmetric(ρ)).U  # Cholesky decomposition s.t. R' R = ρ

    return shocks * R
end

function simulate_brownian_motions(ρ::AbstractMatrix{<:Real}, sim::Int64,
    amount_of_time_steps::Int64, sqrt_δt::Float64)

    N_shocks = size(ρ, 1)  # Number of brownian motions
    dW = Array{Float64}(undef, sim, amount_of_time_steps, N_shocks)
    for n in 1:amount_of_time_steps
        dW[:, n, :] = sqrt_δt * simulate_brownian_shocks_per_time(ρ, sim)
    end

    return dW
end

"""
    resolve_shock_indices(shock_input, shock_map::Dict{Symbol, Int})

Converts a YAML input (String or Vector{String}) into a Vector{Int} of indices.
For example, if you have `shock_map = Dict(:r => 1, :π => 2, :S => 3)`,
then:
- `resolve_shock_indices("r", shock_map)` returns `[1]`
- `resolve_shock_indices(["S", "π"], shock_map)` returns `[3, 2]`
"""
function resolve_shock_indices(shock_input, shock_map::Dict{Symbol, Int})

    # Normalize input to a Vector of Strings
    names = isa(shock_input, Vector) ? String.(shock_input) : [String(shock_input)]

    indices = Int[]
    for name in names
        sym = Symbol(name)
        if !haskey(shock_map, sym)
            error("Shock name '$name' found in Recipe but not in ShockOrder.")
        end
        push!(indices, shock_map[sym])
    end
    return indices
end

# --- "BUILDER" FUNCTIONS ---
function add_vasicek_process!(
    world::SimulationWorld,
    name::Symbol, # Name to store in paths Dict (e.g., :r or :π)
    params::NamedTuple, # The parameters (κ, θ, σ, initial_value)
    shock_indices::Vector{Int64} # Which column of dW to use?
)

    # Unpack parameters
    (; κ, θ, σ, initial_value) = params
    (; sim, M) = world.sim_params

    # Vasicek is a 1-factor model in this implementation.
    # We take the first index provided.
    if length(shock_indices) != 1
        @warn "Vasicek process '$name' received $(length(shock_indices)) shocks. Using the first one."
    end
    shock_index = shock_indices[1]

    # Preallocate path matrix
    path = Matrix{Float64}(undef, sim, world.amount_of_time_steps)
    path[:, 1] .= initial_value

    # Obtain relevant shocks
    dW = @view world.brown_shocks[:, :, shock_index]

    # Run simulation
    @views for n in 2:world.amount_of_time_steps
        path_prev = path[:, n - 1]
        path[:, n] .= path_prev .+ κ .* (θ .- path_prev) * world.δt .+ σ .* dW[:, n]
    end

    # Store the result in the world
    world.paths[name] = path
    return world
end

function add_lognormal_process!(
    world::SimulationWorld,
    name::Symbol, # Name to store in paths Dict (e.g., :S)
    params::NamedTuple, # The parameters (μ_func, σ, initial_value)
    shock_idxs::Vector{Int64}, # Which column of dW to use?
    drift_formula::Function, # The formula for drift
    diffusion_formula::Function # The formula for diffusion
)

    # Unpack parameters
    sim = world.sim_params.sim

    # Preallocate path matrix
    path = Matrix{Float64}(undef, sim, world.amount_of_time_steps)
    path[:, 1] .= params.initial_value

    @views for n in 2:world.amount_of_time_steps

        # Previous value
        path_prev = path[:, n - 1]

        # Compute drift and diffusion
        log_drift = drift_formula(world, params, n)
        log_diffusion = diffusion_formula(world, params, shock_idxs, n)

        # Update lognormal process
        G = exp.(log_drift .* world.δt .+ log_diffusion)
        path[:, n] .= path_prev .* G
    end

    # Store the result in the world
    world.paths[name] = path
    return world

end

function add_CPI_path!(
    world::SimulationWorld,
    name::Symbol
)

    # Unpack parameters
    sim = world.sim_params.sim
    amount_of_time_steps = world.amount_of_time_steps
    δt = world.δt

    # Get dependency path (:π)
    if !haskey(world.paths, :π)
        error("SimulationWorld is missing the :π path. Run `add_vasicek_process!` for :π first.")
    end
    pi_path = world.paths[:π]

    # Preallocate CPI path
    Π_path = Matrix{Float64}(undef, sim, amount_of_time_steps)
    Π_path[:, 1] .= 1.0  # Initialize CPI at 1.0

    # Simulate the path
    @inbounds @views for n = 2:amount_of_time_steps
        # Get inflation from the *start* of the period [t_n-1, t_n]
        pi_prev = pi_path[:, n - 1]

        # Calculate growth factor
        G_Π = exp.(pi_prev .* δt)

        # Update CPI
        Π_path[:, n] .= Π_path[:, n - 1] .* G_Π
    end

    # Store the result and return
    world.paths[name] = Π_path
    return world # Allow chaining
end


function add_nominal_bond_process!(
    world::SimulationWorld,
    name::Symbol, # Name to store in paths Dict (e.g., :P_N)
    vp_r::NamedTuple, # The Vasicek parameters
    mp::NamedTuple, # The Market Price of Risk parameters
    ρ_matrix::AbstractMatrix{<:Real},
    shock_map::Dict{Symbol, Int}
)

    # Unpack parameters
    (; sim, T, M) = world.sim_params
    amount_of_time_steps = world.amount_of_time_steps
    δt = world.δt

    # Obtain the simulated short rate
    if !haskey(world.paths, :r)
        error("SimulationWorld is missing the :r path. Run add_vasicek_process! first.")
    end
    r_path = world.paths[:r]

    # Preallocate path matrix
    P_N_path = Matrix{Float64}(undef, sim, amount_of_time_steps)

    # Pre-calculate A(h) and B(h) for all maturities
    h_vec = [T - (n - 1) * δt for n in 1:amount_of_time_steps]
    A_N_vec = [calculate_A_N(h, vp_r, mp, ρ_matrix, shock_map) for h in h_vec]
    B_r_vec = [B_vasicek(vp_r.κ, h) for h in h_vec]

    @views for n in 1:amount_of_time_steps
        A_N = A_N_vec[n]
        B_r = B_r_vec[n]
        r_t = r_path[:, n]

        # Calculate bond price at time step n
        P_N_path[:, n] .= exp.(A_N .- B_r .* r_t)
    end

    # Store the result in the world
    world.paths[name] = P_N_path
    return world
end

function add_inflation_bond_process!(
    world::SimulationWorld,
    name::Symbol, # Name to store in paths Dict (e.g., :P_I)
    vp_r::NamedTuple, # The Vasicek parameters for r
    vp_π::NamedTuple, # The Vasicek parameters for π
    mp::NamedTuple, # The Market Price of Risk parameters
    ρ_matrix::AbstractMatrix{<:Real},
    shock_map::Dict{Symbol, Int}
)

    # Unpack parameters
    (; sim, T, M) = world.sim_params
    amount_of_time_steps = world.amount_of_time_steps
    δt = world.δt

    # Get the simulated state variable paths
    if !haskey(world.paths, :r) || !haskey(world.paths, :π) || !haskey(world.paths, :Π)
        error("World is missing :r, :π, or :Π paths. Run simulation functions first.")
    end
    r_path = world.paths[:r]
    π_path = world.paths[:π]
    Π_path = world.paths[:Π]

    # Preallocate path matrix
    P_I_path = Matrix{Float64}(undef, sim, amount_of_time_steps)

    # Pre-calculate A(h), B_r(h), and B_π(h) for all maturities
    h_vec = [T - (n - 1) * δt for n in 1:amount_of_time_steps]
    A_I_vec = [calculate_A_I(h, vp_r, vp_π, mp, ρ_matrix, shock_map) for h in h_vec]
    B_r_vec = [B_vasicek(vp_r.κ, h) for h in h_vec]
    B_pi_vec = [B_vasicek(vp_π.κ, h) for h in h_vec]

    @views for n in 1:amount_of_time_steps
        A_I = A_I_vec[n]
        B_r = B_r_vec[n]
        B_pi = B_pi_vec[n]
        r_t = r_path[:, n]
        π_t = π_path[:, n]
        Π_t = Π_path[:, n]

        # Calculate bond price at time step n
        P_I_path[:, n] .= Π_t .* exp.(A_I .- B_r .* r_t .+ B_pi .* π_t)
    end

    # Store the result in the world
    world.paths[name] = P_I_path
    return world
end

# -- RETURN CALCULATION ---
"""
    add_excess_simple_return_path!(
        world::SimulationWorld,
        return_name::Symbol,
        asset_price_name::Symbol,
        risk_free_rate_name::Symbol = :r
    )

Calculates the one-period **simple excess return** path for a given asset
and adds it to the simulation world.

The calculation is:
`R_asset[n] = P_asset[n] / P_asset[n-1]`
`R_riskfree[n] = 1.0 + r[n-1] * δt`
`R_excess[n] = R_asset[n] - R_riskfree[n]`

# Arguments
- `world::SimulationWorld`: The simulation environment.
- `return_name::Symbol`: The name to store the resulting return series
  (e.g., `:Re_Stock`).
- `asset_price_name::Symbol`: The name of the asset's price path
  already in the world (e.g., `:S`, `:P_N`, or `:P_I`).
- `risk_free_rate_name::Symbol`: The name of the short-rate path
  (defaults to `:r`).
"""
function add_excess_simple_return_path!(
    world::SimulationWorld,
    return_name::Symbol,
    asset_price_name::Symbol,
    risk_free_rate_name::Symbol = :r
)
    # Get parameters and paths
    sim = world.sim_params.sim
    amount_of_time_steps = world.amount_of_time_steps
    δt = world.δt

    # Check that paths exist
    if !haskey(world.paths, asset_price_name)
        error("Asset path ':$asset_price_name' not found in SimulationWorld.")
    end
    if !haskey(world.paths, risk_free_rate_name)
        error("Risk-free rate path ':$risk_free_rate_name' not found in SimulationWorld.")
    end

    # Get paths
    price_path = world.paths[asset_price_name]
    rate_path = world.paths[risk_free_rate_name]

    # Preallocate the excess return path
    Re_path = Matrix{Float64}(undef, sim, amount_of_time_steps)
    Re_path[:, 1] .= 0.0  # No return at t=0

    # Calculate returns
    @inbounds @views for n = 2:amount_of_time_steps

        # Get relevant prices
        price_now = price_path[:, n]
        price_prev = price_path[:, n-1]
        rate_prev = rate_path[:, n-1]

        # Calculate simple returns
        R_asset = price_now ./ price_prev
        R_riskfree = 1.0 .+ rate_prev .* δt

        # Simple excess return
        Re_path[:, n] .= R_asset .- R_riskfree
    end

    # Store and return
    world.paths[return_name] = Re_path
    return world
end


function add_simple_return_path!(
    world::SimulationWorld,
    return_name::Symbol,
    asset_price_name::Symbol,
)
    # Get parameters and paths
    sim = world.sim_params.sim
    amount_of_time_steps = world.amount_of_time_steps
    δt = world.δt

    if !haskey(world.paths, asset_price_name)
        error("Asset path ':$asset_price_name' not found in SimulationWorld.")
    end

    price_path = world.paths[asset_price_name]

    # Preallocate the return path
    R_path = Matrix{Float64}(undef, sim, amount_of_time_steps)
    R_path[:, 1] .= 1.0  # No return at t=0


    # Calculate returns
    @inbounds @views for n = 2:amount_of_time_steps

        # Get relevant prices
        price_now = price_path[:, n]
        price_prev = price_path[:, n-1]

        # Calculate simple returns
        R_asset = price_now ./ price_prev

        # Simple excess return
        R_path[:, n] .= R_asset
    end

    # Store and return
    world.paths[return_name] = R_path
    return world
end

"""
    build_correlation_matrix_and_map(config_set::Dict)

Parses the config to build the (N x N) correlation matrix
and the mapping from shock names to matrix indices.

It iterates through all unique pairs of names in `ShockOrder`,
constructs the key it expects to find in `Parameters.Correlations`
(e.g., "ρ_rπ"), and populates the matrix.
"""
function build_correlation_matrix_and_map(config_set::Dict)

    # Get data from config
    shock_order_names = config_set["ShockOrder"]
    corr_params = config_set["Parameters"]["Correlations"]
    n_shocks = length(shock_order_names)

    # Build the matrix
    ρ_matrix = Matrix{Float64}(I, n_shocks, n_shocks) # Start with Identity

    # Iterate over all unique pairs (i, j) where i < j
    for i in 1:n_shocks, j in (i + 1):n_shocks

        # Get the names for this pair, e.g., "r" and "π"
        name1 = shock_order_names[i]
        name2 = shock_order_names[j]

        # Construct the keys we expect to find
        key1 = "ρ_" * name1 * name2  # e.g., "ρ_rπ"
        key2 = "ρ_" * name2 * name1  # e.g., "ρ_πr"

        # Find the value in the YAML
        value = nothing
        if haskey(corr_params, key1)
            value = corr_params[key1]
        elseif haskey(corr_params, key2)
            value = corr_params[key2]
        else
            # This is a missing correlation. We assume 0.
            @warn "Correlation '$key1' or '$key2' not found in parameters.yaml. Defaulting to 0.0."
            value = 0.0
        end

        # Handle sensitivity list: If the value is a dictionary (like `{sensitivity: [0.4, 0.5]}`)
        # then we have an error because this should have been flattened already.
        if (value isa Dict)
            if haskey(value, "sensitivity")
                error("Correlation parameter '$key1' has not been flattened by the sensitivity loader.")
            end
        end

        # Set the value in the matrix
        ρ_matrix[i, j] = ρ_matrix[j, i] = value
    end

    @assert isposdef(ρ_matrix) "Correlation matrix is not positive definite. Check YAML values."

    # Create the Symbol-based map for the rest of the code
    symbol_shock_map = Dict(Symbol(name) => i for (i, name) in enumerate(shock_order_names))

    return ρ_matrix, symbol_shock_map
end

## -- RECIPE EXECUTOR --
"""
    build_world_from_recipe(config_set::Dict)
Builds a `SimulationWorld` from a given configuration set (Dict)
by executing the "Processes" and "Returns" recipes.
"""
function build_world_from_recipe(config_set::Dict)

    # Get Data
    sim_params = load_struct(SimulationParams, config_set["SimulationParams"])
    pantry = config_set["Parameters"]
    recipe_list = config_set["Processes"]

    # Build correlation matrix and shock map
    ρ_matrix, shock_map = build_correlation_matrix_and_map(config_set)

    # Create the World
    world = SimulationWorld(ρ_matrix, sim_params)
    println("SimulationWorld created with $(size(ρ_matrix, 1)) shocks.")

    # --- Execute "Processes" Recipe ---
    println("Building world from recipe...")
    for step in recipe_list
        name = Symbol(step["name"])
        type = step["type"]
        println("... adding process: $name (type: $type)")

        # Find relevant shock indices
        current_shock_indices = Int[]
        if haskey(step, "shocks")
            current_shock_indices = resolve_shock_indices(step["shocks"], shock_map)
        end

        # Merge parameters from pantry
        if isa(step["params_from"], String)
            params = NamedTuple((Symbol(key), value) for (key, value) in pantry[step["params_from"]])
        elseif isa(step["params_from"], Vector)
            merged_params = Dict()
            for param_key in step["params_from"]
                merge!(merged_params, pantry[param_key])
            end
            params = NamedTuple((Symbol(key), value) for (key, value) in merged_params)
        else
            error("Invalid type for 'params_from': $(typeof(step["params_from"]))")
        end

        # -- Dispatch to appropriate builder --
        if type == "vasicek"
            add_vasicek_process!(world, name, params, current_shock_indices)

        elseif type == "cpi"
            add_CPI_path!(world, name)

        elseif type == "lognormal"
            drift_func = DRIFT_RULES[step["drift_rule"]]
            diffusion_func = DIFFUSION_RULES[step["diffusion_rule"]]

            add_lognormal_process!(world, name, params,
                current_shock_indices, drift_func, diffusion_func)

        elseif type == "nominal_bond"
            param_keys = step["params_from"] # ["Vasicek_r", "MPR"]

            # "Grabs" the required pantry blocks
            vp_r = NamedTuple((Symbol(key),value) for (key,value) in pantry["Vasicek_r"])
            mp = NamedTuple((Symbol(key),value) for (key,value) in pantry["MPR"])

            # Passes the matrix and map to the (refactored) builder
            add_nominal_bond_process!(world, name, vp_r, mp, ρ_matrix, shock_map)

        elseif type == "inflation_bond"
            param_keys = step["params_from"] # ["Vasicek_r", "Vasicek_pi", "MPR"]

            vp_r = NamedTuple((Symbol(key),value) for (key,value) in pantry["Vasicek_r"])
            vp_pi = NamedTuple((Symbol(key),value) for (key,value) in pantry["Vasicek_pi"])
            mp = NamedTuple((Symbol(key),value) for (key,value) in pantry["MPR"])

            # Passes the matrix and map
            add_inflation_bond_process!(world, name, vp_r, vp_pi, mp, ρ_matrix, shock_map)

        else
            error("Unknown process type: $type")
        end
    end

    # Execute "Returns" Recipe
    for ret in config_set["ExcessReturns"]
        name = Symbol(ret["name"])
        println("... adding return: $name")
        add_excess_simple_return_path!(world, name, Symbol(ret["asset_path"]))
    end

    for ret in config_set["Returns"]
        name = Symbol(ret["name"])
        println("... adding return: $name")
        add_simple_return_path!(world, name, Symbol(ret["asset_path"]))
    end

    println("Simulation complete.")
    return world
end