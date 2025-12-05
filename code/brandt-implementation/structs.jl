# --- Structs loaded from YAML ---

"""
Defines the grid for the simulation.
(Loaded from YAML)
"""
Base.@kwdef struct SimulationParams
    sim::Int64
    T::Float64
    M::Int64
end

"""
Defines the parameters for the portfolio solver.
(Loaded from YAML)
"""
Base.@kwdef struct SolverParams
    asset_names::Vector{String}
    state_names::Vector{String}
    W_grid::Vector{Float64}
    poly_order::Int
    p_income::Float64
    O_t_real_path::Union{String, Nothing}
    γ::Float64
end


# --- Structs used by the code ---

"""
Holds all simulated data.
(This struct is created by `simulation.jl`, not loaded)
"""
Base.@kwdef struct SimulationWorld
    sim_params::SimulationParams
    brown_shocks::Array{Float64, 3} # The (sim, M+1, N_shocks) array of brownian motions
    paths::Dict{Symbol, Matrix{Float64}} # Stores results, .e.g. :r, :π, :S, :P_N, :P_I

    # Useful data
    δt::Float64
    amount_of_time_steps::Int64

    function SimulationWorld(ρ::AbstractMatrix{<:Real}, sim_params::SimulationParams)

        # Unpacking and Calculation of parameters
        (; sim, T, M) = sim_params
        amount_of_time_steps = M + 1
        δt = T / M
        sqrt_δt = sqrt(δt)

        # Simulation of brownian motions
        dW = simulate_brownian_motions(ρ, sim, amount_of_time_steps, sqrt_δt)

        # Initialize empty paths Dict
        paths = Dict{Symbol, Matrix{Float64}}()

        new(sim_params, dW, paths, δt, amount_of_time_steps)
    end
end


"""
A container for the utility function's derivatives.
(This struct is created by `utils.jl`, not loaded)
"""
Base.@kwdef struct UtilityFunctions
    u::Function
    first_derivative::Function
    second_derivative::Function
    inverse::Function
end
