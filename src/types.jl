using ComponentArrays

# --- Abstract Types ---
abstract type AbstractMarketProcess end

"""
    shock_indices(p::AbstractMarketProcess) -> Vector{Int}

Interface method to retrieve the shock indices used by a process.
Default implementation assumes the struct has a field `shock_idxs`.
Future developers can override this if their struct uses a different field name.
"""
function shock_indices(p::AbstractMarketProcess)
    # Fallback: Try to get the standard field
    if hasfield(typeof(p), :shock_idxs)
        return p.shock_idxs
    else
        error("Process $(typeof(p)) does not have a `shock_idxs` field. " *
              "Please implement `shock_indices(p)` for this type.")
    end
end


# --- Process Definitions ---

"""
    GenericSDEProcess

A flexible SDE process: dX_t = μ(t, X_t)dt + σ(t, X_t)dW_t.
The drift and diffusion are standard Julia functions provided by the user.
We parameterize F1 and F2 to ensure high-performance inlining of the functions.
"""
struct GenericSDEProcess{F1, F2} <: AbstractMarketProcess
    name::Symbol
    drift::F1      # Signature: (t, x) -> Float64
    diffusion::F2  # Signature: (t, x) -> Float64
    initial_value::Float64
    shock_idxs::Vector{Int}
end

"""
    VasicekProcess

A specialized Ornstein-Uhlenbeck process: dr_t = κ(θ - r_t)dt + σ dW_t
"""
struct VasicekProcess <: AbstractMarketProcess
    name::Symbol
    κ::Float64
    θ::Float64
    σ::Float64
    initial_value::Float64
    shock_idx::Int64
end
shock_indices(p::VasicekProcess) = p.shock_idx


# --- Configuration ---

"""
    MarketConfig

The master configuration for the simulation.
"""
Base.@kwdef struct MarketConfig
    sims::Int
    T::Float64
    dt::Float64
    M::Int
    processes::Vector{AbstractMarketProcess}

    # Optional correlation matrix (defaults to Identity if missing)
    correlations::Matrix{Float64} = Matrix{Float64}(I, 0, 0)
end

# --- The World ---

"""
    SimulationWorld

Holds the simulated paths in a memory-efficient ComponentArray.
T is the element type (Float64).
C is the specific ComponentArray type (generated at runtime).
"""
struct SimulationWorld{T, C <: ComponentArray}
    paths::C
    brownian_shocks::Array{T, 3}
    config::MarketConfig
end