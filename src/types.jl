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
    GenericSDEProcess{F1, F2}

A flexible SDE process: dX_t = μ(t, X_t, deps...)dt + σ(t, X_t, deps...) ⋅ dW_t.
Now supports `dependencies`: a list of other process names (symbols) whose values
will be passed to the drift/diffusion functions.
"""
struct GenericSDEProcess{F1, F2} <: AbstractMarketProcess
    name::Symbol
    drift::F1
    diffusion::F2
    initial_value::Float64
    shock_idxs::Vector{Int}
    "List of dependencies (e.g. [:r, :pi])"
    dependencies::Vector{Symbol}
end

# Backward-compatible constructor (defaults to no dependencies)
function GenericSDEProcess(name, drift, diff, val, shocks)
    return GenericSDEProcess(name, drift, diff, val, shocks, Symbol[])
end

"""
    VasicekProcess

A specialized Ornstein-Uhlenbeck process: dr_t = κ(θ - r_t)dt + σ dW_t

# Fields
$(TYPEDFIELDS)
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

"""
    NominalBondProcess{R}

Holds reference to the Rate Process and a vector of Market Risk Factors.

# Fields
$(TYPEDFIELDS)
"""
struct NominalBondProcess{R <: AbstractMarketProcess} <: AbstractMarketProcess
    name::Symbol
    rate_process::R

    # Vector of Market Prices of Risk [ϕ_1, ϕ_2, ..., ϕ_N]
    # Corresponding to the shock indices 1, 2, ..., N
    market_risk_factors::Vector{Float64}
end
shock_indices(::NominalBondProcess) = Int[]

"""
    InflationBondProcess{R, I}

Holds reference to Rate, Inflation, and a vector of Market Risk Factors.
"""
struct InflationBondProcess{R <: AbstractMarketProcess, I <: AbstractMarketProcess} <: AbstractMarketProcess
    name::Symbol
    rate_process::R
    infl_process::I
    cpi_name::Symbol

    market_risk_factors::Vector{Float64}
end
shock_indices(::InflationBondProcess) = Int[]

# --- Return Processes ---

"""
    SimpleReturnProcess

Calculates the simple (discrete) return of an asset:
R_t = (S_t / S_{t-1}) - 1
"""
struct SimpleReturnProcess <: AbstractMarketProcess
    name::Symbol
    asset_name::Symbol  # The asset to track (e.g., :S)
end
shock_indices(::SimpleReturnProcess) = Int[] # Deterministic transformation

"""
    GrossReturnProcess

Calculates the gross return (price relative) of an asset:
G_t = S_t / S_{t-1}
"""
struct GrossReturnProcess <: AbstractMarketProcess
    name::Symbol
    asset_name::Symbol
end
shock_indices(::GrossReturnProcess) = Int[]

"""
    ExcessReturnProcess

Calculates the excess return of an asset over a risk-free rate:
Re_t = (S_t / S_{t-1} - 1) - (r_{t-1} * dt)
"""
struct ExcessReturnProcess <: AbstractMarketProcess
    name::Symbol
    asset_name::Symbol # The risky asset (e.g. :S)
    rate_name::Symbol  # The short rate (e.g. :r)
end
shock_indices(::ExcessReturnProcess) = Int[]

# --- Configuration ---
"""
    MarketConfig

The master configuration for the simulation.
"""
struct MarketConfig
    sims::Int
    T::Float64
    dt::Float64
    M::Int
    processes::Vector{AbstractMarketProcess}
    correlations::Matrix{Float64}
end

# Smart Outer Constructor
function MarketConfig(;
    sims::Int,
    processes::Vector{<:AbstractMarketProcess},
    correlations::AbstractMatrix{Float64} = Matrix{Float64}(I, 0, 0),
    T::Union{Real, Nothing} = nothing,
    dt::Union{Real, Nothing} = nothing,
    M::Union{Int, Nothing} = nothing
)
    # Validate input
    inputs = (T, dt, M)
    provided_count = count(!isnothing, inputs)

    if provided_count < 2
        error("MarketConfig: You must provide at least two of T, dt, and M.")
    end
    if provided_count == 3
        if abs(T - M * dt) > 1e-9
            error("MarketConfig: All of T, dt, and M are provided but inconsistent.")
        end
    end

    # Filling missing values
    local _T, _dt, _M

    if !isnothing(T) && !isnothing(dt)
        # Case 1: Have T, dt -> Calculate M
        _T  = Float64(T)
        _dt = Float64(dt)
        _M  = round(Int, _T / _dt)

        # Consistency check (optional warning)
        if abs(_T - _M * _dt) > 1e-9
            @warn "MarketConfig: T ($_T) is not a perfect multiple of dt ($_dt). Actual horizon will be $(_M * _dt)."
        end

    elseif !isnothing(T) && !isnothing(M)
        # Case 2: Have T, M -> Calculate dt
        _T  = Float64(T)
        _M  = Int(M)
        _dt = _T / _M

    elseif !isnothing(dt) && !isnothing(M)
        # Case 3: Have dt, M -> Calculate T
        _dt = Float64(dt)
        _M  = Int(M)
        _T  = _dt * _M

    else
        # Case 4: All 3 provided? Just take them
        _T, _dt, _M = Float64(T), Float64(dt), Int(M)
    end

    return MarketConfig(sims, _T, _dt, _M, processes, correlations)
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