module FinancialMarketSimulation

using LinearAlgebra
using Statistics
using Random
using ComponentArrays
using CairoMakie
using Statistics
using Printf
using DocStringExtensions

# Export Nouns (Types)
export MarketConfig, SimulationWorld
export AbstractMarketProcess, GenericSDEProcess, VasicekProcess
export NominalBondProcess, InflationBondProcess
export SimpleReturnProcess, ExcessReturnProcess

# Export Verbs (Functions)
export build_world, simulate!
export plot_simulation

# Include Logic
include("types.jl")
include("simulation.jl")
include("processes/generic.jl")
include("processes/vasicek.jl")
include("processes/bonds.jl")
include("processes/returns.jl")
include("plotting.jl")

end