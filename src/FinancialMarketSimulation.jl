module FinancialMarketSimulation

using LinearAlgebra
using Statistics

using ComponentArrays


# 1. Export Nouns (Types)
export MarketConfig, SimulationWorld
export AbstractMarketProcess, GenericSDEProcess, VasicekProcess

# 2. Export Verbs (Functions)
export build_world, simulate!

# 3. Include Logic
include("types.jl")
include("simulation.jl")
include("processes/generic.jl")
include("processes/vasicek.jl")
# include("plotting.jl") # Keep this if you want plotting inside the package

end