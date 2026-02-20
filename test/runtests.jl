using FinancialMarketSimulation
using Statistics
using Test

@testset "FinancialMarketSimulation.jl" begin
    include("engine_test.jl")
    include("processes_test.jl")
    include("bonds_test.jl")
    include("returns_test.jl")
    include("plotting_test.jl")
end