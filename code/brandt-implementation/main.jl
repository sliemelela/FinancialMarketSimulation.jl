using GLMakie
using Statistics
using LinearAlgebra
using Interpolations
using StaticArrays
using ForwardDiff
using Random
using YAML

GLMakie.activate!()

include("structs.jl")
include("loadParameters.jl")
include("utils.jl")
include("simulation.jl")
include("plotting.jl")
include("connectors.jl")
include("solver.jl")


function run_full_analysis(config_filename::String)

    # 1. Load all config sets
    config_sets = load_config_sets(config_filename)
    println("Successfully generated $(length(config_sets)) parameter set(s) to run.")

    all_final_policies = []
    all_worlds = []

    for (i, config_set) in enumerate(config_sets)
        println("\n--- RUNNING ANALYSIS FOR SET $i/$(length(config_sets)) ---")

        # Load the solver struct
        solver_params_dict = config_set["SolverParams"]
        γ = solver_params_dict["γ"]
        solver_params = load_struct(SolverParams, solver_params_dict)
        println("Solver params", solver_params)

        # Build the world
        world = build_world_from_recipe(config_set)

        # Plot the world paths
        plot_world_paths(world)

        # 5. Create the utility function
        crra_base = W -> (W^(1.0 - γ)) / (1.0 - γ)
        my_utility = create_utility_from_ad(crra_base)

        # 6. Run the solver
        println("Starting portfolio solver for Set $i...")
        ω_l = solve_portfolio_problem(world, solver_params, my_utility)
        push!(all_final_policies, ω_l)
        push!(all_worlds, world)

        # Plotting policy
        plot_policy(world, solver_params, ω_l)
    end

    return all_worlds, all_final_policies
end

all_worlds, all_final_policies = run_full_analysis("parameters.yaml")
nothing