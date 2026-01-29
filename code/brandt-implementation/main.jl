using CairoMakie
using Statistics
using ParameterSets
using Random

CairoMakie.activate!()

include("structs.jl")
include("utils.jl")
include("simulation.jl")
include("plotting.jl")
include("diagnostics.jl")
include("connectors.jl")
include("solver.jl")


function run_full_analysis(config_filename::String)

    # Load all config sets
    parameter_sets = load_sets(config_filename)
    println("Successfully generated $(length(parameter_sets)) parameter set(s) to run.")

    # Store results
    all_final_policies = []
    all_worlds = []
    results = Dict{Int, Dict{String, Any}}()

    for parameter_set in parameter_sets

        # Extract config
        config_set = parameter_set.config
        i = parameter_set.id

        println("\n--- RUNNING ANALYSIS FOR SET $i/$(length(parameter_sets)) ---")

        # Load the solver struct
        solver_params_dict = config_set["SolverParams"]
        γ = solver_params_dict["γ"]
        solver_params = load_struct(SolverParams, solver_params_dict)

        # Build the world
        world = build_world_from_recipe(config_set)

        # Plot the world paths
        plot_world_paths(world)

        # Create the utility function
        crra_base = W -> (W^(1.0 - γ)) / (1.0 - γ)
        my_utility = create_utility_from_ad(crra_base)

        # Run the solver
        println("Starting portfolio solver for Set $i...")
        ω_l = solve_portfolio_problem(world, solver_params, my_utility)
        push!(all_final_policies, ω_l)
        push!(all_worlds, world)

        # Plotting policy
        plot_policy(world, solver_params, ω_l)

        # Calculate the value function and certainty equivalent wealth
        for W_1 in config_set["SolverParams"]["W_grid"]
            J_W1, CE_1 = calculate_expected_utility(world, solver_params, ω_l, 1, W_1, nothing, my_utility)

            if !haskey(results, parameter_set.id)
                results[parameter_set.id] = Dict{String, Any}()
            end
            merge!(results[parameter_set.id], Dict(
                "J_W at W_1: $W_1" => J_W1,
                "CE at W_1: $W_1" => CE_1
            ))
        end
    end

    save_sensitivity_reports(parameter_sets, results)
    return all_worlds, all_final_policies
end

config_path = joinpath(@__DIR__, "parameters.yaml")
run_full_analysis(config_path)
nothing