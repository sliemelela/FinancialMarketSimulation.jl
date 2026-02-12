# src/processes/vasicek.jl

"""
    simulate!(storage, process::VasicekProcess, world)

Optimized Euler-Maruyama stepper for the Vasicek model.
"""
function simulate!(storage::AbstractMatrix, p::VasicekProcess, world::SimulationWorld)
    (; dt, M) = world.config

    # 1. Unpack Parameters (The specific variables you were worried about)
    κ, θ, σ = p.κ, p.θ, p.σ

    # 2. Set Initial Value
    storage[:, 1] .= p.initial_value

    # 3. Get the specific shock column
    # Dimensions: (sims, steps)
    # We access the 3D brownian_shocks array [sims, steps, shock_index]
    dW = @view world.brownian_shocks[:, :, p.shock_idx]

    # 4. Simulation Loop
    @views for n in 1:M
        r_prev = storage[:, n]

        # Exact Euler discretization for Vasicek:
        # r_new = r_prev + κ(θ - r_prev)dt + σ * dW

        # Note: dW here is already scaled by sqrt(dt) in build_world!
        # So we just multiply by σ.
        noise = σ .* dW[:, n+1]
        drift = κ .* (θ .- r_prev) .* dt

        storage[:, n+1] .= r_prev .+ drift .+ noise
    end

    return nothing
end