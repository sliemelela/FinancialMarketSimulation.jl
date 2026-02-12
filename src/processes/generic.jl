"""
    simulate!(storage, process, world)

Runs the Euler-Maruyama scheme for a GenericSDEProcess.
"""
function simulate!(storage::AbstractMatrix, p::GenericSDEProcess, world::SimulationWorld)
    (; dt, M) = world.config

    storage[:, 1] .= p.initial_value

    # Extract the subset of shock columns this process needs
    # Result is a 3D view: (sims, time, n_factors)
    dW_all = @view world.brownian_shocks[:, :, p.shock_idxs]

    @views for n in 1:M
        t = (n-1) * dt
        X_prev = storage[:, n]

        # 1. Drift
        mu = p.drift.(t, X_prev)

        # 2. Diffusion (Can be scalar or tuple/vector)
        sigma = p.diffusion.(t, X_prev)

        # 3. Get Shocks for this step
        # Dimensions: (sims, n_factors)
        dZ = dW_all[:, n+1, :]

        # 4. Compute Noise Term (σ ⋅ dZ)
        # We need to handle the dot product efficiently for each simulation
        if size(dZ, 2) == 1
            # Simple case: 1 Factor
            # sigma can be scalar, or we grab the first element
            noise_term = sigma .* dZ[:, 1]
        else
            # Multi-Factor Case: Dot Product
            # We assume sigma[i] is a vector/tuple matching dZ[i, :]
            # Broadcasting `dot` requires slightly careful handling or a loop
            noise_term = map(
                (s, z_row) -> dot(s, z_row),
                sigma,
                eachrow(dZ)
            )
        end

        # 5. Euler Step
        storage[:, n+1] .= X_prev .+ mu .* dt .+ noise_term
    end

    return nothing
end