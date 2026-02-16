"""
    simulate!(storage, process, world)

Runs the Euler-Maruyama scheme for a GenericSDEProcess.
Supports dependency injection: passes values of dependent paths to drift/diffusion.
"""
function simulate!(storage::AbstractMatrix, p::GenericSDEProcess, world::SimulationWorld)
    (; dt, M) = world.config

    storage[:, 1] .= p.initial_value

    # 1. Resolve Dependencies
    # Grab the full paths for every dependency listed in p.dependencies
    # This creates a Tuple of matrices (or vectors) to avoid repeated lookups
    dep_paths = Tuple(getproperty(world.paths, sym) for sym in p.dependencies)

    # 2. Get Shocks (if any)
    has_shocks = !isempty(p.shock_idxs)
    if has_shocks
        dW_all = @view world.brownian_shocks[:, :, p.shock_idxs]
    end

    @views for n in 1:M
        t = (n-1) * dt
        X_prev = storage[:, n]

        # 3. Get values of dependencies at this step
        # We grab the n-th column for each dependency path
        dep_values = map(path -> path[:, n], dep_paths)

        # 4. Calculate Drift & Diffusion
        # We use splatting (...) to pass dependencies as extra arguments
        # User function signature: f(t, x, dep1, dep2...)
        mu    = p.drift.(t, X_prev, dep_values...)
        sigma = p.diffusion.(t, X_prev, dep_values...)

        # 5. Compute Noise Term
        if has_shocks
            dZ = dW_all[:, n+1, :]

            if size(dZ, 2) == 1
                noise = sigma .* dZ[:, 1]
            else
                # Dot product for multi-factor
                noise = map((s, z) -> dot(s, z), sigma, eachrow(dZ))
            end
        else
            # Deterministic (ODE) mode
            noise = 0.0
        end

        # 6. Euler Step
        storage[:, n+1] .= X_prev .+ mu .* dt .+ noise
    end

    return nothing
end