"""
Simulate Simple Returns.
R_t = S_t / S_{t-1} - 1
"""
function simulate!(storage::AbstractMatrix, p::SimpleReturnProcess, world::SimulationWorld)
    (; M) = world.config

    # 1. Get the asset path
    S = getproperty(world.paths, p.asset_name)

    # 2. Initialize t=0 (First step has no return, usually 0)
    storage[:, 1] .= 0.0

    # 3. Loop (or Broadcast)
    # We can do this vectorized for speed since it depends on the full previous step
    # storage[:, n+1] = S[:, n+1] ./ S[:, n] .- 1.0

    @views for n in 1:M
        S_curr = S[:, n+1]
        S_prev = S[:, n]

        storage[:, n+1] .= (S_curr ./ S_prev) .- 1.0
    end

    return nothing
end

"""
Simulate Gross Returns.
G_t = S_t / S_{t-1}
"""
function simulate!(storage::AbstractMatrix, p::GrossReturnProcess, world::SimulationWorld)
    (; M) = world.config

    S = getproperty(world.paths, p.asset_name)

    # Initial value is typically 1.0 (no change) or 0.0 depending on convention.
    # Since it's a multiplier, 1.0 makes mathematical sense for t=0.
    storage[:, 1] .= 1.0

    @views for n in 1:M
        S_curr = S[:, n+1]
        S_prev = S[:, n]

        storage[:, n+1] .= S_curr ./ S_prev
    end

    return nothing
end

"""
Simulate Excess Returns.
Re_t = (S_t / S_{t-1} - 1) - (r_{t-1} * dt)
"""
function simulate!(storage::AbstractMatrix, p::ExcessReturnProcess, world::SimulationWorld)
    (; M, dt) = world.config

    S = getproperty(world.paths, p.asset_name)
    r = getproperty(world.paths, p.rate_name)

    storage[:, 1] .= 0.0

    @views for n in 1:M
        # Asset Return
        S_curr = S[:, n+1]
        S_prev = S[:, n]
        R_asset = (S_curr ./ S_prev) .- 1.0

        # Financing Cost (using rate at START of period)
        cost = r[:, n] .* dt

        storage[:, n+1] .= R_asset .- cost
    end

    return nothing
end