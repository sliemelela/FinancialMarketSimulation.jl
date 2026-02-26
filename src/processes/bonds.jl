# --- Math Helpers ---
function B_vasicek(κ::Float64, h::Float64)
    abs(κ) < 1e-8 ? h : (1.0 - exp(-κ * h)) / κ
end

function get_rho(ρ::AbstractMatrix, i::Int, j::Int)
    (i <= 0 || j <= 0 || i > size(ρ,1) || j > size(ρ,2)) && return 0.0
    return ρ[i, j]
end

# --- Nominal Bond Logic ---

function calc_A_N(h::Float64, p::NominalBondProcess, ρ::AbstractMatrix)

    # 1. Physics
    rate = p.rate_process
    κ, θ, σ = rate.κ, rate.θ, rate.σ
    idx_r = shock_indices(rate)

    # 2. Risk Premium Calculation (Vectorized)
    # λ_r = - sum( ρ_{r,k} * ϕ_k )
    # We sum over all available risk factors provided in the vector
    ϕ = p.market_risk_factors
    λ_r = 0.0
    for k in eachindex(ϕ)
        λ_r -= get_rho(ρ, idx_r, k) * ϕ[k]
    end

    B_h = B_vasicek(κ, h)

    # # Term 1: Drift
    # term1 = (θ - λ_r * σ / κ) * (B_h - h)

    # # Term 2: Volatility
    # exp_kh = exp(-κ * h); exp_2kh = exp(-2.0 * κ * h)
    # term2 = (σ^2 / (2.0 * κ^3)) * (2.0 * exp_kh - 0.5 * exp_2kh - 1.5)

    # # Term 3: Ito
    # term3 = (σ^2 / (2.0 * κ^2)) * h

    part1 = -h * θ + B_h * θ
    exp_kh = exp(-κ * h); exp_2kh = exp(-2.0 * κ * h)
    part2 = (σ^2 / (2.0 * κ^3)) * (2.0 * exp_kh - 0.5 * exp_2kh - 1.5)
    part3 = h * (σ^2 / (2.0 * κ^2))
    part4 = -λ_r * (σ / κ) * (B_h - h)

    return part1 + part2 + part3 + part4
end

function simulate!(storage::AbstractMatrix, p::NominalBondProcess, world::SimulationWorld)
    (; T, M) = world.config
    r_path = getproperty(world.paths, p.rate_process.name)
    T_mat = p.T

    times = range(0, T, length=M+1)
    h_vec = max.(0.0, T_mat .- times)
    ρ = world.config.correlations

    B_vec = B_vasicek.(p.rate_process.κ, h_vec)
    A_vec = [calc_A_N(h, p, ρ) for h in h_vec]

    storage .= exp.(A_vec' .- B_vec' .* r_path)
    return nothing
end

# --- Inflation Bond Logic ---

function calc_A_I(h::Float64, p::InflationBondProcess, ρ::AbstractMatrix)
    # 1. Physics
    r_proc, π_proc = p.rate_process, p.infl_process
    κ_r, σ_r, θ_r = r_proc.κ, r_proc.σ, r_proc.θ
    κ_π, σ_π, θ_π = π_proc.κ, π_proc.σ, π_proc.θ

    # Obtain shocks
    idx_r = shock_indices(r_proc)
    idx_π = shock_indices(π_proc)

    # 2. Risk Premium (Vectorized)
    ϕ = p.market_risk_factors

    λ_r = 0.0
    λ_π = 0.0

    for k in eachindex(ϕ)
        val_k = ϕ[k]
        λ_r -= get_rho(ρ, idx_r, k) * val_k
        λ_π -= get_rho(ρ, idx_π, k) * val_k
    end

    B_r = B_vasicek(κ_r, h)
    B_pi = B_vasicek(κ_π, h)

    # 3. Terms
    term1 = θ_r * (B_r - h) - θ_π * (B_pi - h)

    e_r = exp(-κ_r * h); e_2r = exp(-2κ_r * h)
    term2 = (σ_r^2 / (2κ_r^3)) * (2e_r - 0.5e_2r - 1.5)

    e_pi = exp(-κ_π * h); e_2pi = exp(-2κ_π * h)
    term3 = (σ_π^2 / (2κ_π^3)) * (2e_pi - 0.5e_2pi - 1.5)

    term4 = h * (σ_r^2 / (2κ_r^2) + σ_π^2 / (2κ_π^2))
    term5 = -λ_r * (σ_r / κ_r) * (B_r - h) + λ_π * (σ_π / κ_π) * (B_pi - h)

    # 4. Intrinsic Correlation Term (r vs pi)
    # This requires looking up rho explicitly between the two processes
    ρ_rπ = get_rho(ρ, idx_r, idx_π)

    κ_sum = κ_r + κ_π
    B_sum = (abs(κ_sum) < 1e-8) ? h : (1.0 - exp(-κ_sum * h)) / κ_sum
    term6 = (ρ_rπ * σ_r * σ_π / (κ_r * κ_π)) * (B_r + B_pi - B_sum - h)

    return term1 + term2 + term3 + term4 + term5 + term6
end

function simulate!(storage::AbstractMatrix, p::InflationBondProcess, world::SimulationWorld)
    (; T, dt, M) = world.config
    r_path   = getproperty(world.paths, p.rate_process.name)
    pi_path  = getproperty(world.paths, p.infl_process.name)
    CPI_path = getproperty(world.paths, p.cpi_name)
    T_mat = p.T

    times = range(0, T, length=M+1)

    # 1. Clamp time to maturity (h) so it doesn't drop below 0.0
    h_vec = max.(0.0, T_mat .- times)
    ρ = world.config.correlations

    # 2. Calculate the column index corresponding to maturity
    # We use min(M+1, ...) to prevent out-of-bounds errors if T_mat > T
    mat_idx = min(M + 1, floor(Int, T_mat / dt) + 1)

    # 3. Create an array of column indices that flatlines at mat_idx
    # Example: if mat_idx is 50, it looks like [1, 2, ..., 49, 50, 50, 50...]
    col_indices = min.(1:(M+1), mat_idx)

    # Extract the CPI values using these clamped indices
    clamped_CPI = CPI_path[:, col_indices]

    B_r_vec  = B_vasicek.(p.rate_process.κ, h_vec)
    B_pi_vec = B_vasicek.(p.infl_process.κ, h_vec)
    A_I_vec  = [calc_A_I(h, p, ρ) for h in h_vec]

    # 4. Use clamped_CPI instead of the continuously growing CPI_path
    storage .= clamped_CPI .* exp.(A_I_vec' .- B_r_vec' .* r_path .+ B_pi_vec' .* pi_path)
    return nothing
end