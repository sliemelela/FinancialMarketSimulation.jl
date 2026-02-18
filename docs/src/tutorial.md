# Tutorial: Building a Financial Market
This guide will walk you through building a simulation, starting from a single asset and ending
with a full economy including inflation, bonds, and derivatives.

The general idea of the package is to follow the following steps:
1. Specify a process that you would like to simulate using either some SDE or some built in process
    that is given by this package.
2. Specify the configuration that are needed to run the simulation, e.g. the amount of paths, the
    amount of timesteps etc.
3. Run the builder command.
4. Plot using built in plotter.

Any analysis that one wants to do with the generated paths can be done by simply extracting the paths
from the object in which everything is stored.
Let us make this concrete by building up a more sophisticated model step by step.

## Level 1: The Basics (Interest Rates)
The package gives the user the option to model some stochastic process by specifying some generic
Stochastic Differential Equations (SDE's).
On top of this, the package also has some built in stochastic processes, like the Vasicek process.
To get an overview of all built in stochastic processes, visit the [Models & Math](models.md) section.
We will first how how to built the latter in the code example below.

To create a Vasicek process, we will use the `VasicekProcess` struct.
This struct accepts the parameters $\kappa$, $\theta$ and $\sigma$ such that the process follows
the SDE
```math
    dr_t = \kappa(\theta - r_t) dt + \sigma_r dW_t^1,
```
where $dW_t^1$ is some Brownian motion.
Below is an actual written out example.

```julia
using FinancialMarketSimulation

# dr_t = κ(θ - r_t)dt + σ dW_t
# κ=0.3, θ=0.05, σ_r =0.02, r0=0.05
r_model = VasicekProcess(:r, 0.3, 0.05, 0.02, 0.05, 1)

# We can specify Time (T) and Amount of Intervals (M), and dt is calculated automatically.
config = MarketConfig(
    sims = 500,
    T    = 1.0,
    M    = 100,
    processes = [r_model]
)

world = build_world(config)
plot_simulation(world)
```
Note that the last parameter `1` in the `VasicekProcess` struct specifies 'which shock' to use, i.e.
by writing down `1`, we create a Brownian shock that we index by the number `1` that is used to
generate the process.

To access the paths, use
```julia
r_path = world.paths.r
```
where `r` comes from the name `:r` that you supplied the `VasicekProcess` struct with when it was
initialized.

## Level 2: Adding a Correlated Stock
Now let's add a stock process.
We want the stock market to be negatively correlated with interest rate shocks
(e.g., rates up, stocks down).
We will be creating the stock process using the SDE
```math
    dS_t = \mu S_t dt + \sigma S_t dW_t^2,
```
To create the stochastic process using this SDE, we use the `GenericSDEProcess` struct.
As opposed to before, we cannot directly supply this struct with the parameters $\mu$ and $\sigma$
since it has to be able to create _any_ (Markovian) SDE.
To circumvent this, we supply the struct with two functions that specify the drift and diffusion
terms in the SDE, which correspond to
```math
    (t, S) \mapsto \mu S \\
    (t, S) \mapsto \sigma S,
```
respectively.
Additionally, note that we are now using some other Brownian motion $dW_t^2$ that must be correlated
with $dW_t^1$, i.e. we want to specify $\text{Cor}[dW_t^1, dW_t^2] = \rho \in \mathbb{R}$.
Let us see how this works in practice.

```julia
# Define Stock Process
# dS = μ * S dt + σ * S dW_2 with μ = 0.05 and σ = 0.2 and initial value of 100.
stock = GenericSDEProcess(:S, (t,x)->0.05*x, (t,x)->0.2*x, 100.0, [2])

# Define Correlation Matrix
# Correlation between Shock 1 (Rate) and Shock 2 (Stock) is -0.4
ρ = [ 1.0 -0.4;
     -0.4  1.0]

config = MarketConfig(
    sims=500, T=1.0, M=100,
    processes=[r_model, stock],
    correlations=ρ
)

world = build_world(config)
```
Now note that we have supplied the `GenericSDEProcess` with a `[2]` as the final argument.
Using `2` creates another Brownian shock that is correlated with `1` using the specified correlation
matrix $\rho$.
The reason for supplying it as a vector `[2]`, however, is that in the generic case we might want
to use more than one shock.
So suppose we would like to create a stochastic process using the SDE
```math
    dS_t = \mu S_t dt + \sigma_2 S_t dW_t^2 + \sigma_3 S_t dW_t^3,
```
then we can rewrite this to
```math
    dS_t = \mu S_t dt
        + \begin{pmatrix} \sigma_2 S_t & \sigma_3 S_t \end{pmatrix}
          \begin{pmatrix} dW_t^2 \\ dW_t^3 \end{pmatrix}
```
So in this (more complicated) case, the drift function would remain the same while the
diffusion function would be
```math
    (t, S) \mapsto \begin{pmatrix} \sigma_2 S & \sigma_3 S \end{pmatrix},
```
The code would be then (for example) be
```julia
# Define Stock Process
stock = GenericSDEProcess(:S, (t,x)->0.05*x, (t,x)->(0.2*x, 0.3*x), 100.0, [2, 3])

# Define Correlation Matrix
ρ = [ 1.0 -0.4 -0.2;
     -0.4  1.0 0.5;
     -0.2  0.5 1.0]

config = MarketConfig(
    sims=500, T=1.0, M=100,
    processes=[r_model, stock],
    correlations=ρ
)

world = build_world(config)
```

## Level 3: Bonds & Market Price of Risk
It can also be of interest to model both nominal and inflation linked bonds.
In both cases we must specify the so called factor loadings $\phi$ that are linked to the market
price of risk $\lambda$ via the formula $\lambda = \rho \phi$.

### Nominal Bonds
To create nominal bonds, we must supply it with the underlying interest rate $r_t$ and the
factor loadings $\phi_r, \phi_S$ corresponding to the interest rate $r_t$ and stock process $S$
respectively.
From here, the implementation is quite simple:
```julia
# Risk Factors: [ϕ_rate, ϕ_stock]
# We assume a negative risk premium for both.
market_risk = [-0.1, -0.2]

# Nominal Bond P_N depends on :r
bond = NominalBondProcess(:P_N, r_model, market_risk)

config = MarketConfig(sims=500, T=1.0, M=100, processes=[r_model, stock, bond], correlations=ρ)
world = build_world(config)
```
Note that we do not need to specify a "new" shock as the bond process is completely determined by
$r_t$ and the market prices of risk.

### Inflation Linked Bonds
Similarly, to create inflation linked bonds we must also create an inflation process and a
CPI (Consumer Price Index) process.
The inflation process $\pi_t$ is simple to instantiate:
```julia
pi_model = VasicekProcess(:pi, 0.1, 0.02, 0.01, 0.02, 3)
```
The CPI process however, can be a bit more cumbersome.
Let us treat the case of the CPI process $\Pi_t$ being
```math
d\Pi_t = \Pi_t \pi_t dt.
```
A fair question that will most likely come up is: how do we give the generic SDE struct
access to other stochastic processes?
The answer is simple, we use functions that use more arguments!
The drift process is now simply
```math
(t, \Pi, \pi) \mapsto \Pi \pi.
```
To make sure $\pi$ in the formula corresponds to the path $\pi_t$ we actually created before,
we provide an extra argument to the `GenericSDEProcess` as follows:
```julia
cpi_drift(t, Pi, pi_val) = Pi * pi_val
cpi_diff(t, Pi, pi_val)  = 0.0

cpi_model = GenericSDEProcess(
    :CPI, cpi_drift, cpi_diff, 100.0,
    Int[],   # No Shocks
    [:pi]    # Dependency: Reads the :pi path
)
```
We can finally create the inflation linked bond using
```julia
mprs = [-0.1, -0.2, -0.1]
infl_bond = InflationBondProcess(:P_I, r_model, pi_model, :CPI, mprs)
config = MarketConfig(
    sims=100, T=1.0, dt=0.01, M=100,
    processes=[r_model, stock, bond, pi_model, cpi_model, infl_bond]
)
```
Note that we have to specify another factor loading corresponding to the shock of the inflation
process.
Also note that the order of `processes` in `MarketConfig` matters. For example, since
`cpi_model` is depends on both `r_model` and `pi_model`, it must be supplied later in the `processes`
vector. Something similar holds for `infl_bond` and `bond`.

## Level 4: Simple and Excess Returns
Given an already existing process, we can easily create the simple, gross and excess return process by

```julia
# Simple Return of Stock
ret_S = SimpleReturnProcess(:R_S, :S)

# Excess Return of Stock over Rate
ex_ret_S = ExcessReturnProcess(:Re_S, :S, :r)

# Gross return of Stock
gross_S = GrossReturnProcess(:G_S, :S)
```

## Level 5: Analysis & Plotting
We can zoom in on specific events or check distributions at specific times.
```julia
# Plot only the distribution at the halfway point (t=0.5)
plot_simulation(world,
    plot_traj=false,
    plot_dist=true,
    dist_time=0.5
)

# Plot the trajectories of everything between t=0.5 and t=1.0
plot_simulation(world,
    plot_traj=true,
    plot_dist=false,
    time_range=(0.5, 1.0)
)

# Plot full trajectories and distributions at the final time and  just rates and the nominal bond.
plot_simulation(world, vars=[:r, :P_N])
```