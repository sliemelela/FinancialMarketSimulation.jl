# FinancialMarketSimulation.jl

A flexible, high-performance framework for simulating financial markets using Stochastic Differential Equations (SDEs).

## Features
* **Modular Design**: Mix and match building blocks (Rates, Stocks, Bonds, Inflation).
* **Correlated Shocks**: Simulate multi-asset markets with arbitrary correlation matrices.
* **Context-Aware**: Create processes that depend on others (e.g., Stochastic Volatility, CPI).
* **Visualization**: Built-in plotting for trajectories and distributions.

## Installation
```julia
using Pkg
Pkg.add(url="https://github.com/sliemelela/FinancialMarketSimulation.jl")
```

## Quick Start
```julia
using FinancialMarketSimulation

# Define Physics (Interest Rate)
rate = VasicekProcess(:r, 0.3, 0.05, 0.02, 0.05, 1)

# Build World (1 Year, 100 Steps)
config = MarketConfig(sims=1000, T=1.0, M=100, processes=[rate])
world  = build_world(config)

# Plot
plot_simulation(world)
```

### How to use this manual
- Want to know how to use this package? Visit the [Tutorial](tutorial.md) to get started.
- Want to know how the processes you simulate work in more detail?
    Visit the [Models & Math](models.md) section.
- The API reference can be found [here](api.md).