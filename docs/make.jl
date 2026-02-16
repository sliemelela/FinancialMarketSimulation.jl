using FinancialMarketSimulation
using Documenter

makedocs(;
    modules=[FinancialMarketSimulation],
    authors="Sliem el Ela",
    repo="https://github.com/sliemelela/FinancialMarketSimulation.jl",
    sitename="FinancialMarketSimulation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sliemelela.github.io/FinancialMarketSimulation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md", # Optional: separate page for types/functions
    ],
)

deploydocs(;
    repo="github.com/sliemelela/FinancialMarketSimulation.jl",
    devbranch="main",
)