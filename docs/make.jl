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
    checkdocs = :exports,
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Models & Math" => "models.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/sliemelela/FinancialMarketSimulation.jl",
    devbranch="main",
)