using OptimalBath
using Documenter

DocMeta.setdocmeta!(OptimalBath, :DocTestSetup, :(using OptimalBath); recursive=true)

makedocs(;
    modules=[OptimalBath],
    authors="martinsw01 <martin.s.winther@gmail.com>",
    sitename="OptimalBath.jl",
    format=Documenter.HTML(;
        canonical="https://martisw01.github.io/OptimalBath.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="martinsw01/OptimalBath.jl",
    devbranch="main",
)
