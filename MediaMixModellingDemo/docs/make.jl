using MediaMixModellingDemo
using Documenter

DocMeta.setdocmeta!(MediaMixModellingDemo, :DocTestSetup, :(using MediaMixModellingDemo); recursive=true)

makedocs(;
    modules=[MediaMixModellingDemo],
    authors="Svilup",
    repo="https://github.com/svilup/JuliaCon2022/MediaMixModellingDemo.jl/blob/{commit}{path}#{line}",
    sitename="MediaMixModellingDemo.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://svilup.github.io/MediaMixModellingDemo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Practical Tips and Tricks" => "practical_tips.md",
        "Resources"=>"resources.md"
    ],
)

deploydocs(;
    repo="github.com/svilup/JuliaCon2022/MediaMixModellingDemo.jl",
    devbranch="main",
)
