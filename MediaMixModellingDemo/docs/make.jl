using MediaMixModellingDemo
using Documenter

DocMeta.setdocmeta!(MediaMixModellingDemo, :DocTestSetup, :(using MediaMixModellingDemo);
                    recursive = true)

makedocs(;
         modules = [MediaMixModellingDemo],
         authors = "Svilupp",
         repo = "https://github.com/svilupp/JuliaCon2022/MediaMixModellingDemo.jl/blob/{commit}{path}#{line}",
         sitename = "MediaMixModellingDemo.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://svilupp.github.io/MediaMixModellingDemo.jl",
                                  edit_link = "main",
                                  assets = String[]),
         pages = [
             "Home" => "index.md",
             "Methodology" => "methodology.md",
            #  "Examples" => ["Demo - High-level API" => "demo-high-level.md"],
             "Practical Tips" => "practical_tips.md",
             "Resources" => "resources.md",
             "API Reference" => "api_reference.md",
         ])

deploydocs(;
           repo = "github.com/svilupp/JuliaCon2022/MediaMixModellingDemo.jl",
           devbranch = "main")
