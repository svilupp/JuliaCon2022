using MediaMixModellingDemo
using Test

@testset "MediaMixModellingDemo.jl" begin
    # Write your tests here.
end

# TO DO: Move over test from the old suite
include("test_feature_engineering.jl")
include("marketing_transformations.jl")
include("test_evaluation_stats.jl")
include("evaluation_calculations.jl")
include("test_budget_optimization.jl")
