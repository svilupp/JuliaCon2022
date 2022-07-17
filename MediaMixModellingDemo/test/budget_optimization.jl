using MediaMixModellingDemo: multiply_and_normalize,
                             convert_budget_multiplier_to_spend_multiplier
@testset "budget_optimization - multiply_and_normalize" begin
    input_v = ones(5)
    input_mult = 10ones(5) # relative weights are the same, output should be =input
    output = multiply_and_normalize(input_v, input_mult)
    @test input_v == output
    @test sum(input_v) == sum(output)

    input_v = ones(5)
    input_mult = [10ones(3); ones(2)]
    output = multiply_and_normalize(input_v, input_mult)
    @test sum(input_v) == sum(output)
end

@testset "budget_optimization - convert_budget_multiplier_to_spend_multiplier" begin
    # budget multiplier kept =1
    input_spends = 1:1.0:5
    input_factors = 101:105
    budget_multiplier = ones(5)
    spend_multiplier = convert_budget_multiplier_to_spend_multiplier(input_spends,
                                                                     input_factors,
                                                                     budget_multiplier)
    @test budget_multiplier ≈ spend_multiplier ≈ ones(5) # when budget mult =1 than it doesn't change
    sum_prev = input_spends .* input_factors |> sum
    sum_new = spend_multiplier .* input_spends .* input_factors |> sum
    @test sum_prev ≈ sum_new # same spend before and after

    # budget multiplier kept constant, ie, no spend multiplier change (=1s)
    input_spends = 1:1.0:5
    input_factors = 101:105
    budget_multiplier = 5ones(5)
    spend_multiplier = convert_budget_multiplier_to_spend_multiplier(input_spends,
                                                                     input_factors,
                                                                     budget_multiplier)
    @test spend_multiplier ≈ ones(5) # when budget mult =1 than it doesn't change
    sum_prev = input_spends .* input_factors |> sum
    sum_new = spend_multiplier .* input_spends .* input_factors |> sum
    @test sum_prev ≈ sum_new # same spend before and after

    # change budget multiplier and check that spend is the same
    input_spends = 1:1.0:5
    input_factors = 5:-1:1
    budget_multiplier = [10.0; 1.5; ones(3)]
    spend_multiplier = convert_budget_multiplier_to_spend_multiplier(input_spends,
                                                                     input_factors,
                                                                     budget_multiplier)
    sum_prev = input_spends .* input_factors |> sum
    sum_new = spend_multiplier .* input_spends .* input_factors |> sum
    @test sum_prev ≈ sum_new  # same spend before and after
end
