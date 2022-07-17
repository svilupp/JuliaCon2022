@testset "evaluation_calculations - sum columns" begin
    # sum_columns
    input = ones(Float64, 5, 5)
    @test sum_columns(input) == 5ones(Float64, 5)

    input = ones(Float64, 5, 1)
    @test sum_columns(input) == fill(5.0, 1)

    input = ones(Float64, 5)
    @test sum_columns(input) == fill(5.0, 1)

    input = DataFrame(ones(Float64, 5, 5), :auto)
    @test sum_columns(input) == 5ones(Float64, 5)
end

@testset "evaluation_calculations - percentage share" begin
    # percentage_share
    input = ones(Float64, 5)
    @test percentage_share(input) == 0.2ones(Float64, 5)

    input = Float64[10 20 70]
    @test percentage_share(input) == [0.1 0.2 0.7]
end

@testset "evaluation_calculations - mean_fitted_effects" begin
    # base case
    input = repeat([(; mu_trend = 1:10, mu_org = 1:10)], 3)
    output = mean_fitted_effects(Val(:model_stage2a), input;
                                 extract_keys = [:mu_trend, :mu_org], mask = nothing)
    expected_output = [55.0; 55.0]
    @assert output == expected_output

    # with a mask
    mask_ = vcat(trues(5), falses(5))
    output = mean_fitted_effects(Val(:model_stage2a), input;
                                 extract_keys = [:mu_trend, :mu_org], mask = mask_)
    expected_output = [15.0; 15.0]
    @assert output == expected_output

    # with a mask and random inputs
    input = [(; mu_trend = (1:10) .+ 0.1randn(10), mu_org = (1:10) .+ 0.1randn(10))
             for i in 1:100]
    mask_ = vcat(trues(5), falses(5))
    output = mean_fitted_effects(Val(:model_stage2a), input;
                                 extract_keys = [:mu_trend, :mu_org], mask = mask_)
    expected_output = [15.0; 15.0]
    @assert all(isapprox.(output, expected_output; atol = 0.1))
end
# getflatsamples
# mock=Chains(repeat(reshape(collect(map_estimate.values),1,20,1),outer=(500,)),
# names(map_estimate.values)[1]);

# saturate_adspend

# mroas
