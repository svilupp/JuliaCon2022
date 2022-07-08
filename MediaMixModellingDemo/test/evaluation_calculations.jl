@testset "evaluation_calculations - one-liners" begin
    # sum_columns
    input = ones(Float64, 5, 5)
    @test sum_columns(input) == 5ones(Float64, 1, 5)

    input = ones(Float64, 5)
    @test sum_columns(input) == [5.0]

    # percentage_share
    input = ones(Float64, 5)
    @test percentage_share(input) == 0.2ones(Float64, 5)

    input = Float64[10 20 70]
    @test percentage_share(input) == [0.1 0.2 0.7]
end

# getflatsamples
# mock=Chains(repeat(reshape(collect(map_estimate.values),1,20,1),outer=(500,)),
# names(map_estimate.values)[1]);

# calc_roas_total
# calc_roas

# saturate_adspend

# mroas
