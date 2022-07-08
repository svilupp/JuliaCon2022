@testset "geometric_decay" begin
    # test zero rate
    # TO DO!
    # test rate of 1
    # TO DO!

    # test vector normal
    input = [1.0; zeros(10)]
    decay_rate = 0.1
    exp_result_no_normalization = decay_rate .^ (0:10)
    @test geometric_decay(input, decay_rate, false) ≈ exp_result_no_normalization

    exp_result_with_normalization = exp_result_no_normalization ./
                                    sum(exp_result_no_normalization)
    @test geometric_decay(input, decay_rate, true) ≈ exp_result_with_normalization

    # test_matrix
    input = repeat([1.0; zeros(10)]', outer = 5)'
    decay_rates = [0.1, 0.2, 0.3, 0.1, 0.4]
    exp_result_no_normalization = hcat([decay_ .^ (0:10) for decay_ in decay_rates]...)
    @test geometric_decay(input, decay_rates, false) ≈ exp_result_no_normalization

    exp_result_with_normalization = exp_result_no_normalization ./
                                    sum(exp_result_no_normalization, dims = 1)
    @test geometric_decay(input, decay_rates, true) ≈ exp_result_with_normalization
end
