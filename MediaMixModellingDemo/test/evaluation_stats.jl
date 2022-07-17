@testset "evaluation_stats - pseudor2" begin
    # data avg
    input = randn(100)
    pred_avg = mean(input) * ones(100)
    r2_for_avg = pseudor2(input, pred_avg)
    @test r2_for_avg == 0

    # perfect prediction
    r2_for_perfect = pseudor2(input, input)
    @test r2_for_perfect == 1

    # random
    input_similar = input .+ 0.1
    @test 0 < pseudor2(input, input_similar) < 1
end

@testset "evaluation_stats - nrmse" begin @test 1 == 1
    # TO DO:

    # data avg
    # input=randn(100)
    # pred_avg=mean(input)*ones(100)
    # nrmse_for_avg=nrmse(input,pred_avg)
    # @test nrmse_for_avg == 0

    # # perfect prediction
    # nrmse_for_perfect=nrmse(input,input)
    # @test nrmse_for_perfect == 1

    # # random
    # input_similar=input .+ 0.1
    # @test 0<nrmse(input,input_similar)<1
end
