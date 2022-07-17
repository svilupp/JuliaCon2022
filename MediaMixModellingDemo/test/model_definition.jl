@testset "evaluation_calculations - to_masked_matrix" begin
    # Matrix
    mat = ones(5, 5)
    @test to_masked_matrix(mat) == mat
    mask_ = Bool[0, 0, 1, 1, 0]
    @test to_masked_matrix(mat, mask_) == ones(2, 5)

    # DataFrame
    df = DataFrame(mat, :auto)
    @test to_masked_matrix(df) == Matrix(df)
    mask_ = Bool[0, 0, 1, 1, 0]
    @test to_masked_matrix(df, mask_) == ones(2, 5)

    # Vector
    v = ones(5)
    @test to_masked_matrix(v) == v
    mask_ = Bool[0, 0, 1, 1, 0]
    @test to_masked_matrix(v, mask_) == ones(2)

    # Nothing passthrough
    @test to_masked_matrix(nothing) == nothing
    @test to_masked_matrix(nothing, trues(1)) == nothing
end
