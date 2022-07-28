using FFTW: fftfreq, fft
using Printf: @sprintf
using Formatting: printfmtln

"""
    function plot_periodogram(input_arr,top_k)

Plot Fourier transform coefficients to uncover the most prominent frequencies / seasonalities
Assumes equally spaced data points
Looks only for periods that have seen at least 2 full cycles (ie, `size ÷ 2` at maximum!)
Shows `top-k` values

# Example
p=10 # period is 10
y=sin.(2π/p*collect(1:20)) # generate 20 data points
plot_periodogram(y,1) # plot periodogram, period=10 should be highlighted as maximum
"""
function plot_periodogram(input_arr, top_k = 1::Int64)
    @assert top_k >= 1

    dim = size(input_arr, 1)
    @assert dim > 2

    half_dim = dim ÷ 2 + 1
    periods = (1 ./ fftfreq(dim)) |> x -> x[2:half_dim]

    data = fft(input_arr) .|> abs |> x -> x[2:half_dim]
    topk_idx = sortperm(data, rev = true)

    pl = scatter(periods, data,
                 title = "Periodogram", label = "Coefficients", xlabel = "Period length")
    for i in 1:top_k
        pos = topk_idx[i]
        val = data[pos]
        per = periods[pos]

        vline!(pl, [per], label = "Period: " * @sprintf("%.1f", per))
        printfmtln("#{:d} period: {:.1f} with {:,.1f}", i, per, val)
    end
    return pl
end

"""
    generate_fourier_series(t, p=365.25, n=5)

Generates fourier series with period `p` and degree `n` (the higher, the more flexible it is)
It can be then fitted with coefficients to mimic any period trend

Expects t to be a time index series

Returns array of shape: (size(t,1),2n)

# Example
seaso=generate_fourier_series(1:400,365.25, 5)

"""
function generate_fourier_series(t, p = 365.25, n = 5)
    dim = size(t, 1)
    fourier_base = Array{Float64, 2}(undef, (dim, n))

    for j in axes(fourier_base, 2), i in axes(fourier_base, 1)
        fourier_base[i, j] = 2π / p * j * t[i]
    end
    return hcat(cos.(fourier_base), sin.(fourier_base))
end

"""
    generate_seasonality_features(t, p=365.25, n=5)

Generates seasonality features given an array of tuples in a format (period,degree)
Eg, 7-day period of degree 3 would be

Expects t to be a time index series

Returns array of shape: (size(t,1),2n)

# Example
seaso=generate_fourier_series(1:400,365.25, 5)

"""
function generate_seasonality_features(time_index, seasonality_arr)
    @assert seasonality_arr isa Vector
    @assert eltype(seasonality_arr) <: Tuple{Real, Int64, String}

    results = []
    for (period, deg, label) in seasonality_arr
        data = generate_fourier_series(time_index, period, deg)
        names = [@sprintf("seasonality%s_%02d", label, i) for i in 1:(2 * deg)]
        push!(results, DataFrame(data, names))
    end
    return reduce(hcat, results)
end

"""
    standardize_by_max(X)

Max()-only transform to allow easy scaling between features and the outcome
Uses MinMax() pipe under the hood but overwrites the minimum to be =0

Example:
`y_std,pipe_cache_y=standardize_by_max(select(df,target_label))`
"""
function standardize_by_max(X)
    pipe = MinMax()

    _, pipe_cache = apply(pipe, X)

    # force minimum to be 0. and re-apply
    for i in eachindex(pipe_cache)
        pipe_cache[i] = merge(pipe_cache[i], (; xl = 0.0))
    end

    output = reapply(pipe, X, pipe_cache)

    return output, pipe_cache
end

"""
    standardize_by_zscore(X)

Zscore transform to center the feature to its mean value and scale it (to make it easier to set priors)

Example:
`y_std,pipe_cache_y=standardize_by_zscore(select(df,target_label))`
"""
function standardize_by_zscore(X)
    pipe = ZScore()

    output, pipe_cache = apply(pipe, X)

    return output, pipe_cache
end
