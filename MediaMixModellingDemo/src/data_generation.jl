using Dates: Date, Day

function create_time_index(start_date::Union{Date, String}, length::Integer)
    @assert length > 1
    cycle_length = 7 # weekly

    df = @chain begin
        DataFrame(:dt => Date(start_date):Day(cycle_length):(Date(start_date) + Day(cycle_length *
                                                                                    length -
                                                                                    1)))
        @transform _ :time_std=(eachindex(:dt) .- 1) / (nrow(_) - 1)
    end
    @assert nrow(df) == length
    return df
end

function create_trend(time_idx, trend_offset, configs)
    function trend_factory(time_idx, lowerb, upperb, growth)
        len = length(time_idx)
        growth_factors = zeros(len)
        steps = ones(len) / (len - 1)
        steps[1] = 0 #first is zero  

        mask = (time_idx .>= lowerb) .&& (time_idx .<= upperb)
        growth_factors[mask] .+= growth / sum(steps[mask])
        output = cumsum(growth_factors .* steps)

        return output
    end

    # start with the trend offset
    y_contrib = trend_offset .* ones(length(time_idx))

    for cfg in configs
        lowerb, upperb, growth = cfg
        output = trend_factory(time_idx, lowerb, upperb, growth)
        y_contrib .+= output
    end

    return y_contrib
end

function create_seasonality(time_idx, configs, coefs)
    exp_columns = sum([2 * tup[2] for tup in configs])
    @assert length(coefs) == exp_columns

    X_seas = generate_seasonality_features(time_idx, configs)
    y_contrib = Matrix(X_seas) * coefs

    return y_contrib, X_seas
end

function create_marketing_spend(time_idx, configs, coefs)
    function marketing_spend_factory(time_idx, label, spend_mean, spend_std, spend_sparsity,
                                     unit_scale_ratio)
        len = length(time_idx)
        # center at spend_mean
        input = spend_mean .+ Random.randn(len) .* spend_std
        # bound to 0-1 interval
        input = min.(max.(input, 0), 1)

        # apply desired sparsity (1. = no sparsity, 0. = all zeros)
        sparsity_mask = Random.randsubseq(1:len, 1 - spend_sparsity)
        input[sparsity_mask] .= 0

        # scale data down as required (to reduce its relative effect/share)
        input .*= unit_scale_ratio
        # NO! must scale after effects are applied, as halfpoint depends on previous scale

        return DataFrame(label => input)
    end

    function marketing_effect_factory(input, decay_rate, halfpoint, slope, coef,
                                      unit_scale_ratio)
        # apply adstock, do not normalize yet
        output = geometric_decay(input, decay_rate, false)
        # calculate normalization factor for the necessary scaling later, because we increase the spend by this factor
        normalization_factor = sum(output) / sum(input)

        # apply saturation curve
        # halfpoint was defined on the original scale so have to scale it down as well
        output = hill_curve.(output, halfpoint * unit_scale_ratio, slope)

        # multiply by 2coef because value at halfpoint is 0.5, so we need 2 coef, for the coef = ROAS
        # scale_ratio is the ratio between y_max and x_max (ie, 1 unit of y is 10 units of x)
        # normalization_factor=2/normalization_factor
        return output * coef * 2 / normalization_factor * unit_scale_ratio
    end

    input_arr = []
    output_arr = []
    for (cfg, coef) in zip(configs, coefs)
        # label,spend_mean,spend_std,spend_sparsity=cfg[begin:4]
        input = marketing_spend_factory(time_idx, cfg.label, cfg.mean,
                                        cfg.std, cfg.sparsity, cfg.unit_scale_ratio)
        push!(input_arr, input)

        # decay_rate,halfpoint,slope,unit_scale_ratio=cfg[5:end]
        output = marketing_effect_factory(input[!, 1], cfg.decay_rate, cfg.halfpoint,
                                          cfg.slope, coef, cfg.unit_scale_ratio)
        push!(output_arr, output)
    end

    y_contrib, input_df = hcat(output_arr...), hcat(input_arr...)
    return y_contrib, input_df
end

function create_holidays(start_date, len, coef; cnt_holidays_random = 0)
    temp = zeros(Int64, len)
    if cnt_holidays_random > 0
        mask_for_hols = rand(1:len, cnt_holidays_random)
        temp[mask_for_hols] .= 1
    end

    input_df = DataFrame(:hols_ind => temp)
    y_contrib = temp .* coef
    return y_contrib, input_df
end

function create_dataset(start_date, len, seed = nothing)
    Random.seed!(seed)
    ymax = 1 # not used yet

    # xmax
    # agnostic of scaling
    y_contribs = []

    # CREATE INDEX
    df = create_time_index(start_date, len)

    @info "INDEX with $(len) observations generated (random seed=$(seed))"

    # TREND
    trend_offset = 0.4
    trend_configs = [(0, 0.5, 0.1), (0.5, 1, 0.1)]
    y_contrib = create_trend(df.time_std, trend_offset, trend_configs)
    push!(y_contribs, y_contrib)

    @info begin
        digits = 1
        total_growth = maximum(y_contrib) - trend_offset
        "TREND: Trend offset of $(trend_offset) / Total growth of $(round(total_growth;digits))"
    end

    # SEASONALITY
    seasonality_configs = [(4 / (len - 1), 3, "4")]
    # get 2nd element times num degrees
    num_cols = 2 * sum(getindex.(seasonality_configs, 2))
    coefs = randn(num_cols) * 0.05

    y_contrib, X_seas = create_seasonality(df.time_std, seasonality_configs, coefs)
    push!(y_contribs, y_contrib)

    @info begin
        seasonalities = [cfg[3] for cfg in seasonality_configs] |> x -> join(x, ",")
        "SEASONALITY: Series with periods $(seasonalities) generated"
    end

    # HOLIDAYS
    coef = -0.2 + Random.randn() * 0.05

    cnt_holidays_random = round(Int, min(len * 0.05, 10))
    y_contrib, X_hols = create_holidays(start_date, len, coef; cnt_holidays_random)
    push!(y_contribs, y_contrib)

    @info begin
        digits = 1
        string("HOLIDAYS: Number of days: $(cnt_holidays_random)",
               "\n Average effect: $(round(mean(y_contrib[y_contrib .!= 0]);digits))",
               "\n Total effect: $(round(sum(y_contrib);digits))")
    end

    # EVENTS (BUSINESS-SPECIFIC)
    coef = 0
    events_cnt = 0
    X_events = DataFrame(:events => categorical(repeat(["na"], len)))
    y_contrib = zeros(len)
    push!(y_contribs, y_contrib)

    @info "EVENTS: $(events_cnt) events generated with coefficients: $(coef)"

    # MARKETING SPEND
    # label,spend_mean,spend_std,spend_sparsity,decay_rate,halfpoint,slope,unit_scale_ratio // coef
    # unit_scale_ratio of 0.1 = xmax/ymax => xmax is tenth of ymax
    marketing_configs = [
        (label = "facebook_S", mean = 0.4, std = 0.2, sparsity = 1.0,
         decay_rate = 0.5, halfpoint = 0.4, slope = 1.5,
         unit_scale_ratio = 0.05),
        (label = "tv_S", mean = 0.8, std = 0.2, sparsity = 0.7,
         decay_rate = 0.8, halfpoint = 0.8, slope = 0.9,
         unit_scale_ratio = 0.1),
        (label = "search_S", mean = 0.5, std = 0.3, sparsity = 1.0,
         decay_rate = 0.05, halfpoint = 0.5, slope = 2.0,
         unit_scale_ratio = 0.05),
    ]
    coefs = [2.0, 1.3, 2.5]
    y_contrib, X_spend = create_marketing_spend(df.time_std, marketing_configs, coefs)
    push!(y_contribs, y_contrib)

    @info let marketing_configs = marketing_configs, coefs = coefs
        marketing_msg = [merge(cfg, (; coef))
                         for (cfg, coef) in zip(marketing_configs, coefs)] |>
                        x -> join(x, " |\n ")
        roas_msg = [@sprintf("%s: %.1f", cfg.label, coef)
                    for (cfg, coef) in zip(marketing_configs, coefs)] |>
                   x -> join(x, " | ")
        string("Marketing Variables: \n",
               marketing_msg,
               "\nROAS: ", roas_msg)
    end

    # COMPETITOR SALES and other
    other_vars_cnt = 1
    coefs = [0.1, 0.05, -0.1]
    noise_scale = 0.1

    trend_seasonality_holidays_noise = 5 .* sum([
                                               y_contribs[begin:3]...,
                                               randn(len) .* noise_scale,
                                           ])
    X_context = DataFrame(:market_index => trend_seasonality_holidays_noise,
                          :newsletters => rand(len))
    for i in 1:other_vars_cnt
        X_context[!, @sprintf("other%d", i)] = randn(len)
    end

    @assert length(coefs) == length(names(X_context))
    y_contrib = Matrix(X_context .* coefs')
    push!(y_contribs, y_contrib)

    @info let coefs = coefs
        digits = 1
        coefs_rounded = [round(c; digits) for c in coefs] |> x -> join(x, ",")
        "CONTEXT: Number of vars: $(size(X_context,2)) Coefs: $(coefs_rounded)"
    end

    # NOISE
    noise_scale = 0.1
    y_contrib = randn(len, 1) .* noise_scale
    push!(y_contribs, y_contrib)

    @info let y_ = y_contribs
        digits = 1
        y_ = hcat(y_contribs[begin:(end - 1)]...) |> x -> sum(x, dims = 2)

        "NOISE: Noise scale of $noise_scale (compared to Y STD of $(round(std(y_);digits)))"
    end

    cols_y = [
        "trend",
        "seas",
        "hols",
        "events",
        names(X_spend)...,
        names(X_context)...,
        "noise",
    ]
    Y = DataFrame(hcat(y_contribs...), cols_y)
    X = hcat(df, X_seas, X_hols, X_events, X_spend, X_context)
    col_names = (; cols_time = names(df),
                 cols_other = vcat(names(X_hols), names(X_events)),
                 cols_spend = names(X_spend),
                 cols_context = names(X_context),
                 cols_y = [
                     "trend",
                     "seas",
                     "hols",
                     "events",
                     names(X_spend)...,
                     names(X_context)...,
                     "noise",
                 ])

    # stats
    @info begin
        digits = 1
        total_rev = Matrix(Y) |> sum
        daily_rev = sum.(eachrow(Y))
        string("Total Rev: ", @sprintf("%.1f", total_rev),
               "\n Per-Period Rev AVG: $(round(mean(daily_rev);digits)) ",
               "STD: $(round(std(daily_rev);digits)) ",
               "MIN: $(round(minimum(daily_rev);digits)) ",
               "MAX: $(round(maximum(daily_rev);digits))")
    end
    @info begin
        trend_contib = Y[!, "trend"] |> sum
        mkt_contrib = Y[!, col_names.cols_spend] |> Matrix |> sum
        context_contrib = Y[!, col_names.cols_context] |> Matrix |> sum
        total_rev = Matrix(Y) |> sum
        string("Contributions:",
               "\n Trend: ", @sprintf("%.1f%%", 100 * trend_contib/total_rev),
               "\n Marketing vars: ", @sprintf("%.1f%%", 100 * mkt_contrib/total_rev),
               "\n Context vars: ", @sprintf("%.1f%%", 100 * context_contrib/total_rev))
    end

    return Y, X, col_names
end
