Base.@kwdef struct InputData

    # User Inputs
    col_target::String
    col_datetime::String
    col_time_std::String
    col_cat::String
    cols_context::AbstractVector{String}
    cols_organic::AbstractVector{String}
    cols_hols::AbstractVector{String}
    cols_spend::AbstractVector{String}
    seasonality_periods::AbstractVector{Float64}
    spline_degree::Int
    fit_stage2_mask::BitVector
    optim_mask::BitVector

    # Data
    time_std::Any
    dt::Any
    y_true::Any
    y_std::Any
    X_org::Any
    X_spend::Any
    X_context::Any
    X_trend::Any
    X_cat::Any
    X_hols::Any
    X_seas::Any
    X_feat::Any

    # context assets
    pipe_cache_org::Any
    pipe_cache_spend::Any
    pipe_cache_context::Any
    pipe_cache_y::Any
    cat_levels::Any

    # Transformation utilities
    revert_pipe_spend::Function
    revert_pipe_y::Function
end

"""

    build_inputs(df::AbstractDataFrame;
        cols_spend::AbstractVector{String},
        col_target::String="revenue",
        col_datetime::String="dt",
        col_time_std::String="time_std",
        col_cat::String="",
        cols_hols::AbstractVector{String}=String[],
        cols_context::AbstractVector{String}=String[],
        cols_organic::AbstractVector{String}=String[],
        seasonality_periods::AbstractVector{Float64}=Float64[],
        spline_degree::Int=0,
        fit_stage2_mask::AbstractVector{Bool}=trues(nrow(df)),
        optim_mask::AbstractVector{Bool}=trues(nrow(df))
    )

Builds the input data object for the successing fitting & optimization

Arguments:
- `col_target`::String="revenue" - column name of the target/response variable (eg, revenues)
- `col_datetime`::String="dt" - column name of the variable with dates
- `col_time_std`::String="time_std" - column name of the standardized time index (from 0-1, strict)
- `col_cat`::String="events" - column name of the categorical variable representing various events (ie not available); Set to "" if there isn't any
- `cols_context`::AbstractVector{String}=String[] - column name of the context variables (eg, macroeconomic indicators, market trends, competitors sales or promotions) (variables will be standardized via Z-score); Defaults to empty if there are none
- `cols_organic`::AbstractVector{String}=String[] - column name of the organic marketing activities (eg, email newsletters) (variables will be scaled to maximum=1 via Max() function and their effect on response can be only positive); Defaults to empty if there are none
- cols_hols::AbstractVector{String}=String[] - column name of the holiday indicators; Defaults to empty if there are none
- `cols_spend`::AbstractVector{String} - column names of the Ad spend variables that we want to model (variables will be scaled to maximum=1 via Max() function)
- `seasonality_periods`::AbstractVector{Float64}=Float64[] - what seasonalities are expected in the trendline; Defaults to empty if there are none
- `spline_degree`::Int=0 - if complicated trend modelling is needed, what degree of spline basis should be used (uses cubic bases splines from Splines2.jl)
- `fit_stage2_mask`::AbstractVector{Bool}=trues(nrow(df)) - mask to be apllied to observed data in Stage 2 of modelling (use the observation when `=true`); Defaults to all observations being used
- `optim_mask`::AbstractVector{Bool}=trues(nrow(df)) - mask to be apllied in the budget optimization (use the observation when `=true`); Defaults to all observations being used
"""
function build_inputs(df::AbstractDataFrame;
                      cols_spend::AbstractVector{String},
                      col_target::String = "revenue",
                      col_datetime::String = "dt",
                      col_time_std::String = "time_std",
                      col_cat::String = "",
                      cols_hols::AbstractVector{String} = String[],
                      cols_context::AbstractVector{String} = String[],
                      cols_organic::AbstractVector{String} = String[],
                      seasonality_periods::AbstractVector{Float64} = Float64[],
                      spline_degree::Int = 0,
                      fit_stage2_mask::AbstractVector{Bool} = trues(nrow(df)),
                      optim_mask::AbstractVector{Bool} = trues(nrow(df)))

    ### Datetime column (for nicer plots in the end)
    dt = df[!, col_datetime]

    ### Time index dimension (must be 0-1)
    time_std = df[!, col_time_std]
    @info "Time index stats: " time_std|>size time_std|>extrema

    ### Y transform
    y_true = df[!, col_target]
    y_std, pipe_cache_y = standardize_by_max(select(df, col_target))
    y_std = y_std[!, 1] # extract the vector

    # define revert function
    revert_pipe_y = y -> revert(MinMax(), (; y), pipe_cache_y).y

    ### FEATURES

    # positive coeffcients, transformed
    X_spend, pipe_cache_spend = standardize_by_max(convert.(Float64, df[!, cols_spend]))
    X_feat = X_spend

    # revert function utility
    revert_pipe_spend = x -> revert(MinMax(), x, pipe_cache_spend)

    # positive coefficient, not transformed (saturated etc)
    if length(seasonality_periods) > 0
        X_org, pipe_cache_org = standardize_by_max(convert.(Float64, df[!, cols_organic]))
        X_feat = hcat(X_feat, X_org)
    else
        X_org = nothing
        pipe_cache_org = nothing
    end

    # any coefficient allowed
    if length(cols_context) > 0
        X_context, pipe_cache_context = standardize_by_zscore(convert.(Float64,
                                                                       df[!, cols_context]))
        X_feat = hcat(X_feat, X_context)
    else
        X_context = nothing
        pipe_cache_context = nothing
    end

    # categorical column
    if col_cat != ""
        # must be a categorical vector
        t = typeof(df[!, col_cat])
        @assert t<:CategoricalVector "Column $col_cat must be a categorical vector! (provided: $t)"
        X_cat = df[!, col_cat] .|> levelcode
        cat_levels = df[!, col_cat] |> levels |> length
    else
        X_cat = nothing
        cat_levels = 1
    end

    # No transformation
    if length(cols_hols) > 0
        X_hols = df[!, cols_hols]
    else
        X_hols = nothing
    end

    # Do not provide X_trend if your trend is simple
    # It could overfit - there is a growth term, an offset and a seasonality already!
    # Use it on complicated datasets / with larger datasets
    3 > spline_degree > 0 &&
        @info "Splines ignored. Provided spline degree must be >=3 (given: $(spline_degree))"
    if spline_degree >= 3
        X_trend = Splines2.bs(time_std, df = 3, boundary_knots = (-eps(), 1 + eps()))
        @info "X_trend: " X_trend|>size X_trend|>extrema
    else
        X_trend = nothing
    end

    # Seasonality
    if length(seasonality_periods) > 0
        # assume degree 3 and round the label to decimals (for easier column names)
        seasonality_inputs = [(period / (length(time_std) - 1), 3, @sprintf("%d", period))
                              for period in seasonality_periods]
        X_seas = generate_seasonality_features(time_std, seasonality_inputs)
    else
        X_seas = nothing
    end

    # Masks
    @assert 0<sum(fit_stage2_mask)<=nrow(df) "Optimization mask is wrong! (DF rows: $(nrow(df)) vs Selected rows: $(sum(optim_mask)))"
    @assert 0<sum(optim_mask)<=nrow(df) "Optimization mask is wrong! (DF rows: $(nrow(df)) vs Selected rows: $(sum(optim_mask)))"

    return InputData(;
                     # column names
                     col_target, col_datetime, col_time_std, col_cat, cols_context,
                     cols_organic, cols_hols, cols_spend,
                     seasonality_periods, spline_degree, fit_stage2_mask, optim_mask,

                     # Data
                     dt, time_std, y_true, y_std, X_org, X_spend, X_context, X_trend, X_cat,
                     X_hols, X_seas, X_feat,

                     # context assets
                     pipe_cache_org, pipe_cache_spend, pipe_cache_context, pipe_cache_y,
                     cat_levels,

                     # Transformation utilities
                     revert_pipe_spend, revert_pipe_y)
end
