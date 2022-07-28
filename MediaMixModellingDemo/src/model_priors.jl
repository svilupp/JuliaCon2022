"""
    struct ParamsStage1
        model_name=Val(:any)
        scales_trend_offset::Float64=0. 
        scales_growth_trend::Float64=1.
        scales_trend::Float64 = 0.2
        scales_hols::Float64 = 0.3
        scales_seas::AbstractArray{Float64} = ones(Float64,1)
        scales_feat::AbstractArray{Float64} = ones(Float64,1)
        scales_noise::Float64 = 0.2 
        cat_levels::Int = 1 
    end

Holds priors and relevant parameters for Stage 1 of the modelling

Arguments:
- model_name=Val(:any) : Symbol representing the model version being fitted
- scales_trend_offset::Float64=0. : Scale ("width") of the Normal RV `alpha` (intercepts for different groups)
- scales_growth_trend::Float64=1. : Scale ("width") of the Normal RV `growth_trend` (trend component)
- scales_trend::Float64 = 0.2 : Scale ("width") of the Normal RV `beta_trend` for input X_trend (flexible trend fitting, eg, with splines)
- scales_hols::Float64 = 0.3 : Scale ("width") of the Normal RV `beta_hols` for input X_hols (holidays features)
- scales_seas::AbstractArray{Float64} = ones(Float64,1) : Scale ("width") of the Normal RV `beta_seas` for input X_feat (seasonality components)
- scales_feat::AbstractArray{Float64} = ones(Float64,1) : Scale ("width") of the Normal RV `beta_feat` for input X_feat (concatenated features)
- scales_noise::Float64 = 0.2 : Extent of the random noise around deterministic trend / rate parameter of Exponential distribution for `sigma`
- cat_levels::Int = 1 : Number of categorical levels in RV `alpha` that will be provided in X_cat vector (allows for different intercepts for different groups)

Example:

Set priors with automated priors like this:
```
p1=ParamsStage1(
    model_name=Val(Symbol(MODEL_NAME_PREFIT)),
    scales_trend=0.2,
    scales_hols=0.3,
    scales_noise=0.2,
    cat_levels=1
)
p1=set_priors_stage1_trendline(y_std,p1)
p1=set_priors_auto_scales(y_std,X_seas,:scales_seas,1.0,p1)
p1=set_priors_auto_scales(y_std,X_feat,:scales_feat,1.0,p1)
```
"""
@with_kw struct ParamsStage1
    model_name = Val(:any)
    scales_trend_offset::Float64 = 0.0
    scales_growth_trend::Float64 = 1.0
    scales_trend::Float64 = 0.2
    scales_hols::Float64 = 0.3
    scales_seas::AbstractArray{Float64} = ones(Float64, 1)
    scales_feat::AbstractArray{Float64} = ones(Float64, 1)
    # scale of the Exponential noise term around the deterministic trend
    scales_noise::Float64 = 0.2
    # Number of categorical levels
    cat_levels::Int = 1
end

function set_priors_stage1_trendline(y_std, existing_params = ParamsStage1())
    @assert length(y_std)>5 "y_std is too short for trendline logic to be applied! (length: $(length(y_std)))"
    # pick an average of the first 5 points
    scales_trend_offset = y_std[begin:min(begin + 5, end)] |> mean
    # we know that time_index is 0-1, so let's assume linear growth
    scales_growth_trend = y_std[end] / y_std[begin]
    @info "Trendline:" scales_trend_offset scales_growth_trend

    return reconstruct(existing_params; scales_trend_offset, scales_growth_trend)
end

function clip_boundary(arr::AbstractArray, min_ = 0.05, max_ = 10.0)
    warn_msgs = String["  Warnings: "]
    if any(arr .< min_)
        arr = max.(arr, min_)
        push!(warn_msgs, "A variable hit minimum allowed scale! Set to $(min_)!")
    end
    if any(arr .> max_)
        arr = min.(arr, max_)
        push!(warn_msgs, "A variable hit maximum allowed scale! Set to $(max_)!")
    end
    return arr, warn_msgs
end

# Utility for formatting decimals in debug messages
debug_print_decimals(arr) = join([@sprintf("%.2f", item) for item in arr], " | ")

"""
    set_priors_auto_scales(y::AbstractArray,X,param_key::Symbol,factor::Union{Number,AbstractArray}=1.0,
        existing_params=ParamsStage1())

Sets scales such that 1 standard dev of input data (`X`) could trigger at most 3standard dev. response of Y (`y`)
 (because beta coefficient are Normal). This can be increased by `factor` (defaults to 1.0x).
Result is saved into a Parameters object (as per `constructor`) under a key `param_key`
It can extend (selectively update) existing set of parameters (provide to argument `existing_params`)

Special behaviour:
- Values are clipped to range 0.05 - 10. (warning will be issued if it's exceeded)
- Shortcuts to straight passthrough if `X=nothing`
"""
function set_priors_auto_scales(y::AbstractArray, X, param_key::Symbol,
                                factor::Union{Number, AbstractArray} = 1.0,
                                existing_params = ParamsStage1())
    output = (std(y) ./ std.(eachcol(X))) .* factor
    output, warn_msgs = clip_boundary(output, 0.05, 10.0)
    # Pretty print
    summary_ = debug_print_decimals(output) * (length(warn_msgs) > 1 ? join(warn_msgs) : "")
    @info "Key $param_key: $(summary_)"

    return reconstruct(existing_params; Dict(param_key => output)...)
end

# shortcut behaviour
function set_priors_auto_scales(y::AbstractArray, X::Nothing, param_key::Symbol,
                                factor::Union{Number, AbstractArray} = 1.0,
                                existing_params = ParamsStage1())
    @info "Key $param_key skipped! Provided data is `Nothing`"
    return reconstruct(existing_params)
end

function sanity_check_priors(p::ParamsStage1; X_seas = nothing, X_feat = nothing,
                             X_trend = nothing, X_hols = nothing)

    # X_seas checks
    if !isnothing(X_seas)
        @assert size(X_seas, 2) == length(p.scales_seas)
    end
    # X_feat checks
    if !isnothing(X_feat)
        @assert size(X_feat, 2) == length(p.scales_feat)
    end
end
"""
    set_priors(params::ParamsStage1,inputs::InputData)

Sets automated priors based on input data for Stage 1 fitting
"""
function set_priors(params::ParamsStage1, inputs::InputData)
    @unpack y_std = inputs

    params = set_priors_stage1_trendline(y_std, params)
    params = set_priors_auto_scales(y_std, inputs.X_seas, :scales_seas, 1.0, params)
    params = set_priors_auto_scales(y_std, inputs.X_feat, :scales_feat, 1.0, params)

    sanity_check_priors(params; inputs.X_seas, inputs.X_feat, inputs.X_trend, inputs.X_hols)

    return params
end

"""
    struct ParamsStage2
        model_name=Val(:any)
        scales_trend_offset::Float64 = 0.3
        scales_trend::Float64 = 0.2
        scales_noise::Float64 = 0.3
        scales_context::AbstractArray{Float64} = ones(Float64,1)
        scales_org::AbstractArray{Float64} = ones(Float64,1)

        decay_rate_alphas::AbstractArray{Float64} = ones(Float64,1)
        decay_rate_betas::AbstractArray{Float64} = ones(Float64,1)

        adspend_mean_nonzero::AbstractArray{Float64} = ones(Float64,1)
        adspend_median::AbstractArray{Float64} = ones(Float64,1)

        locs_spend_halfmaxpoint::AbstractArray{Float64} = ones(Float64,1)
        scales_spend_halfmaxpoint::AbstractArray{Float64} = ones(Float64,1)

        locs_spend_beta::AbstractArray{Float64} = ones(Float64,1) 
        scales_spend_beta::AbstractArray{Float64} = ones(Float64,1)
        units_ratio_spend_to_y::AbstractArray{Float64} = ones(Float64,1)
        factor_to_roas_of_one::AbstractArray{Float64} = units_ratio_spend_to_y .* 2
    end

Holds priors and relevant parameters for Stage 2 of the modelling

For easy set up, use utility functions that build it from the inputs

Arguments:
- model_name=Val(:any) : Symbol representing the model version being fitted
- scales_trend_offset::Float64=0. : Scale ("width") of the Normal RV `alpha` (intercepts for different groups)
- scales_trend::Float64 = 0.2 : Scale ("width") of the Normal RV `beta_trend` for input X_trend (flexible trend fitting, eg, with splines)
- scales_noise::Float64 = 0.2 : Extent of the random noise around deterministic trend / rate parameter of Exponential distribution for `sigma`
- scales_context::AbstractArray{Float64} = ones(Float64,1) : Scale ("width") of the Normal RV `beta_context` for input X_context (context variables)
- scales_org::AbstractArray{Float64} = ones(Float64,1) : Scale ("width") of the Normal RV `beta_org` for input X_feat (organic variables - can have only POSITIVE effect)
- decay_rate_alphas::AbstractArray{Float64} = ones(Float64,1) : `decay_rate` RV is modelled by Beta distribution, `alpha` is the corresponding parameter
- decay_rate_betas::AbstractArray{Float64} = ones(Float64,1) :  `decay_rate` RV is modelled by Beta distribution, `beta` is the corresponding parameter
- adspend_mean_nonzero::AbstractArray{Float64} = ones(Float64,1) : Calculated quantity of the average non-zero spend (used to initialize `halfmaxpoint`)
- adspend_median::AbstractArray{Float64} = ones(Float64,1) : Calculated quantity of median of the spend
- locs_spend_halfmaxpoint::AbstractArray{Float64} = ones(Float64,1) : Center of the Normal RV `halfmaxpoint` for the halfmax concentration point in Hill Curve (can be initiated by average of the non-zero spend)
- scales_spend_halfmaxpoint::AbstractArray{Float64} = ones(Float64,1) :  Scale ("width") of the Normal RV `halfmaxpoint` for the halfmax concentration point in Hill Curve
- locs_spend_beta::AbstractArray{Float64} = ones(Float64,1) : Center of the Normal RV 'beta_spend' that represents the ROAS when the ad spend is at `halfmaxpoint`
- scales_spend_beta::AbstractArray{Float64} = ones(Float64,1) : Scale ("width") of the Normal RV 'beta_spend' that represents the ROAS when the ad spend is at `halfmaxpoint` 
- units_ratio_spend_to_y::AbstractArray{Float64} = ones(Float64,1) : Ratio of ad spend to Y to be able to convert unit effect (used for `factor_to_roas_of_one`)
- factor_to_roas_of_one::AbstractArray{Float64} : Conversion factor that ensures that provided `beta_spend` represents the ROAS when the ad spend is at `halfmaxpoint`

Example:
```
p2=ParamsStage2(
    model_name=Val(Symbol(MODEL_NAME)),
    scales_trend_offset=0.3,
    scales_trend=0.2,
    scales_noise=0.3,
)

p2 = set_priors_auto_scales(y_std,X_context,:scales_context,1.0,p2)
p2 = set_priors_auto_scales(y_std,X_org,:scales_org,1.0,p2)
p2 = set_priors_stage2_hill_curves(X_spend,p2;
        units_ratio_spend_to_y=getindex.(pipe_cache_spend,:xh)/pipe_cache_y[1].xh,
        halfmaxpoint_scale=0.3,expected_roas=1.0, expected_roas_scale=1.5)
p2 = set_priors_stage2_decay_rates(["ooh","digital","digital","ooh","ooh"],decay_rates_types_dictionary,p2,cols_spend)

sanity_check_priors(p2;X_spend,X_context,X_org);
```

"""
@with_kw struct ParamsStage2
    model_name = Val(:any)
    scales_trend_offset::Float64 = 0.3
    scales_trend::Float64 = 0.2
    scales_noise::Float64 = 0.3
    scales_context::AbstractArray{Float64} = ones(Float64, 1)
    scales_org::AbstractArray{Float64} = ones(Float64, 1)

    decay_rate_alphas::AbstractArray{Float64} = ones(Float64, 1)
    decay_rate_betas::AbstractArray{Float64} = ones(Float64, 1)

    adspend_mean_nonzero::AbstractArray{Float64} = ones(Float64, 1)
    adspend_median::AbstractArray{Float64} = ones(Float64, 1)

    locs_spend_halfmaxpoint::AbstractArray{Float64} = ones(Float64, 1)
    scales_spend_halfmaxpoint::AbstractArray{Float64} = ones(Float64, 1)

    locs_spend_beta::AbstractArray{Float64} = ones(Float64, 1)
    scales_spend_beta::AbstractArray{Float64} = ones(Float64, 1)
    units_ratio_spend_to_y::AbstractArray{Float64} = ones(Float64, 1)
    # halfmaxpoint is set to mean, then hill curve at that point will be 0.5
    # if we multiply value by 2, the beta coef = ROAS
    factor_to_roas_of_one::AbstractArray{Float64} = units_ratio_spend_to_y .* 2
end

function set_priors_stage2_hill_curves(X_spend, existing_params = ParamsStage2();
                                       units_ratio_spend_to_y::Union{Number, AbstractArray},
                                       halfmaxpoint_scale::Union{Number, AbstractArray} = 0.3,
                                       expected_roas::Union{Number, AbstractArray} = 1.0,
                                       expected_roas_scale::Union{Number, AbstractArray} = 1.5)
    @assert length(units_ratio_spend_to_y)==size(X_spend, 2) "Length of `units_ratio_spend_to_y` ($(length(units_ratio_spend_to_y))) different from number of variables in `X_spend` ($(size(X_spend,2)))"

    # Ad spend statistics for downstream applications in evaluation
    adspend_mean_nonzero = [mean(c[c .!= 0]) for c in eachcol(X_spend)]
    adspend_median = median.(eachcol(X_spend))

    # Halfmax concentration point = the point where saturation flexes
    locs_spend_halfmaxpoint = adspend_mean_nonzero #center at mean spend
    # Default: allow for 3*0.3 movement around that (truncated to 0.1 - 1. range in model_stage2a)
    scales_spend_halfmaxpoint = halfmaxpoint_scale .* ones(size(X_spend, 2))

    # Expected ROAS at the halfmax concentration point 
    # Note that model_stage2a truncates to 0-5 range!
    locs_spend_beta = expected_roas .* ones(size(X_spend, 2))
    scales_spend_beta = expected_roas_scale .* ones(size(X_spend, 2))

    # Normallization
    # units_ratio_spend_to_y reflects the scale translation of 1 unit of spend to 1 unit of Y
    # halfmaxpoint is set to mean, then hill curve at that point will be 0.5
    # if we multiply value by 2, then the beta_spend coef = ROAS
    factor_to_roas_of_one = units_ratio_spend_to_y .* 2

    # Bundle together
    outputs = (; adspend_mean_nonzero, adspend_median, locs_spend_halfmaxpoint,
               scales_spend_halfmaxpoint,
               locs_spend_beta, scales_spend_beta, units_ratio_spend_to_y,
               factor_to_roas_of_one)

    # Pretty print
    summary_ = [string(key, ": ", debug_print_decimals(val))
                for (key, val) in pairs(outputs)] |>
               x -> "\n" * join(x, "\n")
    @info "Hill Curve: $(summary_)"

    return reconstruct(existing_params; outputs...)
end

# "Some rule of thumb estimates we have found from historically building weekly-level models are that
# TV has tended to have adstock between 0.3 - 0.8, OOH/Print/Radio has had 0.1-0.4, and Digital has had 0.0 - 0.3. 
# This is anecdotal advice so please use your best judgement when building your own models."
# Source: https://facebookexperimental.github.io/Robyn/docs/features
decay_rates_types_dictionary = Dict("vague" => Beta(2, 2),
                                    "likely_short" => Beta(1, 5),
                                    "likely_long" => Beta(5, 2),
                                    "very_long" => Beta(40, 5),
                                    "digital" => Beta(1, 10),
                                    "tv" => Beta(25, 20),
                                    "ooh" => Beta(5, 15))

"""
    set_priors_stage2_decay_rates(decay_rate_types::AbstractArray{String},
        decay_rates_types_dictionary::Dict=decay_rates_types_dictionary,
        existing_params=ParamsStage2(),var_names=nothing)

Sets decay rates priors (for Beta distribution) as a dictionary lookup 
 of different options available in `decay_rates_types_dictionary` (eg, "tv", "digital")

Example:
```
types_=["ooh","digital","digital","ooh","ooh"]
set_priors_stage2_decay_rates(types_,decay_rates_types_dictionary,p2,cols_spend)
```
"""
function set_priors_stage2_decay_rates(decay_rate_types::AbstractArray{String},
                                       decay_rates_types_dictionary::Dict = decay_rates_types_dictionary,
                                       existing_params = ParamsStage2(),
                                       var_names = nothing)
    # INIT
    reporting_quantile1 = 0.05
    reporting_quantile2 = 0.95
    format_decimals(x) = @sprintf("%.2f", x)
    summary_ = []

    decay_rate_alphas = Float64[]
    decay_rate_betas = Float64[]
    # check if names are provided
    var_names = isnothing(var_names) ? ["Var_$i" for i in 1:length(decay_rate_types)] :
                var_names

    for (name, rate_type) in zip(var_names, decay_rate_types)
        rate_dist = decay_rates_types_dictionary[rate_type]
        params_ = Distributions.params(rate_dist)
        push!(decay_rate_alphas, params_[1])
        push!(decay_rate_betas, params_[2])

        msg = string(name, ": Mean ", format_decimals(mean(rate_dist)),
                     " | Range ",
                     format_decimals(quantile(rate_dist, reporting_quantile1)),
                     "-",
                     format_decimals(quantile(rate_dist, reporting_quantile2)),
                     # time to reach 0.5 effect
                     " | Halflife ", format_decimals(log(mean(rate_dist), 0.5)), " periods")
        push!(summary_, msg)
    end

    # Pretty print
    summary_ = "\n" * join(summary_, "\n")
    @info "Decay Rates: $(summary_)"

    return reconstruct(existing_params; decay_rate_alphas, decay_rate_betas)
end

function sanity_check_priors(p::ParamsStage2; X_spend = nothing, X_context = nothing,
                             X_org = nothing)

    # X_context checks
    if !isnothing(X_context)
        @assert size(X_context, 2) == length(p.scales_context)
    end
    # X_context checks
    if !isnothing(X_org)
        @assert size(X_org, 2) == length(p.scales_org)
    end

    # X_spend checks
    if !isnothing(X_spend)
        num_spend_vars = size(X_spend, 2)
    else
        num_spend_vars = length(p.factor_to_roas_of_one)
    end
    # Stats
    @assert num_spend_vars == length(p.adspend_mean_nonzero) == length(p.adspend_median)

    # Decay Rates
    @assert num_spend_vars == length(p.decay_rate_alphas) == length(p.decay_rate_betas)

    # Hill curve
    @assert num_spend_vars == length(p.locs_spend_halfmaxpoint) ==
            length(p.scales_spend_halfmaxpoint)
    @assert num_spend_vars == length(p.locs_spend_beta) == length(p.scales_spend_beta)
    @assert num_spend_vars == length(p.units_ratio_spend_to_y) ==
            length(p.factor_to_roas_of_one)
end

"""
    set_priors(params::ParamsStage2,inputs::InputData)

Sets automated priors based on input data for Stage 2 fitting
"""
function set_priors(params::ParamsStage2, inputs::InputData)
    @unpack y_std = inputs

    params = set_priors_auto_scales(y_std, inputs.X_context, :scales_context, 1.0, params)
    params = set_priors_auto_scales(y_std, inputs.X_org, :scales_org, 1.0, params)
    params = set_priors_stage2_hill_curves(inputs.X_spend, params;
                                           units_ratio_spend_to_y = getindex.(inputs.pipe_cache_spend,
                                                                              :xh) /
                                                                    inputs.pipe_cache_y[1].xh,
                                           halfmaxpoint_scale = 0.3, expected_roas = 1.0,
                                           expected_roas_scale = 1.5)

    return params
end
