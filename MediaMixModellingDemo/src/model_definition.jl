
@model function model_stage1a(time_index, X_trend, X_hols, X_seas, X_feat, X_cat, p,
                              ::Type{T} = Float64) where {T}

    # Trend
    growth_trend ~ Normal(0, p.scales_growth_trend)
    mu_trend = time_index .* growth_trend

    if !isnothing(X_trend)
        beta_trend ~ filldist(Normal(0, p.scales_trend), size(X_trend, 2))
        mu_trend .+= X_trend * beta_trend
    end

    if !isnothing(X_hols)
        beta_hols ~ filldist(Normal(0, p.scales_hols), size(X_hols, 2))
        mu_hols = (X_hols * beta_hols)
    else
        mu_hols = zeros(T, length(time_index))
    end

    if !isnothing(X_seas)
        beta_seas ~ arraydist([Normal(0, scale) for scale in p.scales_seas])
        mu_seas = (X_seas * beta_seas)
    else
        mu_seas = zeros(T, length(time_index))
    end

    beta_feat ~ arraydist([Normal(0, scale) for scale in p.scales_feat])
    mu_feat = X_feat * beta_feat

    if !isnothing(X_cat)
        # replaces intercept if there are multiple groups/events
        alpha ~ filldist(Normal(0, p.scales_trend_offset), p.cat_levels)
        mu_cat = alpha[X_cat]
    else
        # fit an intercept
        mu_cat = Normal(0, p.scales_trend_offset)
    end

    mu = mu_trend + mu_hols + mu_seas + mu_feat + mu_cat

    sigma ~ Exponential(p.scales_noise)

    y ~ MvNormal(mu, sigma)

    return (; y, mu_trend, mu_hols, mu_seas, mu_feat, mu_cat)
end;

# add fat tail support
@model function model_stage1b(time_index, X_trend, X_hols, X_seas, X_feat, X_cat, p,
                              ::Type{T} = Float64) where {T}
    growth_trend ~ Normal(0, p.scales_growth_trend)

    # replaces intercept
    alpha ~ filldist(Normal(0, p.scales_trend_offset), p.cat_levels)

    # Trend
    mu_trend = time_index .* growth_trend

    if !isnothing(X_trend)
        beta_trend ~ filldist(Normal(0, p.scales_trend), size(X_trend, 2))
        mu_trend .+= X_trend * beta_trend
    end

    if !isnothing(X_hols)
        beta_hols ~ filldist(Normal(0, p.scales_hols), size(X_hols, 2))
        mu_hols = (X_hols * beta_hols)
    else
        mu_hols = zeros(T, length(time_index))
    end

    if !isnothing(X_seas)
        beta_seas ~ arraydist([Normal(0, scale) for scale in p.scales_seas])
        mu_seas = (X_seas * beta_seas)
    else
        mu_seas = zeros(T, length(time_index))
    end

    beta_feat ~ arraydist([Normal(0, scale) for scale in p.scales_feat])
    mu_feat = X_feat * beta_feat

    mu_cat = alpha[X_cat]

    mu = mu_trend + mu_hols + mu_seas + mu_feat + mu_cat

    sigma ~ Exponential(p.scales_noise)
    nu ~ Gamma(2, 10)
    y ~ arraydist(mu .+ sigma .* TDist.(nu))

    return (; y, mu_trend, mu_hols, mu_seas, mu_feat, mu_cat)
end;

####################################
# Stage 2 Fit
@model function model_stage2a(time_index, X_trend, X_spend, X_org, X_context, p,
                              ::Type{T} = Float64) where {T}
    trend_offset ~ Normal(0, p.scales_trend_offset)

    beta_trend ~ filldist(Normal(1.0, p.scales_trend), size(X_trend, 2))

    # only positive // beta_spend is maximum possible effect
    beta_spend ~ arraydist([Truncated(Normal(loc, scale), 0.0, 5.0)
                            for (loc, scale) in zip(p.locs_spend_beta,
                                                    p.scales_spend_beta)])

    # marketing transforms
    decay_rate ~ arraydist([Beta(alpha, beta)
                            for (alpha, beta) in zip(p.decay_rate_alphas,
                                                     p.decay_rate_betas)])

    slope ~ filldist(Truncated(Normal(1.0, 0.5), 0.5, 3.0), size(X_spend, 2))
    halfmaxpoint ~ arraydist([Truncated(Normal(loc, scale), 0.1, 1.0)
                              for (loc, scale) in zip(p.locs_spend_halfmaxpoint,
                                                      p.scales_spend_halfmaxpoint)])

    X_spend_transformed = geometric_decay(X_spend, decay_rate, false)
    normalization_factor = sum(X_spend_transformed, dims = 1) ./ sum(X_spend, dims = 1)

    # eps_t = eps(T) # can be used below to avoid log(0) instead of the IF condition
    for j in axes(X_spend_transformed, 2)
        @simd for i in axes(X_spend_transformed, 1)
            @inbounds X_spend_transformed[i, j] = X_spend_transformed[i, j] == 0 ? 0 :
                                                  hill_curve(X_spend_transformed[i, j],
                                                             halfmaxpoint[j], slope[j],
                                                             Val(:safe))
        end
    end

    mu_trend = (trend_offset .+ X_trend * beta_trend)
    # because if halfmaxpoint is mean, then logistic will be 0.5 at mean and if we multiply by 2 and the beta will then be ROAS
    mu_spend = X_spend_transformed ./ normalization_factor *
               (beta_spend .* p.factor_to_roas_of_one)
    # allow X_org to be nothing
    if !isnothing(X_org)
        beta_org ~ arraydist([Truncated(Normal(0, scale), 0.0, 5.0)
                              for scale in p.scales_org])
        mu_org = X_org * beta_org
    else
        mu_org = zeros(T, length(time_index))
    end

    # allow X_context to be nothing
    if !isnothing(X_context)
        beta_context ~ arraydist([Normal(0.0, scale) for scale in p.scales_context])
        mu_context = X_context * beta_context
    else
        mu_context = zeros(T, length(time_index))
    end

    mu = mu_trend + mu_spend + mu_org + mu_context

    sigma ~ Exponential(p.scales_noise)

    y ~ MvNormal(mu, sigma)

    # for effect modelling
    mu_spend_by_var = ((X_spend_transformed ./ normalization_factor)
                       .*
                       (beta_spend .* p.factor_to_roas_of_one)')

    return (; y, mu, mu_trend, mu_spend, mu_org, mu_context, mu_spend_by_var)
end;

# Add TDist ala TuringGLM for fatter tails
@model function model_stage2b(time_index, X_trend, X_spend, X_org, X_context, p,
                              ::Type{T} = Float64) where {T}
    trend_offset ~ Normal(0, p.scales_trend_offset)

    beta_trend ~ filldist(Normal(1.0, p.scales_trend), size(X_trend, 2))
    beta_context ~ arraydist([Normal(0.0, scale) for scale in p.scales_context])

    # only positive // beta_spend is maximum possible effect
    beta_spend ~ filldist(Truncated(Normal(1.0, 1.5), 0.0, 5.0), size(X_spend, 2))
    beta_org ~ arraydist([Truncated(Normal(0, scale), 0.0, 5.0) for scale in p.scales_org])

    # marketing transforms
    decay_rate ~ arraydist([Beta(alpha, beta)
                            for (alpha, beta) in zip(p.decay_rate_alphas,
                                                     p.decay_rate_betas)])

    slope ~ filldist(Truncated(Normal(1.0, 0.5), 0.5, 3.0), size(X_spend, 2))
    halfmaxpoint ~ arraydist([Truncated(Normal(loc, scale), 0.1, 1.0)
                              for (loc, scale) in zip(p.locs_spend_halfmaxpoint,
                                                      p.scales_spend_halfmaxpoint)])

    X_spend_transformed = geometric_decay(X_spend, decay_rate, false)
    normalization_factor = sum(X_spend_transformed, dims = 1) ./ sum(X_spend, dims = 1)

    # eps_t = eps(T) # to avoid log(0)
    for j in axes(X_spend_transformed, 2)
        @simd for i in axes(X_spend_transformed, 1)
            @inbounds X_spend_transformed[i, j] = X_spend_transformed[i, j] == 0 ? 0 :
                                                  hill_curve(X_spend_transformed[i, j],
                                                             halfmaxpoint[j], slope[j],
                                                             Val(:safe))
        end
    end

    mu_trend = (trend_offset .+ X_trend * beta_trend)
    # because if halfmaxpoint is mean, then logistic will be 0.5 at mean and if we multiply by 2 and the beta will then be ROAS
    mu_spend = X_spend_transformed ./ normalization_factor *
               (beta_spend .* p.factor_to_roas_of_one)
    mu_org = X_org * beta_org
    mu_context = X_context * beta_context

    mu = mu_trend + mu_spend + mu_org + mu_context

    sigma ~ Exponential(p.scales_noise)

    nu ~ Gamma(2, 10)

    y ~ arraydist(mu .+ sigma .* TDist.(nu))

    # for effect modelling
    mu_spend_by_var = ((X_spend_transformed ./ normalization_factor)
                       .*
                       (beta_spend .* p.factor_to_roas_of_one)')

    return (; y, mu, mu_trend, mu_spend, mu_org, mu_context, mu_spend_by_var)
end;

#######################################
# Fitting functions and objects to hold the results
Base.@kwdef struct Stage1Fit
    chain::Any
    generated_data::Any
    model::Any
    params::Any
    stage1_fitted_trends::Any
end

"""
    fit(inputs::InputData,params::ParamsStage1,model_func::Function,
        algorithm::Turing.Inference.AdaptiveHamiltonian=NUTS(300,0.65;max_depth=10);
        mcmc_samples=100,mcmc_chains=1)

Fits Stage 1 model (trend components) based on model (`model_func`) with provided data (`inputs`) and priors (`params`)
Defaults to NUTS algorithm

"""
function fit(inputs::InputData, params::ParamsStage1, model_func::Function,
             algorithm::Turing.Inference.AdaptiveHamiltonian = NUTS(300, 0.65;
                                                                    max_depth = 10);
             mcmc_samples = 100, mcmc_chains = 1)
    @unpack y_std, time_std, X_trend, X_hols, X_seas, X_org, X_feat, X_cat = inputs

    model = model_func(time_std,
                       to_masked_matrix(X_trend),
                       to_masked_matrix(X_hols),
                       to_masked_matrix(X_seas),
                       to_masked_matrix(X_feat),
                       to_masked_matrix(X_cat), params)
    cond_model = model | (; y = y_std)

    chain = sample(cond_model, algorithm, MCMCThreads(), mcmc_samples, mcmc_chains)

    # produce basic summary
    quick_nuts_diagnostics(chain, algorithm.max_depth)

    # generated quants
    generated_data = generated_quantities(model, Chains(chain, :parameters))

    stage1_fitted_trends = mean([hcat(tup.mu_trend, tup.mu_hols, tup.mu_seas, tup.mu_cat)
                                 for tup in generated_data])

    return Stage1Fit(; chain, generated_data, model, params, stage1_fitted_trends)
end

function Turing.predict(fitted::Stage1Fit)
    predict(fitted.model, fitted.chain, include_all = false) |>
    x -> mean(x.value.data, dims = (1, 3)) |> vec
end

Base.@kwdef struct Stage2Fit
    chain::Any
    generated_data::Any
    model::Any
    params::Any
end
"""
    fit(inputs::InputData,fitted_stage1::Stage1Fit,params::ParamsStage2,model_func::Function,
        algorithm::Turing.Inference.AdaptiveHamiltonian=NUTS(300,0.65;max_depth=10);
        mcmc_samples=250,mcmc_chains=4)

Fits Stage 2 model (marketing transformations of interest) based on model (`model_func`) with provided data (`inputs`) and priors (`params`)
`fitted_stage1` is a Fit from Stage 1 that holds the trend components (see Documentation for the rationale behind 2-stage fit)
Defaults to NUTS algorithm

"""
function fit(inputs::InputData, fitted_stage1::Stage1Fit, params::ParamsStage2,
             model_func::Function,
             algorithm::Turing.Inference.AdaptiveHamiltonian = NUTS(300, 0.65;
                                                                    max_depth = 10);
             mcmc_samples = 250, mcmc_chains = 4)
    @unpack y_std, time_std, X_spend, X_org, X_context, fit_stage2_mask = inputs
    @unpack stage1_fitted_trends = fitted_stage1

    model = model_func(to_masked_matrix(time_std, fit_stage2_mask),
                       to_masked_matrix(stage1_fitted_trends, fit_stage2_mask),
                       to_masked_matrix(X_spend, fit_stage2_mask),
                       to_masked_matrix(X_org, fit_stage2_mask),
                       to_masked_matrix(X_context, fit_stage2_mask),
                       params)

    y_std_masked = to_masked_matrix(y_std, fit_stage2_mask)

    cond_model = model | (; y = y_std_masked)

    metricT = AHMC.DiagEuclideanMetric
    # metricT=AHMC.DenseEuclideanMetric  # optional

    # It's always best to run multiple chains
    chain = sample(cond_model, algorithm, MCMCThreads(), mcmc_samples, mcmc_chains)

    # generated quants
    generated_data = generated_quantities(model, Chains(chain, :parameters))

    return Stage2Fit(; chain, generated_data, model, params)
end

function Turing.predict(fitted::Stage2Fit)
    predict(fitted.model, fitted.chain, include_all = false) |>
    x -> mean(x.value.data, dims = (1, 3)) |> vec
end

### UTILITIES
"""
    quick_nuts_diagnostics(chain::AbstractMCMC.AbstractChains, max_depth::Int)

Prints quick diagnostics of NUTS algorithm - mostly importantly alerting the user if there have been any divergences
"""
function quick_nuts_diagnostics(chain::AbstractMCMC.AbstractChains, max_depth::Int)
    temp = Chains(chain, :internals)[:acceptance_rate] |> mean
    println("Acceptance rate is: ", @sprintf("%.1f%%", 100*temp))
    # temp = (Chains(chain, :internals)[:hamiltonian_energy_error] .>
    #         Chains(chain, :internals)[:max_hamiltonian_energy_error]) |> sum
    # println("Number of Ham energy errors: $temp")
    temp = Chains(chain, :internals)[:numerical_error] |> sum
    println("Number of all numerical errors: $temp")
    temp > 0 && @warn "NUTS Diagnostics WARNING - DIVERGENCES DETECTED ($(round(Int,temp)))"
    temp = Chains(chain, :internals)[:tree_depth] |> x -> sum(x .>= max_depth)
    println("Number of transitions that exceeded max depth of $max_depth: $temp")
end

"""
    to_masked_matrix(x::DataFrame,mask=trues(size(x,1)))

Convert DataFrame to a matrix and apply a `mask` to its rows if provided
"""
to_masked_matrix(x::AbstractDataFrame, mask = trues(size(x, 1))) = Matrix(x)[mask, :]
to_masked_matrix(x::AbstractMatrix, mask = trues(size(x, 1))) = x[mask, :]
to_masked_matrix(x::AbstractVector, mask = trues(size(x, 1))) = x[mask]
# straight pass through if =Nothing
to_masked_matrix(x::Nothing, mask = trues(1)) = x
