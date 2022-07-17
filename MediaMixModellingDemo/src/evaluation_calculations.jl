"""
    sum_columns(x::AbstractMatrix)

Unified interface that sums columns of a provided Matrix/DataFrame/Vector
Returns a Vector (!)
"""
sum_columns(x::AbstractMatrix) = sum(x, dims = 1) |> vec
sum_columns(x::AbstractDataFrame) = sum(Matrix(x), dims = 1) |> vec
sum_columns(x::AbstractVector) = sum(reshape(x, :, 1), dims = 1) |> vec

percentage_share(x) = x ./ sum(x)

"""
    getflatsamples(chain,groupname)

Extract a group of variables under name `groupname` from `chain` 
 and flattens all samples into the first dimension
 ie, outputs a dimension: (num_samples*num_chains,num_variables)
"""
function getflatsamples(chain, groupname)
    temp = chain[namesingroup(chain, groupname)].value.data
    dim_vars = size(temp, 2)

    return temp |>
           # wrap chains dim for data dim
           x -> permutedims(x, (1, 3, 2)) |>
                # flatten chain dim
                # output: num_samples X num_obs
                x -> reshape(x, :, dim_vars)
end

"""
    extract_fitted_effects(::Val{:model_stage2a},generated_quant,
        extract_keys=[:mu_trend,:mu_org,:mu_context,:mu_spend_by_var],mask=nothing)

Extract specific keys from the `generated_quantities` produced by Turing `generated_quantities()` and concatenates them
It should be implemented for each model to ensure the right logic

Returns: a Vector (!)

Example:
```mean_fitted_effects(Val(:model_stage2a),stage2_fit_allsamples)```
```mean_fitted_effects(Val(:model_stage2a),stage2_fit_allsamples;extract_keys=[:mu_spend_by_var])```
"""
function mean_fitted_effects(::Val{:model_stage2a}, generated_quant;
                             extract_keys = [
                                 :mu_trend,
                                 :mu_org,
                                 :mu_context,
                                 :mu_spend_by_var,
                             ], mask = nothing)
    if isnothing(mask)
        mask = Colon()
    end

    # concat the keys -> mask -> sum -> take a mean across samples -> flatten to a vector
    return mean([sum(@view(hcat(getindex.(Ref(tup), extract_keys)...)[mask, :]), dims = 1)
                 for tup in generated_quant]) |> vec
end

"""
    calc_roas(effect::Number, spend::Number, factor_spend_to_effect::Number=1)

Calculates ROAS with an option to normalize the effect to the right scale (if Y and X are scaled)
"""
function calc_roas(effect::Number, spend::Number, factor_spend_to_effect::Number = 1)
    effect / spend / factor_spend_to_effect
end

"""
    calc_roas(effects::AbstractArray, spends::AbstractArray, factors_spend_to_effect::AbstractArray,weights::AbstractArray)
    
Calculates Total ROAS (of all Ad channels) as a weighted-average of individual ROAS' weighted by the raw spend
"""
function calc_roas(effects::AbstractArray, spends::AbstractArray,
                   factors_spend_to_effect::AbstractArray, weights::AbstractArray)
    individual_roas = calc_roas.(effects, spends, factors_spend_to_effect)
    return sum(individual_roas .* weights) / sum(weights)
end

"""
    calc_mroas(x::Number, delta::Number, chain::AbstractMCMC.AbstractChains, p, slice_idx::Int)

Calculate marginal ROAS (mROAS) at a given point `x` with a `delta` (=step size) for a variable under `slice_idx` 
 (ie, 3rd channel would have `slice_idx=3`)

Example: 
Calculate mROAS for all Ad spend variables with `delta=0.01`
```
# p2=ParamsStage2() # model parameters from stage 2
# chain is the result of fitting of stage 2 model
# cols_spend are ad spend column names
mroas_at_mean=[calc_mroas(p2.adspend_mean_nonzero[idx],0.01,chain,p2,idx)[1] for idx in 1:length(cols_spend)]
```
"""
function calc_mroas(x::Number, delta::Number, chain::AbstractMCMC.AbstractChains, p,
                    slice_idx::Int)
    # utility functions
    mean_response = x -> mean(saturate_adspend(x, chain, p.factor_to_roas_of_one)[:,
                                                                                  slice_idx,
                                                                                  :])
    std_response = x -> std(saturate_adspend(x, chain, p.factor_to_roas_of_one)[:,
                                                                                slice_idx,
                                                                                :])

    std_error = (std_response(x + delta) - std_response(x)) / delta /
                p.units_ratio_spend_to_y[slice_idx]
    mroas = (mean_response(x + delta) - mean_response(x)) / delta /
            p.units_ratio_spend_to_y[slice_idx]

    return mroas, std_error
end

"""
    saturate_adspend(x::Number, chain::AbstractMCMC.AbstractChains, factor_to_roas_of_one::AbstractVector)

Extracts Hill Curve parameters and the corresponding beta coefficients from provided `chain`
 and applies them to a provided point (`x`)

WARNING: Depends on model implementation using the same RVs as model_stage2a, ie, `beta_spend`,`halfmaxpoint`,`slope`
"""
function saturate_adspend(x::Number, chain::AbstractMCMC.AbstractChains,
                          factor_to_roas_of_one::AbstractVector)
    beta_spend = chain[namesingroup(chain, "beta_spend")].value.data
    halfmaxpoint = chain[namesingroup(chain, "halfmaxpoint")].value.data
    slope = chain[namesingroup(chain, "slope")].value.data

    # dims: samples X dim_vars X dim_obs
    output = similar(beta_spend)
    for k in axes(beta_spend, 3)
        for j in axes(beta_spend, 2)
            @simd for i in axes(beta_spend, 1)
                @inbounds output[i, j, k] = (hill_curve(x, halfmaxpoint[i, j, k],
                                                        slope[i, j, k])
                                             * beta_spend[i, j, k] *
                                             factor_to_roas_of_one[j])
            end
        end
    end
    return output
end
