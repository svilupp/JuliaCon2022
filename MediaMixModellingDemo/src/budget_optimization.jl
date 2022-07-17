"""
    multiply_and_normalize(v::AbstractVector, multipliers::AbstractVector)

Multiplies a vector `v` (of adspend) by `multipliers` in a way that doesn't change the overall sum
"""
function multiply_and_normalize(v::AbstractVector, multipliers::AbstractVector)
    output = v .* multipliers
    # scale output by the factor of how much it increased, so the sum(output) = sum(a)
    return output .* (sum(v) / sum(output))
end

"""
    convert_budget_multiplier_to_spend_multiplier(spend_prev_trf::AbstractVector,
                                                       factor_to_scale_spend_to_orig::AbstractVector,
                                                       budget_multiplier::AbstractVector)
Converts a relative `budget_multiplier` (a vector of %s for each marketing channel, eg,  [1,1.2,0.8])
 to a `spend_multiplier` (=output), which retains the total amount of spend in the original domain
 Ie, money money between channel while not changing the total spend

Example:
```
spend_prev_trf=X_spend|>Matrix|>sum_columns|>vec
factor_to_scale_spend_to_orig=getindex.(pipe_cache_spend,:xh)
budget_multiplier=ones(size(X_spend,2))
spend_multiplier=convert_budget_multiplier_to_spend_multiplier(spend_prev_trf,factor_to_scale_spend_to_orig,budget_multiplier)
```
"""
function convert_budget_multiplier_to_spend_multiplier(spend_prev_trf::AbstractVector,
                                                       factor_to_scale_spend_to_orig::AbstractVector,
                                                       budget_multiplier::AbstractVector)
    # calculate new spend in original domain
    spend_new_raw = multiply_and_normalize(
                                           # take summed up transformed spend and change to original domain
                                           spend_prev_trf .* factor_to_scale_spend_to_orig,
                                           # multiply by the required weights 
                                           budget_multiplier)
    # apply spend scaling to transformed domain
    spend_new_trf = spend_new_raw ./ factor_to_scale_spend_to_orig

    return spend_multiplier = (spend_new_trf ./ spend_prev_trf)
end

"""
    simulate_revenues_summed(chain_optim::AbstractMCMC.AbstractChains, model_optim::DynamicPPL.Model,
optim_mask::BitVector;extract_key::Symbol=:y)

Samples posterior predictive from model `model_optim` conditioned on Chains `chain_optim`.
Optional - you can provide a mask `optim_mask` (for time dimensions/1st dimension) and Symbol for the posterior predictive value (`extract_key`)

Example:
```
simulations_prev=simulate_revenues_summed(chain_optim,model_orig,optim_mask;extract_key=:y)
```

For budget simulations we replace inputs in the model like:
```
chain_optim=Chains(chain,:parameters)
model_args_prev=model_orig.args;
# replace the old spend with new
model_args_new=merge(model_args_prev,(;X_spend=X_spend_new));
simulations=simulate_revenues_summed(chain_optim,model_stage2a(model_args_new...),optim_mask);
```
"""
function simulate_revenues_summed(chain_optim::AbstractMCMC.AbstractChains,
                                  model_optim::DynamicPPL.Model,
                                  optim_mask::BitVector; extract_key::Symbol = :y)
    simulations = generated_quantities(model_optim, chain_optim)
    simulations_summed = hcat([getfield(s, extract_key) for s in simulations]...) |>
                         x -> @view(x[optim_mask, :]) |> sum_columns |> vec
    return simulations_summed
end

"""
    workflow_budget_to_simulation(
                    chain_optim::AbstractMCMC.AbstractChains, model_orig::DynamicPPL.Model,
                   X_spend::AbstractMatrix, optim_mask::BitVector,
                   factor_to_scale_spend_to_orig::AbstractVector,
                   budget_multiplier::AbstractVector)

For a given context (`budget_multiplier`) produces new adspend (`X_spend_new`) and associated simulated revenues (`smulations_new`)
Used to produced an objective function for the optimization that depends only on `budget_multiplier`
"""
function workflow_budget_to_simulation(chain_optim::AbstractMCMC.AbstractChains,
                                       model_orig::DynamicPPL.Model,
                                       X_spend::AbstractMatrix, optim_mask::BitVector,
                                       factor_to_scale_spend_to_orig::AbstractVector,
                                       budget_multiplier::AbstractVector;
                                       extract_key::Symbol)

    # transformed spend in the optimization period (serves as weights in the original spend domain)
    spend_prev_trf = @view(X_spend[optim_mask, :]) |> sum_columns |> vec
    @assert all(spend_prev_trf .> 0) "Adspend in optimized channels must be larger than zero in the selected period! ($(spend_prev_trf))"

    spend_mutliplier = convert_budget_multiplier_to_spend_multiplier(spend_prev_trf,
                                                                     factor_to_scale_spend_to_orig,
                                                                     budget_multiplier)

    # calculate new spend - change only the rows within optim_mask
    X_spend_new = copy(X_spend)
    X_spend_new[optim_mask, :] .*= spend_mutliplier'

    # replace the old spend with new
    model_args_new = merge(model_orig.args, (; X_spend = X_spend_new))
    simulations_new = simulate_revenues_summed(chain_optim,
                                               model_orig.f(model_args_new...), optim_mask;
                                               extract_key)

    return simulations_new, X_spend_new, spend_mutliplier
end

"""
    generate_objective_func(chain_optim::AbstractMCMC.AbstractChains, 
    model_orig::DynamicPPL.Model,
    X_spend::AbstractMatrix, optim_mask::BitVector,
    factor_to_scale_spend_to_orig::AbstractVector,
    loss_func::Function,
    bounds::AbstractVecOrMat;
    simulations_basecase = nothing,
    extract_key::Symbol=:y)

Generates objective function that depends only on `budget_multiplier` input and returns simulated revenues and other requirements of Metaheuristics
Checks that implied `spend_multiplier_new` (calculated from `budget_multiplier`) is within the provided `bounds` as auxilary objective

Notes:
- This version is suitable if optimization algorithm struggles to find solutions that maintain the same adspend (it balances it under the hood)
- The simplest `loss_func` function that you can use is `identity`!

Optional: `simulation_basecase` can be provided to speed up the optimization (ie, avoid re-computing the revenue under the old budget)

If `optim_mask` subsets data (ie, !=trues(size(X_spend,1))) then only the underlying segment of the `X_spend` matrix is changed 
 and only that portion of simulated revenues is considered for objective function
"""
function generate_objective_func(chain_optim::AbstractMCMC.AbstractChains,
                                 model_orig::DynamicPPL.Model,
                                 X_spend::AbstractMatrix, optim_mask::BitVector,
                                 factor_to_scale_spend_to_orig::AbstractVector,
                                 loss_func::Function,
                                 bounds::AbstractVecOrMat;
                                 simulations_basecase = nothing,
                                 extract_key::Symbol = :y)
    # assumes X_spend is already a Matrix

    return budget_multiplier -> begin
        if isnothing(simulations_basecase)
            simulations_prev = simulate_revenues_summed(chain_optim, model_orig, optim_mask;
                                                        extract_key)
        else
            simulations_prev = simulations_basecase
        end

        simulations_new, X_spend_new, spend_mutliplier_new = workflow_budget_to_simulation(chain_optim,
                                                                                           model_orig,
                                                                                           X_spend,
                                                                                           optim_mask,
                                                                                           factor_to_scale_spend_to_orig,
                                                                                           budget_multiplier;
                                                                                           extract_key)

        # for explanation of this format, see Metaheuristics docs
        # we minimize so put - in front
        fx = -mean(loss_func.(simulations_new .- simulations_prev))

        # check that actual spend_multiplier is within bounds
        if isnothing(bounds)
            # assumes bounds share required by Metaheuristics which is [lower;upper]
            gx = [0.0]
        else
            gx = [
                  # smaller than upper bound
                  spend_mutliplier_new .- bounds'[:, 2];
                  # larger than lower bound
                  bounds'[:, 1] .- spend_mutliplier_new]
        end
        hx = [0.0]

        #returns
        fx, gx, hx
    end
end

"""
    generate_objective_func(
        chain_optim::AbstractMCMC.AbstractChains, 
            model_orig::DynamicPPL.Model,
        X_spend::AbstractMatrix, optim_mask::BitVector,
        spend_raw_sum::AbstractVector,
        loss_func::Function = identity; 
        simulations_basecase = nothing,
        extract_key::Symbol=:y)

Objective function generator which runs directly off a `budget_multiplier` operating on the ad spend in original domain
 and producing uplift in terms of revenues (or a chosen `extract_key`)
Auxilary objective (gx) minimizes the delta between original adspend and new adspend (it must be strictly smaller or equal)

Notes:
- This version leaves ad spend budget balancing to the optimization algorithm
- The simplest `loss_func` function that you can use is `identity`!

Optional: `simulation_basecase` can be provided to speed up the optimization (ie, avoid re-computing the revenue under the old budget)

If `optim_mask` subsets data (ie, !=trues(size(X_spend,1))) then only the underlying segment of the `X_spend` matrix is changed 
 and only that portion of simulated revenues is considered for objective function

Example:

```
# Prepare inputs
chain_optim=Chains(chain,:parameters)
simulations_prev=simulate_revenues_summed(chain_optim,model_orig,optim_mask;extract_key=:mu)

# boundaries on possible solution
lower_bound = 0.5*ones(length(cols_spend)) # max 50% reduction
upper_bound = 1.5*ones(length(cols_spend)) # max 50% increase
bounds = [lower_bound upper_bound]'

# Bayesian Decision Theory -- how to weigh the outcomes across the posterior distribution
# define a simple asymmetric (risk-averse) loss function
loss_func(x)=x>0 ? 0.5x : x

# All channels must have some spend in the optimization period!
@assert all((@view(X_spend[optim_mask,:])|>sum_columns) .>0)

# Method with direct budget multiplier
# spend_raw_sum is masked with optim_mask!
spend_raw_sum=revert_pipe_spend(X_spend[optim_mask,:])|>sum_columns

objective_func=generate_objective_func(
    chain_optim,model_orig,Matrix(X_spend),optim_mask,
    spend_raw_sum,loss_func;simulations_basecase=simulations_prev,extract_key=:mu)

```
"""
function generate_objective_func(chain_optim::AbstractMCMC.AbstractChains,
                                 model_orig::DynamicPPL.Model,
                                 X_spend::AbstractMatrix, optim_mask::BitVector,
                                 spend_raw_sum::AbstractVector,
                                 loss_func::Function = identity;
                                 simulations_basecase = nothing,
                                 extract_key::Symbol = :y)
    # assumes X_spend is already a Matrix

    return budget_multiplier -> begin
        if isnothing(simulations_basecase)
            simulations_prev = simulate_revenues_summed(chain_optim, model_orig, optim_mask;
                                                        extract_key)
        else
            simulations_prev = simulations_basecase
        end

        # calculate new spend - change only the rows within optim_mask
        X_spend_new = copy(X_spend)
        X_spend_new[optim_mask, :] .*= budget_multiplier'

        # replace the old spend with new
        model_args_new = merge(model_orig.args, (; X_spend = X_spend_new))
        simulations_new = simulate_revenues_summed(chain_optim,
                                                   model_orig.f(model_args_new...),
                                                   optim_mask; extract_key)

        # for explanation of this format, see Metaheuristics docs
        # we minimize so put `-` in front
        fx = -mean(loss_func.(simulations_new .- simulations_prev))

        # Adspend must not have increased
        gx = [sum(spend_raw_sum .* budget_multiplier) - sum(spend_raw_sum)]

        hx = [0.0]

        # returns
        fx, gx, hx
    end
end

"""
    threaded_objective_func(budget_multipliers,objective_func)

Parallelizes provided function `objective_func` for Metaheuristics optimization loop
 by leveraging available threads in your Julia instance

Example:

Make sure to change the keyword to `parallel_evaluation=true`
```
options = Metaheuristics.Options(time_limit=10.,debug=false,parallel_evaluation=true)
@time result = Metaheuristics.optimize(x->threaded_objective_func(x,objective_func), bounds, 
    Metaheuristics.ECA(N=7*2*length(cols_spend),K=7,η_max=2.,options=options))
```
"""
function threaded_objective_func(budget_multipliers, objective_func)
    N = size(budget_multipliers, 1)
    fx, gx, hx = zeros(N), zeros(N, 1), zeros(N, 1)

    Threads.@threads for i in 1:size(budget_multipliers, 1)
        @inbounds f_, g_, h_ = objective_func(@view budget_multipliers[i, :])
        @inbounds fx[i] = f_     # objective function
        @inbounds gx[i, :] .= g_
        @inbounds hx[i, :] .= h_
    end
    return fx, gx, hx
end
