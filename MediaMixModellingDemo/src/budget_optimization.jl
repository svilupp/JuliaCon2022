# 1-liners

# multiply in a weight that doesn't change the sum
function multiply_and_normalize(a, multipliers)
    output = a .* multipliers
    # scale output by the factor of how much it increased, so the sum(output) = sum(a)
    return output .* (sum(a) / sum(output))
end

# convert_budget_multiplier_to_spend_multiplier
# spend_prev_trf=X_spend|>Matrix|>sum_columns|>vec
# factor_to_scale_spend_to_orig=getindex.(pipe_cache_spend,:xh)
# example: convert_budget_multiplier_to_spend_multiplier(spend_prev_trf,factor_to_scale_spend_to_orig,budget_multiplier)
function convert_budget_multiplier_to_spend_multiplier(spend_prev_trf,
                                                       factor_to_scale_spend_to_orig,
                                                       budget_multiplier)
    spend_new_raw = multiply_and_normalize(
                                           # take summed up transformed spend and change to original domain
                                           spend_prev_trf .* factor_to_scale_spend_to_orig,
                                           budget_multiplier)
    spend_new_trf = spend_new_raw ./ factor_to_scale_spend_to_orig
    spend_multiplier = (spend_new_trf ./ spend_prev_trf)
end

# Example
# chain_optim=Chains(chain,:parameters)
# model_args_prev=model_orig.args;
# # replace the old spend with new
# model_args_new=merge(model_args_prev,(;X_spend=X_spend_new));
# simulations=simulate_revenues_summed(chain_optim,model_stage2a(model_args_new...),optim_mask);
function simulate_revenues_summed(chain_optim, model_optim, optim_mask)
    simulations = generated_quantities(model_optim, chain_optim)
    simulations_summed = hcat([s.y for s in simulations]...) |>
                         x -> @view(x[optim_mask, :]) |> sum_columns |> vec
    return simulations_summed
end

# wrapper that takes a budget and turns it into simulations
function workflow_budget_to_simulation(chain_optim, model_orig,
                                       X_spend, optim_mask,
                                       spend_prev_trf, factor_to_scale_spend_to_orig,
                                       budget_multiplier)

    # assumes X_spend is already a Matrix
    spend_mutliplier = convert_budget_multiplier_to_spend_multiplier(spend_prev_trf,
                                                                     factor_to_scale_spend_to_orig,
                                                                     budget_multiplier)

    # calculate new spend
    X_spend_new = X_spend .* spend_mutliplier'

    # replace the old spend with new
    model_args_new = merge(model_orig.args, (; X_spend = X_spend_new))
    simulations_new = simulate_revenues_summed(chain_optim,
                                               model_stage2a(model_args_new...), optim_mask)

    return simulations_new, X_spend_new, spend_mutliplier
end

function generate_objective_func(chain_optim, model_orig,
                                 X_spend, optim_mask,
                                 spend_prev_trf, factor_to_scale_spend_to_orig,
                                 loss_func = identity, simulations_basecase = nothing,
                                 bounds = nothing)
    # assumes X_spend is already a Matrix

    return budget_multiplier -> begin
        if isnothing(simulations_basecase)
            simulations_prev = simulate_revenues(chain_optim, model_orig, optim_mask)
        else
            simulations_prev = simulations_basecase
        end

        simulations_new, X_spend_new, spend_mutliplier_new = workflow_budget_to_simulation(chain_optim,
                                                                                           model_orig,
                                                                                           X_spend,
                                                                                           optim_mask,
                                                                                           spend_prev_trf,
                                                                                           factor_to_scale_spend_to_orig,
                                                                                           budget_multiplier)

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

# paralellization of the objective function for faster optimization
# example:
# temp=ones(5,3)
# @btime objective_func_parallel($temp)
# Note: doesn't work for contraint optim (Error in create_child with the sub-criteria)
# depends on objective function being called `objective_func`
function objective_func_parallel(budget_multipliers)
    fx = zeros(Float64, size(budget_multipliers, 1))
    gx = zeros(Float64, size(budget_multipliers, 1), 2 * size(budget_multipliers, 2))
    hx = zeros(Float64, size(budget_multipliers, 1))

    Threads.@threads for i in axes(budget_multipliers, 1)
        @inbounds f_, g_, h_ = objective_func(@view budget_multipliers[i, :])
        @inbounds fx[i] = f_
        @inbounds gx[i, :] .= g_
        # @inbounds hx[i]=h_[1]
    end
    fx, gx, hx
end
