# 1-liners
sum_columns(x)=sum(x,dims=1)
percentage_share(x)=x./sum(x)

function getflatsamples(chain,groupname)
    temp=chain[namesingroup(chain,groupname)].value.data 
    dim_vars=size(temp,2)

    return temp|>
        # wrap chains dim for data dim
        x->permutedims(x,(1,3,2))|>
        # flatten chain dim
        # output: num_samples X num_obs
        x->reshape(x,:,dim_vars)
end

# calculate ROAS from generated model for all variabels
# X_spend_raw=revert(MinMax(),X_spend,pipe_cache_spend)
# example: roas_total=calc_roas_total(stage2_fit_allsamples,X_spend_raw)
# Expects all variables to be masked already (subset)
function calc_roas_total(generated_quantities,X_spend_raw)
    # extract mean effect by each variable
    adspend_effect_mean=[tup.mu_spend_by_var for tup in generated_quantities] |> mean |>vec
    
    effect_raw=sum(revert_pipe_y(adspend_effect_mean).y)
    
    # sum of marketing spend in the period
    adspend_spend_raw=X_spend_raw|>Matrix|>sum

    roas = effect_raw / adspend_spend_raw
    
    return roas
end

# calculate ROAS from generated model for each marketing spend variable (conditioned on fitted MCMCChains)
# any masking has to be done before generating quantities / providing X_spend
# example: roas=calc_roas(stage2_fit_allsamples,X_spend,p2.units_ratio_spend_to_y)
function calc_roas(generated_quantities,X_spend,units_ratio_spend_to_y)
    # extract mean effect by each variable
    adspend_effect_mean=[tup.mu_spend_by_var for tup in generated_quantities] |> mean |>vec
    
    # sum of observed spend in the period
    adspend_spend=(X_spend|>Matrix|>sum_columns|>vec)

    # normalize by ratio of y/x
    roas = adspend_effect_mean ./  adspend_spend ./ units_ratio_spend_to_y
    
    return roas
end

# calculate marginal roas at a given point x, with delta for variable under slice_idx
# example: mroas_at_mean=[calc_mroas(p2.adspend_mean_nonzero[idx],0.01,chain,p2,idx)[1] for idx in 1:length(cols_spend)]
function calc_mroas(x,delta,chain,p,slice_idx)
    # utility functions
    mean_response=x->mean(saturate_adspend(x,chain,p.factor_to_roas_of_one)[:,slice_idx,:])
    std_response=x->std(saturate_adspend(x,chain,p.factor_to_roas_of_one)[:,slice_idx,:])

    std_error=(std_response(x+delta)-std_response(x)) / delta/ p.units_ratio_spend_to_y[slice_idx]
    mroas=(mean_response(x+delta)-mean_response(x)) / delta / p.units_ratio_spend_to_y[slice_idx]

    return mroas,std_error
end

# extracts hill curve parameters and the corresponding beta coefficient
# applies to a given point (=x) all curves based on provided samples in Chain
function saturate_adspend(x::Number,chain,factor_to_roas_of_one)
    beta_spend=chain[namesingroup(chain,"beta_spend")].value.data 
    halfmaxpoint=chain[namesingroup(chain,"halfmaxpoint")].value.data 
    slope=chain[namesingroup(chain,"slope")].value.data 

    # dims: samples X dim_vars X dim_obs
    output=similar(beta_spend)
    for k in axes(beta_spend,3)
        for j in axes(beta_spend,2)
            @simd for i in axes(beta_spend,1)
                @inbounds output[i,j,k] = (
                    hill_curve(x,halfmaxpoint[i,j,k],slope[i,j,k])
                    *beta_spend[i,j,k]*factor_to_roas_of_one[j]
                )
            end
        end
    end
    return output
end