@with_kw struct ParamsStage1
    scales_trend_offset=y_std[begin:min(begin+5,end)]|>mean
    scales_growth_trend=y_std[end]/y_std[begin]
    scales_trend=0.2#/size(X_trend,2)
    scales_hols=0.3
    scales_seas=(std(y_std) ./ std.(eachcol(X_seas))) .|>x->min(max(x,0.05),10)
    scales_feat=(std(y_std) ./ std.(eachcol(X_feat))) .|>x->min(max(x,0.05),10)
    scales_noise=0.2
    
    cat_levels=cat_levels
end

@model function model_stage1a(time_index,X_trend,X_hols,X_seas,X_feat,X_cat,p,::Type{T} = Float64) where {T}

    growth_trend ~ Normal(0, p.scales_growth_trend)
    
    # replaces intercept
    alpha ~ filldist(Normal(0,p.scales_trend_offset),p.cat_levels)
    
    if !isnothing(X_trend)
        beta_trend ~ filldist(Normal(0,p.scales_trend),size(X_trend,2))
    end
    
    beta_hols ~ filldist(Normal(0,p.scales_hols),size(X_hols,2))
    # beta_seas ~ filldist(Normal(0,0.3),size(X_seas,2))
    # beta_feat ~ filldist(Normal(0,0.3),size(X_feat,2))
    beta_seas ~ arraydist([Normal(0,scale) for scale in p.scales_seas])
    beta_feat ~ arraydist([Normal(0,scale) for scale in p.scales_feat])


    
    mu_trend=time_index .* growth_trend 
    
    if !isnothing(X_trend)
        mu_trend .+= X_trend*beta_trend
    end
    
    mu_hols=(X_hols*beta_hols)
    mu_seas=X_seas*beta_seas
    mu_feat=X_feat*beta_feat
    mu_cat=alpha[X_cat]
    
    mu=mu_trend+mu_hols+mu_seas+mu_feat+mu_cat
    
    sigma ~ Exponential(p.scales_noise)
    
    y ~ MvNormal(mu, sigma)
    
    return (;y,mu_trend,mu_hols,mu_seas,mu_feat,mu_cat)
end;


@with_kw struct ParamsStage2
    scales_trend_offset=0.3
    scales_trend=0.2
    scales_noise=0.3
    scales_context=std(y_std) ./ std.(eachcol(X_context))
    scales_org=std(y_std) ./ std.(eachcol(X_org))

    decay_rate_alphas=[10,20,1]
    decay_rate_betas=[10,10,20]
    
    adspend_mean_nonzero=[mean(c[c .!= 0]) for c in eachcol(X_spend)]
    adspend_median=median.(eachcol(X_spend))
    
    locs_spend_halfmaxpoint=adspend_mean_nonzero #center at mean spend
    scales_spend_halfmaxpoint=0.3*ones(size(X_spend,2)) # allow for 0.3 move around that

    #scaling from data generation
    # X_spend is technically already scaled down by 1/20, so we need to mimic (to hit the same beta)
    # if we decreased spend, we need increase effect proportionally - same for Y scaling
    units_ratio_spend_to_y=getindex.(pipe_cache_spend,:xh)/pipe_cache_y[1].xh #./ [20,10,20] 
    # halfmaxpoint is set to mean, then hill curve at that point will be 0.5
    # if we multiply value by 2, the beta coef = ROAS
    factor_to_roas_of_one=units_ratio_spend_to_y .* 2
end


@model function model_stage2a(time_index,X_trend,X_spend,X_org,X_context,p,::Type{T} = Float64) where {T}

    trend_offset ~ Normal(0, p.scales_trend_offset)
    
    beta_trend ~ filldist(Normal(1.,p.scales_trend),size(X_trend,2))
    beta_context ~ arraydist([Normal(0.,scale) for scale in p.scales_context])
    
    # only positive // beta_spend is maximum possible effect
    beta_spend ~ filldist(Truncated(Normal(1.,1.5),0.,5.),size(X_spend,2))
    beta_org ~ arraydist([Truncated(Normal(0,scale),0.,5.) for scale in p.scales_org])
    
    # marketing transforms
    decay_rate ~ arraydist([Beta(alpha,beta)
            for (alpha,beta) in zip(p.decay_rate_alphas,p.decay_rate_betas)])

    slope ~ filldist(Truncated(Normal(1.,.5),0.5,3.),size(X_spend,2))
    halfmaxpoint ~ arraydist([Truncated(Normal(loc,scale),0.1,1.) 
                for (loc,scale) in zip(p.locs_spend_halfmaxpoint,p.scales_spend_halfmaxpoint)])
    
    X_spend_transformed=geometric_decay(X_spend,decay_rate,false)
    normalization_factor=sum(X_spend_transformed,dims=1)./sum(X_spend,dims=1)
    
    eps_t=eps(T) # to avoid log(0)
    for j in axes(X_spend_transformed,2)
        @simd for i in axes(X_spend_transformed,1) 
           @inbounds X_spend_transformed[i,j]=hill_curve(eps_t+X_spend_transformed[i,j],halfmaxpoint[j],slope[j],Val(:safe))
        end
    end
    
    mu_trend=(trend_offset .+ X_trend*beta_trend)
    # because if halfmaxpoint is mean, then logistic will be 0.5 at mean and if we multiply by 2 and the beta will then be ROAS
    mu_spend=X_spend_transformed ./ normalization_factor * (beta_spend .* p.factor_to_roas_of_one)
    mu_org=X_org*beta_org
    mu_context=X_context*beta_context
    
    mu=mu_trend+mu_spend+mu_org+mu_context
    
    sigma ~ Exponential(p.scales_noise)
    
    y ~ MvNormal(mu, sigma)
    
    # for effect modelling
    mu_spend_by_var=((X_spend_transformed ./ normalization_factor)
                    .* (beta_spend .* p.factor_to_roas_of_one )') |> x->sum(x,dims=1)
    
    return (;y,mu,mu_trend,mu_spend,mu_org,mu_context,mu_spend_by_var)
end;




### UTILITIES

function quick_nuts_diagnostics(chain)
    temp=Chains(chain,:internals)[:acceptance_rate]|>mean
    println("Acceptance rate is: ",@sprintf("%.1f%%",100*temp))
    temp=(Chains(chain,:internals)[:hamiltonian_energy_error].>Chains(chain,:internals)[:max_hamiltonian_energy_error])|>sum
    println("Number of Ham energy errors: $temp")
    temp=Chains(chain,:internals)[:numerical_error]|>sum
    println("Number of all numerical errors: $temp")
    temp=Chains(chain,:internals)[:tree_depth] |> x-> sum(x .>= max_depth) 
    println("Number of transitions that exceeded max depth of $max_depth: $temp")
end