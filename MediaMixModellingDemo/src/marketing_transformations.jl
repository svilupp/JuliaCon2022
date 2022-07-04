using StatsFuns: logistic
using Distributions

"""
    geometric_decay(x::Vector{T},decay_rate,normalize=true) where {T<:Real}

Simple geometric decay transformation
if normalize=true it divides the output by the sum of the geometric series
Note: Does NOT check if decay_rate<1 etc to ensure that the sum convergences to the analytic formula
"""
function geometric_decay(x::AbstractVecOrMat{T},decay_rate,normalize=true) where {T<:Real}
    output=similar(x,promote_type(T,eltype(decay_rate)))    
    output[1,:].=x[1,:]

    for i in 2:size(output,1)
        @inbounds output[i,:].=x[i,:] .+ decay_rate.*output[i-1,:]
    end

    if normalize
        # divide by sum of geometric series
        # geom_srs_sum=dropdims(sum(x,dims=1),dims=1) ./ (one(T) .- decay_rate)
        # output ./= geom_srs_sum'

        # Normalize to the same row-sum as before
        if size(x,2)>1
            output ./= dropdims(sum(output,dims=1) ./ sum(x,dims=1),dims=1)'
        else
            # scalar doesn't have adjoint
            # no need to operate column-wise
            output ./=sum(output)/sum(x)
        end
    end

    return output
end    

"""
    weibull_pdf(x::Vector{T},shape,scale,maxwindowlen) where {T<:Real}

Follows https://facebookexperimental.github.io/Robyn/docs/features
"""
function weibull_pdf(x::Vector{T},shape::S,scale::S,maxwindowlen) where {S<:Real,T<:Real}
    dim_x=size(x,1)
    output=zeros(promote_type(T,S),dim_x)

    w=Weibull(shape,scale)
    kernel=pdf.(Ref(w),0:(maxwindowlen-1))
    ex=extrema(kernel)
    # infinity or zeros
    if ex[1]==ex[2] || !isfinite(ex[2])
        kernel=[one(T);zeros(T,maxwindowlen-1)]
    else
        kernel=kernel./(sum(kernel)+eps(T))
    end
       
    for i in axes(output,1)
        window_width=min(maxwindowlen-1,dim_x-i)
        output[i:(i+window_width)]+= x[i]*kernel[1:(1+window_width)]
    end
    return output
end

"""
    weibull_cdf(x,k,window)

Follows https://github.com/annalectnl/weibull-adstock/blob/master/adstock_weibull_annalect.pdf
"""
function weibull_cdf(x,k,window)
    λ=window/((-log(0.001))^(1/k))    
    return exp(-(x/λ)^k)
end

"""
    weibull_cdf_fast(x,λ,k)

Follows https://github.com/annalectnl/weibull-adstock/blob/master/adstock_weibull_annalect.pdf
Pre-calculates lambda outside via: λ=window/((-log(0.001))^(1/k)) 
"""
function weibull_cdf_fast(x,λ,k)  
    return exp(-(x/λ)^k)
end;



# https://www.physiologyweb.com/calculators/hill_equation_interactive_graph.html
function hill_curve(x, half_max_conc,hill_coef)
    
    return  x^hill_coef/(half_max_conc^hill_coef+x^hill_coef)
end;

# safe implementation for Forwarddiff/small numeric inputs
function hill_curve(x, half_max_conc,hill_coef,::Val{:safe})
    
    return  exp(hill_coef*log(x))/(exp(hill_coef*log(half_max_conc))+exp(hill_coef*log(x)))
end;