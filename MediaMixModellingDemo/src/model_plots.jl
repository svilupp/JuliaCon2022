"""
    plot_priors_decay_rate(p,cols=nothing)

Plots priors for decay rate in terms of Beta distributions

Example: 
```
p2=ParamsStage2()
cols_spend=nothing # available from data prep
plot_priors_decay_rate(p2,cols_spend)
```
"""
function plot_priors_decay_rate(p, cols = nothing)
    nvars = length(p.decay_rate_alphas)
    if isnothing(cols)
        cols = ["Var_$i" for i in 1:nvars]
    end

    plot_array = [plot(Beta(a, b), title = "$name",
                       label = @sprintf("Mean %.2f", mean(Beta(a, b))))
                  for (a, b, name) in zip(p.decay_rate_alphas, p.decay_rate_betas, cols)]

    return plot(plot_array...,
                layout = (1, nvars),
                size = (nvars * 150, 150), titlefontsize = 8,
                xformatter = x -> @sprintf("%.1f", x))
end
