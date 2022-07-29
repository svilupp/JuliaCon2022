```@meta
EditURL = "https://github.com/svilupp/JuliaCon2022/blob/main/MediaMixModellingDemo/1-demo-high-level.jl"
```

# JuliaCon - MMM Demo

Example application of Media Mix Modelling to optimize marketing spend

High-level workflow:
- Input and transform data
    - Provide column names of the corresponding to groups of variables (eg, `cols_spend` for Adspend variables to be modelled)
    - Variables that can have positive-only effects will be standardized to 0-1 range
    - Variables with any-signed effects will be at least standardized via Z-Score
- Set priors / conversion factors
    - Ie, use your domain knowledge to set realistic marketing transforms' parameters
- Stage 1: Fit the trend
    - Extract separate series for growth trend, seasonality, holidays, organic variables
- Stage 2: Fit the coefficients for marketing transformation
    - Validate the fit (Rhat, traceplots, etc)
- Quantify the contributions + ROAS of the current marketing spend
- Optimize the marketing budget to maximize the marketing contribution
    - Define a loss function that reflects your business' decision-making process
    - Evaluate the results of the optimization + inherent uncertainty

For more details on the methodology and practical tips visit [MMM Demo Docs](https://svilupp.github.io/JuliaCon2022/dev/)

````@example 1-demo-high-level

import Optim
import Metaheuristics

using Logging
#### Optional: disable logging
# disable_logging(Logging.Info);
import DisplayAs

using MediaMixModellingDemo
# MMMDemo reexports: Turing,DataFramesMeta,Distributions,TableTransforms,Plots,StatsPlots

# Optional for better display:
ENV["LINES"]=200
ENV["COLUMNS"]=600
Threads.nthreads();
nothing #hide
````

## Generate

Let's generate some example data where we know "truth", so we can test this implementation

You can see the "truth" if you have logging enabled for level INFO

````@example 1-demo-high-level
Y,X,col_names=create_dataset("2020-02-01",105,0);

# Let's apply standard notation of df being the source of data
df=X
df[!,:revenue]=vec(sum.(eachrow(Y)));

img=plot(df.revenue,title="Generated revenues",label="",dpi=110)
img=DisplayAs.PNG(img) # trick for Literate.jl
````

# Modelling

We will fit the model in 2 stages to deal with the under-specification.

As a convention the deterministic location of the observed variable (mean of the RV `y`) will be called `mu` and its components will be all prefixed as such, eg,
- `mu_trend` for the trend component,
- `mu_seas` for the seasonality component,
- `mu_hols` for the holidays component,
- `mu_context` for the contributions of the contextual variables,
- `mu_org`
- `mu_spend` for the contribution of all Ad spend variables together
- `mu_spend_by_var` for the contributions of each individual spend variables (ie, number of columns = number of Ad spend variables)

(See `src/model_definition.jl` for more details)

All these deterministic components will be exposed by DynamicPPL in the return statement to be easily extracted via `generated_quantities()` conditioned on fitted samples (a `chains` object). In other words, given a model fit, we can easily extract all the above components by playing the samples through the model.

Let's choose what model implementation we want:

````@example 1-demo-high-level
# Model used in stage1
const MODEL_NAME_PREFIT="model_stage1a"
model_func_stage1=getfield(MediaMixModellingDemo,Symbol(MODEL_NAME_PREFIT))

# Model used in stage2
const MODEL_NAME="model_stage2a"
model_func_stage2=getfield(MediaMixModellingDemo,Symbol(MODEL_NAME))

# Modifier of chart titles
const EXPERIMENT_NAME=" JuliaCon"
pplot=ParamsPlot(title_suffix=EXPERIMENT_NAME)
;
nothing #hide
````

## Data

Let's identify seasonalities in our data (for trend modelling)

Pick the ones with the highest value (usually only 1 or 2 max)

````@example 1-demo-high-level
# what's the seasonality
img=plot_periodogram(df.revenue .- mean(df.revenue),3)
img=DisplayAs.PNG(img) # trick for Literate.jl
````

Let's build our input data object:

````@example 1-demo-high-level
@doc build_inputs
````

````@example 1-demo-high-level
inputs=build_inputs(df;
    col_target="revenue",col_datetime="dt",col_time_std="time_std",col_cat="events",
    cols_context=[c for c in col_names.cols_context if c!="newsletters"],
    cols_organic=["newsletters"],
    cols_hols=["hols_ind"],
    cols_spend=col_names.cols_spend,
    seasonality_periods=[4.],
    spline_degree=0
);
nothing #hide
````

## Stage 1 - Prophet-like model

Let's fit a model to the trend, seasonality and holidays

Note: We do not adstock/saturate the marketing spend at this stage

````@example 1-demo-high-level
p1=set_priors(
    ParamsStage1(
        model_name=Val(Symbol(MODEL_NAME_PREFIT)),
        scales_trend=0.2,
        scales_hols=0.3,
        scales_noise=0.2,
        cat_levels=1
    ),
    inputs
);
nothing #hide
````

### Fit

````@example 1-demo-high-level
stage1fit=fit(inputs,p1,model_func_stage1,NUTS(300,0.65;max_depth=10));
nothing #hide
````

````@example 1-demo-high-level
# Optional: Check the priors used and if they are sensible
# y_prior=mapreduce(x->rand(stage1fit.model).y,hcat,1:100)|>vec
# plot_prior_predictive_histogram(inputs.y_std,y_prior,ParamsPlot())
````

````@example 1-demo-high-level
img=plot_model_fit_by_period(inputs.y_std,predict(stage1fit),ParamsPlot())
img=DisplayAs.PNG(img) # trick for Literate.jl
````

## Stage 2 - Fit the Marketing Drivers

Let's use the fit from the first stage and focus mostly on the marketing variables (including their adstock/saturation transformations)

````@example 1-demo-high-level
img=plot_model_fit_by_period(inputs.y_std,sum(stage1fit.stage1_fitted_trends,dims=2)|>vec,
    ParamsPlot(title_suffix=" for the Trend Components"))
img=DisplayAs.PNG(img) # trick for Literate.jl
````

### Fit

````@example 1-demo-high-level
p2=set_priors(
    ParamsStage2(
        model_name=Val(Symbol(MODEL_NAME)),
        scales_trend_offset=0.3,
        scales_trend=0.2,
        scales_noise=0.3,
    ),
    inputs
);
# quick check of decay_rate priors
p2 = set_priors_stage2_decay_rates(["digital","tv","digital"],decay_rates_types_dictionary,
    p2,inputs.cols_spend)

# let's check if our priors make sense given te data
sanity_check_priors(p2;inputs.X_spend,inputs.X_context,inputs.X_org);

# let's visualize the selected decay rates
img=plot_priors_decay_rate(p2,inputs.cols_spend)
img=DisplayAs.PNG(img) # trick for Literate.jl
````

````@example 1-demo-high-level
stage2fit=fit(inputs,stage1fit,p2,model_func_stage2,
    NUTS(300,0.65;max_depth=10);
    mcmc_samples=250,mcmc_chains=4);
nothing #hide
````

````@example 1-demo-high-level
# Optional1: Check the priors used and if they are sensible
# y_prior=mapreduce(x->rand(stage1fit.model).y,hcat,1:100)|>vec
# plot_prior_predictive_histogram(inputs.y_std,y_prior,ParamsPlot())

## Optional2: Show the posterior samples of quantities of interest
# stage2fit.chain

## Optional3: Diagnostics if the parameter space has been properly explored
# corner(chain[[namesingroup(stage2fit.chain,"slope")...,
#             namesingroup(stage2fit.chain,"decay_rate")...]],
#     guidefontsize=8,size=(1000,1000))

# corner(chain[[namesingroup(chain,"slope")...,
# namesingroup(chain,"beta_spend")...,Symbol("beta_org[1]")]],
#     guidefontsize=8,size=(1000,1000))

# Optional4: Diagnostics of the mixing in the chains
# Display traceplots -- look for "fuzzy caterpillars"! Any flat lines (ie, chain getting stuck on some values) indicate problems!
# plot(chain)
````

## Evaluation

````@example 1-demo-high-level
img=plot(stage2fit,inputs,pplot)
img=DisplayAs.PNG(img) # trick for Literate.jl
#### optional: save the chart
# savefig(pl,joinpath(pwd(),"exports","mmm-1pager-$(pplot.title_suffix).png"))
````

# Optimization

Let's optimize the marketing budget, ie, let's find out by what factor should we increase/decrease our spend on each marketing channel/activity without spending more money overall

## Metaheuristics loop

````@example 1-demo-high-level
###### Metaheuristics Options (runs ECA algorithm under the hood)
# time_limit is in seconds
# debug = true if you want to see each iteration
# parallel_evaluation = true if you have batch-enabled objective function (see docs for `threaded_objective_func`)
metaheuristics_options = Metaheuristics.Options(time_limit=60.,debug=false,parallel_evaluation=true)
optimalbudget=optimize(stage2fit,inputs;metaheuristics_options);

# quick sanity check
sanity_check_optimum(optimalbudget,stage2fit,inputs);
nothing #hide
````

## 1-Pager

````@example 1-demo-high-level
img=plot(optimalbudget,stage2fit,inputs,pplot)
img=DisplayAs.PNG(img) # trick for Literate.jl
#### optional: save the chart
# savefig(pl,joinpath(pwd(),"exports","optimization-1pager-$(pplot.title_suffix).png"))
````

# END

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

