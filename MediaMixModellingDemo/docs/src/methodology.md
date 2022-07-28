# Methodology

## Goals
- Quantify the benefits of your marketing activities (ROAS, mROAS, etc.)
- Re-allocate marketing spend across different marketing channels/activities to maximize revenues

## Challenges
- Insufficient data (eg, unobserved data, untracked data, too little data given the changing dynamics in the market)
- Underspecified problem / weak identifiability (eg, Hill Curves for Ad spend saturation are "too flexible")

## Building blocks		
### Modelling decisions to make it work
- Use Bayesian inference, which allows us to capture the uncertainty / underspecification of the key parameters
- Specify our model with Turing / DynamicPPL, which allows us to apply almost any logic that we can express in Julia / we know from domain expertise
- Leverage structural decomposition of the time series (eg, trend + seasonality + Ad spend + ...)
- Fit the model in two stages to reduce the weak identifiability (effectively cutting the feedback of the first stage model on modelling the plain trend)
- Reparametrize our model and scale all predictors, which enables us to easily leverage our domain knowledge (and existing experiments) and set informative priors
- Operate on fitted results in MCMCChains, which allows to leverage many algorithms to fit our model, eg, HMC, NUTS, VI, MAP, ZigZag, etc
- Apply Bayesian Decision Theory (custom loss function over the samples) to reflect your business’ context, preferences and decision making
- Find the optimal budget with Metaheuristics to overcome that some objectives will be non-convex (ie, there will be local optima)

### Discussion of Some of the Implications 
- To make the pre-defined models re-usable, we must enforce certain naming and grouping conventions (eg, all marketing spend variables must be supplied in the same Dataframe, similarly all context variables must be supplied in another Dataframe) and standardization (context variables are standardized by Zscore, but revenues and marketing spend are Max() scaled to allow for easy conversions and setting of priors) - This is less relevant if you're using the high-level API
- Max() scaling of revenues and marketing spend could be a poor choice if there are many outliers in the data (but that would also complicate matters for the auto priors for the Zscore transformed variables as they are loosely linked to standard deviation of the revenues
- Reparametrization of the model allows directly setting the expected ROAS at an average ad spend level, but it implicitly assumes that the inflection of the Hill Curve (Halfmax concentration point or `halfmaxpoint` RV in the model) is at your average level as well (ie, if the channel is relatively underutilized / very low, it might be a too conservative assumption and it could reduce your fitted ROAS)
- The models might be harder to fit (follow the standard Bayesian workflow and diagnostics - see FAQ for references)
- Two-stage fit implicitly assumes independence between marketing spend and other variables (eg, holidays would not interact with marketing effects)
- Second stage fit can be done on a smaller window if all inputs are masked (subset) consistently, but make sure to reflect this masking in plotting the fit results and in the optimization
- Optimization can be run on a smaller window than the fit (`optim_mask`) but that reduces its potential uplift because there will be a ramp-up/adjustment period to adjust to new spend levels (because of the Geometric Decay) - make sure the optimization window is not too short to see the full effect of more “decayed” variables

## High-level Workflow
- Transform data 
    - Response variable (=revenues) to be standardized to 0-1 range (where y=0 implies that revenue=0)
    - Variables that can have positive-only effects need to be standardized to 0-1 range (where =0 implies that value=0)
    - Variables with any-signed effects need to be at least standardized via Z-Score
    - Note: Code expects certain naming conventions (eg, X_spend for the DataFrame of the ad spend variables to be modelled - more on that below)
- Set priors / conversion factors
    - Scale the priors based on the values of the input data
    - Leverage as much context + business knowledge as possible in setting possible parameters for the marketing transforms
- Stage 1: Fit the trend
    - Extract separate series for growth trend, seasonality, holidays, organic variables 
    - Create a new dataset (growth trend, seasonality, holidays) to use as a fixed input in the second stage 
- Stage 2: Fit the coefficients for marketing transformation
    - Validate the fit (Rhat, traceplots, etc)
    - Validate the model quality (Pseudo-R2, NRMSE) - we ideally need >80% of the variation to be explained by our model to have meaningful results
- Quantify the contributions + ROAS of the current marketing spend
    - Produce model fit summary 1-pager with key plots
- Optimize the marketing budget to maximize the marketing contribution 
    - Define a loss function that reflects your business' decision-making process
    - Produce optimization summary 1-pager with the results of the optimization + inherent uncertainty


## Naming Conventions and Variables

Relevant only for the low-level API
### Input data
- Codebase depends on variables under the following names:
    - `X_spend`: DataFrame of Ad spend to be transformed (only positive coefficients will be allowed)
    - `X_org`: DataFrame of organic variables (ie, only positive coefficients will be allowed)
    - `X_cat`: 1-column DataFrame with a categorical variable (eg, for events)
    - `X_hols`: DataFrame with holiday events (currently, 1-column with dummy encoding of holiday effect)
    - `X_context`: Any context variables (eg, market drivers, market index, competitor popularity)

    - `X_seas`: Created seasonality DataFrame
    - `X_feat`: horizontal concatenation of X_spend,X_org,X_content (order does not matter)
    - `X_trend`: Optional - Spline-basis for flexible trend modelling
    
    - `y_std`: transformed response variable
    - `y_true`: original response variable (=revenues)
    - `time_std`: transformed time index running from 0-1 during the whole period (important for setting priors)
    - `df`: series of datetime values corresponding to the observations

    - `cols_spend`: Vector of ad spend variables in the same order as `X_spend` (ie, `names(X_spend)`)
    - `cat_levels`: Integer of how many levels are in the 1-column DataFrame `X_cat`
    
- Transformations caches saved under pipe_cache_y,pipe_cache_spend
    - `pipe_cache_y`: TableTransforms cache to revert y-variable transform
    - `pipe_cache_spend`: TableTransforms cache to revert X_spend variables transform

- Functions:
    - `revert_pipe_y`: utility to revert transformation of y-variable
    - `revert_pipe_spend`: utility to revert transformation of the ad spend variables
    
Reason: Most of them are used when setting conversion ratios and priors in `ParamsStage1` (variable called `p1`) and `ParamsStage2` (variable called `p2`)


### Turing Models
We will fit the model in 2 stages to deal with the under-specification.

As a convention the deterministic location of the observed variable (mean of the RV `y`) will be called `mu` and its components will be all prefixed as such, eg, 
- `mu_trend` for the trend component, 
- `mu_seas` for the seasonality component, 
- `mu_hols` for the holidays component,
- `mu_context` for the contributions of the contextual variables,
- `mu_org`
- `mu_spend` for the contribution of all Ad spend variables together
- `mu_spend_by_var` for the contributions of each individual spend variable (ie, number of columns = number of Ad spend variables)

(See `src/model_definition.jl` for more details)

All these deterministic components will be exposed by DynamicPPL in the return statement to be easily extracted via `generated_quantities()` conditioned on fitted samples (a `chains` object). In other words, given a model fit, we can easily extract all the above components by playing the samples through the model.

Fitted results from each stage:
- Stage 1: 
    - Fitted parameters (Chains): `chain_stage1`
    - Generated quantities (eg, `mu_trend` etc): `stage1_fit`
- Stage 2:
    - Model used for fitting: `model_orig`
    - Fitted parameters (Chains): `chain`
    - Generated quantities (eg, `mu_trend` etc): `stage2_fit_allsamples` (or simply `stage2fit`) 

### Optimization
`budget_multiplier_optim` holds the recommended multiplier for the `X_spend` (=Ad spend) that yields the maximum revenue uplift.
`simulations_optim` holds the generated quantities resulting from the above optimal Ad spend.

There is an option to run the optimization only on a subset of data using the mask `optim_mask`.

In general, most objects that are derived from this optimal budget are suffixed with `_optim` (eg, `simulations_optim`). In contrast, the un-optimized quantities are often suffixed as `_prev` (eg, `spend_share_prev`)

In addition, if there is a variable that has an original domain (=dollars) and a transformed domain like Ad spend, it will have additional suffix `_raw` and `_trf` for the original and transformed domain, respectively, (eg, optimal Ad spend standardized to 0-1 range -> `X_spend_optim_trf` ). 

To make the contrast really explicit, if some values are in the original domain, they could have the suffix `_raw`

### Other
For plotting consistency, some of the defaults are saved in struct `ParamsPlot`