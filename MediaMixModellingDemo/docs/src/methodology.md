# Methodology

## Goals
- Quantify the benefits of your marketing activities (ROAS, mROAS, etc.)
- Re-allocate marketing spend across different marketing channels/activities to maximize revenues

## Challenges
- Insufficient data (eg, unobserved data, untracked data, too little data given the changing dynamics in the market)
- Underspecified problem / weak identifiability (eg, Hill Curves for Ad spend saturation are "too flexible")

## Building blocks		
### Modelling decisions to make it work
- Use Bayesian inference, which allows us to capture the uncertainty / underspecification of the key parameters
- Specify our model with Turing / DynamicPPL, which allows us to apply almost any logic that we can express in Julia / we know from domain expertise
- Leverage structural decomposition of the time series (eg, trend + seasonality + Ad spend + ...)
- Fit the model in two stages to stabilize it / to reduce the weak identifiability
- Reparametrize our model and scale all predictors, which enables us to easily leverage our domain knowledge (and existing experiments) and set informative priors
- Operate on fitted results in MCMCChains, which allows to leverage many algorithms to fit our model, eg, HMC, NUTS, VI, MAP, ZigZag, etc
- Apply Bayesian Decision Theory (custom loss function over the samples) to reflect your business’ context, preferences and decision making
- Find the optimal budget with Metaheuristics to overcome that some objectives will be non-convex (ie, there will be local optima)

### Discussion of Some of the Implications 
- To make my models re-usable I enforce some naming and grouping conventions (eg, all marketing spend variables must be supplied in the same Dataframe, similarly all context variables must be supplied in another Dataframe) and standardization (context variables are standardized by Zscore, but revenues and marketing spend are Max() scaled to allow for easy conversions and setting of priors)
- Max() scaling of revenues and marketing spend could be a poor choice if there are many outliers in the data (but that would also complicate matters for the auto priors for the Zscore transformed variables as they are loosely linked to standard deviation of the revenues
- Reparametrization of the model allows directly setting the expected ROAS at an average ad spend level, but it implicitly assumes that the inflection of the Hill Curve (Halfmax concentration point or “halfmaxpoint” RV in the model) is at your average level as well (ie, if the channel is relatively underutilized / very low, it might be too conservative assumption and it could reduce your fitted ROAS)
- The models might be harder to fit (follow the standard Bayesian workflow and diagnostics - see FAQ for references)
- Two stage fit implicitly assumes independence between marketing spend and other variables (eg, holidays would not interact with marketing effects)
- Second stage fit can be done on a smaller window if all inputs are masked (subset) consistently, but make sure to reflect this masking in plotting the fit results and in the optimization
- Optimization can be run on a smaller window than the fit (“optim_mask”) but that reduces its potential uplift, because there will be a ramp-up / adjustment period to adjust to new spend levels (because of the Geometric Decay) - make sure the optimization window is not too short to see the full effect of more “decayed” variables