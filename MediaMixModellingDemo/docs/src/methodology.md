# Methodology

## Overview
To be updated...
- Transform data 
    - Variables that can have positive-only effects need to be standardized to 0-1 range
    - Variables with any-signed effects need to be at least standardized via Z-Score
    - Note: Code expects certain naming conventions (eg, X_spend for the DataFrame of the ad spend variables to be modelled)
- Set priors / conversion factors
    - Be restrictive in the marketing transforms' parameters
- Stage 1: Fit the trend
    - Extract separate series for growth trend, seasonality, holidays, organic variables 
- Stage 2: Fit the coefficients for marketing transformation
    - Validate the fit (Rhat, traceplots, etc)
- Quantify the contributions + ROAS of the current marketing spend
- Optimize the marketing budget to maximize the marketing contribution 
    - Define a loss function that reflects your business' decision-making process
    - Evaluate the results of the optimization + inherent uncertainty
    
## Tricks to make it work
- Use structural decomposition of the time series (trend, seasonality, holidays, etc.)
- Setting informed priors
    - Re-parametrization of the $$\beta_{spend}$$ coefficients
- Two-stage fit
- Non-convex optimization