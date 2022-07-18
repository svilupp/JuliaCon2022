# Practical Tips 

## Getting Started 
As the first step, read through the [Robyn package documentation](https://facebookexperimental.github.io/Robyn/docs/quick-start/)

## Frequently Asked Questions
To be updated...
- "I still don't understand the concept of Adstock" \
  Read the following [article on Adstock](https://www.fastcompany.com/1665084/how-long-does-your-ad-have-impact)
- How to set priors for ad-spend variables
    - (Robyn documentation / section on Adstock and Dim. returns](https://facebookexperimental.github.io/Robyn/docs/features) provides some rules of thumb
    - Decay rate priors (implicitly defined via half-life ranges) are also mentioned on [Wikipedia](https://en.wikipedia.org/wiki/Advertising_adstock), however, they imply quite a long lasting effect. Always ask the experts if it's realistic for your business/type of campaign/position in the funnel!
    - For `beta_spend` coefficients use either a conservative range (eg, centered around 1 and from 0 to 5) or leverage data from previous experiments / from experts
    - To help you with the process, there are several utility functions (see the Demo notebook)
- How to set priors for all else
    - Talk to the subject matter experts in your business on what a realistic range of values would be (on the overall modelled quantity / for the implied dynamic). If it's hard to judge, ask them for what values would be impossible, which gives you edges for your prior distributions. If possible, ask them to also give you a sense how quite the likelihood of different values goes up or down within the range (good exercise is to ask them to stack PET bottle caps or post-it note packs to representive the relative likelihood of different values)
    - Once you know some boundary values and the relative shape, play with `plot()` and various distributions to achieve the desired fit, eg, `plot(Beta(10,10))` and visually inspect if it matches the provided knowledge
    - To help you with the process, there are several utility functions (see the Demo notebook)
- How to fit
    - If you're having problems with the inference (eg, divergences - see the bullet below `What are some good diagnostics`)
    - If you're getting bad results, check that your data is correct and appropriately standardized (plots + describe) and that it is not a result of your priors (eg, loosen your assumptions). It that still doesn't work, try to run your model with the generated data - it's a helpful tool for debugging
    - In general, there is an excellent paper on [Bayesian workflow](http://www.stat.columbia.edu/~gelman/research/unpublished/Bayesian_Workflow_article.pdf)
- How to extend / more advanced implementations
    - Complicated trends: splines (example provided for Splines2 package)
    - Big data - Variation Inference in Turing (ELBO!), ZigZag, or simply a MAP
- What are some good diagnostics
    - [Bayesian workflow](http://www.stat.columbia.edu/~gelman/research/unpublished/Bayesian_Workflow_article.pdf)
    - `Rhat` metric should never be above 1.1, ideally close to 1.0 for the parameters that we care about
    - (HMC/NUTS specific) No `divergences`. Divergences indicate that the algorithm was not able to fully explore the posterior distribution. There is a folk theorem that it's usually due to a bad model - investigate pair plots, variables with low `rhat` or with low `n_eff` and try to re-paramterize where possible. You can read more in [Stan Manual](https://mc-stan.org/docs/2_19/reference-manual/divergent-transitions)
    Other diagnostics include traceplots, rankplots, loo-psis, ppc, etc. 
- How to extend this model
    - Try different inference algorithms (default is NUTS): HMC, MAP (for debugging), VI, ZigZag
    - Add more complicated trends (eg, piecewise-linear or spline-based, example provided for the latter)
    - Model explicit dependencies (eg, de-correlate inputs first or model the correlation explicitly but careful with LKJ in Turing as LKJCholesky is not available at the time of writing)
    - Add more fat tailed behaviour to your random variables and to the observed variables (example for the latter provided)
    - If you have too many inputs, consider adding some regularization (eg, Regularized Horse Shoe Prior)

