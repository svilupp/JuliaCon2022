# Practical Tips 

## Getting Started 
As the first step, read through the [Robyn package documentation](https://facebookexperimental.github.io/Robyn/docs/quick-start/)

## Frequently Asked Questions
To be updated...
- "I still don't understand the concept of Adstock" \
  Read the following [article on Adstock](https://www.fastcompany.com/1665084/how-long-does-your-ad-have-impact)
- How to set priors for Ad spend variables
    - (Robyn documentation, section on Adstock and Diminishing returns](https://facebookexperimental.github.io/Robyn/docs/features) provides some rules of thumb
    - Decay rate priors (implicitly defined via half-life ranges) are also mentioned on [Wikipedia](https://en.wikipedia.org/wiki/Advertising_adstock), however, they imply quite a long-lasting effect. Always ask the experts if it's realistic for your business/type of campaign/position in the funnel!
    - For `beta_spend` coefficients use either a conservative range (eg, centered around 1 and from 0 to 5) or leverage data from previous experiments / from experts
    - To help you with the process, there are several utility functions (see the Demo notebooks)
- How to set priors for all else
    - Talk to the subject matter experts in your business on what a realistic range of values would be (on the overall modelled quantity / for the implied dynamic). If it's hard to judge, ask them for what values would be impossible, which gives you edges for your prior distributions. If possible, ask them to also give you a sense of how quite the likelihood of different values goes up or down within the range (a good exercise is to ask them to stack PET bottle caps or post-it note packs to represent the relative likelihoods of different values)
    - Once you know some boundary values and the relative shape, play with `plot()` and various distributions to achieve the desired fit, eg, `plot(Beta(10,10))` and visually inspect if it matches the provided knowledge (bounds, mass, center, skew, etc.)

- Why the two-stage fit
    - While it's not always necessary, it's a really good default that should work well in most cases (there are some implications, see the Discussion in section Methodology)
    - If we were fitting all variables at once, we are likely to get even bigger uncertainties across all RVs because the trendline is fitted as well. That is why we 'cut feedback' between the first and second stage of modelling

What to do when the fitting goes bad
    - If you're having problems with the inference (eg, divergences - see the bullet below `What are some good diagnostics`)
    - If you're getting bad results, check that your data is correct and appropriately standardized (plots + describe) and that it is not a result of your priors (eg, loosen your assumptions). It that still doesn't work, try to run your model with the generated data - it's a super helpful tool for debugging
    - In general, there is an excellent paper on [Bayesian workflow](http://www.stat.columbia.edu/~gelman/research/unpublished/Bayesian_Workflow_article.pdf)
- What are some good diagnostics
    - The best resource: [Bayesian workflow](http://www.stat.columbia.edu/~gelman/research/unpublished/Bayesian_Workflow_article.pdf)
    - `Rhat` metric should never be above 1.1, ideally close to 1.0 for the parameters that we care about
    - (HMC/NUTS specific) No `divergences`. Divergences indicate that the algorithm was not able to fully explore the posterior distribution. There is a folk theorem that it's usually due to a bad model - investigate pair plots, variables with low `rhat` or with low `n_eff` and try to re-parameterize where possible. You can read more in [Stan Manual](https://mc-stan.org/docs/2_19/reference-manual/divergent-transitions)

    If the problem is small (a few divergences), it can be resolved by increasing `target_acceptance` to 0.9 / 0.99, but please note that it only makes the steps smaller, it does not solve the underlying problem if there is some degeneracy in your posterior!
    - If NUTS diagnostics reports that you have a lot of trajectories that exceeded the maximum tree depth, consider increasing it to 12/14. Except that your inference will slow down significantly.
    - Posterior predictive checks - Generate some data from your fitted model. Its distribution should overlap with the observed data.
    - Other diagnostics include traceplots, rankplots, loo-psis, prior predictive checks, etc. 
    - If you want to compare different models/implementations, use Pareto-smoothed Leave-one-out score (PSIS-LOO, use packages: Arviz.jl or ParetoSmooth.jl)
- Why does optimization take so long
	- Our objective function is not convex, so we cannot use any gradient-based methods
    - Instead, we use Metaheuristics package and we provide it with a fixed "budget" of time to try to find the optimum (set to 60 seconds in the Demo notebook)
	- Ideally, you want to make sure to run at least D * 1000 evaluations, where `D=number of Adspend variables` (authors of ECA algorithm that we use in the Demo recommend running D * 10000 evaluations)
	- In terms of speed per evaluation, the objective function calls `generated_quantities` in Turing each time with new inputs (proposed new Ad spend):
        - Make sure you're not re-computing the base case scenario (original Ad spend)
        - If your model is relatively straightforward (like the demo), you can re-write it as a deterministic function of the posterior samples (expected speed up could be even 2-3x per evaluation)
		- If your dataset is large but you care about only a small period (eg, the past few months), you can apply a mask to data provided to the Turing model (Note: Be careful not to lose some of the lagged effects/ramp-up time. Make sure to masked ALL datasets consistently)
- How to extend / more advanced implementations
    - First, try the data-centric approach
        - Are there more internal datasets you can add? More detail? 
        - Any external datasets? (eg, macroeconomic data, market trackers and trends, competitive intelligence)
    - Different inference routine (easy with Turing.jl!), eg, Variation Inference in Turing (ELBO!), ZigZag, or simply a MAP
	- Apply inference tricks, eg, re-parametrization of the distributions but also the functions you use
	- Change the observational model (eg, add fatter tails if you data demands it - examples provided in the `model_definitions` file)
    - Add more dependencies/prior knowledge, eg, correlation, sparsity (Regularized Horse Shoe prior works well), tighter priors
	- Add more flexible relationships and trends, eg, multiplicative model, saturating growth trend, spline-based trends (example provided in the Demo)
    - Different marketing variable transformations, eg, decay models based on Weibull PDF or CDF, piece-wise linear, flip the order of decaying vs saturation (especially if the spend is in rare and large lumps!)
