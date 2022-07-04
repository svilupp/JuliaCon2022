# Resources 

## Various concepts and interpretations
- Concepts 
https://www.ijert.org/marketing-mix-modeling-mmm-concepts-and-model-interpretation

## Industry 
- Google: Bayesian Methods for Media Mix Modeling with Carryover and Shape effects
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46001.pdf
- Google: Challenges and opportunities in media mix modelling
https://research.google/pubs/pub45998/

- !CROWN JEWEL! Robyn Package 
https://facebookexperimental.github.io/Robyn/
It's actually mixed, it uses FB Prophet under the hood to estimate the trend (MAP mode, not MCMC) and then it runs Nevergrad (Python library) to optimize for the lowest errors
Related tutorial: https://towardsdatascience.com/automated-marketing-mix-modeling-with-facebooks-robyn-fd79e60b489d
Output diagnostics: https://facebookexperimental.github.io/Robyn/docs/outputs-diagnostics

- Google: Lightweight MMM in Numpyro
https://github.com/google/lightweight_mmm/blob/main/lightweight_mmm/media_transforms.py

- Hello Fresh: Python with PyMC3 tutorial
https://engineering.hellofresh.com/bayesian-media-mix-modeling-using-pymc3-for-fun-and-profit-2bd4667504e6




## Frequentist MMM
- Python with Statsmodels
https://blog.getcensus.com/you-should-know-introduction-to-marketing-mix-modeling/amp/

- Python with ScikitLearn
https://towardsdatascience.com/introduction-to-marketing-mix-modeling-in-python-d0dd81f4e794
https://link.medium.com/ru5f526OHob

- R
https://towardsdatascience.com/building-a-marketing-mix-model-in-r-3a7004d21239

## Probabilistic MMM
- Stan (Multiplicative model)
https://towardsdatascience.com/python-stan-implementation-of-multiplicative-marketing-mix-model-with-deep-dive-into-adstock-a7320865b334
https://github.com/sibylhe/mmm_stan
- MMM package with CLI and configs
https://github.com/leopoldavezac/BayesianMMM
- MMM in PyMC (no lags, just trivial linear reg)
https://getrecast.com/bayesian-methods-for-mmm/
- PyMC3 and market media mix modelling
https://link.medium.com/MiDctptEvmb
- PyMC Labs post
https://www.pymc-labs.io/blog-posts/bayesian-media-mix-modeling-for-marketing-optimization/
- Orbit / Bayesian Time-Varying Coefficient Regression / BTVC https://arxiv.org/pdf/2106.03322.pdf
package orbit-ml: https://orbit-ml.readthedocs.io/en/latest/tutorials/ktr1.html
- Python tutorial with PyMC3 (and comparison with Robyn)
https://medium.com/towards-data-science/modeling-marketing-mix-using-pymc3-ba18dd9e6e68

- Saturation coefficients changing over time
https://juanitorduz.github.io/pymc_mmm/
- Orbit / KTR model
https://juanitorduz.github.io/orbit_mmm/
Adjacent topics

## On Transformations on the Input Variables
- Weibull transformation case study (for Adstock)
https://github.com/annalectnl/weibull-adstock/blob/master/adstock_weibull_annalect.pdf
http://business-science.pl/dont-get-the-wrong-impression-assessing-the-best-input-variable-to-reflect-meta-in-mmm/

## Other / Attribution
- Survey attribution in Numpyro (touchpoints etc via object-oriented setup)
https://vincentk1991.github.io/survey-attribution-numpyro/
- Attribution models
https://internetrix.github.io/attribution-modelling/methods.html
- Channelattribution package in R
https://cran.r-project.org/web/packages/ChannelAttribution/ChannelAttribution.pdf 
- Eva Anderl, Ingo Becker, Florian v. Wangenheim, Jan H. Schumann (2014)
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2343077
