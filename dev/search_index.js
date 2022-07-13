var documenterSearchIndex = {"docs":
[{"location":"resources/#Resources","page":"Resources","title":"Resources","text":"","category":"section"},{"location":"resources/#Various-Introductory-texts","page":"Resources","title":"Various Introductory texts","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"Marketing Mix Modeling (MMM) – Concepts and Model Interpretation\nLight introduction to Adstock from FastCompany","category":"page"},{"location":"resources/#Industry","page":"Resources","title":"Industry","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"Google: Bayesian Methods for Media Mix Modeling with Carryover and Shape effects\nGoogle: Challenges and opportunities in Media Mix Modelling\n!!CROWN JEWEL!! Robyn Package from Facebook Research team (experimental). Be careful when using the Weibull transformations (it might be fixed now). A related tutorial\nGoogle: Numpyro package Lightweight MMM\nHello Fresh: Tutorial for Python with PyMC3 here","category":"page"},{"location":"resources/#Frequentist-MMM","page":"Resources","title":"Frequentist MMM","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"MMM in Python with Statsmodels here\nMMM in Python with ScikitLearn here The author has a full series on the topic - recommended!\nMMM in R here","category":"page"},{"location":"resources/#Probabilistic-MMM","page":"Resources","title":"Probabilistic MMM","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"MMM with Stan (a multiplicative model) here and here\nMMM package with CLI and configs here\nSimple MMM in PyMC here\nMMM in Python with PyMC3 here Same author as some articles above - he has a full series.\nPyMC Labs post on MMM and also Learning Bayesian Stats podcast / MMM episode\nOrbit / Bayesian Time-Varying Coefficient Regression / BTVC here, package Orbit-ml\nPython tutorial with PyMC3 (and comparison with Robyn) here\nTime-varying saturation coefficients with PyMC3 here\nOrbit / KTR model","category":"page"},{"location":"resources/#Other-/-On-Transformations-on-the-Input-Variables","page":"Resources","title":"Other / On Transformations on the Input Variables","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"Weibull transformation case study (for Adstock) and here","category":"page"},{"location":"resources/#Other-/-Attribution","page":"Resources","title":"Other / Attribution","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"Survey attribution in Numpyro here\nAttribution models here\nChannelattribution package in R\nEva Anderl, Ingo Becker, Florian v. Wangenheim, Jan H. Schumann (2014)","category":"page"},{"location":"api_reference/#API-Reference","page":"API Reference","title":"API Reference","text":"","category":"section"},{"location":"api_reference/#Index","page":"API Reference","title":"Index","text":"","category":"section"},{"location":"api_reference/","page":"API Reference","title":"API Reference","text":"","category":"page"},{"location":"api_reference/","page":"API Reference","title":"API Reference","text":"## Docstrings","category":"page"},{"location":"api_reference/","page":"API Reference","title":"API Reference","text":"Modules = [MediaMixModellingDemo]","category":"page"},{"location":"api_reference/#MediaMixModellingDemo.generate_fourier_series","page":"API Reference","title":"MediaMixModellingDemo.generate_fourier_series","text":"generate_fourier_series(t, p=365.25, n=5)\n\nGenerates fourier series with period p and degree n (the higher, the more flexible it is) It can be then fitted with coefficients to mimic any period trend\n\nExpects t to be a time index series\n\nReturns array of shape: (size(t,1),2n)\n\nExample\n\nseaso=generatefourierseries(1:400,365.25, 5)\n\n\n\n\n\n","category":"function"},{"location":"api_reference/#MediaMixModellingDemo.generate_seasonality_features-Tuple{Any, Any}","page":"API Reference","title":"MediaMixModellingDemo.generate_seasonality_features","text":"generate_seasonality_features(t, p=365.25, n=5)\n\nGenerates seasonality features given an array of tuples in a format (period,degree) Eg, 7-day period of degree 3 would be \n\nExpects t to be a time index series\n\nReturns array of shape: (size(t,1),2n)\n\nExample\n\nseaso=generatefourierseries(1:400,365.25, 5)\n\n\n\n\n\n","category":"method"},{"location":"api_reference/#MediaMixModellingDemo.geometric_decay-Union{Tuple{T}, Tuple{AbstractVecOrMat{T}, Any}, Tuple{AbstractVecOrMat{T}, Any, Any}} where T<:Real","page":"API Reference","title":"MediaMixModellingDemo.geometric_decay","text":"geometric_decay(x::Vector{T},decay_rate,normalize=true) where {T<:Real}\n\nSimple geometric decay transformation if normalize=true it divides the output by the sum of the geometric series Note: Does NOT check if decay_rate<1 etc to ensure that the sum convergences to the analytic formula\n\n\n\n\n\n","category":"method"},{"location":"api_reference/#MediaMixModellingDemo.getflatsamples-Tuple{Any, Any}","page":"API Reference","title":"MediaMixModellingDemo.getflatsamples","text":"getflatsamples(chain,groupname)\n\nExtract a group of variables under name groupname from chain   and flattens all samples into the first dimension  ie, outputs a dimension: (numsamples*numchains,num_variables)\n\n\n\n\n\n","category":"method"},{"location":"api_reference/#MediaMixModellingDemo.plot_periodogram","page":"API Reference","title":"MediaMixModellingDemo.plot_periodogram","text":"function plot_periodogram(input_arr)\n\nPlot Fourier transform coefficients to uncover the most prominent frequencies / seasonalities Assumes equally spaced data points Looks only for periods that have seen at least 2 full cycles (ie, size ÷ 2 at maximum!) Shows top-k values\n\nExample\n\np=10 # period is 10 y=sin.(2π/p*collect(1:20)) # generate 20 data points plot_periodogram(y,1) # plot periodogram, period=10 should be highlighted as maximum\n\n\n\n\n\n","category":"function"},{"location":"api_reference/#MediaMixModellingDemo.standardize_by_max-Tuple{Any}","page":"API Reference","title":"MediaMixModellingDemo.standardize_by_max","text":"standardize_by_max(X)\n\nMax()-only transform to allow easy scaling between features and the outcome Uses MinMax() pipe under the hood but overwrites the minimum to be =0\n\nExample: y_std,pipe_cache_y=standardize_by_max(select(df,target_label))\n\n\n\n\n\n","category":"method"},{"location":"api_reference/#MediaMixModellingDemo.standardize_by_zscore-Tuple{Any}","page":"API Reference","title":"MediaMixModellingDemo.standardize_by_zscore","text":"standardize_by_zscore(X)\n\nZscore transform to center the feature to its mean value and scale it (to make it easier to set priors)\n\nExample: y_std,pipe_cache_y=standardize_by_zscore(select(df,target_label))\n\n\n\n\n\n","category":"method"},{"location":"practical_tips/#Practical-Tips","page":"Practical Tips and Tricks","title":"Practical Tips","text":"","category":"section"},{"location":"practical_tips/#Getting-Started","page":"Practical Tips and Tricks","title":"Getting Started","text":"","category":"section"},{"location":"practical_tips/","page":"Practical Tips and Tricks","title":"Practical Tips and Tricks","text":"As the first step, read through the Robyn package documentation","category":"page"},{"location":"practical_tips/#Frequently-Asked-Questions","page":"Practical Tips and Tricks","title":"Frequently Asked Questions","text":"","category":"section"},{"location":"practical_tips/","page":"Practical Tips and Tricks","title":"Practical Tips and Tricks","text":"To be updated...","category":"page"},{"location":"practical_tips/","page":"Practical Tips and Tricks","title":"Practical Tips and Tricks","text":"\"I still don't understand the concept of Adstock\" \nRead the following article on Adstock\nHow to set priors for ad-spend variables\n(Robyn documentation / section on Adstock and Dim. returns](https://facebookexperimental.github.io/Robyn/docs/features) provides some rules of thumb\nDecay rate priors (implicitly defined via half-life ranges) are also mentioned on Wikipedia, however, they imply quite a long lasting effect. Always ask the experts if it's realistic for your business/type of campaign/position in the funnel!\nFor beta_spend coefficients use either a conservative range (eg, centered around 1 and from 0 to 5) or leverage data from previous experiments / from experts\nHow to set priors for all else\nTalk to the subject matter experts in your business on what a realistic range of values would be (on the overall modelled quantity / for the implied dynamic). If it's hard to judge, ask them for what values would be impossible, which gives you edges for your prior distributions. If possible, ask them to also give you a sense how quite the likelihood of different values goes up or down within the range (good exercise is to ask them to stack PET bottle caps or post-it note packs to representive the relative likelihood of different values)\nOnce you know some boundary values and the relative shape, play with plot() and various distributions to achieve the desired fit, eg, plot(Beta(10,10)) and visually inspect if it matches the provided knowledge\nHow to fit\nDiscussion on 1 vs 2 stages...(TBU)\nExcellent paper on Bayesian workflow\nHow to extend / more advanced implementations\nComplicated trends: splines (example provided for Splines2 package)\nBig data - Variation Inference in Turing (ELBO!), ZigZag, or simply a MAP\nWhat are some good diagnostics\nBayesian workflow\nRhat metric should never be above 1.1, ideally close to 1.0 for the parameters that we care about\n(HMC/NUTS specific) No divergences. Divergences indicate that the algorithm was not able to fully explore the posterior distribution. There is a folk theorem that it's usually due to a bad model - investigate pair plots, variables with low rhat or with low n_eff and try to re-paramterize where possible. You can read more in Stan Manual\nOther diagnostics include traceplots, rankplots, loo-psis, ppc, etc. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = MediaMixModellingDemo","category":"page"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This is a documentation for MediaMixModellingDemo.","category":"page"},{"location":"","page":"Home","title":"Home","text":"It's an example produced to acompany a talk at JuliaCon 2022.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"index.md\", \"methodology.md\",\"practical_tips.md\",\"resources.md\"]\nDepth = 3","category":"page"},{"location":"methodology/#Methodology","page":"Methodology","title":"Methodology","text":"","category":"section"},{"location":"methodology/#Overview","page":"Methodology","title":"Overview","text":"","category":"section"},{"location":"methodology/","page":"Methodology","title":"Methodology","text":"To be updated...","category":"page"},{"location":"methodology/","page":"Methodology","title":"Methodology","text":"Transform data \nVariables that can have positive-only effects need to be standardized to 0-1 range\nVariables with any-signed effects need to be at least standardized via Z-Score\nNote: Code expects certain naming conventions (eg, X_spend for the DataFrame of the ad spend variables to be modelled)\nSet priors / conversion factors\nBe restrictive in the marketing transforms' parameters\nStage 1: Fit the trend\nExtract separate series for growth trend, seasonality, holidays, organic variables \nStage 2: Fit the coefficients for marketing transformation\nValidate the fit (Rhat, traceplots, etc)\nQuantify the contributions + ROAS of the current marketing spend\nOptimize the marketing budget to maximize the marketing contribution \nDefine a loss function that reflects your business' decision-making process\nEvaluate the results of the optimization + inherent uncertainty","category":"page"},{"location":"methodology/#Tricks-to-make-it-work","page":"Methodology","title":"Tricks to make it work","text":"","category":"section"},{"location":"methodology/","page":"Methodology","title":"Methodology","text":"Use structural decomposition of the time series (trend, seasonality, holidays, etc.)\nSetting informed priors\nRe-parametrization of the beta_spend coefficients\nTwo-stage fit\nNon-convex optimization","category":"page"}]
}
