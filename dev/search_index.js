var documenterSearchIndex = {"docs":
[{"location":"resources/#Resources","page":"Resources","title":"Resources","text":"","category":"section"},{"location":"resources/#Various-Introductory-texts","page":"Resources","title":"Various Introductory texts","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"Marketing Mix Modeling (MMM) – Concepts and Model Interpretation\nLight introduction to Adstock from FastCompany","category":"page"},{"location":"resources/#Industry","page":"Resources","title":"Industry","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"Google: Bayesian Methods for Media Mix Modeling with Carryover and Shape effects\nGoogle: Challenges and opportunities in Media Mix Modelling\n!!CROWN JEWEL!! Robyn Package from Facebook Research team (experimental). Be careful when using the Weibull transformations (it might be fixed now). A related tutorial\nGoogle: Numpyro package Lightweight MMM\nHello Fresh: Tutorial for Python with PyMC3 here","category":"page"},{"location":"resources/#Frequentist-MMM","page":"Resources","title":"Frequentist MMM","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"MMM in Python with Statsmodels here\nMMM in Python with ScikitLearn here The author has a full series on the topic - recommended!\nMMM in R here","category":"page"},{"location":"resources/#Probabilistic-MMM","page":"Resources","title":"Probabilistic MMM","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"MMM with Stan (a multiplicative model) here and here\nMMM package with CLI and configs here\nSimple MMM in PyMC here\nMMM in Python with PyMC3 here Same author as some articles above - he has a full series.\nPyMC Labs post on MMM and also Learning Bayesian Stats podcast / MMM episode\nOrbit / Bayesian Time-Varying Coefficient Regression / BTVC here, package Orbit-ml\nPython tutorial with PyMC3 (and comparison with Robyn) here\nTime-varying saturation coefficients with PyMC3 here\nOrbit / KTR model","category":"page"},{"location":"resources/#Other-/-On-Transformations-on-the-Input-Variables","page":"Resources","title":"Other / On Transformations on the Input Variables","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"[Weibull transformation case study (for Adstock)](https://github.com/annalectnl/weibull-adstock/blob/master/adstockweibullannalect.pdf","category":"page"},{"location":"resources/","page":"Resources","title":"Resources","text":"http://business-science.pl/dont-get-the-wrong-impression-assessing-the-best-input-variable-to-reflect-meta-in-mmm/)","category":"page"},{"location":"resources/#Other-/-Attribution","page":"Resources","title":"Other / Attribution","text":"","category":"section"},{"location":"resources/","page":"Resources","title":"Resources","text":"Survey attribution in Numpyro here\nAttribution models here\nChannelattribution package in R\nEva Anderl, Ingo Becker, Florian v. Wangenheim, Jan H. Schumann (2014)","category":"page"},{"location":"practical_tips/#Practical-Tips-and-Tricks","page":"Practical Tips and Tricks","title":"Practical Tips and Tricks","text":"","category":"section"},{"location":"practical_tips/#Getting-Started","page":"Practical Tips and Tricks","title":"Getting Started","text":"","category":"section"},{"location":"practical_tips/","page":"Practical Tips and Tricks","title":"Practical Tips and Tricks","text":"As the first step, read through the Robyn package documentation","category":"page"},{"location":"practical_tips/#Frequently-Asked-Questions","page":"Practical Tips and Tricks","title":"Frequently Asked Questions","text":"","category":"section"},{"location":"practical_tips/","page":"Practical Tips and Tricks","title":"Practical Tips and Tricks","text":"To be updated...","category":"page"},{"location":"practical_tips/","page":"Practical Tips and Tricks","title":"Practical Tips and Tricks","text":"\"I still don't understand the concept of Adstock\" \nRead the following article on Adstock\nHow to set priors for ad-spend variables\n(Robyn documentation / section on Adstock and Dim. returns](https://facebookexperimental.github.io/Robyn/docs/features) provides some rules of thumb\nDecay rate priors (implicitly defined via half-life ranges) are also mentioned on Wikipedia, however, they imply quite a long lasting effect. Always ask the experts if it's realistic for your business/type of campaign/position in the funnel!\nFor beta_spend coefficients use either a conservative range (eg, centered around 1 and from 0 to 5) or leverage data from previous experiments / from experts\nHow to set priors for all else\nTalk to the subject matter experts in your business on what a realistic range of values would be (on the overall modelled quantity / for the implied dynamic). If it's hard to judge, ask them for what values would be impossible, which gives you edges for your prior distributions. If possible, ask them to also give you a sense how quite the likelihood of different values goes up or down within the range (good exercise is to ask them to stack PET bottle caps or post-it note packs to representive the relative likelihood of different values)\nOnce you know some boundary values and the relative shape, play with plot() and various distributions to achieve the desired fit, eg, plot(Beta(10,10)) and visually inspect if it matches the provided knowledge\nHow to fit\nDiscussion on 1 vs 2 stages...(TBU)\nExcellent paper on Bayesian workflow\nHow to extend / more advanced implementations\nComplicated trends: splines (example provided for Splines2 package)\nBig data - Variation Inference in Turing (ELBO!), ZigZag, or simply a MAP\nWhat are some good diagnostics\nBayesian workflow\nRhat metric should never be above 1.1, ideally close to 1.0 for the parameters that we care about\n(HMC/NUTS specific) No divergences. Divergences indicate that the algorithm was not able to fully explore the posterior distribution. There is a folk theorem that it's usually due to a bad model - investigate pair plots, variables with low rhat or with low n_eff and try to re-paramterize where possible. You can read more in Stan Manual\nOther diagnostics include traceplots, rankplots, loo-psis, ppc, etc. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = MediaMixModellingDemo","category":"page"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This is a documentation for MediaMixModellingDemo.","category":"page"},{"location":"","page":"Home","title":"Home","text":"It's an example produced to acompany a talk at JuliaCon 2022.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"index.md\", \"practical_tips.md\",\"resources.md\"]\nDepth = 3","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [MediaMixModellingDemo]","category":"page"}]
}