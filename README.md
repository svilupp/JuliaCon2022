# JuliaCon 2022 talk: Optimize your marketing spend with Julia!

This is a supporting repository for a lightning talk to be given at JuliaCon 2022

[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://svilupp.github.io/JuliaCon2022/dev/)
[![CI](https://github.com/svilupp/JuliaCon2022/actions/workflows/CI.yml/badge.svg)](https://github.com/svilupp/JuliaCon2022/actions/workflows/CI.yml)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

> **"Half the money I spend on advertising is wasted; the trouble is I don't know which half."**
\- J.Wanamaker, 19th-century retailer

Optimizing marketing spend is still difficult, but this [talk I gave at JuliaCon2022](https://youtu.be/nzR5duccxTg) introduces a modern marketing analysis, Media Mix Modelling (MMM), that can help with this difficult task.

We can combine the strength of Julia with Bayesian decision-making to optimize marketing spend.
You can find the example analysis with high-level and low-level API in this repository (`MediaMixModellingDemo/1-demo-high-level.ipynb`). 

If you just want to peek at the tutorial, check out the [![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://svilupp.github.io/JuliaCon2022/dev/).

Presentation given at JuliaCon can be found in folder presentation (`notebook.jl`, use Pluto.jl to re-play).

# Example Outputs
Key questions for any marketer are: 
- What is the marketing's contribution to business results (ie, we need to quantify the Returns on Ad Spend = "ROAS")?
- What would be the optimal allocation of a given marketing spend that would maximize our revenues (ie, we need to run optimization and find how to re-allocate money across channels)?

The first question (and more) are answered by the MMM summary 1-pager:
- We see that our model has managed to capture our revenues reasonably well
- We see the contributions of each marketing channel
- We can quantify the behaviour of each marketing channel and the potential effect of additional $1 of spend
- Most importantly, we can see that Ad spend vs its effects are misaligned, so there is potential for optimization!
![Media Mix Modelling 1-pager summary](/MediaMixModellingDemo/presentation/assets/mmm-1pager_5.png "Media Mix Modelling 1-Pager Summary")

The second question is answered by the Optimization summary 1-pager:
- We can easily see how to re-allocate our spend (eg, if you remember, Search had high ROAS, so its proposed share should go from 27% to 40%)
- We can quantify the expected benefits of such change
- Thanks to the Bayesian framework, we can quantify the uncertainty of the uplift 
![Optimized Marketing Budget 1-pager summary](/MediaMixModellingDemo/presentation/assets/optimization-1pager_4.png "Optimized Marketing Budget 1-Pager Summary")

All that is left to do now, is to run the experiment with new budget!

# Media Mix Modelling
Media Mix Modelling (MMM) is the go-to analysis for deciding how to spend your precious marketing budget. It has been around for more than half a century, but it's constantly being advanced and its importance is poised to increase with the rise of the privacy-conscious consumer.

There are a few key marketing concepts that we are covered in the talk and are important for understanding and performing MMM, e.g., ad stock, saturation, ROAS and mROAS.

Please refer to [![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://svilupp.github.io/JuliaCon2022/dev/) for detailed overview of methodology, practical tips and a lot of resources (and links to other tutorials). 

# Implementation
We will leverage the power of Bayesian inference with Turing.jl to establish the effectiveness of our campaigns (/marketing channels). The main advantage of the Bayesian approach is be the quantification of uncertainty, which we will channel into our decision-making when deciding on the budget allocations.

The "optimal" spend strategy ("budget") is be found with the help of Metaheuristics.jl.
Overall, we will draw on Julia's core strengths, such as composability and speed.

The implementation closely follows the workflow & methodology of the amazing Robyn package ([Robyn Docs](https://facebookexperimental.github.io/Robyn/docs/quick-start/)), and it adds:
- Bayesian inference for the marketing parameters of interest to quantify the uncertainty
- and Bayesian Decision Framework to capture the specifics of your business (custom loss function, switching cost, etc.)

While there are many resources available for Python and R, I believe this is the first tutorial for MMM in Julia.

Please refer to [![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://svilupp.github.io/JuliaCon2022/dev/) for detailed overview of methodology, practical tips and a lot of resources (and links to other tutorials). 

- July, 2022