module MediaMixModellingDemo

using Reexport: @reexport
# temporary fix
@reexport using DataFramesMeta
@reexport using TableTransforms: apply, reapply, revert, MinMax, ZScore
@reexport using Distributions
@reexport using Plots, StatsPlots
@reexport using CategoricalArrays
@reexport using Turing
@reexport using MCMCChains
@reexport import AdvancedHMC
@reexport import Random
const AHMC = AdvancedHMC
export AHMC

# using Turing: Variational
using Parameters
using Printf
using Chain: @chain
using Logging

# *** UNDER CONTRUCTION *** 
# TO DO: Move from Revise into a module
# TO DO: fix dependency on Transformation caches (pipe_cache_y and pipe_cache_spend), esp. in ParamsStage1,2

export create_dataset
include("data_generation.jl")

export plot_periodogram, generate_fourier_series, generate_seasonality_features,
       standardize_by_max, standardize_by_zscore
include("feature_engineering.jl")

export geometric_decay, hill_curve
include("marketing_transformations.jl")

export ParamsStage1, model_stage1a, ParamsStage2, model_stage2a, quick_nuts_diagnostics
include("model_definition.jl")

export pseudor2, rmse, nrmse
include("evaluation_stats.jl")

export sum_columns, percentage_share, getflatsamples
export calc_roas_total, calc_roas, calc_mroas, saturate_adspend
include("evaluation_calculations.jl")

export ParamsPlot, prettify_labels
export plot_prior_predictive_histogram, plot_model_fit_by_period
export plot_contributions, plot_effects_vs_spend, plot_response_curves_table
export plot_mmm_one_pager
export plot_optimized_spend_share_comparison, plot_optimized_contribution,
       plot_optimized_uplift_histogram
export plot_optimization_one_pager
include("evaluation_plots.jl")

export generate_objective_func, simulate_revenues_summed, workflow_budget_to_simulation
include("budget_optimization.jl")

end # Module End
