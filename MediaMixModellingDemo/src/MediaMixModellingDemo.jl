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
import Metaheuristics

export create_dataset
include("data_generation.jl")

export InputData, build_inputs
include("data_processing.jl")

export plot_periodogram, generate_fourier_series, generate_seasonality_features,
       standardize_by_max, standardize_by_zscore
include("feature_engineering.jl")

export geometric_decay, hill_curve
include("marketing_transformations.jl")

export ParamsStage1, set_priors_stage1_trendline, set_priors_auto_scales,
       sanity_check_priors
export ParamsStage2, decay_rates_types_dictionary, set_priors_stage2_decay_rates,
       set_priors_stage2_hill_curves
export set_priors
include("model_priors.jl")

export model_stage1a, model_stage1b
export model_stage2a, model_stage2b
export fit, predict, Stage1Fit, Stage2Fit
export to_masked_matrix, quick_nuts_diagnostics
include("model_definition.jl")

export plot_priors_decay_rate
include("model_plots.jl")

export pseudor2, rmse, nrmse
include("evaluation_stats.jl")

export sum_columns, percentage_share, getflatsamples, mean_fitted_effects
export calc_roas, calc_mroas, saturate_adspend
include("evaluation_calculations.jl")

export generate_objective_func, threaded_objective_func, simulate_revenues_summed,
       workflow_budget_to_simulation, optimize, OptimalBudget, sanity_check_optimum
include("budget_optimization.jl")

export ParamsPlot, prettify_labels
export plot_prior_predictive_histogram, plot_model_fit_by_period
export plot_contributions, plot_effects_vs_spend, plot_response_curves_table
export plot_mmm_one_pager
export plot_optimized_spend_share_comparison, plot_optimized_contribution,
       plot_optimized_uplift_histogram
export plot_optimization_one_pager
export plot
include("evaluation_plots.jl")

end # Module End
