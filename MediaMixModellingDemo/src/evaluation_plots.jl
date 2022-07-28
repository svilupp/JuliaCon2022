using ShiftedArrays: lag

# Parameters
"""
    struct ParamsPlot
        title_suffix = ""
        color_spend = palette(:default)[6]
        color_spend_optimized = palette(:default)[6]
        color_revenues = palette(:default)[1]
        color_revenues_original = palette(:default)[2]
        units_revenues = "USD"
        units_spend = "USD"
        title_fontsize = 10
        table_fontsize_header = 10
        table_fontsize_body = 8
        output_dpi = 150
        output_size_mmm = (1000, 800)
        output_size_optim = (1000, 800)
    end

Holds plotting defaults
"""
@with_kw struct ParamsPlot
    title_suffix = ""
    color_spend = palette(:default)[6]
    color_spend_optimized = palette(:default)[6]
    color_revenues = palette(:default)[1]
    color_revenues_original = palette(:default)[2]
    units_revenues = "USD"
    units_spend = "USD"
    title_fontsize = 10
    table_fontsize_header = 10
    table_fontsize_body = 8
    output_dpi = 150
    output_size_mmm = (1000, 800)
    output_size_optim = (1000, 800)
end

# 1-liners
# to quickly reformat column names from "variable_name" to "Variable Name"
prettify_labels(s) = replace(s, "_" => " ") |> x -> titlecase(x)
pct_formatter0f(x) = @sprintf("%.0f%%", 100*x)
pct_formatter1f(x) = @sprintf("%.1f%%", 100*x)
float_formatter1f(x) = @sprintf("%.1f", x)

# Plot histogram of prior predictive (model implied by the set priors) vs Actuals Y (=True Y)
function plot_prior_predictive_histogram(y_true, y_prior, p)
    pl = histogram(y_prior, normalize = true, label = "Simulated")
    histogram!(pl, y_true, normalize = true, label = "Actuals", color = :red,
               title = "Prior Predictive Check", titlefontsize = p.title_fontsize)

    return pl
end

"""
    plot_model_fit_by_period(y_true, y_pred, p)

Create a plot comparing the modelled revenues(`y_pred`) vs actuals (`y_true`)
"""
function plot_model_fit_by_period(y_true, y_pred, p)
    plt_max = max(maximum(y_true), maximum(y_pred))
    annotation_points = [(0, plt_max), (0, plt_max * 0.95), (0, plt_max * 0.9)]

    pl = plot(y_true, title = "Quality of Model Fit" * p.title_suffix,
              titlefontsize = p.title_fontsize,
              label = "Actuals", color = p.color_revenues_original)
    plot!(pl, y_pred, label = "Fitted",
          color = p.color_revenues,
          yguide = "Revenues")

    annotate!(annotation_points[1]...,
              text("Goodness of Fit", "helvetica bold", 10, :black, :top, :left))
    annotate!(annotation_points[2]...,
              text(@sprintf("Pseudo R2: %.1f%%", pseudor2(y_true, y_pred)*100), "helvetica",
                   8, :black, :top, :left))
    annotate!(annotation_points[3]...,
              text(@sprintf("NRMSE: %.1f%%", nrmse(y_true, y_pred)*100), "helvetica", 8,
                   :black, :top, :left))

    # optional - can add date-based xticks, needs reformating from the long string
    # plot!(xticks=(1:20:105,df.dt[1:20:105]),xrot=45)
    return pl
end

"""
    plot_contributions(effect_shares, cols, p)

Creates a waterfall plot to show % contributions to fitted revenues (no-noise! only deterministic components)
"""
function plot_contributions(effect_shares, cols, p)

    # Prepare data
    temp_to_plot = @chain begin
        DataFrame(:labels => categorical(cols, levels = cols),
                  # add minimum value to show at least some bar (1e-3)
                  :effect_share => effect_shares |> percentage_share |> vec |>
                                   x -> max.(x, 1e-3))
        reverse
        @transform :base = coalesce.(lag(:effect_share), 0) |> cumsum .|> Float64
        DataFramesMeta.stack([:effect_share, :base])
        @transform :variable = categorical(:variable; levels = ["effect_share", "base"])
    end

    yticks_ = 1:length(cols)
    xticks_ = 0:0.2:1 .|> x -> round(x; digits = 1)

    plt_colorscheme = [p.color_revenues, :transparent]
    plt_colors = getindex.(Ref(plt_colorscheme), temp_to_plot.variable .|> levelcode)
    plt_linecolors = getindex.(Ref([:black, nothing]), temp_to_plot.variable .|> levelcode)

    pl = @df temp_to_plot groupedbar(levelcode.(:labels), :value,
                                     group = :variable, bar_position = :stack,
                                     orientation = :horizontal,
                                     linecolor = plt_linecolors,
                                     color = plt_colors,
                                     yticks = (yticks_, cols),
                                     xticks = (xticks_, pct_formatter0f.(xticks_)),
                                     rightmargin = 15Plots.mm,
                                     legend = false,
                                     title = "Revenue Contributions",
                                     titlefontsize = p.title_fontsize)

    # Labels
    for row in eachrow(unstack(temp_to_plot, :variable, :value; fill = 0))
        annotate!((row.base + row.effect_share) + 0.01, levelcode(row.labels),
                  text(pct_formatter1f(row.effect_share), 8, :black, :mid, :left))
    end

    return pl
end

"""
    plot_effects_vs_spend(effect_shares, spend_shares, cols, p)

Plots a comparison between where we spend money (`spend_shares`) vs effect on revenues (`effect_shares`). Expressed in relative terms (% of total)

Big differences within each category (spend % vs effect %) suggest optimization opportunities

"""
function plot_effects_vs_spend(effect_shares, spend_shares, cols, p)

    # Create utility dataframe
    temp_to_plot = @chain begin
        DataFrame(:labels => cols, :effect_share => effect_shares,
                  :spend_share => spend_shares)
        DataFramesMeta.stack(_, [:effect_share, :spend_share])
        @transform :variable=categorical(prettify_labels.(:variable)) :labels=categorical(:labels;
                                                                                          levels = cols)
    end

    plt_groups = temp_to_plot.variable |> levels
    plt_colorscheme = [p.color_revenues, p.color_spend]
    plt_colors = getindex.(Ref(plt_colorscheme), temp_to_plot.variable .|> levelcode)
    plt_xlim = (0, maximum(temp_to_plot.value) + 0.1)

    yticks_ = (0.5 .+ 0):length(cols)
    xticks_ = 0:0.1:plt_xlim[2] .|> x -> round(x; digits = 1)

    pl = @df temp_to_plot groupedbar(:labels, :value,
                                     group = :variable, orientation = :h,
                                     yticks = (yticks_, cols),
                                     xticks = (xticks_, pct_formatter0f.(xticks_)),
                                     color = plt_colors,
                                     xlim = plt_xlim,
                                     title = "Share of Spend vs Share of Effect",
                                     titlefontsize = p.title_fontsize)

    for row in eachrow(temp_to_plot)
        y_offset = row.variable == plt_groups[1] ? -0.7 : -0.3
        annotate!(row.value + 0.01, levelcode(row.labels) + y_offset,
                  text(pct_formatter1f(row.value), 8, :black, :mid, :left))
    end

    return pl
end

"""
    plot_response_curves_table(decay_rates, roass, mroas_at_means, cols, roas_total, p)

Plot a quick summary table of the parameters of the response curves

Expects vectors of parameters - one of each channel

"""
function plot_response_curves_table(decay_rates, roass, mroas_at_means, cols, roas_total, p)
    plt_labels = categorical(cols; levels = cols)
    yticks_ = (-0.5 + 1):length(cols)

    pl = plot(grid = false, showaxis = false,
              yticks = (yticks_, plt_labels), ylim = (0, length(cols)),
              xticks = false, tickfontsize = p.table_fontsize_body,
              titlefontsize = p.title_fontsize,
              title = "Fitted Response Curves")

    # TOTAL ROAS
    annotate!(0.5, 0.1,
              text(@sprintf("Overall ROAS: %.1fx", roas_total), p.table_fontsize_header,
                   :black, :mid, :center))

    # Headers
    annotate!(0.2, length(cols) - 0.15,
              text("Decay Rate", p.table_fontsize_header, :black, :mid, :center))
    annotate!(0.5, length(cols) - 0.15,
              text("ROAS", p.table_fontsize_header, :black, :mid, :center))
    annotate!(0.8, length(cols) - 0.15,
              text("mROAS (@mean)", p.table_fontsize_header, :black, :mid, :center))

    for (label, decay_, roas_, mroas_) in zip(plt_labels, decay_rates, roass,
                                              mroas_at_means)
        annotate!(0.2, levelcode(label) - 0.5,
                  text(pct_formatter1f(decay_), p.table_fontsize_body, :black, :mid,
                       :center))
        annotate!(0.5, levelcode(label) - 0.5,
                  text(@sprintf("%.1fx", roas_), p.table_fontsize_body, :black, :mid,
                       :center))
        annotate!(0.8, levelcode(label) - 0.5,
                  text(@sprintf("%.1fx", mroas_), p.table_fontsize_body, :black, :mid,
                       :center))
    end

    return pl
end

"""
    plot_mmm_one_pager(plot_array, frame_idx::Int64, p)

Combines all plots in plot array into 1 layout for MMM 1-pager

Allows to reveal the plots partially, one by one (provide `frame_idx`)
"""
function plot_mmm_one_pager(plot_array, frame_idx::Int64, p)
    @assert frame_idx > 0

    pl_empty = plot(title = "", grid = false, showaxis = false, ticks = false,
                    bottom_margin = -0Plots.px)

    plot_array_masked = repeat([pl_empty], length(plot_array))
    plot_array_masked[1:frame_idx] .= plot_array[1:frame_idx]

    pl = plot(plot_array_masked...,
              layout = @layout([A{0.01h}; [B C]; [D E]]),
              size = p.output_size_mmm, dpi = p.output_dpi)

    return pl
end

"""
    Plots.plot(fitted::Stage2Fit, inputs::InputData, pplot::ParamsPlot = ParamsPlot())

Wraps all computations to create MMM 1-Pager
"""
function Plots.plot(fitted::Stage2Fit, inputs::InputData, pplot::ParamsPlot = ParamsPlot())

    # TODO: fit_stage2_mask not used for plots!

    @unpack params, generated_data, model, chain = fitted
    @unpack y_std, cols_spend, X_spend = inputs

    adspend_sum_trf = X_spend |> sum_columns
    adspend_sum_raw = inputs.revert_pipe_spend(X_spend) |> sum_columns

    # Header
    pl0 = plot(title = "MMM 1-Pager " * pplot.title_suffix, grid = false, showaxis = false,
               ticks = false, bottom_margin = -0Plots.px)

    ### PLOT 1
    y_pred = predict(fitted)
    # overwrite title suffix not to show the same thing twice
    pl1 = plot_model_fit_by_period(y_std, y_pred, ParamsPlot(pplot, title_suffix = ""))

    ### PLOT 2
    pl2 = let effect_share_mean = mean_fitted_effects(params.model_name, generated_data;
                                                      extract_keys = [
                                                          :mu_trend,
                                                          :mu_org,
                                                          :mu_context,
                                                          :mu_spend_by_var,
                                                      ]),
        cols = ["Trend", "Organic", "Context", cols_spend...] .|> prettify_labels

        plot_contributions(effect_share_mean, cols, pplot)
    end

    ### PLOT 3
    # extract fitted geometric decay rates
    decay_rates = getflatsamples(chain, "decay_rate") |> x -> mean(x, dims = 1) |> vec
    # calculate roas across the whole period
    roass = let effects = mean_fitted_effects(params.model_name, generated_data;
                                              extract_keys = [:mu_spend_by_var]),
        spends = adspend_sum_trf,
        factors = params.units_ratio_spend_to_y

        calc_roas.(effects, spends, factors)
    end
    # calculate mroas at mean spend of each variable with delta of 0.01
    mroas_at_means = [calc_mroas(params.adspend_mean_nonzero[idx], 0.01, chain, params,
                                 idx)[1] for idx in 1:length(cols_spend)]

    # Total ROAS (depends on model2b used)
    roas_total = let effects = mean_fitted_effects(params.model_name, generated_data;
                                                   extract_keys = [:mu_spend_by_var]),
        spends = adspend_sum_trf,
        factors = params.units_ratio_spend_to_y,
        weights = adspend_sum_raw

        calc_roas(effects, spends, factors, weights)
    end

    cols = prettify_labels.(cols_spend)

    pl3 = plot_response_curves_table(decay_rates, roass, mroas_at_means, cols, roas_total,
                                     pplot)

    ### PLOT 4
    pl4 = let ad_effect_share_mean = (mean_fitted_effects(params.model_name, generated_data;
                                                          extract_keys = [:mu_spend_by_var]) |>
                                      percentage_share),
        spend_share = (adspend_sum_raw |> percentage_share),
        cols = prettify_labels.(cols_spend)

        plot_effects_vs_spend(ad_effect_share_mean, spend_share, cols, pplot)
    end

    ####################
    # 1-Pager

    plot_array = [pl0, pl1, pl2, pl3, pl4]

    # show final
    pl = plot_mmm_one_pager(plot_array, 5, pplot)

    return pl
end
###############################
### Optimization

"""
    plot_optimized_spend_share_comparison(spend_share_prev, spend_share_optim, cols, p)

Plots a comparison between previous marketing spend allocation and the optimum discovered

Note that the differences are constrainted by the optimization settings (see `multiplier_bounds` !)
"""
function plot_optimized_spend_share_comparison(spend_share_prev, spend_share_optim, cols, p)

    # Create utility dataframe
    temp_to_plot = @chain begin
        DataFrame(:labels => cols, :previous => spend_share_prev,
                  :optimized => spend_share_optim)
        DataFramesMeta.stack(_, [:optimized, :previous])
        @transform :variable=categorical(prettify_labels.(:variable),
                                         levels = prettify_labels.([
                                                                       "Optimized",
                                                                       "Previous",
                                                                   ])) :labels=categorical(:labels;
                                                                                           levels = cols)
    end

    plt_groups = temp_to_plot.variable |> levels
    plt_colorscheme = [p.color_spend_optimized, p.color_spend]
    plt_colors = getindex.(Ref(plt_colorscheme), temp_to_plot.variable .|> levelcode)
    plt_alphascheme = [1, 0.4]
    plt_alphas = getindex.(Ref(plt_alphascheme), temp_to_plot.variable .|> levelcode)
    plt_xlim = (0, maximum(temp_to_plot.value) + 0.1)

    yticks_ = (0.5 .+ 0):length(cols)
    xticks_ = 0:0.1:plt_xlim[2] .|> x -> round(x; digits = 1)

    pl = @df temp_to_plot groupedbar(:labels, :value,
                                     group = :variable, orientation = :h,
                                     yticks = (yticks_, cols),
                                     xticks = (xticks_, pct_formatter0f.(xticks_)),
                                     color = plt_colors,
                                     alpha = plt_alphas,
                                     xlim = plt_xlim,
                                     title = "Proposed Mkt. Budget Allocation",
                                     titlefontsize = p.title_fontsize)

    for row in eachrow(temp_to_plot)
        y_offset = row.variable == plt_groups[1] ? -0.7 : -0.3
        annotate!(row.value + 0.01, levelcode(row.labels) + y_offset,
                  text(pct_formatter1f(row.value), 8, :black, :mid, :left))
    end

    return pl
end

"""
    plot_optimized_contribution(effect_prev, effect_optim, roas_total, optim_start,
                                     optim_end, revert_y_func, p)

Plots expected delta in revenue if the new marketing budget has been implemented (on the same period)
"""
function plot_optimized_contribution(effect_prev, effect_optim, roas_total, optim_start,
                                     optim_end, revert_y_func, p)

    # utility function
    extract_effect_raw = x -> revert_y_func([x])[1]

    uplift_delta = extract_effect_raw(effect_optim) - extract_effect_raw(effect_prev)
    uplift_perc = extract_effect_raw(effect_optim) / extract_effect_raw(effect_prev) - 1

    pl_max = max(extract_effect_raw(effect_optim), extract_effect_raw(effect_prev))

    pl = @chain begin
        DataFrame(:labels => ["Previous", "Optimized"],
                  :vals => [extract_effect_raw(effect_prev),
                      extract_effect_raw(effect_optim)])

        @df bar(:labels, :vals, title = "Uplift in Marketing Contribution",
                titlefontfize = p.title_fontsize, color = p.color_revenues,
                label = "", bar_width = 0.5, ylim = (0, 1.2 * pl_max))
    end
    # Summary
    annotate!(0, 0, text("", "helvetica bold", 10, :black, :top, :left))
    annotate!(0.2, pl_max * 1.15,
              text("Period: $(optim_start) - $(optim_end)",
                   "helvetica bold", p.table_fontsize_body, :black, :mid, :left))
    annotate!(0.2, pl_max * 1.1,
              text("ROAS: " * @sprintf("%.1fx", roas_total), "helvetica bold",
                   p.table_fontsize_body, :black, :mid, :left))
    annotate!(0.2, pl_max * 1.05,
              text("Uplift (AVG): " * @sprintf("%.1f", uplift_delta) * " / " *
                   pct_formatter1f(uplift_perc),
                   "helvetica bold", p.table_fontsize_body, :black, :mid, :left))

    # Plot labels
    let prev_ = extract_effect_raw(effect_prev), optim_ = extract_effect_raw(effect_optim)
        annotate!(0.5, prev_ * 1.05,
                  text(float_formatter1f(prev_), p.table_fontsize_body, :black, :bottom,
                       :center))
        annotate!(1.5, optim_ * 1.05,
                  text(float_formatter1f(optim_), p.table_fontsize_body, :black, :bottom,
                       :center))
    end

    return pl
end

"""
    plot_optimized_uplift_histogram(revenue_uplift, p)

Plots a histogram of modelled uplift (`revenue_uplift`)
"""
function plot_optimized_uplift_histogram(revenue_uplift, p)
    pl = histogram(revenue_uplift,
                   title = "Modelled Revenue Uplift", bins = :sqrt,
                   label = "Uplift",
                   titlefontfize = p.title_fontsize, color = p.color_revenues)
    pl_max = maximum(yticks(pl)[1][1])
    xmin = let tickvalues = xticks(pl)[1][1]
        x_step_size = (tickvalues[2] - tickvalues[1])
        xmin = minimum(tickvalues) - 0.0 * x_step_size
    end
    plot!(pl, ylim = (0, pl_max * 1.5))

    uplift_avg = mean(revenue_uplift)
    uplift_std = std(revenue_uplift)

    # Summary
    annotate!(0, 0, text("", "helvetica bold", 10, :black, :top, :left))
    # annotate!(0,pl_max*1.2,text("Period: $(optim_start) - $(optim_end)","helvetica bold",10,:black,:top,:left))
    annotate!(xmin, pl_max * 1.4,
              text("Uplift AVG: " * float_formatter1f(uplift_avg) * " / StDev: " *
                   float_formatter1f(uplift_std),
                   "helvetica bold", p.table_fontsize_body, :black, :mid, :left))
    annotate!(xmin, pl_max * 1.35,
              text("Percentage of profitable scenarios: " *
                   pct_formatter1f(mean(revenue_uplift .> 0)),
                   "helvetica bold", p.table_fontsize_body, :black, :mid, :left))

    # guiding lines
    vline!(pl, [0], color = :gray, linestyle = :dash, label = "")
    vline!(pl, [uplift_avg], color = :red, linestyle = :dash, label = "AVG Uplift")

    return pl
end

"""
    plot_optimization_one_pager(plot_array, frame_idx::Int64, p)

Create a layout for Optimization 1-pager given a provided array of plots (`plot_array`)

It is possible to reveal plots one by one (use `frame_idx`)
"""
function plot_optimization_one_pager(plot_array, frame_idx::Int64, p)
    @assert frame_idx > 0

    pl_empty = plot(title = "", grid = false, showaxis = false, ticks = false,
                    bottom_margin = -0Plots.px)

    plot_array_masked = repeat([pl_empty], length(plot_array))
    plot_array_masked[1:frame_idx] .= plot_array[1:frame_idx]

    pl = plot(plot_array_masked...,
              layout = @layout([A{0.01h}; [B{0.5w} grid(2, 1)]]),
              size = p.output_size_optim, dpi = p.output_dpi)
    return pl
end

"""
    Plots.plot(optimal::OptimalBudget, fitted::Stage2Fit, inputs::InputData,
                    pplot::ParamsPlot = ParamsPlot())

Wraps all computations required to plot the Optimization 1-pager
"""
function Plots.plot(optimal::OptimalBudget, fitted::Stage2Fit, inputs::InputData,
                    pplot::ParamsPlot = ParamsPlot())
    @unpack params, chain, model, generated_data = fitted
    @unpack simulations_optim, X_spend_optim_trf = optimal
    @unpack dt, cols_spend, X_spend, optim_mask, revert_pipe_y, revert_pipe_spend, fit_stage2_mask = inputs

    chain_optim = Chains(chain, :parameters)

    adspend_sum_trf = X_spend |> x -> to_masked_matrix(x, optim_mask) |> sum_columns
    adspend_sum_raw = revert_pipe_spend(X_spend) |>
                      x -> to_masked_matrix(x, optim_mask) |> sum_columns
    adspend_sum_optim = revert_pipe_spend(X_spend_optim_trf) |>
                        x -> to_masked_matrix(x, optim_mask) |> sum_columns

    # Header
    pl0 = plot(title = "Budget Optimization 1-Pager " * pplot.title_suffix, grid = false,
               showaxis = false, ticks = false, bottom_margin = -0Plots.px)

    ###
    pl1 = let spend_share_prev = adspend_sum_raw |> percentage_share,
        spend_share_optim = adspend_sum_optim |> percentage_share,
        cols = prettify_labels.(cols_spend)

        plot_optimized_spend_share_comparison(spend_share_prev, spend_share_optim, cols,
                                              pplot)
    end

    ###
    # Total ROAS
    # TODO: in period!
    roas_total = let effects = mean_fitted_effects(params.model_name, simulations_optim;
                                                   extract_keys = [:mu_spend_by_var]),
        spends = adspend_sum_trf,
        factors = params.units_ratio_spend_to_y,
        weights = adspend_sum_raw

        calc_roas(effects, spends, factors, weights)
    end
    pl2 = let roas_total = roas_total,
        effect_prev = mean_fitted_effects(params.model_name, generated_data;
                                          extract_keys = [:mu_spend], mask = optim_mask)[1],
        effect_optim = mean_fitted_effects(params.model_name, simulations_optim;
                                           extract_keys = [:mu_spend], mask = optim_mask)[1],
        optim_start = dt[optim_mask] |> minimum,
        optim_end = dt[optim_mask] |> maximum

        plot_optimized_contribution(effect_prev, effect_optim, roas_total, optim_start,
                                    optim_end, revert_pipe_y, pplot)
    end

    ###
    pl3 = let simulations_prev = simulate_revenues_summed(chain_optim, model, optim_mask;
                                                          extract_key = :mu),
        simulations_optimized = simulate_revenues_summed(chain_optim,
                                                         model.f(merge(model.args,
                                                                       (;
                                                                        X_spend = to_masked_matrix(X_spend_optim_trf,
                                                                                                   fit_stage2_mask)))...),
                                                         optim_mask; extract_key = :mu)

        revenue_uplift = revert_pipe_y(simulations_optimized .- simulations_prev)

        plot_optimized_uplift_histogram(revenue_uplift, pplot)
    end

    ####################
    # 1-Pager

    plot_array = [pl0, pl1, pl2, pl3]

    # show final
    pl = plot_optimization_one_pager(plot_array, 4, pplot)
    return pl
end
