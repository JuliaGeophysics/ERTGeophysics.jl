# 2D ERT data plots
# Author: @pankajkmishra
# Apparent-resistivity pseudosections (observed, predicted, normalised misfit) and a cross-plot, inversion
# convergence, and PlotInversion2D, the standard plots of a run, as PlotData2D.jl of MTGeophysics.jl

using CairoMakie
using Printf
using Statistics

#---------- pseudosections ----------

function _ert2d_pseudo!(axis, data::DataFile2D, values::AbstractVector; colorrange, colormap, keep = trues(ndata(data)))
    p = ert2d_pseudo(data)
    ok = keep .& isfinite.(values)
    scatter!(axis, p.y[ok], p.depth[ok]; color = values[ok], colormap, colorrange, marker = :rect, markersize = 9,
             strokewidth = 0)
end

_ert2d_rhoa_obs(d::DataFile2D) = d.geometric_factor .* d.resistance

"""
    plot_ert2d_pseudosection(data; output_path, values=nothing, colorrange=nothing,
                             label="log10 ρa (Ω·m)", colormap=:Spectral) -> path

Pseudosection of the data, by default log10 ρa = log10 K R (`ert2d_pseudo` positions).
"""
function plot_ert2d_pseudosection(data::DataFile2D; output_path::AbstractString, values = nothing,
                                  colorrange = nothing, label::AbstractString = "log10 ρa (Ω·m)", colormap = :Spectral)
    CairoMakie.activate!()
    v = values === nothing ? log10.(max.(_ert2d_rhoa_obs(data), floatmin())) : Float64.(values)
    v[.!isfinite.(v)] .= NaN
    cr = colorrange === nothing ? _ert2d_auto_range(reshape(v, :, 1)) : colorrange
    figure = Figure(size = (1200, 420))
    axis = _ert_axis(figure[1, 1]; xlabel = "Distance along profile (m)", ylabel = "Pseudo-depth (m)", yreversed = true)
    sc = _ert2d_pseudo!(axis, data, v; colorrange = cr, colormap)
    scatter!(axis, data.electrodes, zeros(nelectrodes(data)); marker = :dtriangle, markersize = 7, color = :black)
    Colorbar(figure[1, 2], sc, label = label, labelfont = :regular)
    _ert_save(output_path, figure)
end

"""
    plot_ert2d_data_fit(observed, predicted; output_path, error_floor=0.0) -> path

Observed and predicted ρa pseudosections on one colour scale, the normalised misfit
(log ρa_pred − log ρa_obs)/σ, and a cross-plot of predicted against observed ρa with
the rms in its legend.
"""
function plot_ert2d_data_fit(observed::DataFile2D, predicted::DataFile2D; output_path::AbstractString,
                             error_floor::Real = 0.0)
    CairoMakie.activate!()
    v = ert2d_valid_data(observed; error_floor)
    obs = log10.(max.(_ert2d_rhoa_obs(observed), floatmin()))
    pred = log10.(max.(predicted.rhoa, floatmin()))
    ok = v.keep .& isfinite.(predicted.rhoa) .& (predicted.rhoa .> 0)
    obs[.!ok] .= NaN
    pred[.!ok] .= NaN
    misfit = fill(NaN, ndata(observed))
    misfit[ok] .= (log(10) .* (pred[ok] .- obs[ok])) ./ v.sigma[ok]
    fit = chi2_rms2d(observed, predicted; error_floor)
    cr = _ert2d_auto_range(reshape(vcat(obs[ok], pred[ok]), :, 1))

    figure = Figure(size = (1300, 1050))
    kw = (xlabel = "Distance along profile (m)", ylabel = "Pseudo-depth (m)", yreversed = true)
    ax1 = _ert_axis(figure[1, 1]; kw...)
    s1 = _ert2d_pseudo!(ax1, observed, obs; colorrange = cr, colormap = :Spectral)
    text!(ax1, 0.01, 0.03; text = "observed", space = :relative, align = (:left, :bottom))
    ax2 = _ert_axis(figure[2, 1]; kw...)
    _ert2d_pseudo!(ax2, observed, pred; colorrange = cr, colormap = :Spectral)
    text!(ax2, 0.01, 0.03; text = "predicted", space = :relative, align = (:left, :bottom))
    Colorbar(figure[1:2, 2], s1, label = "log10 ρa (Ω·m)", labelfont = :regular)
    ax3 = _ert_axis(figure[3, 1]; kw...)
    s3 = _ert2d_pseudo!(ax3, observed, misfit; colorrange = (-3, 3), colormap = :balance)
    text!(ax3, 0.01, 0.03; text = "misfit", space = :relative, align = (:left, :bottom))
    Colorbar(figure[3, 2], s3, label = "(pred − obs)/σ", labelfont = :regular)
    linkxaxes!(ax1, ax2, ax3)
    ax4 = _ert_axis(figure[1:3, 3]; xlabel = "Observed ρa (Ω·m)", ylabel = "Predicted ρa (Ω·m)", xscale = log10,
                    yscale = log10, aspect = 1)
    po, pp = 10 .^ obs[ok], 10 .^ pred[ok]
    lim = (minimum(vcat(po, pp)) / 1.2, maximum(vcat(po, pp)) * 1.2)
    lines!(ax4, [lim[1], lim[2]], [lim[1], lim[2]], color = :gray, linestyle = :dash)
    scatter!(ax4, po, pp; markersize = 5, color = (:navy, 0.6), label = @sprintf("RMS %.3f, %d data", fit.rms, fit.count))
    limits!(ax4, lim..., lim...)
    axislegend(ax4, position = :lt, framevisible = false, labelfont = :regular)
    colsize!(figure.layout, 3, Relative(0.32))
    _ert_save(output_path, figure)
end

"""
    PlotData2D(data_path; predicted_path=nothing, output_path) -> path

Pseudosection of a data file, or with `predicted_path` the data fit.
"""
function PlotData2D(data_path::AbstractString; predicted_path::Union{Nothing, AbstractString} = nothing,
                    output_path::AbstractString)
    observed = load_data2d(data_path)
    predicted_path === nothing && return plot_ert2d_pseudosection(observed; output_path)
    plot_ert2d_data_fit(observed, load_data2d(predicted_path); output_path)
end

#---------- inversion convergence ----------

"""
    plot_inv2d_convergence(history; output_path, target_rms=0.0) -> path

Rms (with the target line when `target_rms > 0`), the objective terms and β per iteration.
"""
function plot_inv2d_convergence(history::AbstractVector; output_path::AbstractString, target_rms::Real = 0.0)
    CairoMakie.activate!()
    it = [h.iteration for h in history]
    figure = Figure(size = (1300, 420))
    ax1 = _ert_axis(figure[1, 1]; xlabel = "Iteration", ylabel = "RMS", yscale = log10)
    scatterlines!(ax1, it, [h.rms for h in history], color = :navy)
    target_rms > 0 && hlines!(ax1, [target_rms], color = :gray, linestyle = :dash)
    ax2 = _ert_axis(figure[1, 2]; xlabel = "Iteration", ylabel = "Objective terms", yscale = log10)
    scatterlines!(ax2, it, [h.objective for h in history], label = "objective")
    scatterlines!(ax2, it, [h.chi2 / 2 for h in history], label = "χ²/2")
    reg = [(h.iteration, h.regularization) for h in history if h.regularization > 0]
    isempty(reg) || scatterlines!(ax2, first.(reg), last.(reg), label = "‖Wm R (m − m_ref)‖²/2")
    axislegend(ax2, position = :rt, framevisible = false, labelfont = :regular)
    ax3 = _ert_axis(figure[1, 3]; xlabel = "Iteration", ylabel = "β", yscale = log10)
    scatterlines!(ax3, it, [h.beta for h in history], color = :darkgreen)
    _ert_save(output_path, figure)
end

#---------- inversion run ----------

"""
    PlotInversion2D(run; true_model_path=nothing, maximum_depth=nothing,
                    resistivity_log10_range=nothing, outlines=[]) -> paths

Standard plots of a run returned by the six-file `Invert2D`,
written to `run.run_dir/plots`: the mesh, the start, final (faded where the coverage is
low) and, when given, true models on one colour scale, the coverage, the data fit, and
the convergence.
"""
function PlotInversion2D(run; true_model_path::Union{Nothing, AbstractString} = nothing,
                         maximum_depth::Union{Nothing, Real} = nothing, resistivity_log10_range = nothing,
                         outlines = NamedTuple[])
    dir = joinpath(run.run_dir, "plots")
    path(name) = joinpath(dir, name)
    water = any(run.water) ? run.water : nothing
    # by default the main domain: the even grid down to its depth (the padding below only follows it)
    main = haskey(run, :padding) && any(run.padding) ? _ert2d_main_depth(run.mesh, run.padding) : nothing
    depth = maximum_depth !== nothing ? maximum_depth : main !== nothing ? main : _ert2d_default_depth(run.mesh)
    range = if resistivity_log10_range === nothing
        rows = 1:searchsortedfirst(run.mesh.z_nodes, depth)
        cols = _core_range(run.mesh.y_cell_sizes)
        v = log10.(run.final[rows, cols])
        v[ert2d_air_mask(run.mesh)[rows, cols]] .= NaN
        _ert2d_auto_range(v)
    else
        resistivity_log10_range
    end
    core = (maximum_depth = depth, resistivity_log10_range = range, water, outlines)
    paths = String[
        plot_ert2d_mesh(run.mesh; output_path = path("Mesh.png"), region = :full, water),
        plot_ert2d_mesh(run.mesh; output_path = path("MeshCore.png"), region = :core, water, core_depth = depth),
        plot_ert2d_model(run.mesh, run.start; output_path = path("ModelStart.png"), core...),
        plot_ert2d_model(run.mesh, run.final; output_path = path("ModelFinal.png"), core...,
                         coverage = run.coverage),
        plot_ert2d_model(run.mesh, run.final; output_path = path("ModelFinalFull.png"), show_padding = true,
                         resistivity_log10_range = range, water),
        plot_ert2d_data_fit(run.observed, run.predicted; output_path = path("DataFit.png"),
                            error_floor = run.ctrl.error_floor),
    ]
    if run.coverage !== nothing
        push!(paths, plot_ert2d_model(run.mesh, log10.(max.(run.coverage, 1e-12)); output_path = path("Coverage.png"),
                                      maximum_depth = depth, log10_values = true, colormap = :viridis,
                                      colorbar_label = "log10 coverage", resistivity_log10_range = (-4.0, 0.0)))
    end
    run.history === nothing || push!(paths, plot_inv2d_convergence(run.history; output_path = path("Convergence.png"),
                                                                    target_rms = run.ctrl.target_rms))
    if true_model_path !== nothing
        truth = ReadModel2D(true_model_path)
        tmesh, _ = Mesh2DFromInputs(truth, run.observed; warn = false)
        push!(paths, plot_ert2d_model(tmesh, truth.resistivity; output_path = path("ModelTrue.png"), core..., water = nothing))
    end
    paths
end
