# 3D ERT plots
# Author: @pankajkmishra
# Depth slices (map view, x north up, y east right), vertical sections, the mesh plan and the data fit of 3D runs

using CairoMakie
using Printf
using Statistics

# log10 values with air blank
function _ert3d_values(mesh, ρ, log10_values)
    v = log10_values ? Array{Float64, 3}(ρ) : log10.(Array{Float64, 3}(ρ))
    v[ert3d_air_mask(mesh)] .= NaN
    v
end

# core column ranges (uniform cells) of a mesh
_ert3d_core(mesh) = (_core_range(mesh.x_cell_sizes), _core_range(mesh.y_cell_sizes))

"""
    plot_ert3d_slices(mesh, ρ; output_path, depths=nothing, resistivity_log10_range=nothing,
                      coverage=nothing, coverage_cutoff=1e-3, log10_values=false, colormap=:Spectral,
                      colorbar_label="log10 ρ (Ω·m)", outlines=[]) -> path

Map-view slices of the core at `depths` (m below the model top; default six depths down
to a third of the survey size), x north up and y east right, electrodes as dots; air is
blank.
`coverage` fades the cells below `coverage_cutoff`; `outlines` are
`(x_range, y_range, z_range)` boxes drawn where they cut a slice.
"""
function plot_ert3d_slices(mesh::ERT3DMesh, ρ::AbstractArray{<:Real, 3}; output_path::AbstractString, depths = nothing,
                           resistivity_log10_range = nothing, coverage = nothing, coverage_cutoff::Real = 1e-3,
                           log10_values::Bool = false, colormap = :Spectral, colorbar_label = "log10 ρ (Ω·m)",
                           outlines = NamedTuple[])
    CairoMakie.activate!()
    v = _ert3d_values(mesh, ρ, log10_values)
    cx, cy = _ert3d_core(mesh)
    zc = ert3d_z_centers(mesh) .- mesh.z_nodes[1]
    size_xy = max(mesh.x_nodes[last(cx)+1] - mesh.x_nodes[first(cx)], mesh.y_nodes[last(cy)+1] - mesh.y_nodes[first(cy)])
    depths = depths === nothing ? collect(range(zc[1], size_xy / 3, length = 6)) : collect(Float64, depths)
    range_ = resistivity_log10_range === nothing ? _ert2d_auto_range(v[cx, cy, 1:searchsortedfirst(zc, maximum(depths))]) :
             resistivity_log10_range
    ncol = min(3, length(depths))
    nrow = cld(length(depths), ncol)
    xe, ye = mesh.x_nodes[first(cx):last(cx)+1], mesh.y_nodes[first(cy):last(cy)+1]
    aspect = (ye[end] - ye[1]) / (xe[end] - xe[1])
    figure = Figure(size = (round(Int, 380 * ncol * clamp(aspect, 0.8, 2.0)) + 120, 380 * nrow))
    for (k, z) in enumerate(depths)
        iz = clamp(searchsortedlast(mesh.z_nodes .- mesh.z_nodes[1], z), 1, length(zc))
        r, c = fldmod1(k, ncol)
        ax = _ert_axis(figure[r, c]; xlabel = "East (m)", ylabel = "North (m)", aspect = DataAspect(),
                       title = @sprintf("%.1f m", zc[iz]), titlefont = :regular)
        heatmap!(ax, ye, xe, permutedims(v[cx, cy, iz]); colormap, colorrange = range_, nan_color = :white)
        if coverage !== nothing
            low = permutedims(Float64.(coverage[cx, cy, iz] .< coverage_cutoff))
            low[low .== 0] .= NaN
            heatmap!(ax, ye, xe, low; colormap = [RGBAf(1, 1, 1, 0.65), RGBAf(1, 1, 1, 0.65)], colorrange = (0, 1))
        end
        for b in outlines
            b.z_range[1] <= zc[iz] <= b.z_range[2] || continue
            lines!(ax, [b.y_range[1], b.y_range[2], b.y_range[2], b.y_range[1], b.y_range[1]],
                   [b.x_range[1], b.x_range[1], b.x_range[2], b.x_range[2], b.x_range[1]], color = :black, linewidth = 1.2)
        end
        scatter!(ax, mesh.electrode_y, mesh.electrode_x; color = :black, markersize = 3)
        xlims!(ax, ye[1], ye[end]); ylims!(ax, xe[1], xe[end])
    end
    Colorbar(figure[1:nrow, ncol+1]; colormap, limits = range_, label = colorbar_label, labelfont = :regular)
    _ert_save(output_path, figure)
end

"""
    plot_ert3d_section(mesh, ρ; output_path, along=:y, at=0.0, maximum_depth=nothing,
                       resistivity_log10_range=nothing, coverage=nothing, coverage_cutoff=1e-3,
                       outlines=[]) -> path

Vertical section of the core: `along = :y` (east-west) through x = `at`, or `along = :x`
(north-south) through y = `at`. Air is blank.
"""
function plot_ert3d_section(mesh::ERT3DMesh, ρ::AbstractArray{<:Real, 3}; output_path::AbstractString, along::Symbol = :y,
                            at::Real = 0.0, maximum_depth = nothing, resistivity_log10_range = nothing, coverage = nothing,
                            coverage_cutoff::Real = 1e-3, outlines = NamedTuple[])
    along in (:x, :y) || throw(ArgumentError("along must be :x or :y"))
    CairoMakie.activate!()
    v = _ert3d_values(mesh, ρ, false)
    cx, cy = _ert3d_core(mesh)
    zn = mesh.z_nodes .- mesh.z_nodes[1]
    if along == :y
        i = _ert3d_column(mesh.x_nodes, at)
        h, edges, cut, label = v[i, cy, :], mesh.y_nodes[first(cy):last(cy)+1], ert3d_x_centers(mesh)[i], "East (m)"
        cov = coverage === nothing ? nothing : coverage[i, cy, :]
    else
        j = _ert3d_column(mesh.y_nodes, at)
        h, edges, cut, label = v[cx, j, :], mesh.x_nodes[first(cx):last(cx)+1], ert3d_y_centers(mesh)[j], "North (m)"
        cov = coverage === nothing ? nothing : coverage[cx, j, :]
    end
    depth = maximum_depth === nothing ? (edges[end] - edges[1]) / 3 : Float64(maximum_depth)
    rows = 1:min(length(zn) - 1, searchsortedfirst(zn, depth))
    range_ = resistivity_log10_range === nothing ? _ert2d_auto_range(h[:, rows]) : resistivity_log10_range
    figure = Figure(size = (1200, clamp(round(Int, 1100 * depth / (edges[end] - edges[1])) + 150, 300, 800)))
    ax = _ert_axis(figure[1, 1]; xlabel = label, ylabel = "Depth (m)", yreversed = true,
                   title = @sprintf("%s = %.1f m", along == :y ? "North" : "East", cut), titlefont = :regular)
    heatmap!(ax, edges, zn[1:last(rows)+1], h[:, rows]; colormap = :Spectral, colorrange = range_, nan_color = :white)
    if cov !== nothing
        low = Float64.(cov[:, rows] .< coverage_cutoff)
        low[low .== 0] .= NaN
        heatmap!(ax, edges, zn[1:last(rows)+1], low; colormap = [RGBAf(1, 1, 1, 0.65), RGBAf(1, 1, 1, 0.65)], colorrange = (0, 1))
    end
    for b in outlines
        (along == :y ? b.x_range : b.y_range)[1] <= cut <= (along == :y ? b.x_range : b.y_range)[2] || continue
        r = along == :y ? b.y_range : b.x_range
        lines!(ax, [r[1], r[2], r[2], r[1], r[1]], [b.z_range[1], b.z_range[1], b.z_range[2], b.z_range[2], b.z_range[1]],
               color = :black, linewidth = 1.2)
    end
    Colorbar(figure[1, 2]; colormap = :Spectral, limits = range_, label = "log10 ρ (Ω·m)", labelfont = :regular)
    xlims!(ax, edges[1], edges[end]); ylims!(ax, zn[last(rows)+1], 0)
    _ert_save(output_path, figure)
end

"""
    plot_ert3d_mesh(mesh; output_path) -> path

Plan view of the mesh: cell edges, the core outlined, the electrodes, and the ground depth
below the model top when there is topography.
"""
function plot_ert3d_mesh(mesh::ERT3DMesh; output_path::AbstractString)
    CairoMakie.activate!()
    cx, cy = _ert3d_core(mesh)
    figure = Figure(size = (1100, 800))
    ax = _ert_axis(figure[1, 1]; xlabel = "East (m)", ylabel = "North (m)", aspect = DataAspect(),
                   title = @sprintf("%d × %d × %d cells (north × east × down)", _ert3d_size(mesh)...), titlefont = :regular)
    if !ert3d_is_flat(mesh)
        hm = heatmap!(ax, mesh.y_nodes, mesh.x_nodes, permutedims(mesh.surface); colormap = Reverse(:greys))
        Colorbar(figure[1, 2], hm; label = "ground depth (m)", labelfont = :regular)
    end
    seg = Point2f[]
    foreach(x -> push!(seg, Point2f(mesh.y_nodes[1], x), Point2f(mesh.y_nodes[end], x)), mesh.x_nodes)
    foreach(y -> push!(seg, Point2f(y, mesh.x_nodes[1]), Point2f(y, mesh.x_nodes[end])), mesh.y_nodes)
    linesegments!(ax, seg; color = (:black, 0.3), linewidth = 0.4)
    x0, x1, y0, y1 = mesh.x_nodes[first(cx)], mesh.x_nodes[last(cx)+1], mesh.y_nodes[first(cy)], mesh.y_nodes[last(cy)+1]
    lines!(ax, [y0, y1, y1, y0, y0], [x0, x0, x1, x1, x0]; color = :firebrick, linewidth = 2)
    scatter!(ax, mesh.electrode_y, mesh.electrode_x; color = :black, markersize = 4)
    _ert_save(output_path, figure)
end

"""
    plot_ert3d_data_fit(observed, predicted; output_path, error_floor=0.0) -> path

Observed against predicted ρa, and the normalised misfit against the pseudo-depth.
"""
function plot_ert3d_data_fit(observed::DataFile3D, predicted::DataFile3D; output_path::AbstractString, error_floor::Real = 0.0)
    CairoMakie.activate!()
    v = ert2d_valid_data(observed; error_floor)
    k = findall(i -> v.keep[i] && isfinite(predicted.rhoa[i]) && predicted.rhoa[i] > 0, 1:ndata(observed))
    o = observed.geometric_factor[k] .* observed.resistance[k]
    p = predicted.rhoa[k]
    e = log.(p ./ o) ./ v.sigma[k]
    figure = Figure(size = (1200, 520))
    a1 = _ert_axis(figure[1, 1]; xlabel = "observed ρa (Ω·m)", ylabel = "predicted ρa (Ω·m)", xscale = log10, yscale = log10,
                   aspect = 1)
    scatter!(a1, o, p; markersize = 4, color = (:black, 0.6))
    lo, hi = extrema(vcat(o, p))
    lines!(a1, [lo, hi], [lo, hi]; color = :firebrick)
    a2 = _ert_axis(figure[1, 2]; xlabel = "normalised misfit", ylabel = "pseudo-depth (m)", yreversed = true,
                   title = @sprintf("RMS %.2f over %d data", sqrt(mean(abs2, e)), length(e)), titlefont = :regular)
    scatter!(a2, e, ert3d_pseudo(observed).depth[k]; markersize = 4, color = (:black, 0.6))
    vlines!(a2, [-1, 1]; color = :gray, linestyle = :dash)
    _ert_save(output_path, figure)
end

"""
    PlotInversion3D(run; true_model_path=nothing, resistivity_log10_range=nothing, depths=nothing,
                    outlines=[]) -> paths

Standard plots of a run returned by the five-file `Invert3D`, written to
`run.run_dir/plots`: the mesh plan, depth slices and central sections of the final model
(faded where the coverage is low) and of the true model when given, the coverage, the
data fit and the convergence.
"""
function PlotInversion3D(run; true_model_path = nothing, resistivity_log10_range = nothing, depths = nothing,
                         outlines = NamedTuple[])
    dir = joinpath(run.run_dir, "plots")
    path(name) = joinpath(dir, name)
    cx, cy = _ert3d_core(run.mesh)
    range_ = resistivity_log10_range === nothing ? _ert2d_auto_range(_ert3d_values(run.mesh, run.final, false)[cx, cy, :]) :
             resistivity_log10_range
    common = (resistivity_log10_range = range_, outlines)
    # by default the main domain: slices and sections down to its depth (the padding below only follows it)
    md = haskey(run, :main_depth) && isfinite(run.main_depth) ? run.main_depth : nothing
    if depths === nothing && md !== nothing
        zc = ert3d_z_centers(run.mesh) .- run.mesh.z_nodes[1]
        depths = collect(range(zc[1], zc[findlast(<(md), zc)], length = 6))
    end
    sec = md === nothing ? (;) : (maximum_depth = md,)
    paths = String[
        plot_ert3d_mesh(run.mesh; output_path = path("Mesh.png")),
        plot_ert3d_slices(run.mesh, run.final; output_path = path("SlicesFinal.png"), depths, coverage = run.coverage, common...),
        plot_ert3d_section(run.mesh, run.final; output_path = path("SectionEW.png"), along = :y, coverage = run.coverage, common..., sec...),
        plot_ert3d_section(run.mesh, run.final; output_path = path("SectionNS.png"), along = :x, coverage = run.coverage, common..., sec...),
        plot_ert3d_slices(run.mesh, log10.(max.(ifelse.(isnan.(run.coverage), 1e-12, run.coverage), 1e-12));
                          output_path = path("Coverage.png"), depths, log10_values = true, colormap = :viridis,
                          colorbar_label = "log10 coverage", resistivity_log10_range = (-4.0, 0.0)),
        plot_ert3d_data_fit(run.observed, run.predicted; output_path = path("DataFit.png"), error_floor = run.ctrl.error_floor),
        plot_inv2d_convergence(run.history; output_path = path("Convergence.png"), target_rms = run.ctrl.target_rms),
    ]
    if true_model_path !== nothing
        tm, tρ = Mesh3DFromInputs(ReadModel3D(true_model_path), run.observed; warn = false)
        push!(paths, plot_ert3d_slices(tm, tρ; output_path = path("SlicesTrue.png"), depths, common...),
              plot_ert3d_section(tm, tρ; output_path = path("SectionEWTrue.png"), along = :y, common..., sec...))
    end
    paths
end
