# 2D ERT model plots
# Author: @pankajkmishra
# Resistivity sections with the electrodes on the ground, coverage shading, and the mesh layout, as
# PlotModel2D.jl of MTGeophysics.jl (metres instead of kilometres)

using CairoMakie
using Printf
using Statistics

#---------- style ----------

const _ERT_PX_PER_UNIT = 3

# no grid, no title, regular-weight labels
_ert_axis(pos; kwargs...) = Axis(pos; xgridvisible = false, ygridvisible = false,
                                 xlabelfont = :regular, ylabelfont = :regular, kwargs...)

_ert_save(path, figure) = (mkpath(dirname(abspath(path))); save(path, figure; px_per_unit = _ERT_PX_PER_UNIT); String(path))

# depth shown by default: a quarter of the electrode spread (of the core without electrodes), at least ten
# first-layer thicknesses, below the deepest electrode
function _ert2d_default_depth(mesh::ERT2DMesh)
    e = mesh.electrode_positions
    core = _core_range(mesh.y_cell_sizes)
    spread = isempty(e) ? mesh.y_nodes[last(core)+1] - mesh.y_nodes[first(core)] : maximum(e) - minimum(e)
    max(0.25spread, 10 * mesh.z_cell_sizes[1]) + (isempty(e) ? 0.0 : maximum(ert2d_electrode_depths(mesh)))
end

# bottom of the main domain (the cells that are not padding) under the core, m below the model top
function _ert2d_main_depth(mesh::ERT2DMesh, padding::AbstractMatrix{Bool})
    core = _core_range(mesh.y_cell_sizes)
    rows = findall(iz -> !all(padding[iz, core]), axes(padding, 1))
    isempty(rows) ? _ert2d_default_depth(mesh) : mesh.z_nodes[maximum(rows)+1] - mesh.z_nodes[1]
end

# 2-98 % range of log10 ρ over the cells of a window, widened to at least half a decade
function _ert2d_auto_range(values::AbstractArray)
    v = filter(isfinite, vec(values))
    isempty(v) && return (0.0, 4.0)
    lo, hi = quantile(v, 0.02), quantile(v, 0.98)
    hi - lo < 0.5 && ((lo, hi) = ((lo + hi) / 2 - 0.25, (lo + hi) / 2 + 0.25))
    (floor(lo * 10) / 10, ceil(hi * 10) / 10)
end

# cell rectangles of rows × columns, depth below the model top
function _ert2d_cell_polys(mesh::ERT2DMesh, rows, columns)
    yn, zn = mesh.y_nodes, mesh.z_nodes .- mesh.z_nodes[1]
    [Point2f[(yn[iy], zn[iz]), (yn[iy+1], zn[iz]), (yn[iy+1], zn[iz+1]), (yn[iy], zn[iz+1])] for iz in rows, iy in columns]
end

"""
    plot_ert2d_model(mesh, resistivity; output_path, show_grid=false, show_padding=false,
                     maximum_depth=nothing, resistivity_log10_range=nothing, coverage=nothing,
                     coverage_cutoff=1e-3, water=nothing, annotation=nothing, log10_values=false,
                     colormap=:Spectral, colorbar_label="log10 ρ (Ω·m)", vertical_exaggeration=1.0,
                     outlines=[]) -> path

Resistivity section in metres, depth below the model top, with the electrodes on the
ground. Air is blank and the staircase ground drawn as a line. `maximum_depth` (default a quarter of the electrode spread
below the deepest electrode) limits the depth, `show_padding = false` shows the core
only. `coverage` (`ert2d_coverage`) fades the cells below `coverage_cutoff`, those the
data barely see. `water` is a mask of water cells, shaded blue. `outlines` are
`(y_range, z_range)` boxes drawn as thin black rectangles, e.g. the true bodies of a
synthetic. With `log10_values` the matrix is plotted as given.
"""
function plot_ert2d_model(
    mesh::ERT2DMesh,
    resistivity::AbstractMatrix{<:Real};
    output_path::AbstractString,
    show_grid::Bool = false,
    show_padding::Bool = false,
    maximum_depth::Union{Nothing, Real} = nothing,
    resistivity_log10_range = nothing,
    coverage::Union{Nothing, AbstractMatrix} = nothing,
    coverage_cutoff::Real = 1e-3,
    water::Union{Nothing, AbstractMatrix{Bool}} = nothing,
    annotation::Union{Nothing, AbstractString} = nothing,
    log10_values::Bool = false,
    colormap = :Spectral,
    colorbar_label::AbstractString = "log10 ρ (Ω·m)",
    vertical_exaggeration::Real = 1.0,
    outlines = NamedTuple[],
)
    CairoMakie.activate!()
    zn = mesh.z_nodes .- mesh.z_nodes[1]
    columns = show_padding ? (1:size(resistivity, 2)) : _core_range(mesh.y_cell_sizes)
    y0, y1 = mesh.y_nodes[first(columns)], mesh.y_nodes[last(columns)+1]
    depth = maximum_depth === nothing ? (show_padding ? zn[end] : _ert2d_default_depth(mesh)) : Float64(maximum_depth)
    depth = min(depth, zn[end])
    rows = 1:min(size(resistivity, 1), searchsortedfirst(mesh.z_nodes .- mesh.z_nodes[1], depth) + 1)
    values = log10_values ? Float64.(resistivity) : log10.(Float64.(resistivity))
    values[ert2d_air_mask(mesh)] .= NaN
    shown = values[rows, columns]
    range = resistivity_log10_range === nothing ? _ert2d_auto_range(shown) : resistivity_log10_range

    width = 1200
    height = clamp(round(Int, 1.1 * width * vertical_exaggeration * depth / (y1 - y0)) + 150, 300, 900)
    figure = Figure(size = (width, height))
    axis = _ert_axis(figure[1, 1]; xlabel = "Distance along profile (m)", ylabel = "Depth (m)", yreversed = true)
    polys = _ert2d_cell_polys(mesh, rows, columns)
    ok = vec(isfinite.(shown))
    # each cell is stroked in its own colour, so anti-aliasing leaves no seams between cells
    cmap = Makie.cgrad(colormap)
    colors = [get(cmap, clamp((v - range[1]) / (range[2] - range[1]), 0, 1)) for v in vec(shown)[ok]]
    poly!(axis, vec(polys)[ok]; color = colors, strokecolor = show_grid ? fill(RGBAf(0, 0, 0, 0.25), length(colors)) : colors,
          strokewidth = show_grid ? 0.3 : 0.6)
    Colorbar(figure[1, 2]; colormap = cmap, limits = range, label = colorbar_label, labelfont = :regular)
    if coverage !== nothing
        faded = vec([isfinite(coverage[iz, iy]) && coverage[iz, iy] < coverage_cutoff for iz in rows, iy in columns])
        any(faded) && poly!(axis, vec(polys)[faded]; color = (:white, 0.65), strokewidth = 0)
    end
    if water !== nothing && any(water)
        wet = vec([water[iz, iy] for iz in rows, iy in columns])
        any(wet) && poly!(axis, vec(polys)[wet]; color = :lightskyblue, strokewidth = 0)
    end
    for b in outlines
        lines!(axis, [b.y_range[1], b.y_range[2], b.y_range[2], b.y_range[1], b.y_range[1]],
               [b.z_range[1], b.z_range[1], b.z_range[2], b.z_range[2], b.z_range[1]], color = :black, linewidth = 1.2)
    end
    _ert2d_ground_line!(axis, mesh, columns)
    isempty(mesh.electrode_positions) ||
        scatter!(axis, mesh.electrode_positions, ert2d_electrode_depths(mesh); marker = :dtriangle, markersize = 7,
                 color = :black)
    xlims!(axis, y0, y1)
    ylims!(axis, depth, min(0.0, -0.02depth))
    annotation === nothing || text!(axis, 0.01, 0.97; text = annotation, space = :relative, align = (:left, :top),
                                    fontsize = 14)
    _ert_save(output_path, figure)
end

# ground surface: the staircase when there are air cells; nothing when flat
function _ert2d_ground_line!(axis, mesh, columns = eachindex(mesh.y_cell_sizes))
    topo = ert2d_topo_air(mesh)
    any(>(0), topo) || return nothing
    depth = [mesh.z_nodes[1+topo[iy]] - mesh.z_nodes[1] for iy in columns]
    stairs!(axis, mesh.y_nodes[first(columns):last(columns)+1], vcat(depth, depth[end]); step = :post, color = :black,
            linewidth = 1.2)
    nothing
end

"""
    PlotModel2D(model_path; output_path, data_path=nothing, kwargs...) -> path

Plot a model file; with `data_path` its electrodes are drawn on the ground
(`Mesh2DFromInputs`). Other keywords go to
`plot_ert2d_model`.
"""
function PlotModel2D(model_path::AbstractString; output_path::AbstractString,
                     data_path::Union{Nothing, AbstractString} = nothing, kwargs...)
    model = ReadModel2D(model_path)
    data_path === nothing && return plot_ert2d_model(_ert2d_earth_mesh(model), model.resistivity; output_path, kwargs...)
    mesh, _ = Mesh2DFromInputs(model, load_data2d(data_path); warn = false)
    plot_ert2d_model(mesh, model.resistivity; output_path, kwargs...)
end

#---------- mesh ----------

"""
    plot_ert2d_mesh(mesh; output_path, region=:full, water=nothing, core_depth=nothing) -> path

Cell edges with the electrodes, the topography and
the main domain outlined: the even core laterally and `core_depth` (the main domain's
depth, `MakeMesh2D`; default a quarter of the electrode spread) vertically; `region = :core`
shows it only.
"""
function plot_ert2d_mesh(mesh::ERT2DMesh; output_path::AbstractString, region::Symbol = :full,
                         water::Union{Nothing, AbstractMatrix{Bool}} = nothing, core_depth::Union{Nothing, Real} = nothing)
    region in (:full, :core) || error("region must be :full or :core")
    CairoMakie.activate!()
    core_y = _core_range(mesh.y_cell_sizes)
    zc = core_depth === nothing ? _ert2d_default_depth(mesh) : Float64(core_depth)
    zn = mesh.z_nodes .- mesh.z_nodes[1]
    yn = mesh.y_nodes
    cols = region == :core ? (first(core_y):last(core_y)+1) : eachindex(yn)
    rows = region == :core ? (1:min(length(zn), searchsortedfirst(zn, zc) + 1)) : eachindex(zn)
    figure = Figure(size = (1200, region == :core ? 520 : 700))
    axis = _ert_axis(figure[1, 1]; xlabel = "Distance along profile (m)", ylabel = "Depth (m)", yreversed = true)
    air = ert2d_air_mask(mesh)
    any(air) && poly!(axis, [Rect2(yn[i[2]], zn[i[1]], yn[i[2]+1] - yn[i[2]], zn[i[1]+1] - zn[i[1]]) for i in findall(air)];
                      color = :aliceblue)
    if water !== nothing && any(water)
        poly!(axis, vec(_ert2d_cell_polys(mesh, 1:size(water, 1), 1:size(water, 2)))[vec(water)]; color = :lightskyblue)
    end
    seg = Point2f[]
    for iz in rows
        push!(seg, Point2f(yn[first(cols)], zn[iz]), Point2f(yn[last(cols)], zn[iz]))
    end
    for k in cols
        push!(seg, Point2f(yn[k], zn[first(rows)]), Point2f(yn[k], zn[last(rows)]))
    end
    linesegments!(axis, seg; color = (:black, 0.35), linewidth = 0.5)
    y_core = (yn[first(core_y)], yn[last(core_y)+1])
    lines!(axis, [y_core[1], y_core[2], y_core[2], y_core[1], y_core[1]], [0, 0, zc, zc, 0], color = :firebrick,
           linewidth = 2, label = "main domain")
    _ert2d_ground_line!(axis, mesh)
    scatter!(axis, mesh.electrode_positions, ert2d_electrode_depths(mesh); marker = :dtriangle, markersize = 7,
             color = :black, label = "electrodes")
    xlims!(axis, yn[first(cols)], yn[last(cols)])
    zmax = zn[last(rows)]
    ylims!(axis, zmax, min(0.0, -0.02zmax))
    Legend(figure[1, 2], axis, @sprintf("%d × %d cells", length(mesh.z_cell_sizes), length(mesh.y_cell_sizes)),
           framevisible = false, titlefont = :regular, labelfont = :regular)
    _ert_save(output_path, figure)
end
