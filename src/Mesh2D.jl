# 2D ERT mesh and model
# Author: @pankajkmishra
# Profile mesh geometry, layer builders, and ModEM-layout model file I/O, as Mesh2D.jl of MTGeophysics.jl
# Every mesh is a rectangular tensor grid. The air is an insulator, so the ground is a no-flux surface; under
# topography the cells above the ground are air, a staircase as in MTGeophysics.jl. The mesh keeps the ground line
# itself (depth below the model top at every y node), so the staircase can be cut at any resolution: the forward
# solve cuts it again on its finer mesh around the electrodes, where the response is sensitive to the ground's shape

using Printf
using Statistics

"""
    ERT2DMesh

Profile mesh and electrodes: y (along the profile) and z (down from the model top) nodes
and cell sizes, the electrode positions y in data order, and `surface`, the ground depth
below the model top at every y node (piecewise linear; empty = flat ground). Cells whose
centre lies above the ground are air (`ert2d_air_mask`); each electrode stands on a flat
tread of the staircase (`ert2d_topo_air`).
"""
Base.@kwdef struct ERT2DMesh
    y_nodes::Vector{Float64}
    z_nodes::Vector{Float64}
    y_cell_sizes::Vector{Float64}
    z_cell_sizes::Vector{Float64}
    electrode_positions::Vector{Float64}
    surface::Vector{Float64} = Float64[]
end

"""
    ModelFile2D

A 2D model file on disk: cell sizes, resistivity `(nz, ny)` in ohm metres with air
tagged 1e17, `origin`, the y of the left edge (the file layout centres the grid on
y = 0), and `surface`, the ground depth below the model top at the ny + 1 y nodes (empty =
flat, or taken from the air cells or the electrode depths; see `Mesh2DFromInputs`).
"""
Base.@kwdef struct ModelFile2D
    title::String = ""
    y_cell_sizes::Vector{Float64}
    z_cell_sizes::Vector{Float64}
    resistivity::Matrix{Float64}
    origin::Float64 = -sum(y_cell_sizes) / 2
    surface::Vector{Float64} = Float64[]
    format::String = "LOGE"
    path::String = ""
end

#---------- topography ----------
#
# as in MTGeophysics: model files hold air as 1e17 ohm m and any cell above 1e15 ohm m reads as air; the cov mask
# gives air 0 and water 9

const ERT2D_AIR_TAG = 1e17
const ERT2D_AIR_THRESHOLD = 1e15
const ERT2D_MASK_WATER = 9
const ERT2D_MASK_PADDING = 2

"""
    ert2d_surface(mesh) -> Vector{Float64}

Ground depth below the model top at each y node (zeros over flat ground).
"""
ert2d_surface(mesh::ERT2DMesh) = isempty(mesh.surface) ? zeros(length(mesh.y_nodes)) : mesh.surface

ert2d_is_flat(mesh::ERT2DMesh) = all(iszero, ert2d_surface(mesh))

"""
    ert2d_ground_depth(mesh, y) -> depth

Depth of the ground line below the model top at profile position y (linear between the
nodes, constant beyond the mesh), 0 over flat ground.
"""
ert2d_ground_depth(mesh::ERT2DMesh, y::Real) =
    ert2d_is_flat(mesh) ? 0.0 : _ert2d_interp(mesh.y_nodes, mesh.surface, y)

# median distance between distinct electrode positions, or the smallest cell without electrodes
_ert_spacing(positions, cells) =
    (u = sort(unique(round.(positions; digits = 6))); length(u) > 1 ? median(diff(u)) : minimum(cells))

"""
    ert2d_electrode_columns(mesh) -> Vector{Int}

Column of each electrode: the cell whose span holds it, an electrode on a node going to
the column on its right.
"""
ert2d_electrode_columns(mesh::ERT2DMesh) =
    [clamp(searchsortedlast(mesh.y_nodes, y), 1, length(mesh.y_cell_sizes)) for y in mesh.electrode_positions]

"""
    ert2d_topo_air(mesh) -> Vector{Int}

Air cells at the top of each column, the staircase of the ground line: a cell is air when
its centre lies above the ground. Every electrode stands on a short flat tread: the
columns within a tenth of the electrode spacing of it (at least the cells it touches) take
the row boundary nearest the electrode's ground (the shallowest where treads overlap), so
no electrode sits on a step corner. Zeros over flat ground.
"""
function ert2d_topo_air(mesh::ERT2DMesh)
    ny, nz = length(mesh.y_cell_sizes), length(mesh.z_cell_sizes)
    ert2d_is_flat(mesh) && return zeros(Int, ny)
    zn = mesh.z_nodes .- mesh.z_nodes[1]
    zc = (zn[1:end-1] .+ zn[2:end]) ./ 2
    yc = (mesh.y_nodes[1:end-1] .+ mesh.y_nodes[2:end]) ./ 2
    air = [count(<(ert2d_ground_depth(mesh, yc[iy])), zc) for iy in 1:ny]
    e = mesh.electrode_positions
    if !isempty(e)
        w = _ert_spacing(e, mesh.y_cell_sizes) / 10
        tread = fill(typemax(Int), ny)
        for y in e
            k = argmin(abs.(zn .- ert2d_ground_depth(mesh, y))) - 1
            for iy in 1:ny
                mesh.y_nodes[iy+1] >= y - w && mesh.y_nodes[iy] <= y + w || continue
                tread[iy] = min(tread[iy], k)
            end
        end
        air = [tread[iy] < typemax(Int) ? tread[iy] : air[iy] for iy in 1:ny]
    end
    clamp.(air, 0, nz - 1)
end

"""
    ert2d_air_mask(mesh) -> BitMatrix

Air cells of the mesh `(nz, ny)`, the staircase of `ert2d_topo_air`.
"""
function ert2d_air_mask(mesh::ERT2DMesh)
    air = falses(length(mesh.z_cell_sizes), length(mesh.y_cell_sizes))
    for (iy, k) in enumerate(ert2d_topo_air(mesh))
        air[1:k, iy] .= true
    end
    air
end

"""
    ert2d_electrode_depths(mesh) -> Vector{Float64}

Depth below the model top of each electrode on the mesh: the tread of its column.
"""
function ert2d_electrode_depths(mesh::ERT2DMesh)
    topo = ert2d_topo_air(mesh)
    [mesh.z_nodes[1+topo[iy]] - mesh.z_nodes[1] for iy in ert2d_electrode_columns(mesh)]
end

ert2d_y_centers(mesh::ERT2DMesh) = 0.5 .* (mesh.y_nodes[1:end-1] .+ mesh.y_nodes[2:end])
ert2d_z_centers(mesh::ERT2DMesh) = 0.5 .* (mesh.z_nodes[1:end-1] .+ mesh.z_nodes[2:end])

"""
    ert2d_electrode_offsets(mesh, data) -> Vector{NamedTuple}

How far each electrode of `data` lies from its tread on the mesh: `name`, `z` (data
depth below the model top), `surface` (the tread depth), `offset = z − surface` and
`tolerance`, half the thickness of the cell under the tread.
"""
function ert2d_electrode_offsets(mesh::ERT2DMesh, data)
    columns, depths, topo = ert2d_electrode_columns(mesh), ert2d_electrode_depths(mesh), ert2d_topo_air(mesh)
    map(eachindex(columns)) do i
        (name = data.electrode_names[i], z = data.z_positions[i], surface = depths[i],
         offset = data.z_positions[i] - depths[i], tolerance = mesh.z_cell_sizes[topo[columns[i]]+1] / 2)
    end
end

# ground depth at the y nodes from electrode depths Z: linear between the electrodes, constant beyond the line ends;
# empty when every Z is zero
function _ert2d_surface_from_electrodes(y_nodes, electrodes, z)
    all(iszero, z) && return Float64[]
    order = sortperm(electrodes)
    ye, ze = electrodes[order], z[order]
    uy = unique(round.(ye; digits = 6))
    uz = [mean(ze[abs.(ye .- u) .< 1e-6]) for u in uy]
    [_ert2d_interp(uy, uz, y) for y in y_nodes]
end

# ground depth at the y nodes of a staircase model (air-tagged cells, e.g. from MTGeophysics.jl): the bottom of the
# air of each column, linear between the column centres
function _ert2d_surface_from_air(y_nodes, z_cells, ρ)
    tagged = ρ .> ERT2D_AIR_THRESHOLD
    any(tagged) || return Float64[]
    zn = vcat(0.0, cumsum(z_cells))
    g = [zn[something(findfirst(!, tagged[:, iy]), size(ρ, 1))] for iy in axes(ρ, 2)]
    yc = (y_nodes[1:end-1] .+ y_nodes[2:end]) ./ 2
    [_ert2d_interp(yc, g, y) for y in y_nodes]
end

_ert2d_check_surface(s, y_nodes, z_nodes) =
    isempty(s) || (length(s) == length(y_nodes) && all(isfinite, s) && maximum(abs.(s)) < 0.5 * (z_nodes[end] - z_nodes[1])) ||
        throw(ArgumentError("surface needs one finite ground depth per y node, within half the mesh depth of the model top"))

#---------- layers ----------

"""
    ert2d_geometric_layers(spacing, depth; first_layer_div=4.0, vertical_factor=1.1) -> thicknesses

Layers from the surface down: the first is `spacing / first_layer_div`, each next one
`vertical_factor` thicker (1 = even layers), down to `depth`.
"""
function ert2d_geometric_layers(spacing::Real, depth::Real; first_layer_div::Real = 4.0, vertical_factor::Real = 1.1)
    spacing > 0 && depth > 0 && first_layer_div > 0 && vertical_factor >= 1 ||
        throw(ArgumentError("need spacing > 0, depth > 0, first_layer_div > 0, vertical_factor >= 1"))
    layers = [spacing / first_layer_div]
    while sum(layers) < depth - 1e-9
        push!(layers, layers[end] * vertical_factor)
    end
    layers
end

"""
    ert2d_padding(width, n, factor) -> thicknesses

`n` padding cells growing by `factor` from a cell of `width`.
"""
ert2d_padding(width::Real, n::Integer, factor::Real) = Float64(width) .* Float64(factor) .^ (1:n)

"""
    ert2d_padding_cells(width, reach, factor) -> n

Smallest number of padding cells growing by `factor` from a cell of `width` that reach
`reach` metres.
"""
function ert2d_padding_cells(width::Real, reach::Real, factor::Real)
    width > 0 && factor > 1 || throw(ArgumentError("need width > 0 and factor > 1"))
    n = 0
    while sum(ert2d_padding(width, n, factor); init = 0.0) < reach
        n += 1
    end
    n
end

"""
    BuildMesh2D(; electrodes, y_core_cell, core_margin_cells=4, n_pad=10, pad_factor=1.5,
                layers, surface=nothing) -> ERT2DMesh

Padded profile mesh centred on y = 0: a uniform core of `y_core_cell` over the
electrodes plus `core_margin_cells` on each side, `n_pad` cells growing by
`pad_factor` beside it, and `layers` (thicknesses, top first) below the model top.
`surface` (a function of y or one value per y node) is the ground depth below the model
top; the cells above it are air. `nothing` = flat.
"""
function BuildMesh2D(; electrodes::AbstractVector{<:Real}, y_core_cell::Real, core_margin_cells::Integer = 4,
                     n_pad::Integer = 10, pad_factor::Real = 1.5, layers::AbstractVector{<:Real}, surface = nothing)
    y_core_cell > 0 || throw(ArgumentError("y_core_cell must be positive"))
    half = ceil(Int, (maximum(abs.(electrodes)) + core_margin_cells * y_core_cell) / y_core_cell - 1e-9)
    dy = vcat(reverse(ert2d_padding(y_core_cell, n_pad, pad_factor)), fill(Float64(y_core_cell), 2half),
              ert2d_padding(y_core_cell, n_pad, pad_factor))
    y_nodes = vcat(0.0, cumsum(dy)) .- sum(dy) / 2
    z_nodes = vcat(0.0, cumsum(Float64.(layers)))
    s = surface === nothing ? Float64[] : surface isa Function ? Float64[surface(y) for y in y_nodes] : Float64.(surface)
    _ert2d_check_surface(s, y_nodes, z_nodes)
    ERT2DMesh(y_nodes = y_nodes, z_nodes = z_nodes, y_cell_sizes = dy, z_cell_sizes = Float64.(layers),
              electrode_positions = Float64.(electrodes), surface = all(iszero, s) ? Float64[] : s)
end

"""
    ert2d_block_model(mesh; background_resistivity=100.0, blocks=[]) -> Matrix

Resistivity `(nz, ny)` of a background with rectangular `blocks`
`(y_range, z_range, resistivity)` holding the cell centres, z the depth below the model
top; air is tagged 1e17. Blocks are applied in order, so later ones overwrite earlier ones.
"""
function ert2d_block_model(mesh::ERT2DMesh; background_resistivity::Real = 100.0, blocks = NamedTuple[])
    ρ = fill(Float64(background_resistivity), length(mesh.z_cell_sizes), length(mesh.y_cell_sizes))
    yc, zc = ert2d_y_centers(mesh), ert2d_z_centers(mesh) .- mesh.z_nodes[1]
    for b in blocks, iy in eachindex(yc), iz in eachindex(zc)
        b.y_range[1] <= yc[iy] <= b.y_range[2] && b.z_range[1] <= zc[iz] <= b.z_range[2] && (ρ[iz, iy] = b.resistivity)
    end
    ρ[ert2d_air_mask(mesh)] .= ERT2D_AIR_TAG
    ρ
end

# fixed-width scientific notation, per_line values per line
function _write_vector_lines(io, values::AbstractVector{<:Real}; per_line::Int = 12)
    for first_index in 1:per_line:length(values)
        last_index = min(first_index + per_line - 1, length(values))
        println(io, join([@sprintf("%.8e", Float64(values[idx])) for idx in first_index:last_index], " "))
    end
end

#---------- ModEM-layout model files ----------
#
# the MTGeophysics.jl 2D layout, after one '#' description line:
#   ny nz LOGE            (or LINEAR)
#   ny cell widths (m)
#   nz layer thicknesses (m), top to bottom
#   0
#   nz rows of ny values, top row first, ln(ρ) for LOGE, ρ for LINEAR; air 1e17
# the grid is centred on y = 0, the same origin as the data file's local y. Under topography the ground line goes to
# '# surface:' comment lines (ny + 1 depths, m below the model top), which ModEM readers skip

# ground depths of '# surface:' comment lines, empty when there are none
function _read_surface_comments(path, n)
    v = Float64[]
    for l in eachline(path)
        s = strip(l)
        startswith(s, "#") || continue
        m = match(r"^#\s*surface\s*:(.*)$"i, s)
        m === nothing || append!(v, parse.(Float64, split(m.captures[1])))
    end
    isempty(v) || length(v) == n || error("$path: '# surface:' holds $(length(v)) values, the grid has $n nodes")
    v
end

function _write_surface_comments(io, s)
    isempty(s) && return
    for part in Iterators.partition(s, 10)
        println(io, "# surface: ", join((@sprintf("%.6f", v) for v in part), ' '))
    end
end

"""
    ReadModel2D(path) -> ModelFile2D

Read a ModEM-layout 2D model (the MTGeophysics.jl layout), centred on y = 0, with its
ground line (`# surface:` lines) when present.
"""
function ReadModel2D(path::AbstractString)
    isfile(path) || error("model file not found: $path")
    lines = filter(l -> !isempty(strip(l)) && !startswith(strip(l), "#"), readlines(path))
    isempty(lines) && error("$path: empty model file")
    head = split(strip(lines[1]))
    length(head) >= 2 || error("$path: first line must be 'ny nz LOGE' or 'ny nz LINEAR'")
    ny, nz = parse(Int, head[1]), parse(Int, head[2])
    format = occursin("LOGE", uppercase(strip(lines[1]))) ? "LOGE" : "LINEAR"
    tokens = parse.(Float64, reduce(vcat, split.(strip.(lines[2:end]))))
    length(tokens) == ny + nz + 1 + ny * nz ||
        error("$path: expected $(ny + nz + 1 + ny * nz) numbers after the header, got $(length(tokens))")
    dy = tokens[1:ny]
    dz = tokens[ny+1:ny+nz]
    values = permutedims(reshape(tokens[ny+nz+2:end], ny, nz))
    ρ = format == "LOGE" ? exp.(values) : values
    all(>(0), dy) && all(>(0), dz) && all(x -> isfinite(x) && x > 0, ρ) ||
        error("$path: cell sizes and resistivities must be positive")
    ModelFile2D(y_cell_sizes = dy, z_cell_sizes = dz, resistivity = ρ, origin = -sum(dy) / 2,
                surface = _read_surface_comments(path, ny + 1), format = format, path = String(path))
end

"""
    WriteModel2D(path, y_cell_sizes, z_cell_sizes, ρ; loge=true, surface=[]) -> path
    WriteModel2D(path, model::ModelFile2D; loge=true) -> path
    WriteModel2D(path, mesh, ρ; loge=true) -> path

Write a ModEM-layout 2D model. With a mesh, its air is written as 1e17 and its ground line
goes to `# surface:` comment lines.
"""
function WriteModel2D(path::AbstractString, dy::AbstractVector{<:Real}, dz::AbstractVector{<:Real},
                      ρ::AbstractMatrix{<:Real}; loge::Bool = true, surface::AbstractVector{<:Real} = Float64[])
    size(ρ) == (length(dz), length(dy)) || throw(DimensionMismatch("model must be (nz, ny) = ($(length(dz)), $(length(dy)))"))
    all(x -> isfinite(x) && x > 0, ρ) || throw(ArgumentError("resistivity must be positive and finite"))
    isempty(surface) || length(surface) == length(dy) + 1 || throw(DimensionMismatch("surface needs ny + 1 values"))
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "# 2D ERT model written by ERTGeophysics.jl in ModEM format")
        _write_surface_comments(io, surface)
        @printf(io, "%5d%5d %s\n", length(dy), length(dz), loge ? "LOGE" : "LINEAR")
        _write_model_rows(io, "%12.3f", dy)
        _write_model_rows(io, "%12.3f", dz)
        println(io, "0")
        for iz in axes(ρ, 1)
            println(io)
            _write_model_rows(io, "%13.5E", loge ? log.(ρ[iz, :]) : ρ[iz, :])
        end
    end
    String(path)
end

WriteModel2D(path::AbstractString, m::ModelFile2D; loge::Bool = true) =
    WriteModel2D(path, m.y_cell_sizes, m.z_cell_sizes, m.resistivity; loge, surface = m.surface)

# ten fixed-width values per line
function _write_model_rows(io, fmt::String, values)
    f = Printf.Format(fmt)
    for row in Iterators.partition(values, 10)
        foreach(v -> Printf.format(io, f, v), row)
        println(io)
    end
end

function WriteModel2D(path::AbstractString, mesh::ERT2DMesh, ρ::AbstractMatrix{<:Real}; loge::Bool = true)
    abs(mesh.y_nodes[1] + mesh.y_nodes[end]) <= 1e-6 * (mesh.y_nodes[end] - mesh.y_nodes[1]) ||
        throw(ArgumentError("mesh must be centred on y = 0, it spans $(mesh.y_nodes[1]) to $(mesh.y_nodes[end]) m"))
    earth = Matrix{Float64}(ρ)
    earth[ert2d_air_mask(mesh)] .= ERT2D_AIR_TAG
    WriteModel2D(path, mesh.y_cell_sizes, mesh.z_cell_sizes, earth; loge, surface = mesh.surface)
end

#---------- mesh from model and data ----------

"""
    Mesh2DFromInputs(model::ModelFile2D, data::DataFile2D; warn=true) -> (mesh, ρ)

Mesh and resistivity for a ModEM-layout model and the electrodes of `data`. The model
is centred on the data's local y = 0. The ground line is the model's `surface` (written by
`MakeMesh2D`); without one, the bottom of the model's air cells (a staircase model, e.g.
from MTGeophysics.jl), or the data's electrode depths Z (linear between the electrodes),
or flat. The air of the result is the staircase of that ground on this mesh, tagged 1e17.
A data depth Z more than half a cell away from its tread is warned about.
"""
function Mesh2DFromInputs(model::ModelFile2D, data; warn::Bool = true)
    y_nodes = model.origin .+ vcat(0.0, cumsum(model.y_cell_sizes))
    all(y -> y_nodes[1] < y < y_nodes[end], data.electrodes) ||
        throw(ArgumentError("electrodes lie outside the model, which spans $(y_nodes[1]) to $(y_nodes[end]) m"))
    z_nodes = vcat(0.0, cumsum(model.z_cell_sizes))
    s = !isempty(model.surface) ? model.surface :
        (a = _ert2d_surface_from_air(y_nodes, model.z_cell_sizes, model.resistivity); isempty(a) ?
         _ert2d_surface_from_electrodes(y_nodes, data.electrodes, data.z_positions) : a)
    _ert2d_check_surface(s, y_nodes, z_nodes)
    mesh = ERT2DMesh(y_nodes = y_nodes, z_nodes = z_nodes,
                     y_cell_sizes = Float64.(model.y_cell_sizes), z_cell_sizes = Float64.(model.z_cell_sizes),
                     electrode_positions = Float64.(data.electrodes), surface = all(iszero, s) ? Float64[] : Float64.(s))
    ρ = copy(model.resistivity)
    air = ert2d_air_mask(mesh)
    ρ[air] .= ERT2D_AIR_TAG
    # earth cells that the file had as air (another staircase of the same ground) take the cell below
    for iy in axes(ρ, 2), iz in size(ρ, 1)-1:-1:1
        !air[iz, iy] && ρ[iz, iy] > ERT2D_AIR_THRESHOLD && (ρ[iz, iy] = ρ[iz+1, iy])
    end
    if warn
        off = filter(o -> abs(o.offset) > o.tolerance, ert2d_electrode_offsets(mesh, data))
        isempty(off) || @warn "$(length(off)) electrode depth(s) Z differ from their tread on the mesh by more than half a cell" *
            join([@sprintf("\n  %s: Z %.2f m, tread %.2f m", o.name, o.z, o.surface) for o in off[1:min(end, 10)]])
    end
    mesh, ρ
end

# mesh of a model file on its own grid, for plotting
_ert2d_earth_mesh(model::ModelFile2D; electrodes = Float64[]) =
    ERT2DMesh(y_nodes = model.origin .+ vcat(0.0, cumsum(model.y_cell_sizes)), z_nodes = vcat(0.0, cumsum(model.z_cell_sizes)),
              y_cell_sizes = model.y_cell_sizes, z_cell_sizes = model.z_cell_sizes,
              electrode_positions = Float64.(electrodes), surface = model.surface)

"""
    RefineMesh2D(mesh, ρ, factor) -> (mesh, ρ)

Split every cell into `factor × factor` cells; the ground line, electrodes and
resistivity carry over (the finer mesh cuts a finer staircase).
"""
function RefineMesh2D(mesh::ERT2DMesh, ρ::AbstractMatrix{<:Real}, factor::Integer)
    factor >= 1 || throw(ArgumentError("refinement factor must be at least 1"))
    factor == 1 && return mesh, Matrix{Float64}(ρ)
    fine, fρ, _ = _ert2d_split(mesh, ρ, fill(factor, length(mesh.z_cell_sizes)), fill(factor, length(mesh.y_cell_sizes)))
    fine, fρ
end

#---------- forward mesh: refined around the sources ----------
#
# the forward solves on a finer copy of the model mesh: every model cell is split so that the forward cells are
# a / refine wide at the electrodes (a = electrode spacing) and grow by `growth` × the distance from the nearest
# electrode (columns) or from the ground (rows), so the sources and the ground under them are resolved while padding and
# deep cells stay whole. The finer mesh cuts its own staircase of the ground line, with the electrodes on short treads

# split count of each interval of `nodes`: its size over the target size at its distance
_ert_splits(nodes, dist, h0, growth) =
    [max(1, ceil(Int, (nodes[i+1] - nodes[i]) / (h0 + growth * dist(nodes[i], nodes[i+1])) - 1e-9)) for i in 1:length(nodes)-1]

# distance of the interval [u, v] from the nearest of the sorted positions e
function _ert_interval_distance(e, u, v)
    j = searchsortedfirst(e, u)
    j <= length(e) && e[j] <= v && return 0.0
    min(j <= length(e) ? e[j] - v : Inf, j > 1 ? u - e[j-1] : Inf)
end

# distance of the interval [u, v] from the band [lo, hi]
_ert_band_distance(lo, hi, u, v) = v < lo ? lo - v : u > hi ? u - hi : 0.0

# fine nodes and the fine index range of every coarse interval
function _ert_split_axis(nodes, splits)
    fine = [nodes[1]]
    ranges = UnitRange{Int}[]
    for i in eachindex(splits)
        append!(fine, nodes[i] .+ (nodes[i+1] - nodes[i]) .* (1:splits[i]) ./ splits[i])
        fine[end] = nodes[i+1]
        push!(ranges, (length(fine) - splits[i]):(length(fine) - 1))
    end
    fine, ranges
end

# model mesh split into sz × sy cells; the fine resistivity takes each cell's model cell (an earth cell that lies in a
# model air cell takes the first earth cell below it); `owner` holds that model cell, 0 for air
function _ert2d_split(mesh::ERT2DMesh, ρ, sz, sy)
    yn, cols = _ert_split_axis(mesh.y_nodes, sy)
    zn, rows = _ert_split_axis(mesh.z_nodes, sz)
    s = ert2d_is_flat(mesh) ? Float64[] : [_ert2d_interp(mesh.y_nodes, mesh.surface, y) for y in yn]
    fine = ERT2DMesh(y_nodes = yn, z_nodes = zn, y_cell_sizes = diff(yn), z_cell_sizes = diff(zn),
                     electrode_positions = copy(mesh.electrode_positions), surface = s)
    rowof = reduce(vcat, [fill(i, length(r)) for (i, r) in enumerate(rows)])
    colof = reduce(vcat, [fill(i, length(c)) for (i, c) in enumerate(cols)])
    coarse_air = ert2d_air_mask(mesh)
    fine_air = ert2d_air_mask(fine)
    first_earth = [something(findfirst(!, coarse_air[:, iy]), size(ρ, 1)) for iy in axes(ρ, 2)]
    lin = LinearIndices((size(ρ, 1), size(ρ, 2)))
    owner = zeros(Int, length(zn) - 1, length(yn) - 1)
    fρ = ones(size(owner))
    for j in axes(owner, 2), i in axes(owner, 1)
        fine_air[i, j] && continue
        I, J = rowof[i], colof[j]
        coarse_air[I, J] && (I = first_earth[J])
        owner[i, j] = lin[I, J]
        fρ[i, j] = ρ[I, J]
    end
    fine, fρ, owner
end

"""
    ert2d_forward_mesh(mesh, ρ; refine=8, growth=0.25) -> (mesh, ρ, owner)

The forward mesh: model cells split so that the forward cells are a/`refine` at the
electrodes (a = median electrode spacing), growing by `growth` × the distance from the
nearest electrode and from the ground (the band of depths the ground spans under the
electrodes); it cuts its own staircase of the ground line. `owner[i, j]` is the model cell
(linear index) whose resistivity forward cell (i, j) carries, 0 for air.
"""
function ert2d_forward_mesh(mesh::ERT2DMesh, ρ::AbstractMatrix{<:Real}; refine::Integer = 8, growth::Real = 0.25)
    refine >= 1 && growth >= 0 || throw(ArgumentError("need refine ≥ 1 and growth ≥ 0"))
    e = sort(mesh.electrode_positions)
    h0 = _ert_spacing(e, mesh.y_cell_sizes) / refine
    sy = isempty(e) ? ones(Int, length(mesh.y_cell_sizes)) :
         _ert_splits(mesh.y_nodes, (u, v) -> _ert_interval_distance(e, u, v), h0, growth)
    g = isempty(e) ? [0.0] : [ert2d_ground_depth(mesh, y) for y in range(e[1], e[end], length = 4length(e))]
    z0 = mesh.z_nodes[1]
    sz = _ert_splits(mesh.z_nodes, (u, v) -> _ert_band_distance(minimum(g), maximum(g), u - z0, v - z0), h0, growth)
    _ert2d_split(mesh, ρ, sz, sy)
end
