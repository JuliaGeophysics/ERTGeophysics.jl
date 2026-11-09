# 3D ERT mesh and model
# Author: @pankajkmishra
# Rectangular tensor mesh (x north, y east, z down) and ModEM 3D model files, as the 3D models of MTGeophysics.jl
# The air is an insulator, so the ground is a no-flux surface; under topography the cells above the ground are air, a
# staircase as in 2D, cut from the ground surface the mesh keeps (depth below the model top at every (x, y) node), so
# the forward solve can cut it again on its finer mesh around the electrodes

using Printf

"""
    ERT3DMesh

3D mesh: x (north), y (east) and reference z (down from the model top) nodes and cell
sizes, the electrode positions `(x, y)` in data order, and `surface`, the ground depth
below the model top at every `(x, y)` node (`(nx + 1, ny + 1)`, bilinear; empty = flat
ground). Cells whose centre lies above the ground are air (`ert3d_air_mask`); each
electrode stands on a short flat tread (`ert3d_topo_air`).
"""
Base.@kwdef struct ERT3DMesh
    x_nodes::Vector{Float64}
    y_nodes::Vector{Float64}
    z_nodes::Vector{Float64}
    x_cell_sizes::Vector{Float64}
    y_cell_sizes::Vector{Float64}
    z_cell_sizes::Vector{Float64}
    electrode_x::Vector{Float64}
    electrode_y::Vector{Float64}
    surface::Matrix{Float64} = zeros(0, 0)
end

_ert3d_size(mesh::ERT3DMesh) = (length(mesh.x_cell_sizes), length(mesh.y_cell_sizes), length(mesh.z_cell_sizes))

"""
    ModelFile3D

A ModEM 3D model file: cell sizes, resistivity `(nx, ny, nz)` in ohm metres, `origin`
(x, y, z of the model's south-west top corner in the data frame, m), `rotation`
(degrees) and `surface`, the ground depth below the model top at the `(nx + 1, ny + 1)`
nodes (empty = flat, or taken from the air cells or the electrode depths; see
`Mesh3DFromInputs`). Air is tagged 1e17. `main_depth` (m, NaN = none) is the depth of the
main domain, the even grid that is inverted; the padding outside it follows it.
"""
Base.@kwdef struct ModelFile3D
    title::String = ""
    x_cell_sizes::Vector{Float64}
    y_cell_sizes::Vector{Float64}
    z_cell_sizes::Vector{Float64}
    resistivity::Array{Float64, 3}
    origin::Vector{Float64} = [-sum(x_cell_sizes) / 2, -sum(y_cell_sizes) / 2, 0.0]
    rotation::Float64 = 0.0
    surface::Matrix{Float64} = zeros(0, 0)
    main_depth::Float64 = NaN
    format::String = "LOGE"
    path::String = ""
end

#---------- topography ----------

"""
    ert3d_surface(mesh) -> Matrix (nx + 1, ny + 1)

Ground depth below the model top at each (x, y) node (zeros over flat ground).
"""
ert3d_surface(mesh::ERT3DMesh) =
    isempty(mesh.surface) ? zeros(length(mesh.x_nodes), length(mesh.y_nodes)) : mesh.surface

ert3d_is_flat(mesh::ERT3DMesh) = all(iszero, ert3d_surface(mesh))

# bilinear value of node values v (nx + 1, ny + 1) at (x, y), clamped to the mesh
function _ert3d_bilinear(xn, yn, v, x, y)
    i = clamp(searchsortedlast(xn, x), 1, length(xn) - 1)
    j = clamp(searchsortedlast(yn, y), 1, length(yn) - 1)
    tx = clamp((x - xn[i]) / (xn[i+1] - xn[i]), 0.0, 1.0)
    ty = clamp((y - yn[j]) / (yn[j+1] - yn[j]), 0.0, 1.0)
    (1 - tx) * (1 - ty) * v[i, j] + tx * (1 - ty) * v[i+1, j] + (1 - tx) * ty * v[i, j+1] + tx * ty * v[i+1, j+1]
end

"""
    ert3d_ground_depth(mesh, x, y) -> depth

Ground depth below the model top at (x, y), bilinear between the nodes.
"""
ert3d_ground_depth(mesh::ERT3DMesh, x::Real, y::Real) =
    ert3d_is_flat(mesh) ? 0.0 : _ert3d_bilinear(mesh.x_nodes, mesh.y_nodes, mesh.surface, x, y)

# column holding a position, a node going to the cell after it
_ert3d_column(nodes, v) = clamp(searchsortedlast(nodes, v), 1, length(nodes) - 1)

"""
    ert3d_electrode_columns(mesh) -> Vector{CartesianIndex{2}}

Column `(ix, iy)` of each electrode: the cell whose footprint holds it, an electrode on a
node going to the cell north and east of it.
"""
ert3d_electrode_columns(mesh::ERT3DMesh) =
    [CartesianIndex(_ert3d_column(mesh.x_nodes, mesh.electrode_x[e]), _ert3d_column(mesh.y_nodes, mesh.electrode_y[e]))
     for e in eachindex(mesh.electrode_x)]

"""
    ert3d_topo_air(mesh) -> Matrix{Int}

Air cells at the top of each `(ix, iy)` column, the staircase of the ground surface: a
cell is air when its centre lies above the ground. Every electrode stands on a short flat
tread: the columns within a tenth of the electrode spacing of it (at least the cells it
touches) take the layer boundary nearest the electrode's ground (the shallowest where
treads overlap). Zeros over flat ground.
"""
function ert3d_topo_air(mesh::ERT3DMesh)
    nx, ny, nz = _ert3d_size(mesh)
    ert3d_is_flat(mesh) && return zeros(Int, nx, ny)
    zn = mesh.z_nodes .- mesh.z_nodes[1]
    zc = (zn[1:end-1] .+ zn[2:end]) ./ 2
    xc, yc = ert3d_x_centers(mesh), ert3d_y_centers(mesh)
    air = [count(<(ert3d_ground_depth(mesh, xc[i], yc[j])), zc) for i in 1:nx, j in 1:ny]
    ne = length(mesh.electrode_x)
    if ne > 0
        w = _ert3d_line_spacing(mesh, minimum(mesh.x_cell_sizes)) / 10
        tread = fill(typemax(Int), nx, ny)
        for e in 1:ne
            x, y = mesh.electrode_x[e], mesh.electrode_y[e]
            k = argmin(abs.(zn .- ert3d_ground_depth(mesh, x, y))) - 1
            for j in 1:ny, i in 1:nx
                mesh.x_nodes[i+1] >= x - w && mesh.x_nodes[i] <= x + w && mesh.y_nodes[j+1] >= y - w && mesh.y_nodes[j] <= y + w ||
                    continue
                tread[i, j] = min(tread[i, j], k)
            end
        end
        air = [tread[c] < typemax(Int) ? tread[c] : air[c] for c in CartesianIndices(air)]
    end
    clamp.(air, 0, nz - 1)
end

"""
    ert3d_air_mask(mesh) -> BitArray{3}

Air cells of the mesh `(nx, ny, nz)`, the staircase of `ert3d_topo_air`.
"""
function ert3d_air_mask(mesh::ERT3DMesh)
    air = falses(_ert3d_size(mesh))
    topo = ert3d_topo_air(mesh)
    for ix in axes(topo, 1), iy in axes(topo, 2)
        air[ix, iy, 1:topo[ix, iy]] .= true
    end
    air
end

"""
    ert3d_electrode_depths(mesh) -> Vector{Float64}

Depth below the model top of each electrode on the mesh: the tread of its column.
"""
function ert3d_electrode_depths(mesh::ERT3DMesh)
    topo = ert3d_topo_air(mesh)
    [mesh.z_nodes[1+topo[c]] - mesh.z_nodes[1] for c in ert3d_electrode_columns(mesh)]
end

ert3d_x_centers(mesh::ERT3DMesh) = 0.5 .* (mesh.x_nodes[1:end-1] .+ mesh.x_nodes[2:end])
ert3d_y_centers(mesh::ERT3DMesh) = 0.5 .* (mesh.y_nodes[1:end-1] .+ mesh.y_nodes[2:end])
ert3d_z_centers(mesh::ERT3DMesh) = 0.5 .* (mesh.z_nodes[1:end-1] .+ mesh.z_nodes[2:end])

"""
    ert3d_electrode_offsets(mesh, data) -> Vector{NamedTuple}

How far each electrode of `data` lies from its tread on the mesh: `name`, `z` (data
depth below the model top), `surface` (the tread depth), `offset = z − surface` and
`tolerance`, half the thickness of the cell under the tread.
"""
function ert3d_electrode_offsets(mesh::ERT3DMesh, data)
    columns, depths, topo = ert3d_electrode_columns(mesh), ert3d_electrode_depths(mesh), ert3d_topo_air(mesh)
    map(eachindex(depths)) do i
        (name = data.electrode_names[i], z = data.z_positions[i], surface = depths[i],
         offset = data.z_positions[i] - depths[i], tolerance = mesh.z_cell_sizes[topo[columns[i]]+1] / 2)
    end
end

_ert3d_check_surface(s, xn, yn, zn) =
    isempty(s) || (size(s) == (length(xn), length(yn)) && all(isfinite, s) && maximum(abs.(s)) < 0.5 * (zn[end] - zn[1])) ||
        throw(ArgumentError("surface needs one finite ground depth per (x, y) node, within half the mesh depth of the model top"))

# ground depth at the (x, y) nodes of a staircase model (air-tagged cells): the bottom of the air of each column,
# bilinear between the column centres (nearest at the edges)
function _ert3d_surface_from_air(xn, yn, z_cells, ρ)
    tagged = ρ .> ERT2D_AIR_THRESHOLD
    any(tagged) || return zeros(0, 0)
    zn = vcat(0.0, cumsum(z_cells))
    g = [zn[something(findfirst(!, tagged[i, j, :]), size(ρ, 3))] for i in axes(ρ, 1), j in axes(ρ, 2)]
    xc, yc = (xn[1:end-1] .+ xn[2:end]) ./ 2, (yn[1:end-1] .+ yn[2:end]) ./ 2
    [_ert3d_bilinear(xc, yc, g, x, y) for x in xn, y in yn]
end

#---------- builders ----------

# uniform core of `cell` over ±(extent + margin cells) around 0, padded both sides
function _ert3d_axis(extent::Real, cell::Real, margin::Integer, n_pad::Integer, pad_factor::Real)
    half = ceil(Int, (extent + margin * cell) / cell - 1e-9)
    d = vcat(reverse(ert2d_padding(cell, n_pad, pad_factor)), fill(Float64(cell), 2half), ert2d_padding(cell, n_pad, pad_factor))
    d, vcat(0.0, cumsum(d)) .- sum(d) / 2
end

"""
    BuildMesh3D(; electrode_x, electrode_y, x_core_cell, y_core_cell=x_core_cell,
                core_margin_cells=4, n_pad=8, pad_factor=1.5, layers, surface=nothing) -> ERT3DMesh

Padded rectangular mesh centred on (0, 0): uniform core cells over the electrodes plus
`core_margin_cells` on each side, `n_pad` cells growing by `pad_factor` beyond, and
`layers` (thicknesses, top first) below the model top. `surface` (a function of (x, y) or
one value per node, `(nx + 1, ny + 1)`) is the ground depth below the model top; the cells
above it are air.
"""
function BuildMesh3D(; electrode_x::AbstractVector{<:Real}, electrode_y::AbstractVector{<:Real}, x_core_cell::Real,
                     y_core_cell::Real = x_core_cell, core_margin_cells::Integer = 4, n_pad::Integer = 8,
                     pad_factor::Real = 1.5, layers::AbstractVector{<:Real}, surface = nothing)
    x_core_cell > 0 && y_core_cell > 0 || throw(ArgumentError("core cells must be positive"))
    dx, xn = _ert3d_axis(maximum(abs.(electrode_x)), x_core_cell, core_margin_cells, n_pad, pad_factor)
    dy, yn = _ert3d_axis(maximum(abs.(electrode_y)), y_core_cell, core_margin_cells, n_pad, pad_factor)
    zn = vcat(0.0, cumsum(Float64.(layers)))
    sf = surface === nothing ? zeros(0, 0) : surface isa Function ? Float64[surface(x, y) for x in xn, y in yn] : Matrix{Float64}(surface)
    _ert3d_check_surface(sf, xn, yn, zn)
    ERT3DMesh(x_nodes = xn, y_nodes = yn, z_nodes = zn, x_cell_sizes = dx, y_cell_sizes = dy, z_cell_sizes = Float64.(layers),
              electrode_x = Float64.(electrode_x), electrode_y = Float64.(electrode_y),
              surface = all(iszero, sf) ? zeros(0, 0) : sf)
end

"""
    ert3d_block_model(mesh; background_resistivity=100.0, blocks=[]) -> Array{Float64,3}

Resistivity `(nx, ny, nz)` of a background with rectangular `blocks`
`(x_range, y_range, z_range, resistivity)` holding the cell centres, z the depth below
the model top; air is tagged 1e17. Later blocks overwrite earlier ones.
"""
function ert3d_block_model(mesh::ERT3DMesh; background_resistivity::Real = 100.0, blocks = NamedTuple[])
    ρ = fill(Float64(background_resistivity), _ert3d_size(mesh))
    xc, yc, zc = ert3d_x_centers(mesh), ert3d_y_centers(mesh), ert3d_z_centers(mesh) .- mesh.z_nodes[1]
    for b in blocks, ix in eachindex(xc), iy in eachindex(yc), iz in eachindex(zc)
        b.x_range[1] <= xc[ix] <= b.x_range[2] && b.y_range[1] <= yc[iy] <= b.y_range[2] &&
            b.z_range[1] <= zc[iz] <= b.z_range[2] && (ρ[ix, iy, iz] = b.resistivity)
    end
    ρ[ert3d_air_mask(mesh)] .= ERT2D_AIR_TAG
    ρ
end

"""
    RefineMesh3D(mesh, ρ, factor) -> (mesh, ρ)

Split every cell horizontally into `factor × factor` cells (the layers are kept); the
ground surface, electrodes and resistivity carry over (the finer mesh cuts a finer staircase).
"""
function RefineMesh3D(mesh::ERT3DMesh, ρ::AbstractArray{<:Real, 3}, factor::Integer)
    factor >= 1 || throw(ArgumentError("refinement factor must be at least 1"))
    factor == 1 && return mesh, Array{Float64, 3}(ρ)
    nx, ny, nz = _ert3d_size(mesh)
    fine, fρ, _ = _ert3d_split(mesh, ρ, fill(factor, nx), fill(factor, ny), ones(Int, nz))
    fine, fρ
end

# split every x, y and z interval into sx, sy, sz cells: the fine mesh (with its own staircase), the resistivity (an
# earth cell in a model air cell takes the first earth cell below it) and `owner`, the model cell of each, 0 for air
function _ert3d_split(mesh::ERT3DMesh, ρ, sx, sy, sz)
    xn, xr = _ert_split_axis(mesh.x_nodes, sx)
    yn, yr = _ert_split_axis(mesh.y_nodes, sy)
    zn, zr = _ert_split_axis(mesh.z_nodes, sz)
    sf = ert3d_is_flat(mesh) ? zeros(0, 0) : [_ert3d_bilinear(mesh.x_nodes, mesh.y_nodes, mesh.surface, x, y) for x in xn, y in yn]
    fine = ERT3DMesh(x_nodes = xn, y_nodes = yn, z_nodes = zn, x_cell_sizes = diff(xn), y_cell_sizes = diff(yn),
                     z_cell_sizes = diff(zn), electrode_x = copy(mesh.electrode_x), electrode_y = copy(mesh.electrode_y),
                     surface = sf)
    of(r) = reduce(vcat, [fill(i, length(q)) for (i, q) in enumerate(r)])
    xo, yo, zo = of(xr), of(yr), of(zr)
    coarse_air, fine_air = ert3d_air_mask(mesh), ert3d_air_mask(fine)
    first_earth = [something(findfirst(!, coarse_air[i, j, :]), size(ρ, 3)) for i in axes(ρ, 1), j in axes(ρ, 2)]
    lin = LinearIndices(size(ρ))
    owner = zeros(Int, length(xn) - 1, length(yn) - 1, length(zn) - 1)
    fρ = ones(size(owner))
    for c in CartesianIndices(owner)
        fine_air[c] && continue
        I, J, K = xo[c[1]], yo[c[2]], zo[c[3]]
        coarse_air[I, J, K] && (K = first_earth[I, J])
        owner[c] = lin[I, J, K]
        fρ[c] = ρ[I, J, K]
    end
    fine, fρ, owner
end

"""
    ert3d_forward_mesh(mesh, ρ; refine=4, growth=0.5) -> (mesh, ρ, owner)

The 3D forward mesh: model cells split so that the forward cells are a/`refine` at the
electrodes (a = electrode spacing along the lines), growing by `growth` × the distance
from the nearest electrode (along x and along y) and from the ground (the band of depths
it spans under the electrodes); it cuts its own staircase. `owner[c]` is the model cell
(linear index) whose resistivity forward cell c carries, 0 for air. As
`ert2d_forward_mesh`, coarser by default for the cost of 3D.
"""
function ert3d_forward_mesh(mesh::ERT3DMesh, ρ::AbstractArray{<:Real, 3}; refine::Integer = 4, growth::Real = 0.5)
    refine >= 1 && growth >= 0 || throw(ArgumentError("need refine ≥ 1 and growth ≥ 0"))
    ex, ey = sort(unique(mesh.electrode_x)), sort(unique(mesh.electrode_y))
    h0 = _ert3d_line_spacing(mesh, minimum(mesh.x_cell_sizes)) / refine
    sx = _ert_splits(mesh.x_nodes, (u, v) -> _ert_interval_distance(ex, u, v), h0, growth)
    sy = _ert_splits(mesh.y_nodes, (u, v) -> _ert_interval_distance(ey, u, v), h0, growth)
    g = [ert3d_ground_depth(mesh, mesh.electrode_x[e], mesh.electrode_y[e]) for e in eachindex(mesh.electrode_x)]
    isempty(g) && (g = [0.0])
    z0 = mesh.z_nodes[1]
    sz = _ert_splits(mesh.z_nodes, (u, v) -> _ert_band_distance(minimum(g), maximum(g), u - z0, v - z0), h0, growth)
    _ert3d_split(mesh, ρ, sx, sy, sz)
end

# electrode spacing along the lines: the median nearest-neighbour distance between electrodes
function _ert3d_line_spacing(mesh::ERT3DMesh, fallback)
    n = length(mesh.electrode_x)
    n < 2 && return fallback
    d = [minimum(hypot(mesh.electrode_x[i] - mesh.electrode_x[j], mesh.electrode_y[i] - mesh.electrode_y[j])
                 for j in 1:n if j != i && (mesh.electrode_x[i], mesh.electrode_y[i]) != (mesh.electrode_x[j], mesh.electrode_y[j]); init = Inf)
         for i in 1:n]
    d = filter(isfinite, d)
    isempty(d) ? fallback : median(d)
end

#---------- ModEM 3D model files ----------
#
# the MTGeophysics.jl / ModEM layout:
#   # description
#   nx ny nz 0 LOGE            (or LINEAR; 0 air layers)
#   nx cell sizes north (m), ny east, nz down
#   nz blocks of ny rows of nx values, i from north to south (nx down to 1), ln ρ for LOGE
#   origin x y z (m), rotation (degrees)
# air is 1e17; under topography the ground surface goes to '# surface:' comment lines: (nx + 1)(ny + 1) depths, m below
# the model top, x (north) fastest from the south-west corner, which ModEM readers skip

"""
    ReadModel3D(path) -> ModelFile3D

Read a ModEM 3D model (the MTGeophysics.jl layout) with its ground surface
(`# surface:` lines) when present.
"""
function ReadModel3D(path::AbstractString)
    isfile(path) || error("model file not found: $path")
    lines = filter(l -> !isempty(strip(l)) && !startswith(strip(l), "#"), readlines(path))
    isempty(lines) && error("$path: empty model file")
    head = split(strip(lines[1]))
    length(head) >= 4 || error("$path: first line must be 'nx ny nz n_air LOGE'")
    nx, ny, nz, nair = parse.(Int, head[1:4])
    nair == 0 || error("$path: models with air layers are not supported (the air is not modelled)")
    format = occursin("LOGE", uppercase(lines[1])) ? "LOGE" : "LINEAR"
    tokens = parse.(Float64, reduce(vcat, split.(strip.(lines[2:end]))))
    nvals = nx + ny + nz + nx * ny * nz
    length(tokens) >= nvals || error("$path: expected at least $nvals numbers after the header, got $(length(tokens))")
    dx, dy, dz = tokens[1:nx], tokens[nx+1:nx+ny], tokens[nx+ny+1:nx+ny+nz]
    v = tokens[nx+ny+nz+1:nvals]
    A = zeros(nx, ny, nz)
    p = 0
    for k in 1:nz, j in 1:ny, i in nx:-1:1
        A[i, j, k] = v[p += 1]
    end
    ρ = format == "LOGE" ? exp.(A) : A
    rest = tokens[nvals+1:end]
    origin = length(rest) >= 3 ? rest[1:3] : [-sum(dx) / 2, -sum(dy) / 2, 0.0]
    rotation = length(rest) >= 4 ? rest[4] : 0.0
    all(>(0), dx) && all(>(0), dy) && all(>(0), dz) && all(x -> isfinite(x) && x > 0, ρ) ||
        error("$path: cell sizes and resistivities must be positive")
    sv = _read_surface_comments(path, (nx + 1) * (ny + 1))
    md = NaN
    for l in eachline(path)
        m = match(r"^#\s*main domain depth\s*:\s*(\S+)"i, strip(l))
        m === nothing || (md = parse(Float64, m.captures[1]))
    end
    ModelFile3D(x_cell_sizes = dx, y_cell_sizes = dy, z_cell_sizes = dz, resistivity = ρ, origin = origin,
                rotation = rotation, surface = isempty(sv) ? zeros(0, 0) : reshape(sv, nx + 1, ny + 1),
                main_depth = md, format = format, path = String(path))
end

"""
    WriteModel3D(path, model::ModelFile3D; loge=true) -> path
    WriteModel3D(path, mesh, ρ; loge=true) -> path

Write a ModEM 3D model; with a mesh the origin is its south-west top corner, its air is
written as 1e17 and its ground surface goes to `# surface:` comment lines.
"""
function WriteModel3D(path::AbstractString, m::ModelFile3D; loge::Bool = true)
    ρ = m.resistivity
    size(ρ) == (length(m.x_cell_sizes), length(m.y_cell_sizes), length(m.z_cell_sizes)) ||
        throw(DimensionMismatch("model must be (nx, ny, nz)"))
    all(x -> isfinite(x) && x > 0, ρ) || throw(ArgumentError("resistivity must be positive and finite"))
    nx, ny, nz = size(ρ)
    isempty(m.surface) || size(m.surface) == (nx + 1, ny + 1) || throw(DimensionMismatch("surface must be (nx + 1, ny + 1)"))
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "# 3D ERT model written by ERTGeophysics.jl in ModEM format")
        isfinite(m.main_depth) && @printf(io, "# main domain depth: %.6f\n", m.main_depth)
        _write_surface_comments(io, vec(m.surface))
        @printf(io, "%5d%5d%5d%5d %s\n", nx, ny, nz, 0, loge ? "LOGE" : "LINEAR")
        for d in (m.x_cell_sizes, m.y_cell_sizes, m.z_cell_sizes)
            _write_model_rows(io, "%12.3f", d)
        end
        for k in 1:nz
            println(io)
            for j in 1:ny
                _write_model_rows(io, "%13.5E", [loge ? log(ρ[i, j, k]) : ρ[i, j, k] for i in nx:-1:1])
            end
        end
        @printf(io, "%12.3f %12.3f %12.3f\n", m.origin...)
        @printf(io, "%12.3f\n", m.rotation)
    end
    String(path)
end

function WriteModel3D(path::AbstractString, mesh::ERT3DMesh, ρ::AbstractArray{<:Real, 3}; loge::Bool = true,
                      main_depth::Real = NaN)
    earth = Array{Float64, 3}(ρ)
    earth[ert3d_air_mask(mesh)] .= ERT2D_AIR_TAG
    WriteModel3D(path, ModelFile3D(x_cell_sizes = mesh.x_cell_sizes, y_cell_sizes = mesh.y_cell_sizes,
                                   z_cell_sizes = mesh.z_cell_sizes, resistivity = earth,
                                   origin = [mesh.x_nodes[1], mesh.y_nodes[1], mesh.z_nodes[1]], surface = mesh.surface,
                                   main_depth = Float64(main_depth)); loge)
end

"""
    ert3d_padding_mask(mesh, main_depth) -> BitArray{3}

Padding cells of a 3D mesh: outside the even core columns or below `main_depth` (m below
the model top). All false when `main_depth` is NaN.
"""
function ert3d_padding_mask(mesh::ERT3DMesh, main_depth::Real)
    pad = falses(_ert3d_size(mesh))
    isfinite(main_depth) || return pad
    cx, cy = _core_range(mesh.x_cell_sizes), _core_range(mesh.y_cell_sizes)
    zc = ert3d_z_centers(mesh) .- mesh.z_nodes[1]
    for c in CartesianIndices(pad)
        pad[c] = !(c[1] in cx && c[2] in cy && zc[c[3]] < main_depth)
    end
    pad
end

"""
    Mesh3DFromInputs(model::ModelFile3D, data::DataFile3D; warn=true) -> (mesh, ρ)

Mesh and resistivity for a ModEM 3D model and the electrodes of `data`. The ground is the
model's `surface` (written by `MakeMesh3D`); without one, the bottom of the model's air
cells, or the data's electrode depths Z (inverse-distance interpolated), or flat. The air
of the result is the staircase of that ground on this mesh, tagged 1e17. A data depth Z
more than half a cell away from its tread is warned about.
"""
function Mesh3DFromInputs(model::ModelFile3D, data; warn::Bool = true)
    iszero(model.rotation) || throw(ArgumentError("rotated models are not supported"))
    xn = model.origin[1] .+ vcat(0.0, cumsum(model.x_cell_sizes))
    yn = model.origin[2] .+ vcat(0.0, cumsum(model.y_cell_sizes))
    zn = vcat(0.0, cumsum(model.z_cell_sizes))
    all(i -> xn[1] < data.x_positions[i] < xn[end] && yn[1] < data.y_positions[i] < yn[end], 1:nelectrodes(data)) ||
        throw(ArgumentError("electrodes lie outside the model"))
    sf = if !isempty(model.surface)
        model.surface
    elseif any(>(ERT2D_AIR_THRESHOLD), model.resistivity)
        _ert3d_surface_from_air(xn, yn, model.z_cell_sizes, model.resistivity)
    elseif any(!iszero, data.z_positions)
        [_ert3d_idw(data.x_positions, data.y_positions, data.z_positions, x, y) for x in xn, y in yn]
    else
        zeros(0, 0)
    end
    _ert3d_check_surface(sf, xn, yn, zn)
    mesh = ERT3DMesh(x_nodes = xn, y_nodes = yn, z_nodes = zn,
                     x_cell_sizes = Float64.(model.x_cell_sizes), y_cell_sizes = Float64.(model.y_cell_sizes),
                     z_cell_sizes = Float64.(model.z_cell_sizes), electrode_x = Float64.(data.x_positions),
                     electrode_y = Float64.(data.y_positions), surface = all(iszero, sf) ? zeros(0, 0) : Matrix{Float64}(sf))
    ρ = copy(model.resistivity)
    air = ert3d_air_mask(mesh)
    ρ[air] .= ERT2D_AIR_TAG
    for i in axes(ρ, 1), j in axes(ρ, 2), k in size(ρ, 3)-1:-1:1     # earth the file had as air takes the cell below
        !air[i, j, k] && ρ[i, j, k] > ERT2D_AIR_THRESHOLD && (ρ[i, j, k] = ρ[i, j, k+1])
    end
    if warn
        off = filter(o -> abs(o.offset) > o.tolerance, ert3d_electrode_offsets(mesh, data))
        isempty(off) || @warn "$(length(off)) electrode depth(s) Z differ from their tread on the mesh by more than half a cell" *
            join([@sprintf("\n  %s: Z %.2f m, tread %.2f m", o.name, o.z, o.surface) for o in off[1:min(end, 10)]])
    end
    mesh, ρ
end

#---------- staircase topography ----------

# inverse-distance elevation of the nearest `k` points
function _ert3d_idw(px, py, pe, x, y; k::Int = 8)
    d2 = (px .- x) .^ 2 .+ (py .- y) .^ 2
    idx = partialsortperm(d2, 1:min(k, length(d2)))
    d2[idx[1]] < 1e-12 && return pe[idx[1]]
    w = 1 ./ d2[idx]
    sum(w .* pe[idx]) / sum(w)
end

"""
    Topography3D(model::ModelFile3D, data::DataFile3D[, topo::Topo2D]) -> (; model, mask, data, datum, surface)

Topography for a ModEM 3D model on its rectangular grid: the ground surface (depth below
the model top at every (x, y) node, kept as the model's `surface`) and its staircase of air
cells (1e17 ohm m, mask 0). The ground comes from `topo` (`topo.dat`, WGS84 lat lon
elevation points, inverse-distance interpolated) or without it from the electrode depths
Z (elevation = −Z).
- The model top (the datum) is the highest electrode; higher ground is flattened to it.
- Each electrode stands on a short flat tread; Z = datum − ground elevation at the electrode.
"""
function Topography3D(model::ModelFile3D, data, topo = nothing)
    if topo === nothing
        px, py, pe = data.x_positions, data.y_positions, -data.z_positions
    else
        length(data.latitudes) == nelectrodes(data) || throw(ArgumentError("the data file has no electrode lat/lon to place the topography"))
        px, py = ert3d_local_frame(data.origin, topo.latitudes, topo.longitudes)
        pe = topo.elevations
    end
    elevation(x, y) = _ert3d_idw(px, py, pe, x, y)
    ne = nelectrodes(data)
    xn = model.origin[1] .+ vcat(0.0, cumsum(model.x_cell_sizes))
    yn = model.origin[2] .+ vcat(0.0, cumsum(model.y_cell_sizes))
    for e in 1:ne
        xn[1] < data.x_positions[e] < xn[end] && yn[1] < data.y_positions[e] < yn[end] ||
            throw(ArgumentError("electrode $(data.electrode_names[e]) lies outside the model"))
    end
    ee = [elevation(data.x_positions[i], data.y_positions[i]) for i in 1:ne]
    datum = maximum(ee)
    surface = [max(datum - elevation(x, y), 0.0) for x in xn, y in yn]
    mesh = ERT3DMesh(x_nodes = xn, y_nodes = yn, z_nodes = vcat(0.0, cumsum(model.z_cell_sizes)),
                     x_cell_sizes = model.x_cell_sizes, y_cell_sizes = model.y_cell_sizes, z_cell_sizes = model.z_cell_sizes,
                     electrode_x = data.x_positions, electrode_y = data.y_positions,
                     surface = all(iszero, surface) ? zeros(0, 0) : surface)
    air = ert3d_air_mask(mesh)
    ρ = copy(model.resistivity)
    ρ[air] .= ERT2D_AIR_TAG
    mask = ones(Int, size(ρ))
    mask[air] .= 0
    newmodel = ModelFile3D(title = model.title, x_cell_sizes = model.x_cell_sizes, y_cell_sizes = model.y_cell_sizes,
                           z_cell_sizes = model.z_cell_sizes, resistivity = ρ, origin = model.origin,
                           rotation = model.rotation, surface = mesh.surface, main_depth = model.main_depth,
                           format = model.format)
    (; model = newmodel, mask, data = _ert3d_with(data; z_positions = datum .- ee), datum, surface)
end
