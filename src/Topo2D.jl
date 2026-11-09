# 2D ERT topography
# Author: @pankajkmishra
# topo.dat (WGS84 lat, lon, elevation) along a profile, and ModEM-layout models with the topographic air and water
# cut in, as Topo2D.jl of MTGeophysics.jl

using Printf

"""
    Topo2D

Ground elevation points along a profile: WGS84 `latitudes`, `longitudes` and
`elevations` in metres above sea level.
"""
Base.@kwdef struct Topo2D
    latitudes::Vector{Float64}
    longitudes::Vector{Float64}
    elevations::Vector{Float64}
end

"""
    ReadTopo2D(path) -> Topo2D

Read `topo.dat`: one `lat lon elevation` point per line, WGS84 degrees and metres
above sea level; `#` lines are skipped.
"""
function ReadTopo2D(path::AbstractString)
    isfile(path) || error("topography file not found: $path")
    lat, lon, elev = Float64[], Float64[], Float64[]
    for line in eachline(path)
        s = strip(first(split(line, '#')))
        isempty(s) && continue
        p = split(s)
        length(p) >= 3 || error("$path: expected 'lat lon elevation', got '$s'")
        push!(lat, parse(Float64, p[1])); push!(lon, parse(Float64, p[2])); push!(elev, parse(Float64, p[3]))
    end
    length(lat) >= 2 || error("$path: at least two topography points are needed")
    Topo2D(latitudes = lat, longitudes = lon, elevations = elev)
end

"""
    WriteTopo2D(path, topo::Topo2D) -> path

Write `topo.dat`: a `#` header, then `lat lon elevation` per point.
"""
function WriteTopo2D(path::AbstractString, t::Topo2D)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "# 2D topography written by ERTGeophysics.jl, WGS84")
        println(io, "# Lat(deg) Lon(deg) Elevation(m a.s.l.)")
        for i in eachindex(t.latitudes)
            @printf(io, "%13.8f %13.8f %10.3f\n", t.latitudes[i], t.longitudes[i], t.elevations[i])
        end
    end
    String(path)
end

#---------- projection onto the profile ----------

"""
    ert2d_profile_topography(topo, data) -> (y, elevation)

Profile position y (the data's local y) of each topography point, sorted along y. Each
point goes to the nearest point of the electrode polyline, so curved lines stay curved;
the end segments extend beyond the first and last electrode.
"""
function ert2d_profile_topography(topo::Topo2D, data)
    ne = length(data.electrodes)
    ne >= 2 || throw(ArgumentError("projecting topography needs at least two electrodes"))
    length(data.latitudes) == ne || throw(ArgumentError("the data file has no electrode lat/lon to place the topography"))
    order = sortperm(data.electrodes)
    sy = data.electrodes[order]
    se, sn = _ert2d_en(data.origin, data.latitudes[order], data.longitudes[order])
    te, tn = _ert2d_en(data.origin, topo.latitudes, topo.longitudes)
    y = map(eachindex(te)) do k
        best, ybest = Inf, NaN
        for i in 1:ne-1
            de, dn = se[i+1] - se[i], sn[i+1] - sn[i]
            len2 = de^2 + dn^2
            len2 > 0 || continue
            t = ((te[k] - se[i]) * de + (tn[k] - sn[i]) * dn) / len2
            t = clamp(t, i == 1 ? -Inf : 0.0, i == ne - 1 ? Inf : 1.0)
            d = hypot(te[k] - se[i] - t * de, tn[k] - sn[i] - t * dn)
            d < best && ((best, ybest) = (d, sy[i] + t * (sy[i+1] - sy[i])))
        end
        ybest
    end
    p = sortperm(y)
    y[p], topo.elevations[p]
end

# linear interpolation, constant beyond the ends
function _ert2d_interp(x::AbstractVector, v::AbstractVector, xi::Real)
    xi <= x[1] && return v[1]
    xi >= x[end] && return v[end]
    j = searchsortedlast(x, xi)
    x[j+1] == x[j] && return v[j]
    v[j] + (v[j+1] - v[j]) * (xi - x[j]) / (x[j+1] - x[j])
end

#---------- staircase topography ----------

"""
    Topography2D(model::ModelFile2D, data::DataFile2D, topo::Topo2D;
                 water=[], water_resistivity=100.0) -> (; model, mask, data, datum, surface)
    Topography2D(model::ModelFile2D, data::DataFile2D; water=[], water_resistivity=100.0)

Topography for a ModEM-layout model on its rectangular grid, as in MTGeophysics.jl: the
ground line (depth below the model top at every y node, kept as the model's `surface`)
and its staircase of air cells (1e17 ohm m, mask 0). The ground comes from `topo`
(`topo.dat`, projected onto the line, linear between the points and constant beyond
them), or without it from the electrode depths Z of the data (linear between the
electrodes, constant beyond the line ends).
- The model top (the datum) is the highest electrode; ground higher than it (beyond the
  line) is flattened to the model top.
- Each electrode stands on a flat tread of the staircase (`ert2d_topo_air`); Z = datum −
  ground elevation at the electrode.
- `water`: lakes, rivers or sea as `(y_range = (y1, y2), level = m a.s.l.)`. Where the
  ground lies below the level the water surface is the top of the earth, and cells whose
  centre lies above the lake bed are water, `water_resistivity`, mask 9. No electrode may
  stand in the water.

Returns the new model, the covariance mask (0 air, 9 water, 1 free), the data with the
electrode depths, the datum (m a.s.l.) and the ground depth of each y node.
"""
Topography2D(model::ModelFile2D, data, topo::Topo2D; kwargs...) =
    _ert2d_staircase(model, data, ert2d_profile_topography(topo, data)...; kwargs...)

function Topography2D(model::ModelFile2D, data; kwargs...)
    order = sortperm(data.electrodes)
    ye, ze = data.electrodes[order], data.z_positions[order]
    keep = vcat(true, diff(ye) .> 1e-9)
    _ert2d_staircase(model, data, ye[keep], -ze[keep]; kwargs...)
end

# staircase of the ground profile (y, elevation) along the line
function _ert2d_staircase(model::ModelFile2D, data, ty, te; water = NamedTuple[], water_resistivity::Real = 100.0)
    ground(y) = _ert2d_interp(ty, te, y)
    y_nodes = model.origin .+ vcat(0.0, cumsum(model.y_cell_sizes))
    for (i, y) in enumerate(data.electrodes)
        y_nodes[1] < y < y_nodes[end] || throw(ArgumentError("electrode $(data.electrode_names[i]) lies outside the model"))
        for w in water
            w.y_range[1] <= y <= w.y_range[2] && ground(y) < w.level &&
                throw(ArgumentError("electrode $(data.electrode_names[i]) at y = $y m stands in water; use a water-borne survey tool"))
        end
    end
    # the top of the earth: the ground, or the water surface over a lake
    function top(y)
        g = ground(y)
        for w in water
            w.y_range[1] <= y <= w.y_range[2] && (g = max(g, Float64(w.level)))
        end
        g
    end
    datum = maximum(ground.(data.electrodes))
    surface = max.(datum .- top.(y_nodes), 0.0)
    nz, ny = size(model.resistivity)
    mesh = ERT2DMesh(y_nodes = y_nodes, z_nodes = vcat(0.0, cumsum(model.z_cell_sizes)),
                     y_cell_sizes = model.y_cell_sizes, z_cell_sizes = model.z_cell_sizes,
                     electrode_positions = data.electrodes, surface = surface)
    air = ert2d_air_mask(mesh)
    yc, zc = ert2d_y_centers(mesh), ert2d_z_centers(mesh)
    ρ = copy(model.resistivity)
    mask = ones(Int, nz, ny)
    for w in water, iy in 1:ny, iz in 1:nz
        w.y_range[1] <= yc[iy] <= w.y_range[2] && !air[iz, iy] && datum - zc[iz] > ground(yc[iy]) || continue
        ρ[iz, iy] = water_resistivity
        mask[iz, iy] = ERT2D_MASK_WATER
    end
    ρ[air] .= ERT2D_AIR_TAG
    mask[air] .= 0
    out = _ert2d_with(data; z_positions = datum .- ground.(data.electrodes))
    newmodel = ModelFile2D(title = model.title, y_cell_sizes = model.y_cell_sizes, z_cell_sizes = model.z_cell_sizes,
                           resistivity = ρ, origin = model.origin, surface = all(iszero, surface) ? Float64[] : surface,
                           format = model.format)
    (; model = newmodel, mask, data = out, datum, surface)
end
