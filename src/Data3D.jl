# 3D ERT data
# Author: @pankajkmishra
# The ModEM-style ERT data file of a 3D survey: electrodes with WGS84 lat/lon and local x (north), y (east), Z (depth
# below the model top), measurements as in 2D; the local frame of MTGeophysics.jl's 3D models; layouts; misfit

using Printf
using Statistics

"""
    DataFile3D

A 3D ERT data file.
- `electrode_names`, `latitudes`, `longitudes`: electrodes, WGS84 degrees
- `x_positions`, `y_positions`, `z_positions`: local metres, x north, y east (the
  ModEM 3D frame), Z depth below the model top (0 when flat)
- `a`, `b`, `m`, `n`: current (A, B) and potential (M, N) electrode indices, 0 = remote
- `resistance`, `resistance_error`: transfer resistance ΔV/I and its standard error (Ω)
- `geometric_factor`: K (m), ρa = K R; `rhoa`: apparent resistivity (Ω·m), K R
- `origin`: WGS84 `[lat, lon]` of the local (0, 0)
"""
Base.@kwdef struct DataFile3D
    title::String = "ERTGeophysics.jl 3D survey data"
    electrode_names::Vector{String}
    latitudes::Vector{Float64} = Float64[]
    longitudes::Vector{Float64} = Float64[]
    x_positions::Vector{Float64}
    y_positions::Vector{Float64}
    z_positions::Vector{Float64} = zeros(length(electrode_names))
    a::Vector{Int}
    b::Vector{Int}
    m::Vector{Int}
    n::Vector{Int}
    resistance::Vector{Float64} = fill(NaN, length(a))
    resistance_error::Vector{Float64} = fill(NaN, length(a))
    geometric_factor::Vector{Float64} = fill(NaN, length(a))
    rhoa::Vector{Float64} = geometric_factor .* resistance
    origin::Vector{Float64} = [0.0, 0.0]
    path::String = ""
end

ndata(d::DataFile3D) = length(d.a)
nelectrodes(d::DataFile3D) = length(d.x_positions)

# copy with some fields replaced; rhoa follows K and R unless given
function _ert3d_with(d::DataFile3D; kwargs...)
    f = Dict{Symbol, Any}(k => deepcopy(getfield(d, k)) for k in fieldnames(DataFile3D))
    merge!(f, Dict(kwargs))
    haskey(kwargs, :rhoa) || (f[:rhoa] = f[:geometric_factor] .* f[:resistance])
    DataFile3D(; f...)
end

function _ert3d_validate(d::DataFile3D)
    ne, nd = nelectrodes(d), ndata(d)
    length(d.electrode_names) == ne == length(d.y_positions) == length(d.z_positions) ||
        throw(DimensionMismatch("electrode names and coordinates must have one entry per electrode"))
    isempty(d.latitudes) || length(d.latitudes) == length(d.longitudes) == ne ||
        throw(DimensionMismatch("latitudes and longitudes must have one entry per electrode"))
    all(v -> length(v) == nd, (d.b, d.m, d.n, d.resistance, d.resistance_error, d.geometric_factor, d.rhoa)) ||
        throw(DimensionMismatch("every measurement column must have one entry per datum"))
    for i in 1:nd
        q = (d.a[i], d.b[i], d.m[i], d.n[i])
        all(e -> 0 <= e <= ne, q) || throw(ArgumentError("datum $i: electrode index outside 0:$ne"))
        d.a[i] == 0 && d.b[i] == 0 && throw(ArgumentError("datum $i: no current electrode"))
        d.m[i] == 0 && d.n[i] == 0 && throw(ArgumentError("datum $i: no potential electrode"))
        nz = filter(!=(0), collect(q))
        length(unique(nz)) == length(nz) || throw(ArgumentError("datum $i: an electrode is used twice"))
    end
    d
end

#---------- local frame ----------

"""
    ert3d_local_frame(origin, latitudes, longitudes) -> (x, y)

Local coordinates (m) of WGS84 points, x north and y east of `origin = [lat, lon]`, in
the local transverse Mercator of MTGeophysics.jl.
"""
function ert3d_local_frame(origin, lat::AbstractVector{<:Real}, lon::AbstractVector{<:Real})
    e, n = _ert2d_en(origin, lat, lon)
    n, e
end

"""
    ert3d_latlon(origin, x, y) -> (latitudes, longitudes)

WGS84 position of local coordinates (x north, y east), the inverse of `ert3d_local_frame`.
"""
ert3d_latlon(origin, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = ert2d_latlon(origin, 90.0, x, y)

#---------- layouts ----------

"""
    ert3d_grid(nx, ny, dx, dy) -> (; x, y, lines_x, lines_y)

Electrode grid of `nx` × `ny` electrodes, `dx` apart northwards and `dy` eastwards,
centred on (0, 0). Electrodes are numbered line by line: the `nx` east-west lines of `ny`
electrodes one after the other. `lines_y` lists the electrode numbers of each east-west
line, `lines_x` those of each north-south line, each in order.
"""
function ert3d_grid(nx::Integer, ny::Integer, dx::Real, dy::Real)
    xs = collect(range(-(nx - 1) * dx / 2, (nx - 1) * dx / 2, length = nx))
    ys = collect(range(-(ny - 1) * dy / 2, (ny - 1) * dy / 2, length = ny))
    id(i, j) = (i - 1) * ny + j
    (; x = [xs[i] for i in 1:nx for j in 1:ny], y = [ys[j] for i in 1:nx for j in 1:ny],
       lines_y = [[id(i, j) for j in 1:ny] for i in 1:nx], lines_x = [[id(i, j) for i in 1:nx] for j in 1:ny])
end

"""
    ert3d_line_quadrupoles(lines, generator) -> (; a, b, m, n)

Quadrupoles of a 2D array generator applied along each line of electrode numbers, e.g.
`ert3d_line_quadrupoles(g.lines_y, n -> ert2d_dipole_dipole(n; a_max = 3, n_max = 6))`.
Lines with too few electrodes for the generator are skipped.
"""
function ert3d_line_quadrupoles(lines, generator)
    a, b, m, n = Int[], Int[], Int[], Int[]
    for line in lines
        q = try
            generator(length(line))
        catch err
            err isa ArgumentError || rethrow()
            continue
        end
        pick(v) = [k == 0 ? 0 : line[k] for k in v]
        append!(a, pick(q.a)); append!(b, pick(q.b)); append!(m, pick(q.m)); append!(n, pick(q.n))
    end
    (; a, b, m, n)
end

"""
    BuildSurvey3D(; origin, x, y, quadrupoles, z_positions=zeros, names=E001..., title) -> DataFile3D

A 3D survey without measurements: electrodes at local x (north), y (east) and depth Z,
WGS84 positions from `origin`, and the quadrupoles `(a, b, m, n)`.
"""
function BuildSurvey3D(; origin, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, quadrupoles,
                       z_positions::AbstractVector{<:Real} = zeros(length(x)),
                       names::AbstractVector{<:AbstractString} = [@sprintf("E%03d", i) for i in eachindex(x)],
                       title::AbstractString = "ERTGeophysics.jl 3D survey")
    lat, lon = ert3d_latlon(origin, x, y)
    _ert3d_validate(DataFile3D(title = title, electrode_names = String.(names), latitudes = lat, longitudes = lon,
                               x_positions = Float64.(x), y_positions = Float64.(y), z_positions = Float64.(z_positions),
                               a = quadrupoles.a, b = quadrupoles.b, m = quadrupoles.m, n = quadrupoles.n,
                               origin = Float64.(collect(origin))))
end

#---------- geometric factors and positions ----------

"""
    ert3d_geometric_factor(data) -> Vector{Float64}

Analytic half-space geometric factor 2π / (1/AM − 1/BM − 1/AN + 1/BN) from the straight
distances between the electrodes (x, y and depth Z); remote electrodes drop their terms.
"""
function ert3d_geometric_factor(d::DataFile3D)
    dist(i, j) = (i == 0 || j == 0) ? Inf :
        sqrt((d.x_positions[i] - d.x_positions[j])^2 + (d.y_positions[i] - d.y_positions[j])^2 +
             (d.z_positions[i] - d.z_positions[j])^2)
    [2π / (1 / dist(d.a[i], d.m[i]) - 1 / dist(d.b[i], d.m[i]) - 1 / dist(d.a[i], d.n[i]) + 1 / dist(d.b[i], d.n[i]))
     for i in 1:ndata(d)]
end

"""
    ert3d_pseudo(data) -> (; x, y, depth, span)

Pseudo-position of each datum: x and y the mean of its electrodes, depth a quarter of the
distance between the current and potential dipole centres, `span` the largest distance
between two of its electrodes.
"""
function ert3d_pseudo(d::DataFile3D)
    P(i) = (d.x_positions[i], d.y_positions[i])
    centre(p, q) = p == 0 ? P(q) : q == 0 ? P(p) : ((P(p)[1] + P(q)[1]) / 2, (P(p)[2] + P(q)[2]) / 2)
    x, y, depth, span = (zeros(ndata(d)) for _ in 1:4)
    for i in 1:ndata(d)
        es = filter(!=(0), [d.a[i], d.b[i], d.m[i], d.n[i]])
        x[i] = mean(d.x_positions[es]); y[i] = mean(d.y_positions[es])
        c1, c2 = centre(d.a[i], d.b[i]), centre(d.m[i], d.n[i])
        depth[i] = hypot(c1[1] - c2[1], c1[2] - c2[2]) / 4
        span[i] = maximum(hypot(P(p)[1] - P(q)[1], P(p)[2] - P(q)[2]) for p in es, q in es)
    end
    (; x, y, depth, span)
end

# median distance from each electrode to its nearest neighbour
function ert3d_spacing(d::DataFile3D)
    ne = nelectrodes(d)
    ne >= 2 || throw(ArgumentError("at least two electrodes are needed"))
    median([minimum(hypot(d.x_positions[i] - d.x_positions[j], d.y_positions[i] - d.y_positions[j]) for j in 1:ne if j != i)
            for i in 1:ne])
end

#---------- file ----------
#
#   # title
#   > ERT_Resistance_3D
#   > DC
#   > [Ohm]
#   > 0.0                  (rotation of the x axis from north, degrees; always 0)
#   > origin lat lon
#   > n_electrodes n_data
#   n_electrodes lines: Name Lat Lon X(m, north) Y(m, east) Z(m, depth)
#   n_data lines:       A B M N R(Ohm) Error(Ohm) K(m) Rhoa(Ohm.m)

"""
    write_data3d(path, data::DataFile3D) -> path

Write a ModEM-style 3D ERT data file.
"""
function write_data3d(path::AbstractString, d::DataFile3D)
    _ert3d_validate(d)
    ne, nd = nelectrodes(d), ndata(d)
    lat = isempty(d.latitudes) ? fill(NaN, ne) : d.latitudes
    lon = isempty(d.longitudes) ? fill(NaN, ne) : d.longitudes
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "# ", isempty(d.title) ? "3D ERT data written by ERTGeophysics.jl" : d.title)
        println(io, "> ERT_Resistance_3D")
        println(io, "> DC")
        println(io, "> [Ohm]")
        println(io, "> 0.0")
        @printf(io, "> %.8f %.8f\n", d.origin[1], d.origin[2])
        @printf(io, "> %d %d\n", ne, nd)
        println(io, "# Electrode       Lat(deg)       Lon(deg)  X(m,north)   Y(m,east)  Z(m,depth)")
        for i in 1:ne
            @printf(io, "%-12s %14.8f %14.8f %11.3f %11.3f %11.3f\n", d.electrode_names[i], lat[i], lon[i],
                    d.x_positions[i], d.y_positions[i], d.z_positions[i])
        end
        println(io, "#    A     B     M     N          R(Ohm)      Error(Ohm)            K(m)     Rhoa(Ohm.m)")
        for i in 1:nd
            @printf(io, "%6d%6d%6d%6d %15.7E %15.7E %15.7E %15.7E\n", d.a[i], d.b[i], d.m[i], d.n[i],
                    d.resistance[i], d.resistance_error[i], d.geometric_factor[i], d.rhoa[i])
        end
    end
    String(path)
end

"""
    load_data3d(path) -> DataFile3D

Read a ModEM-style 3D ERT data file. When every K is NaN, K is set to the analytic value
over the electrode distances.
"""
function load_data3d(path::AbstractString)
    isfile(path) || error("data file not found: $path")
    header, body = String[], Vector{Vector{SubString{String}}}()
    title = ""
    for raw in eachline(path)
        s = strip(raw)
        isempty(s) && continue
        if startswith(s, "#")
            isempty(title) && isempty(header) && (title = strip(s[2:end]))
        elseif startswith(s, ">")
            push!(header, strip(s[2:end]))
        else
            push!(body, split(first(split(s, '#'))))
        end
    end
    length(header) >= 6 || error("$path: expected 6 '>' header lines, found $(length(header))")
    uppercase(header[1]) == "ERT_RESISTANCE_3D" || error("$path: data type '$(header[1])' is not ERT_Resistance_3D")
    iszero(parse(Float64, split(header[4])[1])) || error("$path: rotated 3D frames are not supported")
    origin = parse.(Float64, split(header[5])[1:2])
    ne, nd = parse.(Int, split(header[6])[1:2])
    length(body) == ne + nd || error("$path: expected $ne electrode and $nd data lines, found $(length(body)) lines")
    el, ms = body[1:ne], body[ne+1:end]
    all(t -> length(t) >= 6, el) || error("$path: electrode lines must read 'Name Lat Lon X Y Z'")
    all(t -> length(t) >= 6, ms) || error("$path: data lines must read 'A B M N R Error [K Rhoa]'")
    col(t, j) = length(t) >= j ? _ert2d_float(t[j]) : NaN
    lat, lon = [col(t, 2) for t in el], [col(t, 3) for t in el]
    d = DataFile3D(title = title, electrode_names = [String(t[1]) for t in el],
                   latitudes = all(isnan, lat) ? Float64[] : lat, longitudes = all(isnan, lon) ? Float64[] : lon,
                   x_positions = [col(t, 4) for t in el], y_positions = [col(t, 5) for t in el],
                   z_positions = [col(t, 6) for t in el],
                   a = [parse(Int, t[1]) for t in ms], b = [parse(Int, t[2]) for t in ms],
                   m = [parse(Int, t[3]) for t in ms], n = [parse(Int, t[4]) for t in ms],
                   resistance = [col(t, 5) for t in ms], resistance_error = [col(t, 6) for t in ms],
                   geometric_factor = [col(t, 7) for t in ms], rhoa = [col(t, 8) for t in ms],
                   origin = origin, path = String(path))
    _ert3d_validate(d)
    all(isnan, d.geometric_factor) ? _ert3d_with(d; geometric_factor = ert3d_geometric_factor(d)) : d
end

#---------- selection, misfit, frame ----------

"""
    select_data3d(data, keep) -> DataFile3D

The data whose `keep` entry (Bool vector or indices) is true; the electrodes are kept.
"""
function select_data3d(d::DataFile3D, keep)
    idx = keep isa AbstractVector{Bool} ? findall(keep) : collect(keep)
    _ert3d_with(d; a = d.a[idx], b = d.b[idx], m = d.m[idx], n = d.n[idx], resistance = d.resistance[idx],
                resistance_error = d.resistance_error[idx], geometric_factor = d.geometric_factor[idx], rhoa = d.rhoa[idx])
end

chi2_rms3d(observed::DataFile3D, predicted::DataFile3D; error_floor::Real = 0.0) =
    chi2_rms2d(observed, predicted; error_floor)
chi2_rms3d(observed::AbstractString, predicted::AbstractString; kwargs...) =
    chi2_rms3d(load_data3d(observed), load_data3d(predicted); kwargs...)

"""
    recentre_data3d(data) -> DataFile3D

Move the local origin to the centre of the electrodes' bounding box, so that a model
centred on (0, 0) is centred on the survey.
"""
function recentre_data3d(d::DataFile3D)
    cx = (minimum(d.x_positions) + maximum(d.x_positions)) / 2
    cy = (minimum(d.y_positions) + maximum(d.y_positions)) / 2
    hypot(cx, cy) < 1e-6 && return d
    origin = if length(d.latitudes) == nelectrodes(d)
        lat, lon = ert3d_latlon(d.origin, [cx], [cy])
        [lat[1], lon[1]]
    else
        d.origin
    end
    _ert3d_with(d; x_positions = d.x_positions .- cx, y_positions = d.y_positions .- cy, origin)
end

# predicted data on the observed survey: ρa, and R = ρa / K with the observed K and errors
_ert3d_predicted(observed::DataFile3D, rhoa::AbstractVector; title = "ERTGeophysics.jl predicted data") =
    _ert3d_with(observed; title = title, resistance = rhoa ./ observed.geometric_factor, rhoa = Float64.(rhoa))
