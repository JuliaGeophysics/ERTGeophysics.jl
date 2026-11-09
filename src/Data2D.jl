# 2D ERT data
# Author: @pankajkmishra
# The ModEM-style ERT data file: electrodes with WGS84 lat/lon and local x y z, four-electrode measurements with
# transfer resistance, error, geometric factor and apparent resistivity; the local profile frame; arrays; misfit

using Printf
using Statistics

"""
    DataFile2D

A 2D ERT data file.
- `electrode_names`, `latitudes`, `longitudes`: electrodes, WGS84 degrees
- `x_positions`, `electrodes`, `z_positions`: local metres, x across the profile, y
  (`electrodes`) along it, Z depth below the model top (0 when flat)
- `a`, `b`, `m`, `n`: current (A, B) and potential (M, N) electrode indices, 0 = remote
- `resistance`, `resistance_error`: transfer resistance ΔV/I and its standard error (Ω)
- `geometric_factor`: K (m), ρa = K R; NaN = not yet known
- `rhoa`: apparent resistivity (Ω·m), K R
- `origin`: WGS84 `[lat, lon]` of the local (0, 0); `azimuth`: profile direction
  (y axis), degrees clockwise from north
"""
Base.@kwdef struct DataFile2D
    title::String = "ERTGeophysics.jl 2D profile data"
    electrode_names::Vector{String}
    latitudes::Vector{Float64} = Float64[]
    longitudes::Vector{Float64} = Float64[]
    x_positions::Vector{Float64} = zeros(length(electrode_names))
    electrodes::Vector{Float64}
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
    azimuth::Float64 = 90.0
    path::String = ""
end

ndata(d::DataFile2D) = length(d.a)
nelectrodes(d::DataFile2D) = length(d.electrodes)

# copy with some fields replaced; rhoa follows K and R unless given
function _ert2d_with(d::DataFile2D; kwargs...)
    f = Dict{Symbol, Any}(k => deepcopy(getfield(d, k)) for k in fieldnames(DataFile2D))
    merge!(f, Dict(kwargs))
    haskey(kwargs, :rhoa) || (f[:rhoa] = f[:geometric_factor] .* f[:resistance])
    DataFile2D(; f...)
end

function _ert2d_validate(d::DataFile2D)
    ne, nd = length(d.electrodes), length(d.a)
    length(d.electrode_names) == ne == length(d.x_positions) == length(d.z_positions) ||
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

_local_tm_proj_string(lat0::Real, lon0::Real) =
    "+proj=tmerc +lat_0=$(float(lat0)) +lon_0=$(float(lon0)) +k=0.9996 +x_0=500000 +y_0=0 +datum=WGS84 +units=m +no_defs"

# local transverse Mercator metres (east, north) around the origin
function _ert2d_en(origin, lat, lon)
    t = Proj.Transformation("EPSG:4326", _local_tm_proj_string(origin[1], origin[2]); always_xy = true)
    p = [t(lon[i], lat[i]) for i in eachindex(lat)]
    [q[1] - 500000.0 for q in p], [q[2] for q in p]
end

"""
    ert2d_local_frame(origin, azimuth, latitudes, longitudes) -> (x, y)

Local profile coordinates (m) of WGS84 points: y along `azimuth` (degrees clockwise
from north), x 90° to its left, both from `origin = [lat, lon]`. The projection is the
local transverse Mercator of MTGeophysics.jl around the origin; azimuth 90 gives x north,
y east.
"""
function ert2d_local_frame(origin, azimuth::Real, lat::AbstractVector{<:Real}, lon::AbstractVector{<:Real})
    e, n = _ert2d_en(origin, lat, lon)
    s, c = sind(azimuth), cosd(azimuth)
    (-e .* c .+ n .* s, e .* s .+ n .* c)
end

"""
    ert2d_latlon(origin, azimuth, x, y) -> (latitudes, longitudes)

WGS84 position of local profile coordinates, the inverse of `ert2d_local_frame`.
"""
function ert2d_latlon(origin, azimuth::Real, x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    s, c = sind(azimuth), cosd(azimuth)
    e = -x .* c .+ y .* s
    n = x .* s .+ y .* c
    t = Proj.Transformation(_local_tm_proj_string(origin[1], origin[2]), "EPSG:4326"; always_xy = true)
    p = [t(e[i] + 500000.0, n[i]) for i in eachindex(e)]
    [q[2] for q in p], [q[1] for q in p]
end

"""
    ert2d_fit_profile(latitudes, longitudes) -> (; origin, azimuth, x, y)

Profile frame of a line of electrodes: the origin halfway between the end electrodes'
projections on the best-fitting line, the azimuth of that line (0-180°, pointing from
the first electrode to the last), and the electrodes' local coordinates.
"""
function ert2d_fit_profile(lat::AbstractVector{<:Real}, lon::AbstractVector{<:Real})
    length(lat) >= 2 || throw(ArgumentError("a profile needs at least two electrodes"))
    o = [mean(lat), mean(lon)]
    e, n = _ert2d_en(o, lat, lon)
    C = cov(hcat(e .- mean(e), n .- mean(n)))
    v = eigen(Symmetric(C)).vectors[:, end]
    dot(v, [e[end] - e[1], n[end] - n[1]]) < 0 && (v = -v)
    az = mod(atand(v[1], v[2]), 360)
    x, y = ert2d_local_frame(o, az, lat, lon)
    mid = (minimum(y) + maximum(y)) / 2
    lat2, lon2 = ert2d_latlon(o, az, [mean(x)], [mid])
    origin = [lat2[1], lon2[1]]
    x, y = ert2d_local_frame(origin, az, lat, lon)
    (; origin, azimuth = az, x, y)
end

#---------- arrays ----------

"""
    ert2d_dipole_dipole(n_electrodes; a_max=3, n_max=6) -> (; a, b, m, n)

Dipole-dipole quadrupoles on a line of equally spaced electrodes: dipole length
`s = 1..a_max` electrode spacings and separation `n = 1..n_max` dipole lengths. Each
datum is ordered B A M N along the line, so A is next to M and K > 0.
"""
function ert2d_dipole_dipole(ne::Integer; a_max::Integer = 3, n_max::Integer = 6)
    A, B, M, N = Int[], Int[], Int[], Int[]
    for s in 1:a_max, nn in 1:n_max, i in 1:ne
        b, a = i, i + s
        m = a + nn * s
        n = m + s
        n <= ne || continue
        push!(A, a); push!(B, b); push!(M, m); push!(N, n)
    end
    (a = A, b = B, m = M, n = N)
end

"""
    ert2d_wenner(n_electrodes; a_max=n_electrodes÷3) -> (; a, b, m, n)

Wenner-alpha quadrupoles A M N B with spacing 1..`a_max` electrode spacings.
"""
function ert2d_wenner(ne::Integer; a_max::Integer = (ne - 1) ÷ 3)
    A, B, M, N = Int[], Int[], Int[], Int[]
    for s in 1:a_max, i in 1:ne-3s
        push!(A, i); push!(M, i + s); push!(N, i + 2s); push!(B, i + 3s)
    end
    (a = A, b = B, m = M, n = N)
end

"""
    ert2d_schlumberger(n_electrodes; n_max=6, a_max=1) -> (; a, b, m, n)

Wenner-Schlumberger quadrupoles A M N B: potential dipole of `s = 1..a_max` spacings,
current electrodes `n s` outside it, `n = 1..n_max`.
"""
function ert2d_schlumberger(ne::Integer; n_max::Integer = 6, a_max::Integer = 1)
    A, B, M, N = Int[], Int[], Int[], Int[]
    for s in 1:a_max, nn in 1:n_max, i in 1:ne
        a, m = i, i + nn * s
        n, b = m + s, m + s + nn * s
        b <= ne || continue
        push!(A, a); push!(M, m); push!(N, n); push!(B, b)
    end
    (a = A, b = B, m = M, n = N)
end

"""
    ert2d_pole_dipole(n_electrodes; a_max=1, n_max=6) -> (; a, b, m, n)

Pole-dipole quadrupoles A M N with B remote (0), forward direction.
"""
function ert2d_pole_dipole(ne::Integer; a_max::Integer = 1, n_max::Integer = 6)
    A, B, M, N = Int[], Int[], Int[], Int[]
    for s in 1:a_max, nn in 1:n_max, i in 1:ne
        m = i + nn * s
        n = m + s
        n <= ne || continue
        push!(A, i); push!(B, 0); push!(M, m); push!(N, n)
    end
    (a = A, b = B, m = M, n = N)
end

"""
    BuildSurvey2D(; origin, azimuth=90.0, positions, quadrupoles, names, x_positions,
                  z_positions, title) -> DataFile2D

A survey without measurements: electrodes at local `positions` y (m) along the profile
through `origin = [lat, lon]` at `azimuth`, their lat/lon from `ert2d_latlon`, and the
`quadrupoles` `(a, b, m, n)` of an array generator. R, errors and K are NaN.
"""
function BuildSurvey2D(; origin, azimuth::Real = 90.0, positions::AbstractVector{<:Real}, quadrupoles,
                       names::AbstractVector{<:AbstractString} = [@sprintf("E%03d", i) for i in eachindex(positions)],
                       x_positions::AbstractVector{<:Real} = zeros(length(positions)),
                       z_positions::AbstractVector{<:Real} = zeros(length(positions)),
                       title::AbstractString = "ERTGeophysics.jl 2D survey")
    lat, lon = ert2d_latlon(origin, azimuth, x_positions, positions)
    _ert2d_validate(DataFile2D(title = title, electrode_names = String.(names), latitudes = lat, longitudes = lon,
                               x_positions = Float64.(x_positions), electrodes = Float64.(positions),
                               z_positions = Float64.(z_positions), a = quadrupoles.a, b = quadrupoles.b,
                               m = quadrupoles.m, n = quadrupoles.n, origin = Float64.(collect(origin)),
                               azimuth = Float64(azimuth)))
end

#---------- geometric factors and pseudosections ----------

"""
    ert2d_geometric_factor(data) -> Vector{Float64}

Analytic geometric factor of a half-space, 2π / (1/AM − 1/BM − 1/AN + 1/BN), from the
straight distances between the electrodes (local x, y and depth Z); remote electrodes drop
their terms. Exact over flat ground and over a uniform slope only: over curved ground the
topography changes K by several per cent, so `MakeMesh2D` uses the numerical K of the
mesh's ground there (`ert2d_numerical_geometric_factor`).
"""
function ert2d_geometric_factor(d::DataFile2D)
    dist(i, j) = (i == 0 || j == 0) ? Inf : hypot(d.electrodes[i] - d.electrodes[j], d.x_positions[i] - d.x_positions[j],
                                                  d.z_positions[i] - d.z_positions[j])
    map(1:ndata(d)) do i
        g = 1 / dist(d.a[i], d.m[i]) - 1 / dist(d.b[i], d.m[i]) - 1 / dist(d.a[i], d.n[i]) + 1 / dist(d.b[i], d.n[i])
        2π / g
    end
end

"""
    ert2d_pseudo(data) -> (; y, depth, span)

Pseudosection position of each datum: y, the mean of the electrodes it uses; depth, a
quarter of the distance between the current and potential dipole centres (pole terms
take the pole); `span`, the distance between its outermost electrodes.
"""
function ert2d_pseudo(d::DataFile2D)
    pos(i) = d.electrodes[i]
    centre(p, q) = p == 0 ? pos(q) : q == 0 ? pos(p) : (pos(p) + pos(q)) / 2
    y, depth, span = Float64[], Float64[], Float64[]
    for i in 1:ndata(d)
        used = [pos(e) for e in (d.a[i], d.b[i], d.m[i], d.n[i]) if e != 0]
        push!(y, mean(used))
        push!(depth, abs(centre(d.a[i], d.b[i]) - centre(d.m[i], d.n[i])) / 4)
        push!(span, maximum(used) - minimum(used))
    end
    (; y, depth, span)
end

# median electrode spacing along the line
ert2d_spacing(d::DataFile2D) = (y = sort(unique(d.electrodes)); length(y) > 1 ? median(diff(y)) : 1.0)

#---------- file I/O ----------
#
# a '#' comment line, then
#   > ERT_Resistance
#   > DC
#   > [Ohm]
#   > azimuth (deg clockwise from north of the profile y axis)
#   > origin lat lon
#   > n_electrodes n_data
#   n_electrodes lines: Name Lat Lon X(m) Y(m) Z(m)
#   n_data lines:       A B M N R(Ohm) Error(Ohm) K(m) Rhoa(Ohm.m)
# '#' lines may appear anywhere; K and Rhoa are NaN when unknown

"""
    write_data2d(path, data::DataFile2D) -> path

Write a ModEM-style ERT data file.
"""
function write_data2d(path::AbstractString, d::DataFile2D)
    _ert2d_validate(d)
    ne, nd = nelectrodes(d), ndata(d)
    lat = isempty(d.latitudes) ? fill(NaN, ne) : d.latitudes
    lon = isempty(d.longitudes) ? fill(NaN, ne) : d.longitudes
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "# ", isempty(d.title) ? "2D ERT data written by ERTGeophysics.jl" : d.title)
        println(io, "> ERT_Resistance")
        println(io, "> DC")
        println(io, "> [Ohm]")
        @printf(io, "> %.6f\n", d.azimuth)
        @printf(io, "> %.8f %.8f\n", d.origin[1], d.origin[2])
        @printf(io, "> %d %d\n", ne, nd)
        println(io, "# Electrode       Lat(deg)       Lon(deg)        X(m)        Y(m)        Z(m)")
        for i in 1:ne
            @printf(io, "%-12s %14.8f %14.8f %11.3f %11.3f %11.3f\n", d.electrode_names[i], lat[i], lon[i],
                    d.x_positions[i], d.electrodes[i], d.z_positions[i])
        end
        println(io, "#    A     B     M     N          R(Ohm)      Error(Ohm)            K(m)     Rhoa(Ohm.m)")
        for i in 1:nd
            @printf(io, "%6d%6d%6d%6d %15.7E %15.7E %15.7E %15.7E\n", d.a[i], d.b[i], d.m[i], d.n[i],
                    d.resistance[i], d.resistance_error[i], d.geometric_factor[i], d.rhoa[i])
        end
    end
    String(path)
end

_ert2d_float(s) = (v = tryparse(Float64, s); v === nothing ? (lowercase(s) == "nan" ? NaN : error("not a number: '$s'")) : v)

"""
    load_data2d(path) -> DataFile2D

Read a ModEM-style ERT data file. K and Rhoa may be NaN; when every K is NaN and the
electrodes are flat (Z = 0), K is set to the analytic half-space value.
"""
function load_data2d(path::AbstractString)
    isfile(path) || error("data file not found: $path")
    header, body = String[], Vector{Vector{SubString{String}}}()
    lines = readlines(path)
    title = ""
    for (k, raw) in enumerate(lines)
        s = strip(raw)
        isempty(s) && continue
        if startswith(s, "#")
            isempty(title) && isempty(header) && (title = strip(s[2:end]))
            continue
        end
        if startswith(s, ">")
            push!(header, strip(s[2:end]))
        else
            push!(body, split(first(split(s, '#'))))
        end
    end
    length(header) >= 6 || error("$path: expected 6 '>' header lines, found $(length(header))")
    uppercase(header[1]) == "ERT_RESISTANCE" || error("$path: data type '$(header[1])' is not ERT_Resistance")
    azimuth = parse(Float64, split(header[4])[1])
    origin = parse.(Float64, split(header[5])[1:2])
    ne, nd = parse.(Int, split(header[6])[1:2])
    length(body) == ne + nd || error("$path: expected $ne electrode and $nd data lines, found $(length(body)) lines")
    el, ms = body[1:ne], body[ne+1:end]
    all(t -> length(t) >= 6, el) || error("$path: electrode lines must read 'Name Lat Lon X Y Z'")
    all(t -> length(t) >= 6, ms) || error("$path: data lines must read 'A B M N R Error [K Rhoa]'")
    col(t, j) = length(t) >= j ? _ert2d_float(t[j]) : NaN
    lat = [col(t, 2) for t in el]
    lon = [col(t, 3) for t in el]
    d = DataFile2D(title = title, electrode_names = [String(t[1]) for t in el],
                   latitudes = all(isnan, lat) ? Float64[] : lat, longitudes = all(isnan, lon) ? Float64[] : lon,
                   x_positions = [col(t, 4) for t in el], electrodes = [col(t, 5) for t in el],
                   z_positions = [col(t, 6) for t in el],
                   a = [parse(Int, t[1]) for t in ms], b = [parse(Int, t[2]) for t in ms],
                   m = [parse(Int, t[3]) for t in ms], n = [parse(Int, t[4]) for t in ms],
                   resistance = [col(t, 5) for t in ms], resistance_error = [col(t, 6) for t in ms],
                   geometric_factor = [col(t, 7) for t in ms], rhoa = [col(t, 8) for t in ms],
                   origin = origin, azimuth = azimuth, path = String(path))
    _ert2d_validate(d)
    if all(isnan, d.geometric_factor)
        d = _ert2d_with(d; geometric_factor = ert2d_geometric_factor(d))
    end
    d
end

"""
    ImportData2D(electrodes_csv, data_csv; out_path, value=:resistance, error_fraction=0.03,
                 error_floor=0.0, title="", topo_path="") -> (; data, path, topo)

Build a data file from two comma- or whitespace-separated tables with a header row:
- electrodes: `name, lat, lon` and optionally `elevation` (m a.s.l.);
- data: `a, b, m, n` (electrode numbers in the electrode table's order, 0 = remote)
  and `value` (`:resistance` in Ω, or `:rhoa` in Ω·m, converted with the analytic
  K), optionally `error` (same unit; else `error_fraction` of |value|, at least
  `error_floor`).
The profile frame comes from `ert2d_fit_profile`. With elevations, `topo_path` (if
given) gets a `topo.dat` of the electrode elevations for `MakeMesh2D`.
"""
function ImportData2D(electrodes_csv::AbstractString, data_csv::AbstractString; out_path::AbstractString,
                      value::Symbol = :resistance, error_fraction::Real = 0.03, error_floor::Real = 0.0,
                      title::AbstractString = "", topo_path::AbstractString = "")
    value in (:resistance, :rhoa) || throw(ArgumentError("value must be :resistance or :rhoa"))
    et, dt = _ert2d_read_table(electrodes_csv), _ert2d_read_table(data_csv)
    for k in ("name", "lat", "lon")
        haskey(et, k) || error("$electrodes_csv: missing column '$k'")
    end
    for k in ("a", "b", "m", "n", "value")
        haskey(dt, k) || error("$data_csv: missing column '$k'")
    end
    lat, lon = parse.(Float64, et["lat"]), parse.(Float64, et["lon"])
    frame = ert2d_fit_profile(lat, lon)
    q = (a = parse.(Int, dt["a"]), b = parse.(Int, dt["b"]), m = parse.(Int, dt["m"]), n = parse.(Int, dt["n"]))
    d = DataFile2D(title = isempty(title) ? "Imported from $(basename(data_csv))" : title,
                   electrode_names = String.(et["name"]), latitudes = lat, longitudes = lon, x_positions = frame.x,
                   electrodes = frame.y, a = q.a, b = q.b, m = q.m, n = q.n, origin = frame.origin,
                   azimuth = frame.azimuth)
    _ert2d_validate(d)
    K = ert2d_geometric_factor(d)
    v = _ert2d_float.(dt["value"])
    R = value == :resistance ? v : v ./ K
    err = haskey(dt, "error") ? abs.(_ert2d_float.(dt["error"])) : fill(NaN, length(v))
    value == :rhoa && (err = err ./ abs.(K))
    err = [isfinite(err[i]) ? err[i] : max(error_fraction * abs(R[i]), error_floor) for i in eachindex(R)]
    d = _ert2d_with(d; resistance = R, resistance_error = err, geometric_factor = K)
    topo = ""
    if !isempty(topo_path) && haskey(et, "elevation")
        topo = WriteTopo2D(topo_path, Topo2D(latitudes = lat, longitudes = lon, elevations = parse.(Float64, et["elevation"])))
    end
    (; data = d, path = write_data2d(out_path, d), topo)
end

# header row and columns of a comma- or whitespace-separated table, lowercase keys
function _ert2d_read_table(path::AbstractString)
    isfile(path) || error("table not found: $path")
    rows = [strip(l) for l in eachline(path) if !isempty(strip(l)) && !startswith(strip(l), "#")]
    isempty(rows) && error("$path: empty table")
    splitrow(s) = occursin(',', s) ? strip.(split(s, ',')) : split(s)
    head = lowercase.(splitrow(rows[1]))
    cols = Dict(h => String[] for h in head)
    for (k, r) in enumerate(rows[2:end])
        t = splitrow(r)
        length(t) == length(head) || error("$path:$(k + 1): $(length(t)) values for $(length(head)) columns")
        foreach(((h, v),) -> push!(cols[h], v), zip(head, t))
    end
    cols
end

#---------- selection and misfit ----------

"""
    select_data2d(data, keep) -> DataFile2D

The data whose `keep` entry (Bool vector or indices) is true; the electrodes are kept.
"""
function select_data2d(d::DataFile2D, keep)
    idx = keep isa AbstractVector{Bool} ? findall(keep) : collect(keep)
    _ert2d_with(d; a = d.a[idx], b = d.b[idx], m = d.m[idx], n = d.n[idx], resistance = d.resistance[idx],
                resistance_error = d.resistance_error[idx], geometric_factor = d.geometric_factor[idx],
                rhoa = d.rhoa[idx])
end

"""
    ert2d_valid_data(data; error_floor=0.0) -> (; keep, sigma, reasons)

Data usable for inversion: finite R, positive error and K, positive ρa. `sigma` is the
standard error of log ρa, σ_R/|R|, raised to `error_floor` (a fraction, 0.03 = 3%).
`reasons` counts the rejected data by cause.
"""
function ert2d_valid_data(d; error_floor::Real = 0.0)
    reasons = Dict("non-finite R, error or K" => 0, "zero or negative error" => 0, "zero or negative ρa" => 0)
    keep = falses(ndata(d))
    sigma = fill(NaN, ndata(d))
    for i in 1:ndata(d)
        R, e, K = d.resistance[i], d.resistance_error[i], d.geometric_factor[i]
        if !(isfinite(R) && isfinite(e) && isfinite(K))
            reasons["non-finite R, error or K"] += 1
        elseif e <= 0 && error_floor <= 0
            reasons["zero or negative error"] += 1
        elseif K * R <= 0
            reasons["zero or negative ρa"] += 1
        else
            keep[i] = true
            sigma[i] = max(abs(e / R), error_floor)
        end
    end
    (; keep, sigma, reasons)
end

# chi2 over the data, rms = sqrt(chi2 / count)
Base.@kwdef struct FitSummary2D
    chi2::Float64
    rms::Float64
    count::Int
end

"""
    chi2_rms2d(observed, predicted; error_floor=0.0) -> FitSummary2D

Misfit of log apparent resistivity, (log ρa_pred − log K R_obs)/σ with σ = σ_R/|R_obs|
(at least `error_floor`), over the observed data valid for inversion. Also takes two
data file paths.
"""
function chi2_rms2d(observed, predicted; error_floor::Real = 0.0)
    ndata(observed) == ndata(predicted) || error("observed and predicted surveys do not match")
    v = ert2d_valid_data(observed; error_floor)
    χ², count = 0.0, 0
    for i in findall(v.keep)
        p = predicted.rhoa[i]
        (isfinite(p) && p > 0) || continue
        χ² += (log(p / (observed.geometric_factor[i] * observed.resistance[i])) / v.sigma[i])^2
        count += 1
    end
    FitSummary2D(chi2 = χ², rms = count > 0 ? sqrt(χ² / count) : NaN, count = count)
end

chi2_rms2d(observed::AbstractString, predicted::AbstractString; kwargs...) =
    chi2_rms2d(load_data2d(observed), load_data2d(predicted); kwargs...)

"""
    recentre_data2d(data) -> DataFile2D

Move the local origin to the middle of the electrode line (same azimuth), so that a
model centred on y = 0 is centred on the electrodes.
"""
function recentre_data2d(d::DataFile2D)
    mid = (minimum(d.electrodes) + maximum(d.electrodes)) / 2
    abs(mid) < 1e-6 && return d
    origin = if length(d.latitudes) == nelectrodes(d)
        lat, lon = ert2d_latlon(d.origin, d.azimuth, [0.0], [mid])
        [lat[1], lon[1]]
    else
        d.origin
    end
    _ert2d_with(d; electrodes = d.electrodes .- mid, origin = origin)
end
