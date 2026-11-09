# 2D control files
# Author: @pankajkmishra
# fwd.ctrl, inv.ctrl (GN, NLCG) and the model covariance of the 2D workflows,
# as Control2D.jl of MTGeophysics.jl
# Control files are "Key : value" lines; unknown, duplicate and missing required keys are errors

using Printf

#---------- key-value parsing ----------

_ctrl_key(s) = lowercase(join(split(strip(s)), " "))

function _ctrl_parse_bool(s)
    v = lowercase(strip(s))
    v in ("yes", "true", "1") && return true
    v in ("no", "false", "0") && return false
    throw(ArgumentError("expected yes or no, got '$s'"))
end

function _ctrl_parse_pair(s)
    t = split(strip(s))
    length(t) == 2 || throw(ArgumentError("expected two numbers, got '$s'"))
    (parse(Float64, t[1]), parse(Float64, t[2]))
end

# spec: display key => (field, parser, required)
function _read_ctrl(path::AbstractString, spec, kind::AbstractString)
    isfile(path) || error("$kind file not found: $path")
    lookup = Dict(_ctrl_key(k) => (k, v...) for (k, v) in spec)
    values = Dict{Symbol, Any}()
    for (n, raw) in enumerate(eachline(path))
        line = strip(first(split(raw, '#')))
        isempty(line) && continue
        occursin(':', line) || error("$path:$n: expected 'Key : value', got '$line'")
        key, value = strip.(split(line, ':'; limit = 2))
        entry = get(lookup, _ctrl_key(key), nothing)
        entry === nothing && error("$path:$n: unknown $kind key '$key'")
        _, field, parser, _ = entry
        haskey(values, field) && error("$path:$n: duplicate $kind key '$key'")
        values[field] = try
            parser(value)
        catch err
            error("$path:$n: bad value for '$key': $(sprint(showerror, err))")
        end
    end
    for (k, (field, _, required)) in spec
        required && !haskey(values, field) && error("$path: missing required $kind key '$k'")
    end
    values
end

function _write_ctrl(io, rows)
    width = maximum(length ∘ first, rows)
    foreach(((k, v),) -> println(io, rpad(k, width), " : ", v), rows)
end

_ctrl_float(s) = parse(Float64, s)
_ctrl_int(s) = parse(Int, s)
_ctrl_yesno(b::Bool) = b ? "yes" : "no"
_ctrl_algorithm(s) = (a = Symbol(lowercase(strip(s))); a in (:gn, :nlcg) ? a :
                      throw(ArgumentError("algorithm must be GN or NLCG")))
_ctrl_norm(s) = (v = Symbol(uppercase(strip(s))); v in (:L1, :L2) ? v : throw(ArgumentError("norm must be L1 or L2")))
_ctrl_lambda(s) = lowercase(strip(s)) == "auto" ? NaN : parse(Float64, s)
_ctrl_percent(s) = parse(Float64, s) / 100
_ctrl_g(x) = @sprintf("%.6g", x)

#---------- fwd.ctrl ----------

"""
    FwdCtrl2D

Forward control, read from `fwd.ctrl`; every key has a default.
- `wavenumbers`: number of cross-profile wavenumbers of the 2.5D transform (default 16)
- `write_frechet`: also write the sensitivity next to the predicted data
- `refine`: forward cells per electrode spacing at the electrodes (default 8; `MakeMesh2D`
  writes 16 under topography): the solve runs on the model mesh with its cells split,
  finest at the electrodes and the ground (`ert2d_forward_mesh`), where it also cuts a fine
  staircase; the model cells keep one resistivity
- `growth`: growth of the forward cells with the distance from the electrodes and the
  ground, as a fraction of that distance (default 0.25)
- `k_refine`: forward cells per spacing for the geometric factors K over topography
  (`MakeMesh2D`, default 32)
"""
Base.@kwdef struct FwdCtrl2D
    wavenumbers::Int = 16
    refine::Int = 8
    growth::Float64 = 0.25
    write_frechet::Bool = false
    k_refine::Int = 32
end

const _FWD_CTRL_SPEC = (
    "Wavenumbers"                   => (:wavenumbers, _ctrl_int, false),
    "Forward refinement"            => (:refine, _ctrl_int, false),
    "Refinement growth"             => (:growth, _ctrl_float, false),
    "Write Frechet derivative"      => (:write_frechet, _ctrl_parse_bool, false),
    "Geometric factor refinement"   => (:k_refine, _ctrl_int, false),
)

function _validate_ctrl(c::FwdCtrl2D)
    4 <= c.wavenumbers <= 40 || throw(ArgumentError("wavenumbers must lie in 4:40"))
    1 <= c.refine <= 32 || throw(ArgumentError("forward refinement (cells per electrode spacing) must lie in 1:32"))
    0 <= c.growth <= 2 || throw(ArgumentError("refinement growth must lie in [0, 2]"))
    c.k_refine >= 1 || throw(ArgumentError("geometric factor refinement must be at least 1"))
    c
end

"""
    ReadFwdCtrl2D(path) -> FwdCtrl2D

Read a 2D forward control file.
"""
ReadFwdCtrl2D(path::AbstractString) = _validate_ctrl(FwdCtrl2D(; _read_ctrl(path, _FWD_CTRL_SPEC, "fwd.ctrl")...))

"""
    WriteFwdCtrl2D(path, ctrl::FwdCtrl2D) -> path

Write a 2D forward control file.
"""
function WriteFwdCtrl2D(path::AbstractString, c::FwdCtrl2D)
    _validate_ctrl(c)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        _write_ctrl(io, [
            ("Wavenumbers", string(c.wavenumbers)),
            ("Forward refinement", string(c.refine)),
            ("Refinement growth", _ctrl_g(c.growth)),
            ("Write Frechet derivative", _ctrl_yesno(c.write_frechet)),
            ("Geometric factor refinement", string(c.k_refine)),
        ])
    end
    String(path)
end

#---------- inv.ctrl ----------

"""
    InvCtrl2D

Deterministic inversion control, read from `inv.ctrl`. The lambda, rms and iteration
keys follow ModEM's control file; the rest have defaults.
- `algorithm`: `:gn` or `:nlcg`
- `lambda`: starting regularization weight β; NaN (`auto`) balances the data and
  model terms at the start model
- `lambda_cooling`: β is multiplied by this after every iteration that ends above the
  target rms (1 = fixed β), down to `min_lambda_ratio` × the starting β
- `target_rms`, `max_iter`: stopping controls
- `relax_lambda`: when an iteration lands below 0.9 × the target rms, raise β again and
  keep the smoothest model that still fits (default yes)
- `error_floor`: minimum relative error of the resistances (fraction; the file
  key is in percent)
- `data_norm`, `model_norm`: `:L2`, or `:L1` (robust, by iteratively reweighted least
  squares): L1 data resist outliers, an L1 model gives blocky sections
- `log_bounds`: log10 ρ box the model is held in
- `max_step`, `max_linesearch`: largest log10 change per iteration and halvings per search
- `smallness`, `smooth_y`, `smooth_z`: weights of the regularization terms
- `gn_damping`, `nlcg_restart`, `nlcg_precondition`: per-algorithm settings
"""
Base.@kwdef struct InvCtrl2D
    algorithm::Symbol
    lambda::Float64 = NaN
    target_rms::Float64
    max_iter::Int
    relax_lambda::Bool = true
    lambda_cooling::Float64 = 0.6
    min_lambda_ratio::Float64 = 1e-3
    error_floor::Float64 = 0.0
    data_norm::Symbol = :L2
    model_norm::Symbol = :L2
    log_bounds::Tuple{Float64, Float64} = (-1.0, 6.0)
    max_step::Float64 = 0.5
    max_linesearch::Int = 12
    smallness::Float64 = 0.01
    smooth_y::Float64 = 1.0
    smooth_z::Float64 = 1.0
    gn_damping::Float64 = 1e-2
    nlcg_restart::Int = 30
    nlcg_precondition::Bool = true
end

const _INV_CTRL_SPEC = (
    "Algorithm"                          => (:algorithm, _ctrl_algorithm, true),
    "Initial damping factor lambda"      => (:lambda, _ctrl_lambda, false),
    "Lambda cooling factor"              => (:lambda_cooling, _ctrl_float, false),
    "Minimum lambda ratio"               => (:min_lambda_ratio, _ctrl_float, false),
    "Exit search when rms is less than"  => (:target_rms, _ctrl_float, true),
    "Maximum number of iterations"       => (:max_iter, _ctrl_int, true),
    "Relax lambda at the target"         => (:relax_lambda, _ctrl_parse_bool, false),
    "Error floor (%)"                    => (:error_floor, _ctrl_percent, false),
    "Data norm"                          => (:data_norm, _ctrl_norm, false),
    "Model norm"                         => (:model_norm, _ctrl_norm, false),
    "Log10 resistivity bounds"           => (:log_bounds, _ctrl_parse_pair, false),
    "Max log10 step"                     => (:max_step, _ctrl_float, false),
    "Max line search steps"              => (:max_linesearch, _ctrl_int, false),
    "Smallness weight"                   => (:smallness, _ctrl_float, false),
    "Smoothing weight y"                 => (:smooth_y, _ctrl_float, false),
    "Smoothing weight z"                 => (:smooth_z, _ctrl_float, false),
    "GN damping"                         => (:gn_damping, _ctrl_float, false),
    "NLCG restart"                       => (:nlcg_restart, _ctrl_int, false),
    "NLCG precondition"                  => (:nlcg_precondition, _ctrl_parse_bool, false),
)

function _validate_ctrl(c::InvCtrl2D)
    c.algorithm in (:gn, :nlcg) || throw(ArgumentError("algorithm must be GN or NLCG"))
    isnan(c.lambda) || c.lambda >= 0 || throw(ArgumentError("lambda must be auto or nonnegative"))
    0 < c.lambda_cooling <= 1 || throw(ArgumentError("lambda cooling factor must lie in (0, 1]"))
    0 < c.min_lambda_ratio <= 1 || throw(ArgumentError("minimum lambda ratio must lie in (0, 1]"))
    c.max_iter >= 0 || throw(ArgumentError("maximum number of iterations must be nonnegative"))
    0 <= c.error_floor < 1 || throw(ArgumentError("error floor must lie in [0, 100) %"))
    c.log_bounds[1] < c.log_bounds[2] || throw(ArgumentError("log10 bounds must be increasing"))
    c
end

"""
    ReadInvCtrl2D(path) -> InvCtrl2D

Read a GN or NLCG inversion control file.
"""
ReadInvCtrl2D(path::AbstractString) = _validate_ctrl(InvCtrl2D(; _read_ctrl(path, _INV_CTRL_SPEC, "inv.ctrl")...))

"""
    WriteInvCtrl2D(path, ctrl::InvCtrl2D) -> path

Write a GN or NLCG inversion control file with the shared keys and those of its
algorithm.
"""
function WriteInvCtrl2D(path::AbstractString, c::InvCtrl2D)
    _validate_ctrl(c)
    mkpath(dirname(abspath(path)))
    g = _ctrl_g
    rows = [
        ("Algorithm", uppercase(string(c.algorithm))),
        ("Initial damping factor lambda", isnan(c.lambda) ? "auto" : g(c.lambda)),
        ("Lambda cooling factor", g(c.lambda_cooling)),
        ("Minimum lambda ratio", g(c.min_lambda_ratio)),
        ("Exit search when rms is less than", g(c.target_rms)),
        ("Maximum number of iterations", string(c.max_iter)),
        ("Relax lambda at the target", _ctrl_yesno(c.relax_lambda)),
        ("Error floor (%)", g(100 * c.error_floor)),
        ("Data norm", string(c.data_norm)),
        ("Model norm", string(c.model_norm)),
        ("Log10 resistivity bounds", "$(g(c.log_bounds[1])) $(g(c.log_bounds[2]))"),
        ("Max log10 step", g(c.max_step)),
        ("Max line search steps", string(c.max_linesearch)),
        ("Smallness weight", g(c.smallness)),
        ("Smoothing weight y", g(c.smooth_y)),
        ("Smoothing weight z", g(c.smooth_z)),
    ]
    c.algorithm == :gn ? push!(rows, ("GN damping", g(c.gn_damping))) :
        append!(rows, [("NLCG restart", string(c.nlcg_restart)),
                       ("NLCG precondition", _ctrl_yesno(c.nlcg_precondition))])
    open(io -> _write_ctrl(io, rows), path, "w")
    String(path)
end

#---------- model covariance ----------

"""
    Cov2D

Model covariance, read from the covariance file. The smoothing values keep the
ModEM layout; the inversion reads only the mask.
- `sy`: horizontal smoothing, one value per layer, top to bottom
- `sz`: vertical smoothing
- `n_smooth`: number of smoothing passes
- `exceptions`: `(class_a, class_b, smoothing)` rules between mask classes
- `mask`: `(nz, ny)` cells, top row first; 0 = fixed, 1 = free,
  9 = water (fixed, no regularization), other values = exception classes, inverted like 1
"""
Base.@kwdef struct Cov2D
    sy::Vector{Float64}
    sz::Float64
    n_smooth::Int
    exceptions::Vector{Tuple{Int, Int, Float64}} = Tuple{Int, Int, Float64}[]
    mask::Matrix{Int}
end

function _validate_ctrl(c::Cov2D)
    nz, ny = size(c.mask)
    length(c.sy) == nz || throw(ArgumentError("cov: $(length(c.sy)) y smoothing values for $nz layers"))
    all(s -> 0 <= s <= 1, c.sy) && 0 <= c.sz <= 1 || throw(ArgumentError("cov: smoothing must lie in [0, 1]"))
    c.n_smooth >= 0 || throw(ArgumentError("cov: number of smoothing passes must be nonnegative"))
    all(>=(0), c.mask) || throw(ArgumentError("cov: mask values must be nonnegative"))
    c
end

"""
    Cov2D(nz, ny; smoothing=0.3, n_smooth=1)

Uniform covariance for an `nz × ny` grid, every cell free.
"""
Cov2D(nz::Integer, ny::Integer; smoothing::Real = 0.3, n_smooth::Integer = 1) =
    Cov2D(sy = fill(Float64(smoothing), nz), sz = Float64(smoothing), n_smooth = n_smooth, mask = ones(Int, nz, ny))

"""
    ReadCov2D(path) -> Cov2D

Read a 2D model covariance file: `ny nz`, `nz` y smoothings, the z smoothing, the number
of passes, the number of exception rules and the rules, then `nz` mask rows of `ny`
integers.
"""
function ReadCov2D(path::AbstractString)
    isfile(path) || error("covariance file not found: $path")
    tokens = String[]
    for raw in eachline(path)
        append!(tokens, split(first(split(raw, '#'))))
    end
    k = 0
    next() = (k += 1; k <= length(tokens) || error("$path: covariance file ends early"); tokens[k])
    ny, nz = parse(Int, next()), parse(Int, next())
    sy = [parse(Float64, next()) for _ in 1:nz]
    sz = parse(Float64, next())
    n_smooth = parse(Int, next())
    exceptions = [(parse(Int, next()), parse(Int, next()), parse(Float64, next())) for _ in 1:parse(Int, next())]
    mask = Matrix{Int}(undef, nz, ny)
    for iz in 1:nz, iy in 1:ny
        mask[iz, iy] = parse(Int, next())
    end
    k == length(tokens) || error("$path: $(length(tokens) - k) extra values after the mask")
    _validate_ctrl(Cov2D(; sy, sz, n_smooth, exceptions, mask))
end

"""
    WriteCov2D(path, cov::Cov2D) -> path

Write a 2D model covariance file.
"""
function WriteCov2D(path::AbstractString, c::Cov2D)
    _validate_ctrl(c)
    nz, ny = size(c.mask)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        @printf(io, "%d %d\n", ny, nz)
        _write_vector_lines(io, c.sy)
        @printf(io, "%.6g\n", c.sz)
        println(io, c.n_smooth)
        println(io, length(c.exceptions))
        for (a, b, s) in c.exceptions
            @printf(io, "%d %d %.6g\n", a, b, s)
        end
        for iz in 1:nz
            println(io, join(c.mask[iz, :], ' '))
        end
    end
    String(path)
end
