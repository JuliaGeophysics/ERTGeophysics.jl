# 3D control files
# Author: @pankajkmishra
# fwd.ctrl and inv.ctrl of the 3D workflows, "Key : value" lines as in 2D; the inversion is Gauss–Newton

using Printf

#---------- fwd.ctrl ----------

"""
    FwdCtrl3D

3D forward control, read from `fwd.ctrl`; every key has a default.
- `refine`: forward cells per electrode spacing at the electrodes (default 4): the solve
  runs on the model mesh with its cells split, finest at the electrodes and the ground
  (`ert3d_forward_mesh`)
- `growth`: growth of the forward cells with the distance from the electrodes and the
  ground, as a fraction of that distance (default 0.5; 3D cost, the same accuracy as 0.25)
- `k_refine`: forward cells per spacing for the geometric factors K over topography
  (`MakeMesh3D`, default 8)
- `write_frechet`: also write the sensitivity next to the predicted data
"""
Base.@kwdef struct FwdCtrl3D
    refine::Int = 4
    growth::Float64 = 0.5
    k_refine::Int = 8
    write_frechet::Bool = false
end

const _FWD3D_CTRL_SPEC = (
    "Forward refinement"          => (:refine, _ctrl_int, false),
    "Refinement growth"           => (:growth, _ctrl_float, false),
    "Geometric factor refinement" => (:k_refine, _ctrl_int, false),
    "Write Frechet derivative"    => (:write_frechet, _ctrl_parse_bool, false),
)

function _validate_ctrl(c::FwdCtrl3D)
    1 <= c.refine <= 16 || throw(ArgumentError("forward refinement (cells per electrode spacing) must lie in 1:16"))
    0 <= c.growth <= 2 || throw(ArgumentError("refinement growth must lie in [0, 2]"))
    c.k_refine >= 1 || throw(ArgumentError("geometric factor refinement must be at least 1"))
    c
end

"""
    ReadFwdCtrl3D(path) -> FwdCtrl3D
"""
ReadFwdCtrl3D(path::AbstractString) = _validate_ctrl(FwdCtrl3D(; _read_ctrl(path, _FWD3D_CTRL_SPEC, "fwd.ctrl")...))

"""
    WriteFwdCtrl3D(path, ctrl::FwdCtrl3D) -> path
"""
function WriteFwdCtrl3D(path::AbstractString, c::FwdCtrl3D)
    _validate_ctrl(c)
    mkpath(dirname(abspath(path)))
    rows = [("Forward refinement", string(c.refine)), ("Refinement growth", _ctrl_g(c.growth)),
            ("Geometric factor refinement", string(c.k_refine)), ("Write Frechet derivative", _ctrl_yesno(c.write_frechet))]
    open(io -> _write_ctrl(io, rows), path, "w")
    String(path)
end

#---------- inv.ctrl ----------

"""
    InvCtrl3D

3D Gauss–Newton inversion control, read from `inv.ctrl`. The keys are those of the 2D
`InvCtrl2D` with smoothing in x, y and z, `regularization` (`index`, as ModEM, or
`volume`; see `Inv3DOptions`), and no algorithm choice (always GN).
"""
Base.@kwdef struct InvCtrl3D
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
    smooth_x::Float64 = 1.0
    smooth_y::Float64 = 1.0
    smooth_z::Float64 = 1.0
    regularization::Symbol = :index
    gn_damping::Float64 = 1e-2
end

_ctrl_regularization(s) = (v = Symbol(lowercase(strip(s))); v in (:index, :volume) ? v :
                           throw(ArgumentError("regularization must be index or volume")))
_ctrl_gn3d(s) = uppercase(strip(s)) == "GN" ? :gn : throw(ArgumentError("the 3D inversion is Gauss–Newton only (Algorithm : GN)"))

const _INV3D_CTRL_SPEC = (
    "Algorithm"                          => (:algorithm, _ctrl_gn3d, false),
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
    "Smoothing weight x"                 => (:smooth_x, _ctrl_float, false),
    "Smoothing weight y"                 => (:smooth_y, _ctrl_float, false),
    "Smoothing weight z"                 => (:smooth_z, _ctrl_float, false),
    "Regularization"                     => (:regularization, _ctrl_regularization, false),
    "GN damping"                         => (:gn_damping, _ctrl_float, false),
)

function _validate_ctrl(c::InvCtrl3D)
    isnan(c.lambda) || c.lambda >= 0 || throw(ArgumentError("lambda must be auto or nonnegative"))
    0 < c.lambda_cooling <= 1 || throw(ArgumentError("lambda cooling factor must lie in (0, 1]"))
    0 < c.min_lambda_ratio <= 1 || throw(ArgumentError("minimum lambda ratio must lie in (0, 1]"))
    c.max_iter >= 0 || throw(ArgumentError("maximum number of iterations must be nonnegative"))
    0 <= c.error_floor < 1 || throw(ArgumentError("error floor must lie in [0, 100) %"))
    c.log_bounds[1] < c.log_bounds[2] || throw(ArgumentError("log10 bounds must be increasing"))
    c
end

"""
    ReadInvCtrl3D(path) -> InvCtrl3D
"""
function ReadInvCtrl3D(path::AbstractString)
    v = _read_ctrl(path, _INV3D_CTRL_SPEC, "inv.ctrl")
    delete!(v, :algorithm)
    _validate_ctrl(InvCtrl3D(; v...))
end

"""
    WriteInvCtrl3D(path, ctrl::InvCtrl3D) -> path
"""
function WriteInvCtrl3D(path::AbstractString, c::InvCtrl3D)
    _validate_ctrl(c)
    mkpath(dirname(abspath(path)))
    g = _ctrl_g
    rows = [
        ("Algorithm", "GN"),
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
        ("Smoothing weight x", g(c.smooth_x)),
        ("Smoothing weight y", g(c.smooth_y)),
        ("Smoothing weight z", g(c.smooth_z)),
        ("Regularization", string(c.regularization)),
        ("GN damping", g(c.gn_damping)),
    ]
    open(io -> _write_ctrl(io, rows), path, "w")
    String(path)
end
