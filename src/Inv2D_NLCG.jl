# 2D ERT nonlinear conjugate-gradient inversion
# Author: @pankajkmishra
# Preconditioned Polak–Ribière+ NLCG direction for the Inv2D driver, as Inv2D_NLCG.jl of MTGeophysics.jl
# Gradient only (from the pole fields), G is formed only for the automatic β and the final coverage

using LinearAlgebra
using SparseArrays

#---------- configuration ----------

"""
    NLCG2DConfig(; restart=30, precondition=true, epsilon=1e-2)

Nonlinear conjugate-gradient algorithm for `Invert2D`.
- `restart`: reset to preconditioned steepest descent every `restart` iterations
  (and whenever β or the IRLS weights change); β cools only once the rms drops less
  than 5% in an iteration
- `precondition`: apply `P = (βRᵗWm²R + εI)⁻¹` to the gradient, the model-covariance
  preconditioner; smooths updates the same way the regularization does
- `epsilon`: `ε` relative to the mean diagonal of `βRᵗWm²R`

Each iteration costs one gradient plus the line-search forward solves, so NLCG needs
many more iterations than Gauss–Newton (typically 100-200 for a 2D ERT line) but each
one is cheaper and memory stays linear in the number of cells: use it when the active
model is too large for GN.
"""
Base.@kwdef struct NLCG2DConfig <: AbstractInversion2D
    restart::Int = 30
    precondition::Bool = true
    epsilon::Float64 = 1e-2
end

const NLCG2DResult = Inv2DResult{NLCG2DConfig}

inv2d_tag(::NLCG2DConfig) = "nlcg"

# β cools only once the rms stalls at the current β, so the conjugate directions can build up
inv2d_cool_always(::NLCG2DConfig) = false

function inv2d_validate(alg::NLCG2DConfig)
    alg.restart > 0 || throw(ArgumentError("restart must be positive"))
    isfinite(alg.epsilon) && alg.epsilon > 0 || throw(ArgumentError("epsilon must be finite and positive"))
    nothing
end

#---------- interface ----------

mutable struct NLCG2DWork
    P::Any                      # preconditioner factor, nothing = identity
    g::Vector{Float64}          # this iteration: gradient,
    h::Vector{Float64}          # preconditioned gradient,
    d::Vector{Float64}          # search direction
    g_prev::Vector{Float64}     # last accepted iteration
    h_prev::Vector{Float64}
    d_prev::Vector{Float64}
    scale::Float64              # step length that the last accepted step implied
    k::Int                      # iterations since the last restart
    steepest::Bool              # current d is -h
end

function _nlcg2d_preconditioner(alg::NLCG2DConfig, problem)
    (alg.precondition && problem.beta > 0) || return nothing
    WR = problem.wm .* problem.R_active
    A = problem.beta * (WR' * WR)
    ε = alg.epsilon * max(mean(diag(A)), eps())
    cholesky(Symmetric(A + ε * I))
end

function inv2d_init(alg::NLCG2DConfig, problem, state)
    e = Float64[]
    NLCG2DWork(_nlcg2d_preconditioner(alg, problem), e, e, e, e, e, e, 0.0, 0, true)
end

# a new objective: steepest descent again, but the last step length carries over
function inv2d_restart!(alg::NLCG2DConfig, work::NLCG2DWork, problem)
    work.P = _nlcg2d_preconditioner(alg, problem)
    work.k = 0
    nothing
end

inv2d_info(::NLCG2DConfig, work::NLCG2DWork) = (; restart = work.steepest ? 1 : 0)

inv2d_prepare!(alg::NLCG2DConfig, work::NLCG2DWork, problem, state) = inv2d_gradient(problem, state)

function inv2d_direction(alg::NLCG2DConfig, work::NLCG2DWork, problem, state, gradient)
    g = gradient
    h = work.P === nothing ? copy(g) : work.P \ g

    # polak-ribière+; restart on schedule or when the result is not a descent direction
    d = -h
    work.steepest = true
    if work.k > 0 && work.k % alg.restart != 0
        β = max(0.0, dot(g, h - work.h_prev) / dot(work.g_prev, work.h_prev))
        candidate = -h + β * work.d_prev
        if β > 0 && dot(candidate, g) < 0
            d = candidate
            work.steepest = false
        end
    end
    work.g, work.h, work.d = g, h, d

    # initial step length: the first iteration takes max_step, later ones carry over the last step's slope,
    # doubled so a step that was cut short can grow back; the driver only caps and backtracks
    s = if work.scale > 0 && !isempty(work.d_prev)
        2 * work.scale * dot(work.g_prev, work.d_prev) / dot(g, d)
    else
        problem.options.max_step / max(norm(d, Inf), eps())
    end
    s = isfinite(s) && s > 0 ? s : problem.options.max_step / max(norm(d, Inf), eps())
    s .* d
end

# a failed conjugate direction gets one retry as steepest descent, then the driver cools β or stops
function inv2d_reject!(alg::NLCG2DConfig, work::NLCG2DWork)
    work.steepest && return false
    work.k = 0
    work.scale = 0.0
    true
end

function inv2d_accept!(alg::NLCG2DConfig, work::NLCG2DWork, problem, old, new)
    delta = new.m - old.m
    work.scale = max(dot(delta, work.d) / dot(work.d, work.d), 0.0)
    work.g_prev, work.h_prev, work.d_prev = work.g, work.h, work.d
    work.k = work.steepest ? 1 : work.k + 1
    nothing
end

#---------- entry points ----------

"""
    NLCG2D(mesh, initial_resistivity, observed; config=NLCG2DConfig(), kwargs...)

NLCG shorthand for `Invert2D(...; algorithm=config)`.
"""
NLCG2D(mesh::ERT2DMesh, initial_resistivity::AbstractMatrix{<:Real}, observed::DataFile2D;
       config::NLCG2DConfig = NLCG2DConfig(), kwargs...) =
    Invert2D(mesh, initial_resistivity, observed; algorithm = config, kwargs...)
