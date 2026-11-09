# 2D ERT Gauss–Newton inversion
# Author: @pankajkmishra
# Damped (Levenberg–Marquardt) Gauss–Newton direction for the Inv2D driver, as Inv2D_GN.jl of MTGeophysics.jl
# Explicit sensitivity G; the step solves the normal equations densely for small models and by
# Jacobi-preconditioned conjugate gradients (G products, no normal matrix) above `dense_limit` active cells

using LinearAlgebra
using SparseArrays

#---------- configuration ----------

"""
    GaussNewton2DConfig(; damping=1e-2, max_damping_trials=6, dense_limit=4000,
                        cg_tolerance=1e-3, cg_max_iter=200)

Gauss–Newton algorithm for `Invert2D`.
- `damping`: initial Levenberg–Marquardt damping, relative to the diagonal of
  `GᵗWd²G + βRᵗWm²R`; divided by 3 after an accepted step, multiplied by 10 after a
  failed search
- `max_damping_trials`: failed searches per iteration before the driver cools β or stops
- `dense_limit`: up to this many active cells the normal equations are factorised,
  above it they are solved by conjugate gradients
- `cg_tolerance`, `cg_max_iter`: relative residual and iteration cap of the CG solve
"""
Base.@kwdef struct GaussNewton2DConfig <: AbstractInversion2D
    damping::Float64 = 1e-2
    max_damping_trials::Int = 6
    dense_limit::Int = 4000
    cg_tolerance::Float64 = 1e-3
    cg_max_iter::Int = 200
end

const GaussNewton2DResult = Inv2DResult{GaussNewton2DConfig}

inv2d_tag(::GaussNewton2DConfig) = "gn"

function inv2d_validate(alg::GaussNewton2DConfig)
    isfinite(alg.damping) && alg.damping > 0 || throw(ArgumentError("damping must be finite and positive"))
    alg.max_damping_trials > 0 || throw(ArgumentError("max_damping_trials must be positive"))
    alg.dense_limit >= 0 && alg.cg_max_iter > 0 && 0 < alg.cg_tolerance < 1 ||
        throw(ArgumentError("need dense_limit ≥ 0, cg_max_iter > 0 and 0 < cg_tolerance < 1"))
    nothing
end

#---------- interface ----------

mutable struct GaussNewton2DWork
    damping::Float64
    trials::Int
    G::Matrix{Float64}                          # Wd G
    WR::SparseMatrixCSC{Float64, Int}           # Wm R
    beta::Float64
    H::Matrix{Float64}                          # GᵗG + βRᵗR, dense mode only
    diagonal::Vector{Float64}                   # its diagonal
end

inv2d_init(alg::GaussNewton2DConfig, problem, state) =
    GaussNewton2DWork(alg.damping, 0, zeros(0, 0), spzeros(0, 0), 0.0, zeros(0, 0), Float64[])

inv2d_info(::GaussNewton2DConfig, work::GaussNewton2DWork) = (; damping = work.damping)

function inv2d_prepare!(alg::GaussNewton2DConfig, work::GaussNewton2DWork, problem, state)
    G = problem.wd .* inv2d_frechet(problem, state)
    WR = problem.wm .* problem.R_active
    work.G, work.WR, work.beta = G, WR, problem.beta
    if size(G, 2) <= alg.dense_limit
        work.H = G' * G + problem.beta * Matrix(WR' * WR)
        work.diagonal = diag(work.H)
    else
        work.H = zeros(0, 0)
        work.diagonal = vec(sum(abs2, G; dims = 1)) .+ problem.beta .* vec(sum(abs2, WR; dims = 1))
    end
    work.diagonal .= max.(work.diagonal, 1e-12 * max(maximum(work.diagonal), eps()))
    work.trials = 0
    G' * (problem.wd .* state.r) + problem.beta * (problem.R_active' * (problem.wm .^ 2 .* state.reg))
end

# (GᵗG + βRᵗR + λ diag) d = -g; the same step as the damped augmented least squares
function inv2d_direction(alg::GaussNewton2DConfig, work::GaussNewton2DWork, problem, state, gradient)
    λ = work.damping
    if !isempty(work.H)
        A = copy(work.H)
        for i in axes(A, 1)
            A[i, i] += λ * work.diagonal[i]
        end
        return -(cholesky!(Symmetric(A)) \ gradient)
    end
    G, WR, β = work.G, work.WR, work.beta
    D = (1 + λ) .* work.diagonal
    apply(x) = G' * (G * x) + β * (WR' * (WR * x)) + λ .* work.diagonal .* x
    -_inv2d_pcg(apply, gradient, D; tol = alg.cg_tolerance, maxiter = alg.cg_max_iter)
end

# Jacobi-preconditioned conjugate gradients for a symmetric positive definite operator
function _inv2d_pcg(apply, b::AbstractVector, D::AbstractVector; tol::Real = 1e-3, maxiter::Int = 200)
    x = zeros(length(b))
    r = copy(b)
    z = r ./ D
    p = copy(z)
    rz = dot(r, z)
    nb = norm(b)
    nb == 0 && return x
    for _ in 1:maxiter
        Ap = apply(p)
        α = rz / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        norm(r) <= tol * nb && break
        z = r ./ D
        rz_new = dot(r, z)
        p .= z .+ (rz_new / rz) .* p
        rz = rz_new
    end
    x
end

function inv2d_reject!(alg::GaussNewton2DConfig, work::GaussNewton2DWork)
    work.damping *= 10
    work.trials += 1
    work.trials < alg.max_damping_trials
end

function inv2d_accept!(alg::GaussNewton2DConfig, work::GaussNewton2DWork, problem, old, new)
    work.damping = max(work.damping / 3, 1e-6)
    nothing
end

#---------- entry points ----------

"""
    GaussNewton2D(mesh, initial_resistivity, observed; config=GaussNewton2DConfig(), kwargs...)

Gauss–Newton shorthand for `Invert2D(...; algorithm=config)`.
"""
GaussNewton2D(mesh::ERT2DMesh, initial_resistivity::AbstractMatrix{<:Real}, observed::DataFile2D;
              config::GaussNewton2DConfig = GaussNewton2DConfig(), kwargs...) =
    Invert2D(mesh, initial_resistivity, observed; algorithm = config, kwargs...)
