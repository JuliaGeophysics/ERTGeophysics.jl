# 2D ERT deterministic inversion framework
# Author: @pankajkmishra
# Algorithm-agnostic problem setup, objective, regularization, line search, and driver loop, as Inv2D.jl of
# MTGeophysics.jl; each algorithm lives in its own Inv2D_<Name>.jl and plugs in through the interface below
# The data are log apparent resistivities; β cools towards the target misfit, L1 data or model norms run by
# iteratively reweighted least squares, the model stays inside log10 bounds

using Dates
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics

#---------- algorithm interface ----------
#
# an algorithm is a config struct `A <: AbstractInversion2D` plus these methods:
#
#   inv2d_init(alg, problem, state)                    -> work   (mutable scratch)
#   inv2d_prepare!(alg, work, problem, state)          -> gradient, once per iteration
#   inv2d_direction(alg, work, problem, state, grad)   -> search direction in log10 ρ
#   inv2d_reject!(alg, work)                           -> true = retry with a new direction
#   inv2d_accept!(alg, work, problem, old, new)        -> nothing
#   inv2d_restart!(alg, work, problem)                 -> nothing, after β or the IRLS weights change
#   inv2d_info(alg, work)                              -> NamedTuple added to history
#   inv2d_tag(alg)                                     -> short label for the log, e.g. "gn"
#   inv2d_cool_always(alg)                             -> true: cool β after every iteration above the target;
#                                                         false: only once the rms drops less than 5% per iteration
#   inv2d_validate(alg)                                -> throws on bad settings
#
# the driver owns stopping tests, cooling, reweighting, step capping, bounds and the Armijo line search

"""
    AbstractInversion2D

Supertype of 2D deterministic inversion algorithms. See `Inv2D.jl` for the interface.
"""
abstract type AbstractInversion2D end

inv2d_reject!(::AbstractInversion2D, work) = false
inv2d_accept!(::AbstractInversion2D, work, problem, old, new) = nothing
inv2d_restart!(::AbstractInversion2D, work, problem) = nothing
inv2d_info(::AbstractInversion2D, work) = (;)
inv2d_validate(::AbstractInversion2D) = nothing
inv2d_cool_always(::AbstractInversion2D) = true

#---------- options and result ----------

"""
    Inv2DOptions(; kwargs...)

Algorithm-independent inversion controls.
- `max_iter`: maximum accepted iterations; 0 evaluates the start model only
- `beta`: starting regularization weight; NaN = automatic, the ratio of the data and
  model terms of the normal equations for a random model change
- `beta_cooling`, `min_beta_ratio`: β is multiplied by `beta_cooling` after every
  iteration that ends above the target, down to `min_beta_ratio` × its start
- `smallness`, `smooth_y`, `smooth_z`: weights of the regularization terms
- `data_norm`, `model_norm`: `:L2` or `:L1` (IRLS, Ekblom weights)
- `error_floor`: minimum relative error of the resistances (fraction)
- `log_bounds`: log10 ρ box of the model
- `max_step`: cap on the largest log10 change per iteration
- `max_linesearch`: backtracking halvings per direction
- `target_rms`: stop at this rms; 0 = off
- `relax_beta`: once an iteration lands below 0.9 × `target_rms`, raise β again
  (× `relax_factor`, at most `max_relax` times) and keep the smoothest model that still
  fits, so the result is not over-fitted (the second phase of Occam's inversion)
- `wavenumbers`, `refine`, `growth`: forward settings, see `FwdCtrl2D`
- `gradient_tolerance`, `step_tolerance`, `objective_tolerance`: remaining stopping tests
- `verbose`: print one line per accepted iteration
"""
Base.@kwdef struct Inv2DOptions
    max_iter::Int = 20
    beta::Float64 = NaN
    beta_cooling::Float64 = 0.6
    min_beta_ratio::Float64 = 1e-3
    smallness::Float64 = 0.01
    smooth_y::Float64 = 1.0
    smooth_z::Float64 = 1.0
    data_norm::Symbol = :L2
    model_norm::Symbol = :L2
    error_floor::Float64 = 0.0
    log_bounds::Tuple{Float64, Float64} = (-1.0, 6.0)
    max_step::Float64 = 0.5
    max_linesearch::Int = 12
    target_rms::Float64 = 1.0
    relax_beta::Bool = true
    relax_factor::Float64 = 3.0
    max_relax::Int = 5
    wavenumbers::Int = 16
    refine::Int = 8
    growth::Float64 = 0.25
    gradient_tolerance::Float64 = 1e-8
    step_tolerance::Float64 = 1e-5
    objective_tolerance::Float64 = 1e-3
    verbose::Bool = true
end

"""
    Inv2DResult

Result of `Invert2D`.
- `resistivity`: recovered model in ohm metres, air tagged 1e17
- `rhoa`: predicted apparent resistivity of every observed datum (NaN where unused)
- `fit`: chi2 and rms over the inverted data
- `history`: accepted iterations, iteration zero first
- `active_cells`: inverted cells (the parameters) in model `(z, y)` order
- `used`: indices of the observed data that were inverted
- `coverage`: normalised cumulative sensitivity per model cell, NaN outside the active cells
- `converged`: false for `:max_iter`, `:line_search_failed` and `:stalled`
- `reason`: termination reason
- `algorithm`: the algorithm config that produced it
"""
struct Inv2DResult{A<:AbstractInversion2D}
    resistivity::Matrix{Float64}
    rhoa::Vector{Float64}
    fit::FitSummary2D
    history::Vector{NamedTuple}
    active_cells::Vector{CartesianIndex{2}}
    used::Vector{Int}
    coverage::Matrix{Float64}
    converged::Bool
    reason::Symbol
    algorithm::A
end

#---------- problem ----------

"""
    Inv2DProblem

The inversion: mesh, the inverted data (log ρa, errors, homogeneous response), active
cells, regularization operator, reference model, options, and the parts that change
during a run: β and the IRLS weights of the data (`wd`) and the model terms (`wm`).
"""
mutable struct Inv2DProblem
    mesh::ERT2DMesh
    data::DataFile2D
    used::Vector{Int}
    dobs::Vector{Float64}
    sigma::Vector{Float64}
    homogeneous::Vector{Float64}
    cells::Vector{CartesianIndex{2}}
    groups::Vector{Vector{CartesianIndex{2}}} # per parameter: its cell, then the padding cells that follow it
    R::SparseMatrixCSC{Float64, Int}          # all model cells
    R_active::SparseMatrixCSC{Float64, Int}   # active columns only
    mref::Matrix{Float64}
    options::Inv2DOptions
    beta::Float64
    beta0::Float64
    wd::Vector{Float64}
    wm::Vector{Float64}
    cache::ERT2DCache                         # forward system and factors, reused by every evaluation
end

function _inv2d_validate(opt::Inv2DOptions)
    opt.max_iter >= 0 || throw(ArgumentError("max_iter must be nonnegative"))
    isnan(opt.beta) || (isfinite(opt.beta) && opt.beta >= 0) || throw(ArgumentError("beta must be NaN (auto) or ≥ 0"))
    0 < opt.beta_cooling <= 1 && 0 < opt.min_beta_ratio <= 1 ||
        throw(ArgumentError("need 0 < beta_cooling ≤ 1 and 0 < min_beta_ratio ≤ 1"))
    all(x -> isfinite(x) && x >= 0, (opt.smallness, opt.smooth_y, opt.smooth_z, opt.target_rms, opt.error_floor,
        opt.gradient_tolerance, opt.step_tolerance, opt.objective_tolerance)) ||
        throw(ArgumentError("regularization and stopping controls must be finite and nonnegative"))
    opt.data_norm in (:L1, :L2) && opt.model_norm in (:L1, :L2) || throw(ArgumentError("norms must be :L1 or :L2"))
    opt.log_bounds[1] < opt.log_bounds[2] || throw(ArgumentError("log_bounds must be increasing"))
    isfinite(opt.max_step) && opt.max_step > 0 || throw(ArgumentError("max_step must be finite and positive"))
    opt.max_linesearch > 0 || throw(ArgumentError("max_linesearch must be positive"))
    opt.max_relax >= 0 && opt.relax_factor > 1 || throw(ArgumentError("need max_relax ≥ 0 and relax_factor > 1"))
    nothing
end

# active cells: default all earth cells; mask, cartesian, or linear indices otherwise
function _inv2d_cells(mesh, rho, active)
    indices = CartesianIndices(rho)
    air = ert2d_air_mask(mesh)
    cells = if active === nothing
        [i for i in indices if !air[i]]
    elseif active isa AbstractArray{Bool}
        size(active) == size(rho) || throw(DimensionMismatch("active mask must match model"))
        findall(active)
    else
        [i isa CartesianIndex{2} ? i : indices[i] for i in active]
    end
    isempty(cells) && throw(ArgumentError("at least one active cell is required"))
    all(i -> checkbounds(Bool, rho, i), cells) || throw(BoundsError(rho, cells))
    all(i -> !air[i], cells) || throw(ArgumentError("air cells cannot be inverted"))
    length(unique(cells)) == length(cells) || throw(ArgumentError("active cells must be unique"))
    cells
end

# each padding cell follows the nearest free cell (centre distance): one group per free cell, its padding after it
function _inv2d_padding_groups(mesh, cells, pad)
    isempty(cells) && throw(ArgumentError("padding needs at least one free cell"))
    yc, zc = ert2d_y_centers(mesh), ert2d_z_centers(mesh)
    groups = [[c] for c in cells]
    for p in pad
        d = [hypot(yc[c[2]] - yc[p[2]], zc[c[1]] - zc[p[1]]) for c in cells]
        push!(groups[argmin(d)], p)
    end
    groups
end

# the inverted data: valid rows, log ρa, its error; invalid observations are dropped here once, never later
function _inv2d_data(mesh, data::DataFile2D, opt; verbose = opt.verbose)
    length(data.electrodes) == length(mesh.electrode_positions) && data.electrodes ≈ mesh.electrode_positions ||
        throw(ArgumentError("data electrodes must match the mesh, including order"))
    v = ert2d_valid_data(data; error_floor = opt.error_floor)
    used = findall(v.keep)
    isempty(used) && throw(ArgumentError("no valid data: need finite R, K, positive errors and positive ρa"))
    if verbose && length(used) < ndata(data)
        @printf("Data: %d of %d used;", length(used), ndata(data))
        foreach(((k, n),) -> n > 0 && @printf(" %d %s;", n, k), v.reasons)
        println()
    end
    used, log.(data.geometric_factor[used] .* data.resistance[used]), v.sigma[used]
end

# finite-volume regularization of log10(m/mref): gradient terms approximate the area integral, smallness uses
# cell area over the median core-cell area; excluded cells (air, water, padding that follows a free cell) take no term
# and no smoothing pair
function _inv2d_regularizer(mesh, rho, opt; excluded = ert2d_air_mask(mesh))
    nz, ny = size(rho)
    linear = LinearIndices(rho)
    I, J, V = Int[], Int[], Float64[]
    row = 0
    area0 = median(mesh.z_cell_sizes) * median(mesh.y_cell_sizes)
    for iy in 1:ny, iz in 1:nz
        excluded[iz, iy] && continue
        cell = linear[iz, iy]
        if opt.smallness > 0
            row += 1
            push!(I, row); push!(J, cell)
            push!(V, sqrt(opt.smallness * mesh.z_cell_sizes[iz] * mesh.y_cell_sizes[iy] / area0))
        end
        if iy < ny && opt.smooth_y > 0 && !excluded[iz, iy+1]
            row += 1
            w = sqrt(opt.smooth_y * mesh.z_cell_sizes[iz] / ((mesh.y_cell_sizes[iy] + mesh.y_cell_sizes[iy+1]) / 2))
            append!(I, (row, row)); append!(J, (cell, linear[iz, iy+1])); append!(V, (-w, w))
        end
        if iz < nz && opt.smooth_z > 0 && !excluded[iz+1, iy]
            row += 1
            w = sqrt(opt.smooth_z * mesh.y_cell_sizes[iy] / ((mesh.z_cell_sizes[iz] + mesh.z_cell_sizes[iz+1]) / 2))
            append!(I, (row, row)); append!(J, (cell, linear[iz+1, iy])); append!(V, (-w, w))
        end
    end
    sparse(I, J, V, row, length(rho))
end

#---------- state: one evaluated model ----------

# objective of a state with the problem's current β and weights
_inv2d_objective(p::Inv2DProblem, r, reg) = (sum(abs2, p.wd .* r) + p.beta * sum(abs2, p.wm .* reg)) / 2

_inv2d_rescore(p::Inv2DProblem, state) = merge(state, (objective = _inv2d_objective(p, state.r, state.reg),))

# evaluate log10 model m on the active cells, other cells taken from rho0
function _inv2d_evaluate(problem::Inv2DProblem, rho0::AbstractMatrix, m::AbstractVector)
    opt = problem.options
    rho = copy(rho0)
    for (g, v) in zip(problem.groups, m)
        rho[g] .= 10.0^v
    end
    sub = problem.data
    f = ForwardSolve2D(problem.mesh, rho, sub; wavenumbers = opt.wavenumbers, refine = opt.refine, growth = opt.growth,
                       homogeneous = problem.homogeneous, keep_fields = true, cache = problem.cache)
    # a sign flip (ρa ≤ 0) is a huge but finite misfit, so the line search backs off
    r = (log.(max.(f.rhoa, floatmin(Float64))) .- problem.dobs) ./ problem.sigma
    fullm = log10.(rho)
    reg = problem.R * vec(fullm .- problem.mref)
    (; m = Vector{Float64}(m), rho, fwd = f.fwd, resistance = f.resistance, rhoa = f.rhoa, r, reg,
       objective = _inv2d_objective(problem, r, reg))
end

#---------- derivatives shared by all algorithms ----------

"""
    inv2d_frechet(problem, state) -> Matrix

Error-weighted sensitivity ∂r/∂m of the residuals r = (log ρa − d)/σ to log10 ρ of the
active cells, `n_data × n_active`.
"""
function inv2d_frechet(problem::Inv2DProblem, state)
    Jσ = _inv2d_group_sum(problem, ert2d_sensitivity(state.fwd, problem.data, reduce(vcat, problem.groups);
                                                     model_size = size(state.rho)))
    scale = [-log(10.0) / state.rho[c] for c in problem.cells]
    (Jσ ./ (state.resistance .* problem.sigma)) .* scale'
end

# columns (or entries) over all grouped cells summed per parameter: a padding cell's sensitivity goes to its free cell
function _inv2d_group_sum(problem::Inv2DProblem, J::AbstractVecOrMat)
    all(g -> length(g) == 1, problem.groups) && return J
    offsets = cumsum(vcat(0, length.(problem.groups)))
    J isa AbstractVector ? [sum(J[offsets[p]+1:offsets[p+1]]) for p in eachindex(problem.groups)] :
        reduce(hcat, [vec(sum(view(J, :, offsets[p]+1:offsets[p+1]); dims = 2)) for p in eachindex(problem.groups)])
end

"""
    inv2d_gradient(problem, state[, G])

Gradient of the objective in active log10 resistivity. With `G` (`inv2d_frechet`) it is
`Gᵗ Wd² r + β Rᵗ Wm² reg`; without it the data term comes from the pole fields without
forming G, which is what gradient-only algorithms use.
"""
inv2d_gradient(problem::Inv2DProblem, state, G::AbstractMatrix) =
    G' * (problem.wd .^ 2 .* state.r) + problem.beta * (problem.R_active' * (problem.wm .^ 2 .* state.reg))

function inv2d_gradient(problem::Inv2DProblem, state)
    c = problem.wd .^ 2 .* state.r ./ (problem.sigma .* state.resistance)
    gσ = _inv2d_group_sum(problem, ert2d_sensitivity_transpose(state.fwd, problem.data, c, reduce(vcat, problem.groups);
                                                               model_size = size(state.rho)))
    gσ .* [-log(10.0) / state.rho[cell] for cell in problem.cells] +
        problem.beta * (problem.R_active' * (problem.wm .^ 2 .* state.reg))
end

#---------- β and IRLS ----------

# β that balances ‖Wd G x‖² and ‖Wm R x‖² for a random model change x
function _inv2d_auto_beta(problem::Inv2DProblem, G::AbstractMatrix)
    x = randn(MersenneTwister(20261019), size(G, 2))
    d = sum(abs2, problem.wd .* (G * x))
    m = sum(abs2, problem.wm .* (problem.R_active * x))
    m > 0 ? d / m : 1.0
end

# Ekblom IRLS weights: data wd² = 1/√(1 + r²) (r in standard errors), model wm² = ε/√(reg² + ε²)
function _inv2d_reweight!(problem::Inv2DProblem, state)
    opt = problem.options
    changed = false
    if opt.data_norm == :L1
        problem.wd .= (1 .+ state.r .^ 2) .^ (-0.25)
        changed = true
    end
    if opt.model_norm == :L1 && any(!iszero, state.reg)
        nz = filter(!iszero, abs.(state.reg))
        ε = max(1e-4, 0.1 * sqrt(mean(abs2, nz)))
        problem.wm .= (ε ./ sqrt.(state.reg .^ 2 .+ ε^2)) .^ 0.5
        changed = true
    end
    changed
end

#---------- line search ----------

# armijo backtracking along a capped direction inside the log10 bounds; nothing when no step is accepted
function _inv2d_linesearch(problem::Inv2DProblem, state, direction, gradient)
    opt = problem.options
    lo, hi = opt.log_bounds
    d = direction .* min(1.0, opt.max_step / max(norm(direction, Inf), eps()))
    for ls in 0:opt.max_linesearch-1
        alpha = 0.5^ls
        candidate = clamp.(state.m + alpha .* d, lo, hi)
        delta = candidate - state.m
        slope = dot(gradient, delta)
        slope < 0 || continue
        trial = _inv2d_evaluate(problem, state.rho, candidate)
        if isfinite(trial.objective) && trial.objective <= state.objective + 1e-4 * slope
            return trial, norm(delta, Inf), alpha
        end
    end
    nothing
end

#---------- coverage ----------

"""
    ert2d_coverage(G, cells, mesh) -> Matrix

Normalised cumulative sensitivity, Σᵢ |Gᵢⱼ| per unit cell area, scaled to 1 at its
maximum, on the model grid (NaN outside `cells`). Low values mark cells the data barely
see, a guide to the depth of investigation.
"""
function ert2d_coverage(G::AbstractMatrix, cells, mesh::ERT2DMesh)
    s = vec(sum(abs, G; dims = 1)) ./ [mesh.y_cell_sizes[c[2]] * mesh.z_cell_sizes[c[1]] for c in cells]
    cov = fill(NaN, length(mesh.z_cell_sizes), length(mesh.y_cell_sizes))
    cov[cells] .= s ./ max(maximum(s), eps())
    cov
end

#---------- driver ----------

"""
    Invert2D(mesh, initial_resistivity, observed::DataFile2D;
             algorithm=GaussNewton2DConfig(), options=Inv2DOptions(),
             active_cells=nothing, reference_resistivity=initial_resistivity, water_cells=nothing,
             padding_cells=nothing)

Invert log apparent resistivities for log10 resistivity, minimising
`0.5‖Wd (log ρa(m) − log ρa_obs)/σ‖² + 0.5 β ‖Wm R (m − m_ref)‖²` with β cooling to the
target misfit, optional L1 norms (IRLS) and the model held inside `log_bounds`.

- `observed`: data with finite R, K and errors; invalid data are dropped and reported
- `algorithm`: `GaussNewton2DConfig`, `NLCG2DConfig` or any `AbstractInversion2D`
- `options`: `Inv2DOptions`, shared by every algorithm
- `active_cells`: model-shaped Boolean mask or vector of cartesian/linear indices;
  default all earth cells. Other cells stay fixed, air is always fixed
- `reference_resistivity`: model the regularization pulls toward; default the start
- `water_cells`: model-shaped Boolean mask of fixed water; it takes no
  regularization term and no smoothing pair
- `padding_cells`: model-shaped Boolean mask of padding outside the main domain: each
  takes the value of the nearest active cell (its sensitivity is added to that cell) and
  is neither inverted nor regularized

Returns `Inv2DResult`; no files are written by this method, the six-file method below
writes a run directory.
"""
function Invert2D(mesh::ERT2DMesh, initial_resistivity::AbstractMatrix{<:Real}, observed::DataFile2D;
                  algorithm::AbstractInversion2D = GaussNewton2DConfig(),
                  options::Inv2DOptions = Inv2DOptions(),
                  active_cells = nothing, reference_resistivity = initial_resistivity,
                  water_cells::Union{Nothing, AbstractMatrix{Bool}} = nothing,
                  padding_cells::Union{Nothing, AbstractMatrix{Bool}} = nothing)
    opt = options
    _inv2d_validate(opt)
    inv2d_validate(algorithm)

    #---------- setup ----------
    rho0 = Matrix{Float64}(initial_resistivity)
    size(reference_resistivity) == size(rho0) || throw(DimensionMismatch("reference must match initial model"))
    air = ert2d_air_mask(mesh)
    all(x -> isfinite(x) && x > 0, reference_resistivity[.!air]) ||
        throw(ArgumentError("reference earth resistivity must be positive and finite"))
    all(x -> isfinite(x) && x > 0, rho0[.!air]) || throw(ArgumentError("initial earth resistivity must be positive and finite"))
    cells = _inv2d_cells(mesh, rho0, active_cells)
    used, dobs, sigma = _inv2d_data(mesh, observed, opt)
    excluded = copy(air)
    if water_cells !== nothing
        size(water_cells) == size(rho0) || throw(DimensionMismatch("water mask must match initial model"))
        active_cells === nothing && filter!(i -> !water_cells[i], cells)
        any(i -> water_cells[i], cells) && throw(ArgumentError("water cells cannot be inverted"))
        excluded .|= water_cells
    end
    groups = [[c] for c in cells]
    if padding_cells !== nothing
        size(padding_cells) == size(rho0) || throw(DimensionMismatch("padding mask must match initial model"))
        pad = findall(padding_cells .& .!air)
        filter!(c -> !padding_cells[c], cells)
        groups = _inv2d_padding_groups(mesh, cells, pad)
        excluded .|= padding_cells
    end
    lo, hi = opt.log_bounds
    m0 = log10.(rho0[cells])
    for (g, v) in zip(groups, m0)              # padding starts from its free cell
        rho0[g] .= 10.0^v
    end
    any(x -> !(lo <= x <= hi), m0) && @warn "start model outside log10 bounds $(opt.log_bounds); clamped"
    m0 = clamp.(m0, lo, hi)

    rho0[air] .= ERT2D_AIR_TAG
    reference = Matrix{Float64}(reference_resistivity)
    reference[air] .= ERT2D_AIR_TAG
    R = _inv2d_regularizer(mesh, rho0, opt; excluded)
    R_active = R[:, LinearIndices(rho0)[cells]]
    sub = select_data2d(observed, used)
    cache = ERT2DCache()
    homogeneous = ert2d_homogeneous(mesh, sub; wavenumbers = opt.wavenumbers, refine = opt.refine, growth = opt.growth, cache)
    problem = Inv2DProblem(mesh, sub, used, dobs, sigma, homogeneous, cells, groups, R, R_active, log10.(reference), opt,
                           isnan(opt.beta) ? 1.0 : opt.beta, 1.0, ones(length(used)), ones(size(R, 1)), cache)

    state = _inv2d_evaluate(problem, rho0, m0)
    all(isfinite, state.r) || error("initial forward response is not finite")
    if isnan(opt.beta)
        problem.beta = _inv2d_auto_beta(problem, inv2d_frechet(problem, state))
        opt.verbose && @printf("Automatic β = %.4g\n", problem.beta)
    end
    problem.beta0 = problem.beta
    beta_min = problem.beta0 * opt.min_beta_ratio
    state = _inv2d_rescore(problem, state)
    work = inv2d_init(algorithm, problem, state)

    #---------- history ----------
    history = NamedTuple[]
    rms_of(s) = sqrt(sum(abs2, s.r) / length(s.r))
    function record(iteration, step, alpha, info)
        chi2 = sum(abs2, state.r)
        entry = (; iteration, objective = state.objective, chi2, rms = sqrt(chi2 / length(state.r)),
                   regularization = sum(abs2, problem.wm .* state.reg) / 2, beta = problem.beta, step, alpha, info...)
        push!(history, entry)
        opt.verbose && @printf("%s %3d  RMS %8.4f  β %.3g  objective %.6g  step %.3g\n",
                               uppercase(inv2d_tag(algorithm)), iteration, entry.rms, problem.beta, entry.objective, step)
        opt.verbose && flush(stdout)
    end
    record(0, 0.0, 0.0, inv2d_info(algorithm, work))

    #---------- iterations ----------
    reason = :max_iter
    stalls = 0
    relaxed = 0                     # β raises after landing below the target
    fitted = nothing                # the latest model within the target, the answer of the relaxation
    for iteration in 1:opt.max_iter
        if opt.target_rms > 0 && rms_of(state) <= opt.target_rms && relaxed == 0
            reason = :target_rms
            break
        end
        if iteration > 1 && _inv2d_reweight!(problem, state)
            state = _inv2d_rescore(problem, state)
            inv2d_restart!(algorithm, work, problem)
        end
        gradient = inv2d_prepare!(algorithm, work, problem, state)
        if norm(gradient, Inf) <= opt.gradient_tolerance
            reason = :gradient_tolerance
            break
        end

        # the algorithm may retry with a new direction (e.g. more damping) after a failed search
        found = nothing
        while true
            direction = inv2d_direction(algorithm, work, problem, state, gradient)
            found = _inv2d_linesearch(problem, state, direction, gradient)
            found === nothing || break
            inv2d_reject!(algorithm, work) || break
        end
        if found === nothing
            if problem.beta > beta_min * (1 + 1e-9) && opt.beta_cooling < 1
                problem.beta = max(problem.beta * opt.beta_cooling, beta_min)
                state = _inv2d_rescore(problem, state)
                inv2d_restart!(algorithm, work, problem)
                continue
            end
            reason = :line_search_failed
            break
        end

        old = state
        state, step, alpha = found
        inv2d_accept!(algorithm, work, problem, old, state)
        record(iteration, step, alpha, inv2d_info(algorithm, work))

        if opt.target_rms > 0 && relaxed > 0 && rms_of(state) > opt.target_rms
            # relaxed past the target: the previous model is the smoothest that fits
            state = fitted
            reason = :target_rms
            break
        end
        if opt.target_rms > 0 && rms_of(state) <= opt.target_rms
            fitted = state
            if opt.relax_beta && rms_of(state) < 0.9opt.target_rms && relaxed < opt.max_relax
                # over-fitted: raise β and step again from here towards a smoother model at the target
                relaxed += 1
                problem.beta *= opt.relax_factor
                state = _inv2d_rescore(problem, state)
                inv2d_restart!(algorithm, work, problem)
                continue
            end
            reason = :target_rms
            break
        elseif step <= opt.step_tolerance
            reason = :step_tolerance
            break
        end
        at_floor = problem.beta <= beta_min * (1 + 1e-9) || opt.beta_cooling == 1
        rel = (rms_of(old) - rms_of(state)) / max(rms_of(old), eps())
        stalls = at_floor && rel < opt.objective_tolerance ? stalls + 1 : 0
        if stalls >= 3
            reason = :stalled
            break
        end
        if !at_floor && rms_of(state) > opt.target_rms && (inv2d_cool_always(algorithm) || rel < 0.05)
            problem.beta = max(problem.beta * opt.beta_cooling, beta_min)
            state = _inv2d_rescore(problem, state)
            inv2d_restart!(algorithm, work, problem)
        end
    end
    fitted === nothing || rms_of(state) <= opt.target_rms || (state = fitted)
    if opt.target_rms > 0 && rms_of(state) <= opt.target_rms
        reason = :target_rms
    end

    chi2 = sum(abs2, state.r)
    fit = FitSummary2D(chi2 = chi2, rms = sqrt(chi2 / length(state.r)), count = length(state.r))
    rhoa = fill(NaN, ndata(observed))
    rhoa[used] .= state.rhoa
    coverage = ert2d_coverage(inv2d_frechet(problem, state), cells, mesh)
    final = copy(state.rho)
    final[air] .= ERT2D_AIR_TAG
    Inv2DResult(final, rhoa, fit, history, cells, used, coverage,
                reason ∉ (:max_iter, :line_search_failed, :stalled), reason, algorithm)
end

#---------- file workflow ----------

# shared options and the algorithm config from inv.ctrl and fwd.ctrl
function _inv2d_from_ctrl(c::InvCtrl2D, fwd::FwdCtrl2D)
    options = Inv2DOptions(max_iter = c.max_iter, beta = c.lambda, beta_cooling = c.lambda_cooling,
                           min_beta_ratio = c.min_lambda_ratio, smallness = c.smallness, smooth_y = c.smooth_y,
                           smooth_z = c.smooth_z, data_norm = c.data_norm, model_norm = c.model_norm,
                           error_floor = c.error_floor, log_bounds = c.log_bounds, max_step = c.max_step,
                           max_linesearch = c.max_linesearch, target_rms = c.target_rms, relax_beta = c.relax_lambda,
                           wavenumbers = fwd.wavenumbers, refine = fwd.refine, growth = fwd.growth)
    algorithm = c.algorithm == :gn ? GaussNewton2DConfig(damping = c.gn_damping) :
                NLCG2DConfig(restart = c.nlcg_restart, precondition = c.nlcg_precondition)
    options, algorithm
end

# run_YYYYmmdd_HHMMSS next to the data, suffixed _2, _3 when started in the same second
function _inv2d_run_dir(data_path::AbstractString)
    base = joinpath(dirname(abspath(data_path)), "run_" * Dates.format(now(), "yyyymmdd_HHMMSS"))
    dir, k = base, 1
    while ispath(dir)
        k += 1
        dir = "$(base)_$k"
    end
    mkpath(dir)
    dir
end

_same_grid(a::ModelFile2D, b::ModelFile2D) =
    size(a.resistivity) == size(b.resistivity) && a.y_cell_sizes ≈ b.y_cell_sizes && a.z_cell_sizes ≈ b.z_cell_sizes

# mesh, start model, active, water and padding cells of the file workflows from the model and its mask
# (cov.ctrl): 0 = air or fixed, 9 = water, 2 = padding that follows the nearest free cell, others free
function _inv2d_file_setup(start::ModelFile2D, observed::DataFile2D, mask::AbstractMatrix{<:Integer}, kind)
    size(mask) == size(start.resistivity) ||
        error("the $kind is $(size(mask)) cells, the model is $(size(start.resistivity))")
    any(isnan, observed.geometric_factor) && error("the data have no geometric factors; run MakeMesh2D on the data file")
    mesh, ρ0 = Mesh2DFromInputs(start, observed)
    air = ert2d_air_mask(mesh)
    water = mask .== ERT2D_MASK_WATER
    padding = (mask .== ERT2D_MASK_PADDING) .& .!air
    wet = intersect(ert2d_electrode_columns(mesh), findall(vec(any(water; dims = 1))))
    isempty(wet) || error("electrodes stand over water (mask $(ERT2D_MASK_WATER)) in model column(s) $wet")
    active = (mask .!= 0) .& (mask .!= ERT2D_MASK_WATER) .& .!air
    any(active .& .!padding) || error("the $kind fixes every cell")
    (; mesh, ρ0, active, water, padding, offsets = ert2d_electrode_offsets(mesh, observed))
end

function _inv2d_write_history(path, history)
    open(path, "w") do io
        println(io, join(string.(keys(first(history))), ','))
        foreach(h -> println(io, join(values(h), ',')), history)
    end
    path
end

function _inv2d_open_run(run_dir, data_path, inputs)
    dir = run_dir === nothing ? _inv2d_run_dir(data_path) : mkpath(String(run_dir))
    mkpath(joinpath(dir, "inputs"))
    foreach(p -> cp(p, joinpath(dir, "inputs", basename(p)); force = true), inputs)
    println("Run directory: ", dir)
    dir
end

# data use, cells and electrode snapping, the tail of every Summary.txt
function _inv2d_summary_cells(io, s, observed, used)
    println(io, "Data used: ", length(used), " of ", ndata(observed))
    println(io, "Inverted cells: ", count(s.active .& .!s.padding))
    println(io, "Padding cells following the nearest inverted cell: ", count(s.padding))
    println(io, "Topographic air cells: ", count(ert2d_air_mask(s.mesh)))
    println(io, "Water cells: ", count(s.water))
    any(o -> abs(o.offset) > 1e-6, s.offsets) || return
    println(io, "Electrode treads (Z = data depth below the model top, tread = its step on the model mesh, m):")
    @printf(io, "  %-12s %10s %10s %10s %10s\n", "electrode", "Z", "tread", "offset", "tolerance")
    for o in s.offsets
        @printf(io, "  %-12s %10.3f %10.3f %10.3f %10.3f%s\n", o.name, o.z, o.surface, o.offset, o.tolerance,
                abs(o.offset) > o.tolerance ? "  beyond tolerance" : "")
    end
end

"""
    Invert2D(start_path, data_path, fwd_path, inv_path, cov_path, prior_path; run_dir=nothing)

ModEM-style deterministic inversion (GN or NLCG) from six files: the start model, the
observed data, `fwd.ctrl`, `inv.ctrl`, the model covariance and the prior model. The
algorithm comes from `inv.ctrl`, the inverted cells from the covariance mask (0 = air or
fixed, 9 = water, fixed and unregularized, 2 = padding, following the nearest free cell,
others free). No electrode may stand over water. The prior is the reference model of the
regularization; the topography is the start model's.

Everything is written to `run_dir`, by default `run_YYYYmmdd_HHMMSS/` next to the data:
`model.rho` (ModEM layout, restartable), `data.pred`, `coverage.rho` (normalised
cumulative sensitivity, LINEAR), `History.csv`, `Summary.txt` and copies of the inputs
in `inputs/`. Returns a named tuple with the run directory, mesh, models, data,
history, rms and termination reason.
"""
function Invert2D(start_path::AbstractString, data_path::AbstractString, fwd_path::AbstractString,
                  inv_path::AbstractString, cov_path::AbstractString, prior_path::AbstractString;
                  run_dir::Union{Nothing, AbstractString} = nothing)
    fwd, ctrl, cov = ReadFwdCtrl2D(fwd_path), ReadInvCtrl2D(inv_path), ReadCov2D(cov_path)
    observed = load_data2d(data_path)
    start, prior = ReadModel2D(start_path), ReadModel2D(prior_path)
    _same_grid(start, prior) || error("the prior model must be on the start model's grid")
    s = _inv2d_file_setup(start, observed, cov.mask, "covariance mask")
    priormesh, ρref = Mesh2DFromInputs(prior, observed; warn = false)
    ert2d_air_mask(priormesh) == ert2d_air_mask(s.mesh) || error("the prior model's topography differs from the start model's")
    dir = _inv2d_open_run(run_dir, data_path, (start_path, data_path, fwd_path, inv_path, cov_path, prior_path))

    options, algorithm = _inv2d_from_ctrl(ctrl, fwd)
    result = Invert2D(s.mesh, s.ρ0, observed; algorithm, options, active_cells = s.active, reference_resistivity = ρref,
                      water_cells = any(s.water) ? s.water : nothing, padding_cells = any(s.padding) ? s.padding : nothing)
    predicted = _ert2d_predicted(observed, result.rhoa)
    fit = chi2_rms2d(observed, predicted; error_floor = ctrl.error_floor)

    WriteModel2D(joinpath(dir, "model.rho"), s.mesh, result.resistivity)
    write_data2d(joinpath(dir, "data.pred"), predicted)
    WriteModel2D(joinpath(dir, "coverage.rho"), s.mesh, ifelse.(isnan.(result.coverage), 1.0, max.(result.coverage, 1e-12));
                 loge = false)
    _inv2d_write_history(joinpath(dir, "History.csv"), result.history)
    open(joinpath(dir, "Summary.txt"), "w") do io
        println(io, "Algorithm: ", uppercase(string(ctrl.algorithm)))
        println(io, "Termination: ", result.reason)
        println(io, "Converged: ", result.converged)
        @printf(io, "RMS: %.6f\n", fit.rms)
        @printf(io, "Final lambda: %.6g\n", result.history[end].beta)
        println(io, "Data norm: ", ctrl.data_norm, ", model norm: ", ctrl.model_norm)
        println(io, "Accepted iterations: ", length(result.history) - 1)
        _inv2d_summary_cells(io, s, observed, result.used)
    end
    (; run_dir = dir, algorithm = ctrl.algorithm, ctrl, fwd, mesh = s.mesh, observed, predicted, start = s.ρ0,
       prior = ρref, final = result.resistivity, active = s.active, water = s.water, padding = s.padding,
       coverage = result.coverage,
       history = result.history, rms = fit.rms, reason = result.reason, converged = result.converged)
end
