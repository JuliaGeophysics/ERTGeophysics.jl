# 3D ERT Gauss–Newton inversion
# Author: @pankajkmishra
# The 2D scheme in 3D: log apparent resistivities inverted for log10 resistivity, β automatic and cooled to the
# target misfit, optional L1 data and model norms (IRLS), log10 bounds, damped Gauss–Newton steps with an Armijo line
# search. The normal equations are factorised for small models and solved by Jacobi-preconditioned conjugate
# gradients (products with the explicit sensitivity, no normal matrix) above `dense_limit` cells

using Dates
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics

"""
    Inv3DOptions(; kwargs...)

3D inversion controls, as `Inv2DOptions` with `smooth_x` and the Gauss–Newton settings:
- `max_iter`, `beta` (NaN = automatic), `beta_cooling`, `min_beta_ratio`, `target_rms`
- `relax_beta`, `relax_factor`, `max_relax`: raise β again after landing below 0.9 ×
  the target and keep the smoothest model that fits (see `Inv2DOptions`)
- `smallness`, `smooth_x`, `smooth_y`, `smooth_z`: regularization weights
- `regularization`: `:index` (default, as ModEM: every neighbour pair and every cell weighs the
  same, so thin surface layers are smoothed as strongly as deep ones) or `:volume` (the
  finite-volume integral of the gradient, as in 2D)
- `data_norm`, `model_norm`: `:L2` or `:L1` (IRLS)
- `error_floor`, `log_bounds`, `max_step`, `max_linesearch`
- `refine`, `growth`: forward refinement (`FwdCtrl3D`)
- `damping`: initial Levenberg–Marquardt damping relative to the diagonal
- `dense_limit`, `cg_tolerance`, `cg_max_iter`: normal-equation solve
- `step_tolerance`, `objective_tolerance`, `verbose`
"""
Base.@kwdef struct Inv3DOptions
    max_iter::Int = 15
    beta::Float64 = NaN
    beta_cooling::Float64 = 0.6
    min_beta_ratio::Float64 = 1e-3
    smallness::Float64 = 0.01
    smooth_x::Float64 = 1.0
    smooth_y::Float64 = 1.0
    smooth_z::Float64 = 1.0
    regularization::Symbol = :index
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
    refine::Int = 4
    growth::Float64 = 0.5
    damping::Float64 = 1e-2
    max_damping_trials::Int = 6
    dense_limit::Int = 3000
    cg_tolerance::Float64 = 1e-3
    cg_max_iter::Int = 200
    step_tolerance::Float64 = 1e-5
    objective_tolerance::Float64 = 1e-3
    verbose::Bool = true
end

function _inv3d_validate(opt::Inv3DOptions)
    opt.max_iter >= 0 || throw(ArgumentError("max_iter must be nonnegative"))
    isnan(opt.beta) || (isfinite(opt.beta) && opt.beta >= 0) || throw(ArgumentError("beta must be NaN (auto) or ≥ 0"))
    0 < opt.beta_cooling <= 1 && 0 < opt.min_beta_ratio <= 1 ||
        throw(ArgumentError("need 0 < beta_cooling ≤ 1 and 0 < min_beta_ratio ≤ 1"))
    all(x -> isfinite(x) && x >= 0, (opt.smallness, opt.smooth_x, opt.smooth_y, opt.smooth_z, opt.target_rms, opt.error_floor)) ||
        throw(ArgumentError("regularization and stopping controls must be finite and nonnegative"))
    opt.data_norm in (:L1, :L2) && opt.model_norm in (:L1, :L2) || throw(ArgumentError("norms must be :L1 or :L2"))
    opt.max_relax >= 0 && opt.relax_factor > 1 || throw(ArgumentError("need max_relax ≥ 0 and relax_factor > 1"))
    opt.regularization in (:index, :volume) || throw(ArgumentError("regularization must be :index or :volume"))
    opt.log_bounds[1] < opt.log_bounds[2] || throw(ArgumentError("log_bounds must be increasing"))
    opt.max_step > 0 && opt.max_linesearch > 0 && opt.damping > 0 && opt.max_damping_trials > 0 ||
        throw(ArgumentError("max_step, max_linesearch, damping and max_damping_trials must be positive"))
    nothing
end

# regularization of log10(m/mref) over the earth cells, excluded cells take no term. :index weighs every cell and
# neighbour pair alike (ModEM); :volume is the finite-volume form, smallness by cell volume over the median volume,
# gradient pairs by face area over the centre distance
function _inv3d_regularizer(mesh::ERT3DMesh, opt::Inv3DOptions, excluded::AbstractArray{Bool, 3})
    nx, ny, nz = size(excluded)
    L = LinearIndices(excluded)
    dx, dy, dz = mesh.x_cell_sizes, mesh.y_cell_sizes, mesh.z_cell_sizes
    vol0 = median(dx) * median(dy) * median(dz)
    index = opt.regularization == :index
    I, J, V = Int[], Int[], Float64[]
    row = 0
    pair!(c1, c2, w) = (row += 1; append!(I, (row, row)); append!(J, (c1, c2)); append!(V, (-w, w)))
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        excluded[ix, iy, iz] && continue
        c = L[ix, iy, iz]
        if opt.smallness > 0
            row += 1
            push!(I, row); push!(J, c); push!(V, sqrt(opt.smallness * (index ? 1.0 : dx[ix] * dy[iy] * dz[iz] / vol0)))
        end
        ix < nx && opt.smooth_x > 0 && !excluded[ix+1, iy, iz] &&
            pair!(c, L[ix+1, iy, iz], sqrt(opt.smooth_x * (index ? 1.0 : dy[iy] * dz[iz] / ((dx[ix] + dx[ix+1]) / 2) / cbrt(vol0))))
        iy < ny && opt.smooth_y > 0 && !excluded[ix, iy+1, iz] &&
            pair!(c, L[ix, iy+1, iz], sqrt(opt.smooth_y * (index ? 1.0 : dx[ix] * dz[iz] / ((dy[iy] + dy[iy+1]) / 2) / cbrt(vol0))))
        iz < nz && opt.smooth_z > 0 && !excluded[ix, iy, iz+1] &&
            pair!(c, L[ix, iy, iz+1], sqrt(opt.smooth_z * (index ? 1.0 : dx[ix] * dy[iy] / ((dz[iz] + dz[iz+1]) / 2) / cbrt(vol0))))
    end
    sparse(I, J, V, row, length(excluded))
end

"""
    ert3d_coverage(G, cells, mesh) -> Array{Float64,3}

Normalised cumulative sensitivity, Σᵢ |Gᵢⱼ| per unit cell volume, scaled to 1 at its
maximum, on the model grid (NaN outside `cells`).
"""
function ert3d_coverage(G::AbstractMatrix, cells, mesh::ERT3DMesh)
    s = vec(sum(abs, G; dims = 1)) ./ [mesh.x_cell_sizes[c[1]] * mesh.y_cell_sizes[c[2]] * mesh.z_cell_sizes[c[3]] for c in cells]
    cov = fill(NaN, _ert3d_size(mesh))
    cov[cells] .= s ./ max(maximum(s), eps())
    cov
end

"""
    Invert3D(mesh, initial_resistivity, observed::DataFile3D; options=Inv3DOptions(),
             active_cells=nothing, reference_resistivity=initial_resistivity, padding_cells=nothing)
        -> (; resistivity, rhoa, fit, history, active_cells, used, coverage, converged, reason)

Gauss–Newton inversion of log apparent resistivities for log10 resistivity, minimising
`0.5‖Wd (log ρa(m) − log ρa_obs)/σ‖² + 0.5 β ‖Wm R (m − m_ref)‖²` with β cooling to the
target misfit. `active_cells`: Boolean mask `(nx, ny, nz)` or cartesian indices, default
all earth cells; air is always fixed. `padding_cells`: Boolean mask of padding outside the
main domain; each takes the value of the nearest active cell (its sensitivity is added to
that cell) and is neither inverted nor regularized. Invalid data are dropped and reported.
"""
function Invert3D(mesh::ERT3DMesh, initial_resistivity::AbstractArray{<:Real, 3}, observed::DataFile3D;
                  options::Inv3DOptions = Inv3DOptions(), active_cells = nothing,
                  reference_resistivity = initial_resistivity,
                  padding_cells::Union{Nothing, AbstractArray{Bool, 3}} = nothing)
    opt = options
    _inv3d_validate(opt)
    nelectrodes(observed) == length(mesh.electrode_x) && observed.x_positions ≈ mesh.electrode_x &&
        observed.y_positions ≈ mesh.electrode_y || throw(ArgumentError("data electrodes must match the mesh, including order"))
    air = ert3d_air_mask(mesh)
    rho0 = Array{Float64, 3}(initial_resistivity)
    rho0[air] .= ERT2D_AIR_TAG
    cells = active_cells === nothing ? [c for c in CartesianIndices(air) if !air[c]] :
            active_cells isa AbstractArray{Bool} ? findall(active_cells) : collect(active_cells)
    any(c -> air[c], cells) && throw(ArgumentError("air cells cannot be inverted"))
    excluded = copy(air)
    groups = [[c] for c in cells]
    if padding_cells !== nothing
        size(padding_cells) == size(rho0) || throw(DimensionMismatch("padding mask must match the model"))
        pad = [c for c in cells if padding_cells[c]]
        filter!(c -> !padding_cells[c], cells)
        isempty(cells) && throw(ArgumentError("padding needs at least one free cell"))
        xc, yc, zc = ert3d_x_centers(mesh), ert3d_y_centers(mesh), ert3d_z_centers(mesh)
        groups = [[c] for c in cells]
        for p in pad                            # each padding cell follows the nearest free cell
            k = argmin([hypot(xc[c[1]] - xc[p[1]], yc[c[2]] - yc[p[2]], zc[c[3]] - zc[p[3]]) for c in cells])
            push!(groups[k], p)
        end
        excluded .|= padding_cells
    end
    isempty(cells) && throw(ArgumentError("at least one active cell is required"))

    v = ert2d_valid_data(observed; error_floor = opt.error_floor)
    used = findall(v.keep)
    isempty(used) && throw(ArgumentError("no valid data: need finite R, K, positive errors and positive ρa"))
    opt.verbose && length(used) < ndata(observed) && @printf("Data: %d of %d used\n", length(used), ndata(observed))
    sub = select_data3d(observed, used)
    dobs = log.(sub.geometric_factor .* sub.resistance)
    sigma = v.sigma[used]

    lo, hi = opt.log_bounds
    m0 = clamp.(log10.(rho0[cells]), lo, hi)
    reference = Array{Float64, 3}(reference_resistivity)
    reference[air] .= ERT2D_AIR_TAG
    mref = log10.(reference)
    R = _inv3d_regularizer(mesh, opt, excluded)
    R_active = R[:, LinearIndices(rho0)[cells]]
    wd, wm = ones(length(used)), ones(size(R, 1))
    cache = ERT3DCache()
    t0 = time()
    homogeneous = ert3d_homogeneous(mesh, sub; refine = opt.refine, growth = opt.growth, cache)
    opt.verbose && @printf("3D forward: %d active cells, %d data, %.1f s per solve\n", length(cells), length(used), time() - t0)

    beta = 1.0
    objective(r, reg) = (sum(abs2, wd .* r) + beta * sum(abs2, wm .* reg)) / 2
    function evaluate(rho_base, m)
        rho = copy(rho_base)
        for (g, v) in zip(groups, m)
            rho[g] .= 10.0^v
        end
        f = ForwardSolve3D(mesh, rho, sub; refine = opt.refine, growth = opt.growth, homogeneous, keep_fields = true, cache)
        r = (log.(max.(f.rhoa, floatmin(Float64))) .- dobs) ./ sigma
        reg = R * vec(log10.(rho) .- mref)
        (; m = Vector{Float64}(m), rho, fwd = f.fwd, resistance = f.resistance, rhoa = f.rhoa, r, reg, objective = objective(r, reg))
    end
    rescore(s) = merge(s, (objective = objective(s.r, s.reg),))
    function frechet(s)                       # in place: the sensitivity is the largest array of a 3D run
        Gall = ert3d_sensitivity(s.fwd, sub, reduce(vcat, groups); model_size = size(s.rho))
        G = all(g -> length(g) == 1, groups) ? Gall :
            (o = cumsum(vcat(0, length.(groups))); reduce(hcat, [vec(sum(view(Gall, :, o[p]+1:o[p+1]); dims = 2)) for p in eachindex(groups)]))
        G ./= s.resistance .* sigma
        G .*= [-log(10.0) / s.rho[c] for c in cells]'
        G
    end
    rms(s) = sqrt(sum(abs2, s.r) / length(s.r))

    state = evaluate(rho0, m0)              # padding starts from its free cell
    all(isfinite, state.r) || error("initial forward response is not finite")
    G = frechet(state)
    if isnan(opt.beta)
        x = randn(MersenneTwister(20261019), length(cells))
        mr = sum(abs2, R_active * x)
        beta = mr > 0 ? sum(abs2, G * x) / mr : 1.0
        opt.verbose && @printf("Automatic β = %.4g\n", beta)
    else
        beta = opt.beta
    end
    beta_min = beta * opt.min_beta_ratio
    state = rescore(state)

    history = NamedTuple[]
    damping = opt.damping
    function record(iteration, step)
        chi2 = sum(abs2, state.r)
        push!(history, (; iteration, objective = state.objective, chi2, rms = sqrt(chi2 / length(state.r)),
                          regularization = sum(abs2, wm .* state.reg) / 2, beta, step, damping))
        opt.verbose && @printf("GN3D %3d  RMS %8.4f  β %.3g  objective %.6g  step %.3g\n", iteration, history[end].rms, beta,
                               state.objective, step)
        opt.verbose && flush(stdout)
    end
    record(0, 0.0)

    reason, stalls = :max_iter, 0
    relaxed, fitted = 0, nothing          # β raises after landing below the target, the latest model within it
    for iteration in 1:opt.max_iter
        relaxed == 0 && rms(state) <= opt.target_rms && (reason = :target_rms; break)
        if iteration > 1
            changed = false
            opt.data_norm == :L1 && (wd .= (1 .+ state.r .^ 2) .^ (-0.25); changed = true)
            if opt.model_norm == :L1 && any(!iszero, state.reg)
                ε = max(1e-4, 0.1 * sqrt(mean(abs2, filter(!iszero, state.reg))))
                wm .= (ε ./ sqrt.(state.reg .^ 2 .+ ε^2)) .^ 0.5
                changed = true
            end
            changed && (state = rescore(state))
        end
        WG = opt.data_norm == :L1 ? wd .* G : G
        WR = wm .* R_active
        gradient = WG' * (wd .* state.r) + beta * (R_active' * (wm .^ 2 .* state.reg))
        dense = length(cells) <= opt.dense_limit
        H = dense ? WG' * WG + beta * Matrix(WR' * WR) : zeros(0, 0)
        D = dense ? diag(H) : vec(sum(abs2, WG; dims = 1)) .+ beta .* vec(sum(abs2, WR; dims = 1))
        D .= max.(D, 1e-12 * max(maximum(D), eps()))
        b = beta
        function direction(λ)
            if dense
                A = copy(H)
                foreach(i -> A[i, i] += λ * D[i], eachindex(D))
                return -(cholesky!(Symmetric(A)) \ gradient)
            end
            -_inv2d_pcg(x -> WG' * (WG * x) + b * (WR' * (WR * x)) + λ .* D .* x, gradient, (1 + λ) .* D;
                        tol = opt.cg_tolerance, maxiter = opt.cg_max_iter)
        end

        found = nothing
        for _ in 1:opt.max_damping_trials
            d = direction(damping)
            d .*= min(1.0, opt.max_step / max(norm(d, Inf), eps()))
            for ls in 0:opt.max_linesearch-1
                candidate = clamp.(state.m .+ 0.5^ls .* d, lo, hi)
                delta = candidate .- state.m
                slope = dot(gradient, delta)
                slope < 0 || continue
                trial = evaluate(state.rho, candidate)
                if isfinite(trial.objective) && trial.objective <= state.objective + 1e-4 * slope
                    found = (trial, norm(delta, Inf))
                    break
                end
            end
            found === nothing || break
            damping *= 10
        end
        if found === nothing
            if beta > beta_min * (1 + 1e-9) && opt.beta_cooling < 1
                beta = max(beta * opt.beta_cooling, beta_min)
                state = rescore(state)
                continue
            end
            reason = :line_search_failed
            break
        end
        old = state
        state, step = found
        damping = max(damping / 3, 1e-6)
        G = nothing
        G = frechet(state)
        record(iteration, step)

        if relaxed > 0 && rms(state) > opt.target_rms
            state, G = fitted                 # relaxed past the target: the previous model is the smoothest that fits
            reason = :target_rms
            break
        end
        if rms(state) <= opt.target_rms
            fitted = (state, G)
            if opt.relax_beta && rms(state) < 0.9opt.target_rms && relaxed < opt.max_relax
                relaxed += 1
                beta *= opt.relax_factor
                state = rescore(state)
                continue
            end
            reason = :target_rms
            break
        end
        step <= opt.step_tolerance && (reason = :step_tolerance; break)
        at_floor = beta <= beta_min * (1 + 1e-9) || opt.beta_cooling == 1
        rel = (rms(old) - rms(state)) / max(rms(old), eps())
        stalls = at_floor && rel < opt.objective_tolerance ? stalls + 1 : 0
        stalls >= 3 && (reason = :stalled; break)
        if !at_floor
            beta = max(beta * opt.beta_cooling, beta_min)
            state = rescore(state)
        end
    end
    fitted === nothing || rms(state) <= opt.target_rms || ((state, G) = fitted)
    rms(state) <= opt.target_rms && (reason = :target_rms)

    chi2 = sum(abs2, state.r)
    rhoa = fill(NaN, ndata(observed))
    rhoa[used] .= state.rhoa
    final = copy(state.rho)
    final[air] .= ERT2D_AIR_TAG
    (; resistivity = final, rhoa, fit = FitSummary2D(chi2 = chi2, rms = sqrt(chi2 / length(state.r)), count = length(state.r)),
       history, active_cells = cells, used, coverage = ert3d_coverage(G, cells, mesh),
       converged = reason ∉ (:max_iter, :line_search_failed, :stalled), reason)
end

#---------- file workflow ----------

_same_grid(a::ModelFile3D, b::ModelFile3D) =
    size(a.resistivity) == size(b.resistivity) && a.x_cell_sizes ≈ b.x_cell_sizes && a.y_cell_sizes ≈ b.y_cell_sizes &&
    a.z_cell_sizes ≈ b.z_cell_sizes && a.origin ≈ b.origin

"""
    Invert3D(start_path, data_path, fwd_path, inv_path, prior_path; run_dir=nothing)

ModEM-style 3D Gauss–Newton inversion from five files: the start model, the observed
data, `fwd.ctrl`, `inv.ctrl` and the prior model (the reference of the regularization).
Every earth cell of the main domain is inverted: the even core columns down to the start
model's `# main domain depth` (written by `MakeMesh3D`); the padding outside it follows the
nearest inverted cell. Without that line every earth cell is inverted. Air stays fixed.

Everything is written to `run_dir`, by default `run_YYYYmmdd_HHMMSS/` next to the data:
`model.rho` (ModEM 3D, restartable), `data.pred`, `coverage.rho` (LINEAR), `History.csv`,
`Summary.txt` and copies of the inputs in `inputs/`. Returns a named tuple with the run
directory, mesh, models, data, history, rms and termination reason.
"""
function Invert3D(start_path::AbstractString, data_path::AbstractString, fwd_path::AbstractString,
                  inv_path::AbstractString, prior_path::AbstractString; run_dir::Union{Nothing, AbstractString} = nothing)
    fwd, ctrl = ReadFwdCtrl3D(fwd_path), ReadInvCtrl3D(inv_path)
    observed = load_data3d(data_path)
    any(isnan, observed.geometric_factor) && error("the data have no geometric factors; run MakeMesh3D on the data file")
    start, prior = ReadModel3D(start_path), ReadModel3D(prior_path)
    _same_grid(start, prior) || error("the prior model must be on the start model's grid")
    mesh, ρ0 = Mesh3DFromInputs(start, observed)
    pmesh, ρref = Mesh3DFromInputs(prior, observed; warn = false)
    ert3d_air_mask(pmesh) == ert3d_air_mask(mesh) || error("the prior model's topography differs from the start model's")
    padding = ert3d_padding_mask(mesh, start.main_depth) .& .!ert3d_air_mask(mesh)
    dir = _inv2d_open_run(run_dir, data_path, (start_path, data_path, fwd_path, inv_path, prior_path))

    options = Inv3DOptions(max_iter = ctrl.max_iter, beta = ctrl.lambda, beta_cooling = ctrl.lambda_cooling,
                           min_beta_ratio = ctrl.min_lambda_ratio, smallness = ctrl.smallness, smooth_x = ctrl.smooth_x,
                           smooth_y = ctrl.smooth_y, smooth_z = ctrl.smooth_z, regularization = ctrl.regularization, data_norm = ctrl.data_norm,
                           model_norm = ctrl.model_norm, error_floor = ctrl.error_floor, log_bounds = ctrl.log_bounds,
                           max_step = ctrl.max_step, max_linesearch = ctrl.max_linesearch, target_rms = ctrl.target_rms,
                           relax_beta = ctrl.relax_lambda, refine = fwd.refine, growth = fwd.growth, damping = ctrl.gn_damping)
    result = Invert3D(mesh, ρ0, observed; options, reference_resistivity = ρref, padding_cells = any(padding) ? padding : nothing)
    predicted = _ert3d_predicted(observed, result.rhoa)
    fit = chi2_rms3d(observed, predicted; error_floor = ctrl.error_floor)

    WriteModel3D(joinpath(dir, "model.rho"), mesh, result.resistivity; main_depth = start.main_depth)
    write_data3d(joinpath(dir, "data.pred"), predicted)
    WriteModel3D(joinpath(dir, "coverage.rho"), mesh, ifelse.(isnan.(result.coverage), 1.0, max.(result.coverage, 1e-12));
                 loge = false)
    _inv2d_write_history(joinpath(dir, "History.csv"), result.history)
    open(joinpath(dir, "Summary.txt"), "w") do io
        println(io, "Algorithm: GN (3D)")
        println(io, "Termination: ", result.reason)
        println(io, "Converged: ", result.converged)
        @printf(io, "RMS: %.6f\n", fit.rms)
        @printf(io, "Final lambda: %.6g\n", result.history[end].beta)
        println(io, "Data norm: ", ctrl.data_norm, ", model norm: ", ctrl.model_norm)
        println(io, "Accepted iterations: ", length(result.history) - 1)
        println(io, "Data used: ", length(result.used), " of ", ndata(observed))
        println(io, "Inverted cells: ", length(result.active_cells))
        println(io, "Padding cells following the nearest inverted cell: ", count(padding))
        println(io, "Topographic air cells: ", count(ert3d_air_mask(mesh)))
    end
    (; run_dir = dir, ctrl, fwd, mesh, observed, predicted, start = ρ0, prior = ρref, final = result.resistivity,
       padding, main_depth = start.main_depth,
       coverage = result.coverage, history = result.history, rms = fit.rms, reason = result.reason, converged = result.converged)
end
