# 2.5D ERT forward solver
# Author: @pankajkmishra
# Point electrodes over a 2D earth: the potential is cosine-transformed across the profile, φ = (2/π) ∫ φ̃(k) dk,
# and each wavenumber solves -∇·σ∇φ̃ + k²σφ̃ = I/2 δ with bilinear (Q1) elements on the rectangular mesh, potentials
# on the mesh nodes, σ per cell. The ground (flat, or the staircase of the topography) is a natural no-flux surface,
# air cells drop out; the left, right and bottom boundaries take the mixed condition of Dey and Morrison (1979)
# seen from the centre of the electrode spread, so every wavenumber has one symmetric matrix for all sources:
# one Cholesky factor, pole fields of every electrode, reciprocity and exact sensitivities
# Wavenumbers and weights: nonnegative least squares so that Σ wₖ K₀(k r) = π/(2r) from a quarter of the electrode
# spacing to the line length
# The solve runs on a refinement of the model mesh around the electrodes (ert2d_forward_mesh), which resolves the
# sources and cuts a fine staircase under them

using LinearAlgebra
using Printf
using SparseArrays

#---------- modified Bessel functions ----------

# e^x K₀(x) and e^x K₁(x), Abramowitz and Stegun 9.8.1-9.8.8 (relative error below 1e-7)
function _besselk01e(x::Float64)
    x > 0 || throw(DomainError(x, "the modified Bessel functions need x > 0"))
    if x <= 2
        t = (x / 3.75)^2
        i0 = 1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
        i1 = x * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))))
        y = x * x / 4
        k0 = -log(x / 2) * i0 + (-0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.03488590 +
             y * (0.00262698 + y * (0.00010750 + y * 0.0000074))))))
        k1 = (x * log(x / 2) * i1 + (1 + y * (0.15443144 + y * (-0.67278579 + y * (-0.18156897 +
             y * (-0.01919402 + y * (-0.00110404 + y * (-0.00004686)))))))) / x
        return k0 * exp(x), k1 * exp(x)
    end
    y = 2 / x
    k0 = (1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446 + y * (0.00587872 +
         y * (-0.00251540 + y * 0.00053208)))))) / sqrt(x)
    k1 = (1.25331414 + y * (0.23498619 + y * (-0.03655620 + y * (0.01504268 + y * (-0.00780353 +
         y * (0.00325614 + y * (-0.00068245))))))) / sqrt(x)
    k0, k1
end

_besselk0(x) = _besselk01e(Float64(x))[1] * exp(-x)

#---------- wavenumbers ----------

# Lawson-Hanson nonnegative least squares
function _nnls(A::AbstractMatrix, b::AbstractVector; maxit::Int = 10size(A, 2))
    n = size(A, 2)
    x, P = zeros(n), falses(n)
    w = A' * (b - A * x)
    for _ in 1:maxit
        (all(P) || maximum(w[.!P]; init = -Inf) <= 1e-12) && break
        P[argmax(ifelse.(P, -Inf, w))] = true
        while true
            z = zeros(n)
            z[P] = A[:, P] \ b
            all(z[P] .> 0) && (x = z; break)
            α = minimum(x[i] / (x[i] - z[i]) for i in findall(P) if z[i] <= 0)
            x .+= α .* (z .- x)
            P .&= x .> 1e-14
        end
        w = A' * (b - A * x)
    end
    x
end

"""
    ert2d_wavenumbers(r_min, r_max; n=16) -> (k, w, error)

Cross-profile wavenumbers (1/m) and weights of the inverse cosine transform: `n`
log-spaced k over [0.1/r_max, 5/r_min], weights by nonnegative least squares so that
Σ wₖ K₀(k r) = π/(2r) for r over [r_min, r_max]; wavenumbers of zero weight are
dropped. `error` is the largest relative error of the fit.
"""
function ert2d_wavenumbers(rmin::Real, rmax::Real; n::Integer = 16)
    0 < rmin <= rmax || throw(ArgumentError("need 0 < r_min ≤ r_max"))
    rmax = max(rmax, 1.5rmin)
    k = 10 .^ range(log10(0.1 / rmax), log10(5 / rmin), length = n)
    fit(r) = [_besselk0(kj * ri) * 2ri / π for ri in r, kj in k]
    w = _nnls(fit(10 .^ range(log10(rmin), log10(rmax), length = 200)), ones(200))
    used = w .> 0
    r = 10 .^ range(log10(rmin), log10(rmax), length = 1000)
    err = maximum(abs.(fit(r)[:, used] * w[used] .- 1))
    k[used], w[used], err
end

#---------- element matrices ----------

# unit-σ stiffness and mass of a dy × dz bilinear rectangle; local nodes (z1,y1), (z1,y2), (z2,y1), (z2,y2)
function _ert2d_element(dy::Real, dz::Real)
    sy, sz = (-1.0, 1.0, -1.0, 1.0), (-1.0, -1.0, 1.0, 1.0)
    S, M = zeros(4, 4), zeros(4, 4)
    for a in 1:4, b in 1:4
        py, pz = sy[a] * sy[b], sz[a] * sz[b]          # 1 for nodes on the same side, -1 across
        ly, lz = (3 + py) / 6, (3 + pz) / 6              # 1D mass factors: 2/3 or 1/3 of the half width
        S[a, b] = py * dz / dy * lz / 2 + pz * dy / dz * ly / 2
        M[a, b] = dy * dz * ly * lz / 4
    end
    S, M
end

# boundary edges of a cell: (local nodes, length, midpoint y, z, outward normal y, z); the ground is a no-flux surface
function _ert2d_cell_edges(yn, Z, iz::Int, iy::Int)
    nz, ny = size(Z, 1) - 1, size(Z, 2) - 1
    edges = NTuple{6, Any}[]
    iy == 1 && push!(edges, ((1, 3), Z[iz+1, 1] - Z[iz, 1], yn[1], (Z[iz, 1] + Z[iz+1, 1]) / 2, -1.0, 0.0))
    iy == ny && push!(edges, ((2, 4), Z[iz+1, end] - Z[iz, end], yn[end], (Z[iz, end] + Z[iz+1, end]) / 2, 1.0, 0.0))
    iz == nz && push!(edges, ((3, 4), yn[iy+1] - yn[iy], (yn[iy] + yn[iy+1]) / 2, Z[end, iy], 0.0, 1.0))
    edges
end

# mixed boundary coefficient α = k K₁(kr)/K₀(kr) cos θ seen from the reference point
function _ert2d_alpha(k::Float64, ref::NTuple{2, Float64}, y::Float64, z::Float64, ny::Float64, nz::Float64)
    dy, dz = y - ref[1], z - ref[2]
    r = hypot(dy, dz)
    k0, k1 = _besselk01e(k * r)
    max(k * k1 / k0 * (dy * ny + dz * nz) / r, 0.0)
end

#---------- forward ----------

"""
    ERT2DForward

A solved 2.5D forward problem on one mesh and resistivity: the element matrices, the node numbering, the electrode interpolation `P`, the wavenumbers `k` and weights `w`, the pole fields `G`
(one matrix per wavenumber, unknowns × electrodes, empty unless kept) and the
pole-pole transfer resistances `Φ[e, s]`, the potential at electrode e for a unit
current at electrode s (Ω).
"""
struct ERT2DForward
    mesh::ERT2DMesh
    σ::Matrix{Float64}
    S::Array{Float64, 3}            # unit-σ element stiffness and mass per cell (4, 4, nz ny)
    M::Array{Float64, 3}
    node::Matrix{Int}
    cells::Vector{CartesianIndex{2}}
    P::SparseMatrixCSC{Float64, Int}
    k::Vector{Float64}
    w::Vector{Float64}
    G::Vector{Matrix{Float64}}
    Φ::Matrix{Float64}
    ref::NTuple{2, Float64}
    Z::Matrix{Float64}              # node depths below the model top (nz + 1, ny + 1)
    members::Vector{Vector{CartesianIndex{2}}}   # solve-mesh earth cells of each model cell (linear index)
end

# wavenumbers of a mesh's electrodes: the transform is fitted from a quarter of the smallest electrode distance (the
# fields in the cells next to an electrode enter the sensitivities) to the length of the line
function _ert2d_mesh_wavenumbers(mesh::ERT2DMesh, n::Integer)
    y = sort(unique(round.(mesh.electrode_positions; digits = 6)))
    length(y) >= 2 || throw(ArgumentError("at least two distinct electrode positions are needed"))
    k, w, _ = ert2d_wavenumbers(minimum(diff(y)) / 4, y[end] - y[1]; n)
    k, w
end

#---------- σ-independent system ----------

# everything of a forward that does not depend on σ: nodes, electrodes, wavenumbers, element matrices, the sparse
# pattern and where each element and boundary entry lands in it; built once per mesh and wavenumber count
struct ERT2DSystem
    mesh::ERT2DMesh
    earth::BitMatrix
    wavenumbers::Int
    S::Array{Float64, 3}                    # unit-σ element stiffness and mass per cell (4, 4, nz ny)
    M::Array{Float64, 3}
    node::Matrix{Int}
    cells::Vector{CartesianIndex{2}}
    P::SparseMatrixCSC{Float64, Int}
    Pd::Matrix{Float64}
    k::Vector{Float64}
    w::Vector{Float64}
    ref::NTuple{2, Float64}
    Z::Matrix{Float64}
    pattern::SparseMatrixCSC{Float64, Int}  # structure of the system matrix
    emap::Matrix{Int}                       # (16, cells): position of each element entry in pattern.nzval
    boundary::Vector{Vector{Tuple{Int, Int, Float64}}}   # per wavenumber: (nz index, cell number, unit-σ value)
end

function _ert2d_system(mesh::ERT2DMesh, earth::BitMatrix, wavenumbers::Integer)
    nz, ny = size(earth)
    cells = [CartesianIndex(iz, iy) for iy in 1:ny for iz in 1:nz if earth[iz, iy]]
    used = falses(nz + 1, ny + 1)
    for c in cells
        iz, iy = Tuple(c)
        used[iz:iz+1, iy:iy+1] .= true
    end
    node = zeros(Int, nz + 1, ny + 1)
    node[used] .= 1:count(used)
    nunk = count(used)
    zn = mesh.z_nodes .- mesh.z_nodes[1]
    Z = [zn[iz] for iz in 1:nz+1, _ in 1:ny+1]

    # electrodes: on the tread of their column, linear along its top edge
    cols = ert2d_electrode_columns(mesh)
    topo = ert2d_topo_air(mesh)
    ne = length(mesh.electrode_positions)
    I, J, V = Int[], Int[], Float64[]
    for (e, iy) in enumerate(cols)
        g = topo[iy] + 1
        t = clamp((mesh.electrode_positions[e] - mesh.y_nodes[iy]) / mesh.y_cell_sizes[iy], 0.0, 1.0)
        for (jy, wt) in ((iy, 1 - t), (iy + 1, t))
            wt > 1e-12 || continue
            node[g, jy] > 0 || error("electrode $e is not on an earth node")
            push!(I, node[g, jy]); push!(J, e); push!(V, wt)
        end
    end
    P = sparse(I, J, V, nunk, ne)
    ref = ((minimum(mesh.electrode_positions) + maximum(mesh.electrode_positions)) / 2,
           sum(ert2d_electrode_depths(mesh)) / ne)

    # element matrices and the pattern
    lin = LinearIndices((nz, ny))
    Se, Me = zeros(4, 4, nz * ny), zeros(4, 4, nz * ny)
    Is, Js = Int[], Int[]
    locs = Vector{NTuple{4, Int}}(undef, length(cells))
    for (j, c) in enumerate(cells)
        iz, iy = Tuple(c)
        S, M = _ert2d_element(mesh.y_cell_sizes[iy], mesh.z_cell_sizes[iz])
        Se[:, :, lin[c]] .= S
        Me[:, :, lin[c]] .= M
        locs[j] = (node[iz, iy], node[iz, iy+1], node[iz+1, iy], node[iz+1, iy+1])
        for a in 1:4, b in 1:4
            push!(Is, locs[j][a]); push!(Js, locs[j][b])
        end
    end
    pattern = sparse(Is, Js, zeros(length(Is)), nunk, nunk)
    position(r, c) = pattern.colptr[c] - 1 + searchsortedfirst(view(pattern.rowval, pattern.colptr[c]:pattern.colptr[c+1]-1), r)
    emap = Matrix{Int}(undef, 16, length(cells))       # entry ab = (a, b) = (mod1(ab, 4), cld(ab, 4))
    for j in eachindex(cells), ab in 1:16
        emap[ab, j] = position(locs[j][mod1(ab, 4)], locs[j][cld(ab, 4)])
    end

    # mixed boundary entries per wavenumber
    k, w = _ert2d_mesh_wavenumbers(mesh, wavenumbers)
    boundary = map(k) do kk
        entries = Tuple{Int, Int, Float64}[]
        for (j, c) in enumerate(cells), (loc, len, y, z, nyv, nzv) in _ert2d_cell_edges(mesh.y_nodes, Z, Tuple(c)...)
            α = _ert2d_alpha(kk, ref, y, z, nyv, nzv)
            p, q = locs[j][loc[1]], locs[j][loc[2]]
            push!(entries, (position(p, p), j, α * len / 3), (position(q, q), j, α * len / 3),
                  (position(p, q), j, α * len / 6), (position(q, p), j, α * len / 6))
        end
        entries
    end
    ERT2DSystem(mesh, earth, Int(wavenumbers), Se, Me, node, cells, P, Matrix(P), k, w, ref, Z, pattern, emap, boundary)
end

"""
    ERT2DCache()

Reusable state for repeated forward solves on one mesh (inversions): the
σ-independent system (nodes, element matrices, sparse pattern) and one Cholesky factor
per wavenumber, whose symbolic analysis later solves reuse. It is rebuilt automatically
when the mesh, the air or the wavenumber count changes. One cache per thread.
"""
mutable struct ERT2DCache
    system::Union{Nothing, ERT2DSystem}
    factors::Vector{Any}
end

ERT2DCache() = ERT2DCache(nothing, Any[])

_ert2d_same_mesh(a::ERT2DMesh, b::ERT2DMesh) =
    a === b || (a.y_nodes == b.y_nodes && a.z_nodes == b.z_nodes && a.surface == b.surface &&
                a.electrode_positions == b.electrode_positions)

function _ert2d_cached_system(cache::Union{Nothing, ERT2DCache}, mesh, earth, wavenumbers)
    cache === nothing && return _ert2d_system(mesh, earth, wavenumbers)
    s = cache.system
    if s === nothing || s.wavenumbers != wavenumbers || s.earth != earth || !_ert2d_same_mesh(s.mesh, mesh)
        cache.system = _ert2d_system(mesh, earth, wavenumbers)
        cache.factors = Any[]
    end
    cache.system
end

#---------- forward ----------

"""
    ert2d_forward(mesh, ρ; wavenumbers=16, keep_fields=true, threaded=false, cache=nothing, owner=nothing) -> ERT2DForward

Solve the 2.5D problem for a unit current at every electrode of `mesh` in the
resistivity `ρ` `(nz, ny)` (ohm m; the air of the mesh's staircase drops out). `keep_fields` keeps the
pole fields for sensitivities; `threaded` spreads the wavenumbers over the Julia threads
(off by default: on the test machine, Apple silicon, they gave no gain).
`cache` (`ERT2DCache`) keeps the σ-independent system and the factors between calls.
`owner` (from `ert2d_forward_mesh`) records the model cell of every cell of a refined mesh.
"""
function ert2d_forward(mesh::ERT2DMesh, ρ::AbstractMatrix{<:Real}; wavenumbers::Integer = 16,
                       keep_fields::Bool = true, threaded::Bool = false, cache::Union{Nothing, ERT2DCache} = nothing,
                       owner = nothing)
    nz, ny = length(mesh.z_cell_sizes), length(mesh.y_cell_sizes)
    size(ρ) == (nz, ny) || throw(DimensionMismatch("model must be (nz, ny) = ($nz, $ny)"))
    earth = .!ert2d_air_mask(mesh)
    all(x -> isfinite(x) && x > 0, ρ[earth]) || throw(ArgumentError("resistivity must be positive and finite"))
    σ = [earth[c] ? 1 / Float64(ρ[c]) : 0.0 for c in CartesianIndices(ρ)]
    sys = _ert2d_cached_system(cache, mesh, earth, wavenumbers)

    # σ-weighted stiffness and mass values in the fixed pattern
    lin = LinearIndices(σ)
    snz, mnz = zeros(nnz(sys.pattern)), zeros(nnz(sys.pattern))
    @inbounds for (j, c) in enumerate(sys.cells)
        sc, l = σ[c], lin[c]
        for ab in 1:16
            a, b = mod1(ab, 4), cld(ab, 4)
            snz[sys.emap[ab, j]] += sc * sys.S[a, b, l]
            mnz[sys.emap[ab, j]] += sc * sys.M[a, b, l]
        end
    end

    k, w, Pd = sys.k, sys.w, sys.Pd
    nunk = size(sys.pattern, 1)
    G = Vector{Matrix{Float64}}(undef, length(k))
    Φk = Vector{Matrix{Float64}}(undef, length(k))
    factors = cache === nothing ? nothing : cache.factors
    function solve(ik)
        nzv = snz .+ k[ik]^2 .* mnz
        @inbounds for (idx, j, v) in sys.boundary[ik]
            nzv[idx] += σ[sys.cells[j]] * v
        end
        A = Symmetric(SparseMatrixCSC(nunk, nunk, sys.pattern.colptr, sys.pattern.rowval, nzv))
        F = if factors !== nothing && isassigned(factors, ik)
            cholesky!(factors[ik], A)
        else
            F0 = cholesky(A)
            factors === nothing || (factors[ik] = F0)
            F0
        end
        U = F \ Pd
        Φk[ik] = (w[ik] / π) .* (Pd' * U)
        keep_fields && (G[ik] = U)
        nothing
    end
    factors === nothing || length(factors) >= length(k) || resize!(factors, length(k))
    if threaded && Threads.nthreads() > 1
        Threads.@threads for ik in eachindex(k)
            solve(ik)
        end
    else
        foreach(solve, eachindex(k))
    end
    Φ = sum(Φk)
    ERT2DForward(mesh, σ, sys.S, sys.M, sys.node, sys.cells, sys.P, k, w, keep_fields ? G : Matrix{Float64}[],
                 (Φ + Φ') ./ 2, sys.ref, sys.Z, _ert2d_members(owner === nothing ? _ert2d_identity(earth) : owner))
end

# Φ[e, s], 0 for a remote electrode
@inline _ert2d_pot(Φ, e::Int, s::Int) = (e == 0 || s == 0) ? 0.0 : Φ[e, s]

"""
    ert2d_resistance(fwd, data) -> Vector{Float64}

Transfer resistance V_MN / I_AB (Ω) of every datum of `data` from the pole-pole
potentials of `fwd`; electrode 0 is remote.
"""
function ert2d_resistance(f::ERT2DForward, d)
    Φ = f.Φ
    [_ert2d_pot(Φ, d.m[i], d.a[i]) - _ert2d_pot(Φ, d.m[i], d.b[i]) - _ert2d_pot(Φ, d.n[i], d.a[i]) +
     _ert2d_pot(Φ, d.n[i], d.b[i]) for i in eachindex(d.a)]
end

#---------- sensitivities ----------

# unit-σ element matrix of cell (iz, iy) at wavenumber k, its mixed boundary edges included
function _ert2d_cell_matrix(f::ERT2DForward, iz::Int, iy::Int, k::Float64)
    l = LinearIndices(f.σ)[iz, iy]
    A = f.S[:, :, l] .+ k^2 .* f.M[:, :, l]
    for (loc, len, y, z, ny, nz) in _ert2d_cell_edges(f.mesh.y_nodes, f.Z, iz, iy)
        α = _ert2d_alpha(k, f.ref, y, z, ny, nz)
        A[loc[1], loc[1]] += α * len / 3
        A[loc[2], loc[2]] += α * len / 3
        A[loc[1], loc[2]] += α * len / 6
        A[loc[2], loc[1]] += α * len / 6
    end
    A
end

# local unknowns of an earth cell
@inline _ert2d_loc(f::ERT2DForward, iz, iy) = (f.node[iz, iy], f.node[iz, iy+1], f.node[iz+1, iy], f.node[iz+1, iy+1])

# the model cell (linear index) of every earth cell of an unrefined mesh, 0 for air
_ert2d_identity(earth) = [earth[c] ? LinearIndices(earth)[c] : 0 for c in CartesianIndices(earth)]

# the solve-mesh cells of every model cell, from the owner map
function _ert2d_members(owner)
    members = [CartesianIndex{2}[] for _ in 1:maximum(owner; init = 0)]
    for c in CartesianIndices(owner)
        owner[c] > 0 && push!(members[owner[c]], c)
    end
    members
end

# the solve-mesh cells of a model cell
_ert2d_subcells(f::ERT2DForward, c::CartesianIndex{2}, model_size) =
    (l = LinearIndices(model_size)[c]; l <= length(f.members) ? f.members[l] : CartesianIndex{2}[])

"""
    ert2d_sensitivity(fwd, data, cells) -> Matrix

∂R/∂σ, the derivative of every transfer resistance of `data` with respect to the
conductivity of each of `cells` (`CartesianIndex(iz, iy)`):
−(1/π) Σₖ wₖ (G_M − G_N)ᵀ Kₑ (G_A − G_B) over the cell's element matrix. `cells` are
model cells; on a refined solve mesh (`ForwardSolve2D`) their sub-cells are summed. Needs
the pole fields (`keep_fields = true`).
"""
function ert2d_sensitivity(f::ERT2DForward, d, cells::AbstractVector{CartesianIndex{2}}; model_size = size(f.σ))
    isempty(f.G) && throw(ArgumentError("the forward was solved without its pole fields"))
    nd, nc = length(d.a), length(cells)
    J = zeros(nd, nc)
    quads = [(d.a[i], d.b[i], d.m[i], d.n[i]) for i in 1:nd]
    Threads.@threads for j in 1:nc
        u, v, Kv = zeros(4), zeros(4), zeros(4)
        for sc in _ert2d_subcells(f, cells[j], model_size)
            iz, iy = Tuple(sc)
            loc = _ert2d_loc(f, iz, iy)
            for ik in eachindex(f.k)
                Ke = _ert2d_cell_matrix(f, iz, iy, f.k[ik])
                Gk = f.G[ik]
                c = -f.w[ik] / π
                for (i, (a, b, m, n)) in enumerate(quads)
                    for l in 1:4
                        p = loc[l]
                        v[l] = (a == 0 ? 0.0 : Gk[p, a]) - (b == 0 ? 0.0 : Gk[p, b])
                        u[l] = (m == 0 ? 0.0 : Gk[p, m]) - (n == 0 ? 0.0 : Gk[p, n])
                    end
                    mul!(Kv, Ke, v)
                    J[i, j] += c * dot(u, Kv)
                end
            end
        end
    end
    J
end

"""
    ert2d_sensitivity_transpose(fwd, data, weights, cells) -> Vector

Σᵢ weightsᵢ ∂Rᵢ/∂σ for each of `cells`, without forming the sensitivity matrix.
"""
function ert2d_sensitivity_transpose(f::ERT2DForward, d, weights::AbstractVector{<:Real},
                                     cells::AbstractVector{CartesianIndex{2}}; model_size = size(f.σ))
    isempty(f.G) && throw(ArgumentError("the forward was solved without its pole fields"))
    ne = size(f.P, 2)
    C = zeros(ne, ne)                                   # C[e, s]: weight of G_eᵀ K G_s
    for i in eachindex(d.a)
        c = weights[i]
        c == 0 && continue
        for (e, se) in ((d.m[i], 1.0), (d.n[i], -1.0)), (s, ss) in ((d.a[i], 1.0), (d.b[i], -1.0))
            (e == 0 || s == 0) || (C[e, s] += c * se * ss)
        end
    end
    H = [Gk * C for Gk in f.G]                          # (G C)[:, s] = Σ_e C[e, s] G_e
    g = zeros(length(cells))
    Threads.@threads for j in eachindex(cells)
        for sc in _ert2d_subcells(f, cells[j], model_size)
            iz, iy = Tuple(sc)
            loc = collect(_ert2d_loc(f, iz, iy))
            for ik in eachindex(f.k)
                Ke = _ert2d_cell_matrix(f, iz, iy, f.k[ik])
                g[j] += -f.w[ik] / π * sum(H[ik][loc, :] .* (Ke * f.G[ik][loc, :]))
            end
        end
    end
    g
end

#---------- responses ----------

"""
    ForwardSolve2D(mesh, ρ, data; wavenumbers=16, refine=8, growth=0.25, homogeneous=nothing,
                   keep_fields=false, cache=nothing) -> (; resistance, rhoa, homogeneous, fwd)

Transfer resistances of `data` over `ρ`, and the apparent resistivity normalised by the
same mesh's response to a 1 ohm m earth with the same topography,
ρa = R(ρ) / R(1 Ω·m): discretisation errors near the electrodes cancel and a
homogeneous earth returns its resistivity exactly. The data's geometric factors must be
those of the real ground (analytic over flat ground, `ert2d_numerical_geometric_factor`
over topography), so that ρa_obs = K R_obs compares with this ρa. The solve runs on the
graded forward mesh (`ert2d_forward_mesh`: cells a/`refine` at the electrodes, growing by
`growth` × the distance), the model cells keep their resistivity. `homogeneous` passes
R(1 Ω·m) when it is already known; `cache` (`ERT2DCache`) speeds up repeated solves.
"""
function ForwardSolve2D(mesh::ERT2DMesh, ρ::AbstractMatrix{<:Real}, d; wavenumbers::Integer = 16, refine::Integer = 8,
                        growth::Real = 0.25, homogeneous = nothing, keep_fields::Bool = false, threaded::Bool = false,
                        cache::Union{Nothing, ERT2DCache} = nothing)
    fm, fρ, owner = ert2d_forward_mesh(mesh, ρ; refine, growth)
    f = ert2d_forward(fm, fρ; wavenumbers, keep_fields, threaded, cache, owner)
    R = ert2d_resistance(f, d)
    Rh = homogeneous === nothing ? ert2d_homogeneous(mesh, d; wavenumbers, refine, growth, threaded, cache) : homogeneous
    (; resistance = R, rhoa = R ./ Rh, homogeneous = Rh, fwd = f)
end

"""
    ert2d_homogeneous(mesh, data; wavenumbers=16, refine=8, growth=0.25) -> Vector

Transfer resistances over a 1 ohm m earth with the mesh's topography, 1/K of the mesh
(solved on the graded forward mesh, `ert2d_forward_mesh`).
"""
function ert2d_homogeneous(mesh::ERT2DMesh, d; wavenumbers::Integer = 16, refine::Integer = 8, growth::Real = 0.25,
                           threaded::Bool = false, cache::Union{Nothing, ERT2DCache} = nothing)
    fm, fρ, _ = ert2d_forward_mesh(mesh, ones(length(mesh.z_cell_sizes), length(mesh.y_cell_sizes)); refine, growth)
    ert2d_resistance(ert2d_forward(fm, fρ; wavenumbers, keep_fields = false, threaded, cache), d)
end

"""
    ert2d_numerical_geometric_factor(mesh, data; refine=32, growth=0.15, wavenumbers=20) -> Vector

Geometric factors K = 1/R(1 Ω·m) of `data` over the mesh's ground, solved on a forward
mesh finer than the inversion's (cells a/`refine` at the electrodes, with its fine
staircase): the K of the topography, which `ForwardSolve2D` normalises against. Over flat
ground the analytic `ert2d_geometric_factor` is exact.
"""
ert2d_numerical_geometric_factor(mesh::ERT2DMesh, d; refine::Integer = 32, growth::Real = 0.15, wavenumbers::Integer = 20) =
    1 ./ ert2d_homogeneous(mesh, d; wavenumbers, refine, growth)

# predicted data on the observed survey: ρa, and R = ρa / K with the observed K and errors
function _ert2d_predicted(observed::DataFile2D, rhoa::AbstractVector; title = "ERTGeophysics.jl predicted data")
    K = observed.geometric_factor
    _ert2d_with(observed; title = title, resistance = rhoa ./ K, rhoa = Float64.(rhoa))
end

"""
    WriteFrechet2D(path, J, cells) -> path

Write a sensitivity matrix: a header with the size, one line per cell `iz iy` (the
model rows and columns), then one row per datum.
"""
function WriteFrechet2D(path::AbstractString, J::AbstractMatrix, cells::AbstractVector{CartesianIndex{2}})
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "# ∂log ρa / ∂log10 ρ written by ERTGeophysics.jl: n_data n_cells, cells (iz iy), rows")
        @printf(io, "%d %d\n", size(J, 1), size(J, 2))
        foreach(c -> @printf(io, "%d %d\n", c[1], c[2]), cells)
        for i in axes(J, 1)
            println(io, join((@sprintf("%.6e", x) for x in J[i, :]), ' '))
        end
    end
    String(path)
end

"""
    FrechetDerivative2D(mesh, ρ, data; wavenumbers=16, refine=8, growth=0.25, cells=all cells) -> (J, cells)

Sensitivity of log ρa to log10 ρ of the model `cells`.
"""
function FrechetDerivative2D(mesh::ERT2DMesh, ρ::AbstractMatrix{<:Real}, d; wavenumbers::Integer = 16,
                             refine::Integer = 8, growth::Real = 0.25, cells = nothing)
    r = ForwardSolve2D(mesh, ρ, d; wavenumbers, refine, growth, keep_fields = true)
    air = ert2d_air_mask(mesh)
    cells = cells === nothing ? [c for c in CartesianIndices(air) if !air[c]] : cells
    J = ert2d_sensitivity(r.fwd, d, cells; model_size = size(ρ)) ./ r.resistance
    J .*= [-log(10.0) / ρ[c] for c in cells]'
    J, cells
end

"""
    ForwardSolve2D(model_path, data_path, fwd_path; output_path=nothing) -> path

ModEM-style forward run: the model, the data file (electrodes, quadrupoles, K and
errors) and `fwd.ctrl`. Writes the predicted data, by default `data.pred` next to the
data file: ρa from the normalised response and R = ρa / K with the observed K. With
`Write Frechet derivative : yes` the sensitivity goes to `frechet.dat` beside it.
"""
function ForwardSolve2D(model_path::AbstractString, data_path::AbstractString, fwd_path::AbstractString;
                        output_path::Union{Nothing, AbstractString} = nothing)
    fwd = ReadFwdCtrl2D(fwd_path)
    data = load_data2d(data_path)
    mesh, ρ = Mesh2DFromInputs(ReadModel2D(model_path), data)
    any(isnan, data.geometric_factor) && error("$data_path: geometric factors are missing; run MakeMesh2D on it")
    out = output_path === nothing ? joinpath(dirname(abspath(data_path)), "data.pred") : String(output_path)
    if fwd.write_frechet
        J, cells = FrechetDerivative2D(mesh, ρ, data; wavenumbers = fwd.wavenumbers, refine = fwd.refine, growth = fwd.growth)
        WriteFrechet2D(joinpath(dirname(out), "frechet.dat"), J, cells)
    end
    r = ForwardSolve2D(mesh, ρ, data; wavenumbers = fwd.wavenumbers, refine = fwd.refine, growth = fwd.growth)
    write_data2d(out, _ert2d_predicted(data, r.rhoa))
end
