# 3D ERT forward solver
# Author: @pankajkmishra
# Point electrodes over a 3D earth: -∇·σ∇φ = I δ with trilinear (Q1) elements on the rectangular mesh, potentials on
# the mesh nodes, σ per cell. The ground (flat, or the staircase of the topography) is a natural no-flux surface, air
# cells drop out; the sides and the bottom take the mixed condition ∂φ/∂n + (cos θ / r) φ = 0 of Dey and
# Morrison (1979) seen from the centre of the electrodes, so one symmetric matrix serves every source: one Cholesky
# factor, pole fields of every electrode, reciprocity and exact sensitivities. The solve runs on a refinement of the
# model mesh around the electrodes (ert3d_forward_mesh). As in 2D, ρa = R(m) / R(1 Ω·m) on the same mesh

using LinearAlgebra
using Printf
using SparseArrays

#---------- element matrices ----------

const _ERT3D_SIGNS = [(2((a - 1) & 1) - 1, 2(((a - 1) >> 1) & 1) - 1, 2(((a - 1) >> 2) & 1) - 1) for a in 1:8]

# unit-σ stiffness of a trilinear hexahedron, 2 × 2 × 2 Gauss points (exact for the bricks of the mesh); local node
# 1 + i + 2j + 4k for corner offsets (i, j, k) in x, y, z, corner coordinates X (8 × 3)
function _ert3d_element(X::AbstractMatrix{Float64})
    K = zeros(8, 8)
    dN, G = zeros(8, 3), zeros(8, 3)
    g = 1 / sqrt(3)
    for ξ in (-g, g), η in (-g, g), ζ in (-g, g)
        for (a, (sx, sy, sz)) in enumerate(_ERT3D_SIGNS)
            dN[a, 1] = sx * (1 + sy * η) * (1 + sz * ζ) / 8
            dN[a, 2] = sy * (1 + sx * ξ) * (1 + sz * ζ) / 8
            dN[a, 3] = sz * (1 + sx * ξ) * (1 + sy * η) / 8
        end
        J = X' * dN                                   # J[r, c] = ∂x_r / ∂ξ_c
        d = det(J)
        d > 0 || throw(ArgumentError("a mesh cell is folded (non-positive Jacobian)"))
        mul!(G, dN, inv(J))                           # ∇N_a = J⁻ᵀ ∂N_a/∂ξ, as rows
        K .+= (G * G') .* d
    end
    K
end

_ert3d_element(dx::Real, dy::Real, dz::Real) =
    _ert3d_element([(_ERT3D_SIGNS[a][r] + 1) / 2 * (dx, dy, dz)[r] for a in 1:8, r in 1:3])

# mass matrix of a bilinear quadrilateral face in 3D, corners in tensor order (u fastest), 2 × 2 Gauss points
function _ert3d_face_mass(P::AbstractMatrix{Float64})
    M = zeros(4, 4)
    g = 1 / sqrt(3)
    su, sv = (-1.0, 1.0, -1.0, 1.0), (-1.0, -1.0, 1.0, 1.0)
    for u in (-g, g), v in (-g, g)
        N = [(1 + su[a] * u) * (1 + sv[a] * v) / 4 for a in 1:4]
        pu = sum(su[a] * (1 + sv[a] * v) / 4 .* P[a, :] for a in 1:4)
        pv = sum(sv[a] * (1 + su[a] * u) / 4 .* P[a, :] for a in 1:4)
        M .+= (N * N') .* norm(cross(pu, pv))
    end
    M
end

# corner coordinates (8 × 3) of a cell from the node arrays
function _ert3d_corners(xn, yn, Z, c::CartesianIndex{3})
    X = zeros(8, 3)
    for a in 1:8
        i, j, k = c[1] + ((a - 1) & 1), c[2] + (((a - 1) >> 1) & 1), c[3] + (((a - 1) >> 2) & 1)
        X[a, 1], X[a, 2], X[a, 3] = xn[i], yn[j], Z[i, j, k]
    end
    X
end

# outer boundary faces of a cell: (local nodes, face mass, centre, outward normal)
function _ert3d_cell_faces(xn, yn, Z, c::CartesianIndex{3})
    nx, ny, nz = length(xn) - 1, length(yn) - 1, size(Z, 3) - 1
    ix, iy, iz = Tuple(c)
    X = _ert3d_corners(xn, yn, Z, c)
    faces = Tuple{NTuple{4, Int}, Matrix{Float64}, NTuple{3, Float64}, NTuple{3, Float64}}[]
    face(loc, n) = (P = X[collect(loc), :]; push!(faces, (loc, _ert3d_face_mass(P), Tuple(vec(sum(P; dims = 1)) ./ 4), n)))
    ix == 1 && face((1, 3, 5, 7), (-1.0, 0.0, 0.0))
    ix == nx && face((2, 4, 6, 8), (1.0, 0.0, 0.0))
    iy == 1 && face((1, 2, 5, 6), (0.0, -1.0, 0.0))
    iy == ny && face((3, 4, 7, 8), (0.0, 1.0, 0.0))
    iz == nz && face((5, 6, 7, 8), (0.0, 0.0, 1.0))
    faces
end

# mixed boundary coefficient cos θ / r seen from the reference point
function _ert3d_alpha(ref::NTuple{3, Float64}, p::NTuple{3, Float64}, n::NTuple{3, Float64})
    d = p .- ref
    max(sum(d .* n) / sum(abs2, d), 0.0)
end

#---------- σ-independent system ----------

# nodes, electrodes, element matrices (shared by cells of the same shape), the sparse pattern and where each element and
# boundary entry lands in it; built once per mesh
struct ERT3DSystem
    mesh::ERT3DMesh
    earth::BitArray{3}
    elements::Vector{Matrix{Float64}}
    element_of::Array{Int, 3}                  # cell → index into elements
    node::Array{Int, 3}
    cells::Vector{CartesianIndex{3}}
    P::SparseMatrixCSC{Float64, Int}
    Pd::Matrix{Float64}
    ref::NTuple{3, Float64}
    Z::Array{Float64, 3}                       # node depths below the model top
    pattern::SparseMatrixCSC{Float64, Int}
    emap::Matrix{Int}                          # (64, cells): position of each element entry in pattern.nzval
    boundary::Vector{Tuple{Int, Int, Float64}} # (nz index, cell number, unit-σ value)
end

@inline _ert3d_loc(node, c::CartesianIndex{3}) =
    ntuple(a -> node[c[1]+((a-1)&1), c[2]+(((a-1)>>1)&1), c[3]+(((a-1)>>2)&1)], 8)

function _ert3d_system(mesh::ERT3DMesh, earth::BitArray{3})
    nx, ny, nz = size(earth)
    cells = [CartesianIndex(ix, iy, iz) for iz in 1:nz for iy in 1:ny for ix in 1:nx if earth[ix, iy, iz]]
    used = falses(nx + 1, ny + 1, nz + 1)
    for c in cells
        used[c[1]:c[1]+1, c[2]:c[2]+1, c[3]:c[3]+1] .= true
    end
    node = zeros(Int, nx + 1, ny + 1, nz + 1)
    node[used] .= 1:count(used)
    nunk = count(used)
    zn = mesh.z_nodes .- mesh.z_nodes[1]
    Z = [zn[k] for _ in 1:nx+1, _ in 1:ny+1, k in 1:nz+1]
    xn, yn = mesh.x_nodes, mesh.y_nodes

    # electrodes: on the tread of their column, bilinear over its top face
    topo = ert3d_topo_air(mesh)
    ne = length(mesh.electrode_x)
    I, J, V = Int[], Int[], Float64[]
    for (e, c) in enumerate(ert3d_electrode_columns(mesh))
        ix, iy = Tuple(c)
        g = topo[ix, iy] + 1
        tx = clamp((mesh.electrode_x[e] - xn[ix]) / mesh.x_cell_sizes[ix], 0.0, 1.0)
        ty = clamp((mesh.electrode_y[e] - yn[iy]) / mesh.y_cell_sizes[iy], 0.0, 1.0)
        for (jx, wx) in ((ix, 1 - tx), (ix + 1, tx)), (jy, wy) in ((iy, 1 - ty), (iy + 1, ty))
            wx * wy > 1e-12 || continue
            node[jx, jy, g] > 0 || error("electrode $e is not on an earth node")
            push!(I, node[jx, jy, g]); push!(J, e); push!(V, wx * wy)
        end
    end
    P = sparse(I, J, V, nunk, ne)
    depths = ert3d_electrode_depths(mesh)
    ref = ((minimum(mesh.electrode_x) + maximum(mesh.electrode_x)) / 2,
           (minimum(mesh.electrode_y) + maximum(mesh.electrode_y)) / 2, sum(depths) / ne)

    # element matrices per distinct cell shape (corner offsets from the first corner), and the pattern
    keys = Dict{Vector{Float64}, Int}()
    elements = Matrix{Float64}[]
    element_of = zeros(Int, nx, ny, nz)
    shapes = Vector{Matrix{Float64}}(undef, length(cells))
    Threads.@threads for j in eachindex(cells)
        shapes[j] = _ert3d_corners(xn, yn, Z, cells[j])
    end
    for (j, c) in enumerate(cells)
        X = shapes[j]
        key = vec(X .- X[1:1, :])
        element_of[c] = get!(keys, key) do
            push!(elements, zeros(0, 0))
            length(elements)
        end
    end
    first_cell = Dict(v => j for (j, v) in Iterators.reverse(collect(enumerate(element_of[cells]))))
    Threads.@threads for e in eachindex(elements)
        elements[e] = _ert3d_element(shapes[first_cell[e]])
    end
    Is = Vector{Int}(undef, 64 * length(cells))
    Js = similar(Is)
    locs = Vector{NTuple{8, Int}}(undef, length(cells))
    for (j, c) in enumerate(cells)
        locs[j] = _ert3d_loc(node, c)
        for b in 1:8, a in 1:8
            k = 64(j - 1) + 8(b - 1) + a
            Is[k] = locs[j][a]; Js[k] = locs[j][b]
        end
    end
    pattern = sparse(Is, Js, zeros(length(Is)), nunk, nunk)
    position(r, col) = pattern.colptr[col] - 1 + searchsortedfirst(view(pattern.rowval, pattern.colptr[col]:pattern.colptr[col+1]-1), r)
    emap = Matrix{Int}(undef, 64, length(cells))       # entry ab = (a, b) = (mod1(ab, 8), cld(ab, 8))
    Threads.@threads for j in eachindex(cells)
        for ab in 1:64
            emap[ab, j] = position(locs[j][mod1(ab, 8)], locs[j][cld(ab, 8)])
        end
    end

    boundary = Tuple{Int, Int, Float64}[]
    for (j, c) in enumerate(cells)
        (c[1] in (1, nx) || c[2] in (1, ny) || c[3] == nz) || continue
        for (loc, Mf, p, n) in _ert3d_cell_faces(xn, yn, Z, c)
            α = _ert3d_alpha(ref, p, n)
            α > 0 || continue
            for b in 1:4, a in 1:4
                push!(boundary, (position(locs[j][loc[a]], locs[j][loc[b]]), j, α * Mf[a, b]))
            end
        end
    end
    ERT3DSystem(mesh, earth, elements, element_of, node, cells, P, Matrix(P), ref, Z, pattern, emap, boundary)
end

"""
    ERT3DCache()

Reusable state for repeated 3D forward solves on one mesh (inversions): the σ-independent
system and the Cholesky factor, whose symbolic analysis later solves reuse. Rebuilt
automatically when the mesh or the air changes.
"""
mutable struct ERT3DCache
    system::Union{Nothing, ERT3DSystem}
    factor::Any
end

ERT3DCache() = ERT3DCache(nothing, nothing)

_ert3d_same_mesh(a::ERT3DMesh, b::ERT3DMesh) =
    a === b || (a.x_nodes == b.x_nodes && a.y_nodes == b.y_nodes && a.z_nodes == b.z_nodes && a.surface == b.surface &&
                a.electrode_x == b.electrode_x && a.electrode_y == b.electrode_y)

function _ert3d_cached_system(cache::Union{Nothing, ERT3DCache}, mesh, earth)
    cache === nothing && return _ert3d_system(mesh, earth)
    s = cache.system
    if s === nothing || s.earth != earth || !_ert3d_same_mesh(s.mesh, mesh)
        cache.system = _ert3d_system(mesh, earth)
        cache.factor = nothing
    end
    cache.system
end

#---------- forward ----------

"""
    ERT3DForward

A solved 3D forward problem: the conductivity, element matrices, node numbering,
electrode interpolation `P`, the pole fields `G` (unknowns × electrodes, empty unless
kept), the pole-pole transfer resistances `Φ[e, s]` (Ω), and the solve-mesh earth cells of
every model cell (`members`, by linear index).
"""
struct ERT3DForward
    mesh::ERT3DMesh
    σ::Array{Float64, 3}
    elements::Vector{Matrix{Float64}}
    element_of::Array{Int, 3}
    node::Array{Int, 3}
    cells::Vector{CartesianIndex{3}}
    P::SparseMatrixCSC{Float64, Int}
    G::Matrix{Float64}
    Φ::Matrix{Float64}
    ref::NTuple{3, Float64}
    Z::Array{Float64, 3}
    members::Vector{Vector{CartesianIndex{3}}}
end

"""
    ert3d_forward(mesh, ρ; keep_fields=true, cache=nothing, owner=nothing) -> ERT3DForward

Solve the 3D problem for a unit current at every electrode of `mesh` in the resistivity
`ρ` `(nx, ny, nz)` (ohm m; the air of the mesh's staircase drops out). `keep_fields` keeps
the pole fields for sensitivities; `cache` (`ERT3DCache`) keeps the system and the factor
between calls; `owner` (from `ert3d_forward_mesh`) records the model cell of every cell of
a refined mesh.
"""
function ert3d_forward(mesh::ERT3DMesh, ρ::AbstractArray{<:Real, 3}; keep_fields::Bool = true,
                       cache::Union{Nothing, ERT3DCache} = nothing, owner = nothing)
    size(ρ) == _ert3d_size(mesh) || throw(DimensionMismatch("model must be (nx, ny, nz) = $(_ert3d_size(mesh))"))
    earth = .!ert3d_air_mask(mesh)
    all(x -> isfinite(x) && x > 0, ρ[earth]) || throw(ArgumentError("resistivity must be positive and finite"))
    σ = [earth[c] ? 1 / Float64(ρ[c]) : 0.0 for c in CartesianIndices(ρ)]
    sys = _ert3d_cached_system(cache, mesh, earth)
    nzv = zeros(nnz(sys.pattern))
    @inbounds for (j, c) in enumerate(sys.cells)
        sc, E = σ[c], sys.elements[sys.element_of[c]]
        for ab in 1:64
            nzv[sys.emap[ab, j]] += sc * E[ab]
        end
    end
    @inbounds for (idx, j, v) in sys.boundary
        nzv[idx] += σ[sys.cells[j]] * v
    end
    n = size(sys.pattern, 1)
    A = Symmetric(SparseMatrixCSC(n, n, sys.pattern.colptr, sys.pattern.rowval, nzv))
    F = if cache !== nothing && cache.factor !== nothing
        cholesky!(cache.factor, A)
    else
        F0 = cholesky(A)
        cache === nothing || (cache.factor = F0)
        F0
    end
    U = F \ sys.Pd
    Φ = sys.Pd' * U
    own = owner === nothing ? [earth[c] ? LinearIndices(earth)[c] : 0 for c in CartesianIndices(earth)] : owner
    members = [CartesianIndex{3}[] for _ in 1:maximum(own; init = 0)]
    for c in CartesianIndices(own)
        own[c] > 0 && push!(members[own[c]], c)
    end
    ERT3DForward(mesh, σ, sys.elements, sys.element_of, sys.node, sys.cells, sys.P, keep_fields ? U : zeros(0, 0),
                 (Φ + Φ') ./ 2, sys.ref, sys.Z, members)
end

"""
    ert3d_resistance(fwd, data) -> Vector{Float64}

Transfer resistance V_MN / I_AB (Ω) of every datum of `data`; electrode 0 is remote.
"""
ert3d_resistance(f::ERT3DForward, d) =
    [_ert2d_pot(f.Φ, d.m[i], d.a[i]) - _ert2d_pot(f.Φ, d.m[i], d.b[i]) - _ert2d_pot(f.Φ, d.n[i], d.a[i]) +
     _ert2d_pot(f.Φ, d.n[i], d.b[i]) for i in eachindex(d.a)]

#---------- sensitivities ----------

# unit-σ element matrix of a cell, its mixed boundary faces included
function _ert3d_cell_matrix(f::ERT3DForward, c::CartesianIndex{3})
    A = copy(f.elements[f.element_of[c]])
    nx, ny, nz = size(f.σ)
    (c[1] in (1, nx) || c[2] in (1, ny) || c[3] == nz) || return A
    for (loc, Mf, p, n) in _ert3d_cell_faces(f.mesh.x_nodes, f.mesh.y_nodes, f.Z, c)
        α = _ert3d_alpha(f.ref, p, n)
        α > 0 || continue
        for b in 1:4, a in 1:4
            A[loc[a], loc[b]] += α * Mf[a, b]
        end
    end
    A
end

# the solve-mesh cells of a model cell
_ert3d_subcells(f::ERT3DForward, c::CartesianIndex{3}, model_size) =
    (l = LinearIndices(model_size)[c]; l <= length(f.members) ? f.members[l] : CartesianIndex{3}[])

"""
    ert3d_sensitivity(fwd, data, cells) -> Matrix

∂R/∂σ of every transfer resistance of `data` with respect to the conductivity of each of
`cells` (`CartesianIndex(ix, iy, iz)`): −(G_M − G_N)ᵀ Kₑ (G_A − G_B) over the cell's
element matrix. `cells` are model cells; on a refined solve mesh (`ForwardSolve3D`) their
sub-cells are summed. Needs the pole fields.
"""
function ert3d_sensitivity(f::ERT3DForward, d, cells::AbstractVector{CartesianIndex{3}}; model_size = size(f.σ))
    isempty(f.G) && throw(ArgumentError("the forward was solved without its pole fields"))
    nd, ne = length(d.a), size(f.G, 2)
    J = zeros(nd, length(cells))
    Threads.@threads for j in eachindex(cells)
        E = zeros(ne, ne)
        for sc in _ert3d_subcells(f, cells[j], model_size)
            loc = collect(_ert3d_loc(f.node, sc))
            Gc = f.G[loc, :]
            E .+= Gc' * (_ert3d_cell_matrix(f, sc) * Gc)
        end
        @inbounds for i in 1:nd
            a, b, m, n = d.a[i], d.b[i], d.m[i], d.n[i]
            s = 0.0
            for (e, se) in ((m, 1.0), (n, -1.0)), (q, sq) in ((a, 1.0), (b, -1.0))
                (e == 0 || q == 0) || (s += se * sq * E[e, q])
            end
            J[i, j] = -s
        end
    end
    J
end

#---------- responses ----------

"""
    ForwardSolve3D(mesh, ρ, data; refine=4, growth=0.5, homogeneous=nothing, keep_fields=false, cache=nothing)
        -> (; resistance, rhoa, homogeneous, fwd)

Transfer resistances of `data` over `ρ`, and ρa = R(ρ) / R(1 Ω·m), normalised by the same
mesh's response to a 1 ohm m earth with the same topography, so discretisation and
boundary errors cancel and a homogeneous earth returns its resistivity exactly; the data's
K must be those of the real ground (analytic when flat, numerical over topography). The
solve runs on the graded forward mesh (`ert3d_forward_mesh`: cells a/`refine` at the
electrodes, growing by `growth` × the distance).
"""
function ForwardSolve3D(mesh::ERT3DMesh, ρ::AbstractArray{<:Real, 3}, d; refine::Integer = 4, growth::Real = 0.5,
                        homogeneous = nothing, keep_fields::Bool = false, cache::Union{Nothing, ERT3DCache} = nothing)
    fm, fρ, owner = ert3d_forward_mesh(mesh, ρ; refine, growth)
    f = ert3d_forward(fm, fρ; keep_fields, cache, owner)
    R = ert3d_resistance(f, d)
    Rh = homogeneous === nothing ? ert3d_homogeneous(mesh, d; refine, growth, cache) : homogeneous
    (; resistance = R, rhoa = R ./ Rh, homogeneous = Rh, fwd = f)
end

"""
    ert3d_homogeneous(mesh, data; refine=4, growth=0.5, cache=nothing) -> Vector

Transfer resistances over a 1 ohm m earth with the mesh's topography, 1/K of the mesh.
"""
function ert3d_homogeneous(mesh::ERT3DMesh, d; refine::Integer = 4, growth::Real = 0.5,
                           cache::Union{Nothing, ERT3DCache} = nothing)
    fm, fρ, _ = ert3d_forward_mesh(mesh, ones(_ert3d_size(mesh)); refine, growth)
    ert3d_resistance(ert3d_forward(fm, fρ; keep_fields = false, cache), d)
end

"""
    ert3d_numerical_geometric_factor(mesh, data; refine=8, growth=0.5) -> Vector

Geometric factors K = 1/R(1 Ω·m) of `data` over the mesh's ground, solved on a forward mesh
finer than the inversion's (with its fine staircase): the K of the topography. Over flat
ground the analytic `ert3d_geometric_factor` is exact.
"""
ert3d_numerical_geometric_factor(mesh::ERT3DMesh, d; refine::Integer = 8, growth::Real = 0.5) =
    1 ./ ert3d_homogeneous(mesh, d; refine, growth)

"""
    FrechetDerivative3D(mesh, ρ, data; refine=4, growth=0.5, cells=all cells) -> (J, cells)

Sensitivity of log ρa to log10 ρ of the model `cells`.
"""
function FrechetDerivative3D(mesh::ERT3DMesh, ρ::AbstractArray{<:Real, 3}, d; refine::Integer = 4, growth::Real = 0.5,
                             cells = nothing)
    r = ForwardSolve3D(mesh, ρ, d; refine, growth, keep_fields = true)
    air = ert3d_air_mask(mesh)
    cells = cells === nothing ? [c for c in CartesianIndices(air) if !air[c]] : cells
    J = ert3d_sensitivity(r.fwd, d, cells; model_size = size(ρ)) ./ r.resistance
    J .*= [-log(10.0) / ρ[c] for c in cells]'
    J, cells
end

"""
    ForwardSolve3D(model_path, data_path, fwd_path; output_path=nothing) -> path

ModEM-style 3D forward run: the model, the data file and `fwd.ctrl` (3D keys). Writes the
predicted data, by default `data.pred` next to the data file.
"""
function ForwardSolve3D(model_path::AbstractString, data_path::AbstractString, fwd_path::AbstractString;
                        output_path::Union{Nothing, AbstractString} = nothing)
    fwd = ReadFwdCtrl3D(fwd_path)
    data = load_data3d(data_path)
    any(isnan, data.geometric_factor) && error("$data_path: geometric factors are missing; run MakeMesh3D on it")
    mesh, ρ = Mesh3DFromInputs(ReadModel3D(model_path), data)
    out = output_path === nothing ? joinpath(dirname(abspath(data_path)), "data.pred") : String(output_path)
    r = ForwardSolve3D(mesh, ρ, data; refine = fwd.refine, growth = fwd.growth)
    write_data3d(out, _ert3d_predicted(data, r.rhoa))
end
