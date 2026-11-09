# Data errors from normal and reciprocal measurements
# Author: @pankajkmishra

using Printf
using Statistics

# dipole with its smaller electrode first (remote last), and the sign of the swap
_ert2d_dipole(p, q) = (p == 0 || (q != 0 && q < p)) ? ((q, p), -1) : ((p, q), 1)

"""
    Reciprocals2D(data; max_relative=0.1, floor_fraction=0.01, keep_unpaired=true, bins=20)
        -> (; data, pairs, dropped, e0, e1, relative, unpaired)

Errors from normal and reciprocal measurements (current and potential dipoles swapped,
R(AB, MN) = R(MN, AB) by reciprocity).
- Pairs (any orientation; repeated measurements averaged) give R = the mean and the
  reciprocal difference |ΔR| = |R_normal − R_reciprocal|.
- Pairs with |ΔR|/|R| above `max_relative` are dropped as unreliable.
- The error model σ = e0 + e1 |R| is fitted to the mean |ΔR| in `bins` bins of |R|
  (equal counts), a standard model for ERT (Slater et al. 2000; Koestel et al. 2008), and
  applied to every datum, at least `floor_fraction` × |R|.
- The result has one datum per pair (the normal's orientation) and, with
  `keep_unpaired`, the unpaired data with modelled errors.
"""
function Reciprocals2D(d::DataFile2D; max_relative::Real = 0.1, floor_fraction::Real = 0.01, keep_unpaired::Bool = true,
                       bins::Integer = 20)
    groups = Dict{NTuple{4, Int}, Vector{Tuple{Int, Bool, Float64}}}()     # key => (index, normal, signed R)
    for i in 1:ndata(d)
        isfinite(d.resistance[i]) || continue
        C, sc = _ert2d_dipole(d.a[i], d.b[i])
        P, sp = _ert2d_dipole(d.m[i], d.n[i])
        normal = C <= P
        key = normal ? (C..., P...) : (P..., C...)
        push!(get!(groups, key, Tuple{Int, Bool, Float64}[]), (i, normal, sc * sp * d.resistance[i]))
    end
    paired, unpaired = Tuple{Int, Float64, Float64}[], Int[]       # (index of a normal, mean R in its sign, |ΔR|)
    for (_, g) in groups
        nrm = [x for x in g if x[2]]
        rec = [x for x in g if !x[2]]
        if isempty(nrm) || isempty(rec)
            append!(unpaired, first.(g))
            continue
        end
        Rn, Rr = mean(last.(nrm)), mean(last.(rec))
        i = nrm[1][1]
        sgn = d.resistance[i] / nrm[1][3]                           # back to the normal's orientation
        push!(paired, (i, sgn * (Rn + Rr) / 2, abs(Rn - Rr)))
    end
    isempty(paired) && error("no normal/reciprocal pairs in the data")
    rel = [p[3] / max(abs(p[2]), eps()) for p in paired]
    good = rel .<= max_relative
    count(good) >= 3 || error("fewer than 3 reciprocal pairs within $(100max_relative)%")

    # σ = e0 + e1 |R| through the mean |ΔR| of equal-count bins of |R|
    gp = paired[good]
    order = sortperm([abs(p[2]) for p in gp])
    nb = clamp(length(gp) ÷ 10, 1, bins)
    chunks = [order[round(Int, (b - 1) * length(order) / nb)+1:round(Int, b * length(order) / nb)] for b in 1:nb]
    Rb = [mean(abs(gp[j][2]) for j in c) for c in chunks]
    Eb = [mean(gp[j][3] for j in c) for c in chunks]
    e0, e1 = if nb >= 2
        A = hcat(ones(nb), Rb)
        c = A \ Eb
        c[1] < 0 ? (0.0, max(dot(Rb, Eb) / dot(Rb, Rb), floor_fraction)) : (c[1], max(c[2], 0.0))
    else
        (0.0, max(Eb[1] / Rb[1], floor_fraction))
    end
    σ(R) = max(e0 + e1 * abs(R), floor_fraction * abs(R))

    idx = [p[1] for p in gp]
    R = [p[2] for p in gp]
    if keep_unpaired
        append!(idx, unpaired)
        append!(R, d.resistance[unpaired])
    end
    out = select_data2d(d, idx)
    out = _ert2d_with(out; resistance = R, resistance_error = σ.(R))
    (; data = out, pairs = count(good), dropped = count(!, good), e0, e1, relative = rel, unpaired = length(unpaired))
end
