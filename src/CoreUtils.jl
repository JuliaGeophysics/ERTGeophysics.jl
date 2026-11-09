# Core and padding utilities
# Author: @pankajkmishra
# The uniform-width core of a padded mesh and depth ranges, copied from MTGeophysics.jl CoreUtils3D.jl

"""
    _mode_width(w::AbstractVector)

Most frequently occurring cell width in `w`, rounded to 6 significant digits before
tallying. Ties are broken toward the smaller width (finer cells).
"""
function _mode_width(w::AbstractVector)
    counts = Dict{Float64,Int}()
    @inbounds for v in w
        key = round(float(v); sigdigits = 6)
        counts[key] = get(counts, key, 0) + 1
    end
    best_w, best_c = first(keys(counts)), 0
    for (k, c) in counts
        if c > best_c || (c == best_c && k < best_w)
            best_w, best_c = k, c
        end
    end
    return best_w
end

# longest run of cells within tol of the modal width: the core of a padded axis
function _core_range(w::AbstractVector; tol::Real = 0.2)
    n = length(w)
    n <= 4 && return 1:n
    w_ref = _mode_width(w)
    is_core = abs.(w .- w_ref) .<= tol * w_ref
    best_i, best_len = 1, 0
    i = 1
    while i <= n
        if is_core[i]
            j = i
            while j <= n && is_core[j]; j += 1; end
            if (j - i) > best_len
                best_i, best_len = i, (j - i)
            end
            i = j
        else
            i += 1
        end
    end
    best_len > 0 ? (best_i:(best_i + best_len - 1)) : (1:n)
end

# layers 1:k whose bottom lies nearest max_depth
function _depth_range(dz::AbstractVector, max_depth::Real)
    cum = cumsum(dz)
    if max_depth <= cum[1]
        return 1:1
    elseif max_depth >= cum[end]
        return 1:length(dz)
    else
        k = findfirst(>=(max_depth), cum)::Int
        if k > 1 && (max_depth - cum[k-1]) < (cum[k] - max_depth)
            return 1:(k-1)
        else
            return 1:k
        end
    end
end
