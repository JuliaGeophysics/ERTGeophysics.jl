# 2D covariance masks
# Author: @pankajkmishra
# The cov.ctrl mask of a model, as Mask2D.jl of MTGeophysics.jl: 0 for air and fixed cells, 9 for water, 1 for free
# cells, 2 for padding cells that follow the nearest free cell; and the ground line a model file describes

"""
    Mask2D(model::ModelFile2D; water=nothing, fixed_below_m=Inf, fixed=[]) -> Matrix{Int}

Covariance mask `(nz, ny)` of a ModEM-layout model:
- 0 for air (cells above 1e15 ohm m), for cells whose top lies below `fixed_below_m`
  (depth below the model top) and for cells whose centre lies in a `fixed` box
  `(y_range, z_range)` (m, the model's y frame and depth below the model top);
- 9 for `water`, a model-shaped Bool mask such as `Topography2D(...).mask .== 9`;
- 2 for `padding`, a model-shaped Bool mask: cells outside the main domain that take the
  value of the nearest free cell instead of being inverted;
- 1 elsewhere (free).
"""
function Mask2D(model::ModelFile2D; water::Union{Nothing, AbstractMatrix{Bool}} = nothing, fixed_below_m::Real = Inf,
                fixed = NamedTuple[], padding::Union{Nothing, AbstractMatrix{Bool}} = nothing)
    nz, ny = size(model.resistivity)
    mask = ones(Int, nz, ny)
    ztop = vcat(0.0, cumsum(model.z_cell_sizes)[1:end-1])
    if padding !== nothing
        size(padding) == (nz, ny) || throw(DimensionMismatch("padding mask must match the model"))
        mask[padding] .= ERT2D_MASK_PADDING
    end
    mask[ztop .>= fixed_below_m, :] .= 0
    yc = model.origin .+ cumsum(model.y_cell_sizes) .- model.y_cell_sizes ./ 2
    zc = ztop .+ model.z_cell_sizes ./ 2
    for b in fixed, iy in 1:ny, iz in 1:nz
        b.y_range[1] <= yc[iy] <= b.y_range[2] && b.z_range[1] <= zc[iz] <= b.z_range[2] && (mask[iz, iy] = 0)
    end
    if water !== nothing
        size(water) == (nz, ny) || throw(DimensionMismatch("water mask must match the model"))
        mask[water] .= ERT2D_MASK_WATER
    end
    mask[model.resistivity .> ERT2D_AIR_THRESHOLD] .= 0
    mask
end

"""
    ert2d_ground(model::ModelFile2D) -> (y, depth)

The y nodes (the model's y frame) and the ground depth below the model top at each, the
model's ground line (zeros when flat).
"""
function ert2d_ground(model::ModelFile2D)
    y = model.origin .+ vcat(0.0, cumsum(model.y_cell_sizes))
    y, isempty(model.surface) ? zeros(length(y)) : copy(model.surface)
end
