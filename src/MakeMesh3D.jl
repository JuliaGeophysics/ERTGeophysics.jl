# 3D mesh tool
# Author: @pankajkmishra
# Inversion inputs from a 3D data file, as MakeMesh2D: a padded rectangular mesh sized by the electrode spacing and the
# widest quadrupole, an even main domain with padding that follows it, a staircase cut from topo.dat or the electrode
# depths, the geometric factors of the real ground, and model.start, model.prior, fwd.ctrl, inv.ctrl and data.dat
# written next to each other

using Printf
using Statistics

const _MAKEMESH3D_DEFAULTS = (
    cells_per_spacing = 2, core_margin_cells = 3, n_pad = 0, pad_factor = 1.5, pad_reach = 1.0, first_layer_div = 0.0,
    vertical_factor = 1.0, depth_frac = 0.35, n_pad_z = 0, background_resistivity = 0.0, topography = :staircase,
    geometric_factor = :auto, refine = 0, growth = 0.5, k_refine = 12, invert_padding = false,
)

# electrode spacing along x and y: the median nearest-neighbour distance along each axis, the other axis held
function _ert3d_axis_spacing(v, w)
    gaps = Float64[]
    for i in eachindex(v)
        same = [j for j in eachindex(v) if j != i && abs(w[j] - w[i]) < 1e-6 && abs(v[j] - v[i]) > 1e-6]
        isempty(same) || push!(gaps, minimum(abs(v[j] - v[i]) for j in same))
    end
    isempty(gaps) ? NaN : median(gaps)
end

function _makemesh3d_build(data::DataFile3D, topo, p)
    p.cells_per_spacing >= 1 || throw(ArgumentError("cells_per_spacing must be at least 1"))
    p.topography in (:staircase, :flat) || throw(ArgumentError("topography must be :staircase or :flat"))
    p.geometric_factor in (:auto, :file, :analytic, :numerical) ||
        throw(ArgumentError("geometric_factor must be :auto, :file, :analytic or :numerical"))
    p.pad_factor > 1 && p.pad_reach > 0 || throw(ArgumentError("need pad_factor > 1 and pad_reach > 0"))
    data = recentre_data3d(data)
    a = ert3d_spacing(data)
    sx, sy = _ert3d_axis_spacing(data.x_positions, data.y_positions), _ert3d_axis_spacing(data.y_positions, data.x_positions)
    sx = isfinite(sx) ? sx : a
    sy = isfinite(sy) ? sy : a
    dx, dy = sx / p.cells_per_spacing, sy / p.cells_per_spacing
    span = maximum(ert3d_pseudo(data).span)
    extent = max(maximum(data.x_positions) - minimum(data.x_positions), maximum(data.y_positions) - minimum(data.y_positions))
    core_depth = max(p.depth_frac * span, 3a)
    relief = if p.topography == :flat
        0.0
    elseif topo !== nothing
        px, py = ert3d_local_frame(data.origin, topo.latitudes, topo.longitudes)
        ee = [_ert3d_idw(px, py, topo.elevations, data.x_positions[i], data.y_positions[i]) for i in 1:nelectrodes(data)]
        maximum(ee) - minimum(ee)
    else
        maximum(data.z_positions) - minimum(data.z_positions)
    end
    p.topography == :flat && (data = _ert3d_with(data; z_positions = zeros(nelectrodes(data))))
    # padding reaching pad_reach × the electrode extent (or the widest span) beside and below the core
    reach = p.pad_reach * max(extent, span)
    # the main domain: an even grid, layers as thick as the smaller horizontal core cell (first_layer_div = 0,
    # vertical_factor = 1); the cells grow only in the padding outside it
    fld = p.first_layer_div > 0 ? p.first_layer_div : a / min(dx, dy)
    layers = ert2d_geometric_layers(a, core_depth; first_layer_div = fld, vertical_factor = p.vertical_factor)
    n_pad = p.n_pad > 0 ? p.n_pad : ert2d_padding_cells(min(dx, dy), reach, p.pad_factor)
    n_pad_z = p.n_pad_z > 0 ? p.n_pad_z : ert2d_padding_cells(layers[end], reach + relief, p.pad_factor)
    dz = vcat(layers, ert2d_padding(layers[end], n_pad_z, p.pad_factor))
    grid = BuildMesh3D(electrode_x = data.x_positions, electrode_y = data.y_positions, x_core_cell = dx, y_core_cell = dy,
                       core_margin_cells = p.core_margin_cells, n_pad = n_pad, pad_factor = p.pad_factor, layers = dz)
    ρbg = p.background_resistivity > 0 ? Float64(p.background_resistivity) :
          (v = filter(x -> isfinite(x) && x > 0, ert3d_geometric_factor(data) .* data.resistance);
           isempty(v) ? 100.0 : 10^median(log10.(v)))
    model = ModelFile3D(x_cell_sizes = grid.x_cell_sizes, y_cell_sizes = grid.y_cell_sizes, z_cell_sizes = dz,
                        resistivity = fill(ρbg, length(grid.x_cell_sizes), length(grid.y_cell_sizes), length(dz)),
                        origin = [grid.x_nodes[1], grid.y_nodes[1], 0.0], main_depth = p.invert_padding ? NaN : sum(layers))
    if p.topography == :staircase && (topo !== nothing || relief > 0)
        t = Topography3D(model, data, topo)
        model, data = t.model, t.data
    end
    mesh, _ = Mesh3DFromInputs(model, data; warn = false)
    flat = ert3d_is_flat(mesh)
    # geometric factors of the real ground: analytic when flat, numerical over topography (a finer forward staircase)
    mode = p.geometric_factor == :auto ? (flat ? :analytic : :numerical) : p.geometric_factor
    K = if mode == :file
        any(isnan, data.geometric_factor) && error("geometric_factor = :file, but the data file has no K for every datum")
        data.geometric_factor
    elseif mode == :analytic
        ert3d_geometric_factor(data)
    else
        ert3d_numerical_geometric_factor(mesh, data; refine = p.k_refine, growth = p.growth)
    end
    data = _ert3d_with(data; geometric_factor = K)

    notes = String[]
    off = count(i -> minimum(abs.(mesh.x_nodes .- data.x_positions[i])) > 0.01dx ||
                     minimum(abs.(mesh.y_nodes .- data.y_positions[i])) > 0.01dy, 1:nelectrodes(data))
    off > 0 && push!(notes, "$off electrode(s) between nodes (irregular spacing): they are interpolated")
    cols = ert3d_electrode_columns(mesh)
    length(unique(cols)) < nelectrodes(data) && push!(notes, "more than one electrode in a cell: raise cells_per_spacing")
    padded = min(sum(ert2d_padding(dx, n_pad, p.pad_factor)), sum(ert2d_padding(dy, n_pad, p.pad_factor)))
    p.topography == :flat && (topo !== nothing || relief > 0) && push!(notes, "topography = :flat: the relief is ignored")
    !flat && mode == :analytic && push!(notes, "geometric_factor = :analytic over topography: K ignores the topographic effect; :auto uses the numerical K")
    nc = prod(_ert3d_size(mesh))
    nc > 150_000 && push!(notes, @sprintf("%d cells: the 3D solve will be slow; consider cells_per_spacing = 1 or a smaller pad_reach", nc))
    # forward cells a/4 at the electrodes over flat ground, a/8 under topography (the staircase under the sources)
    refine = p.refine > 0 ? p.refine : flat ? 4 : 8
    (; model, data, mesh, notes, fwd = FwdCtrl3D(; refine, growth = p.growth, k_refine = p.k_refine), main_depth = sum(layers),
       summary = @sprintf("%d × %d × %d cells (N × E × down, %d); main domain of even %.2f × %.2f × %.2f m cells to %.1f m depth; padding %.0f m beside and %.0f m below it%s; ρ %.0f Ω·m, K %s%s",
                          _ert3d_size(mesh)..., nc, dx, dy, layers[1], sum(layers), padded, sum(dz) - sum(layers),
                          p.invert_padding ? " (inverted)" : " (following the main domain)", ρbg, mode,
                          flat ? ", flat" : @sprintf(", staircase over %.1f m of relief, %d air cells", relief, count(ert3d_air_mask(mesh)))))
end

"""
    MakeMesh3D(data_path; out_dir=dirname(data_path), topo_path="",
               inv_ctrl=examples/ctrl/3D/InvCtrl.GN, kwargs...) -> (; paths, summary, notes, mesh, data)

Inversion inputs for a 3D ERT data file:
- the data are re-centred on the middle of the electrodes (`recentre_data3d`);
- lateral: uniform core cells of the electrode spacing along x and along y divided by
  `cells_per_spacing` over the electrodes plus `core_margin_cells`, then padding cells
  growing by `pad_factor` that reach `pad_reach` × the electrode extent (`n_pad > 0` fixes
  their number);
- vertical: an even grid down to `depth_frac` × the widest span, layers as thick as the
  smaller horizontal core cell (`first_layer_div > 0` sets the first layer to
  a/`first_layer_div` and `vertical_factor > 1` lets the layers grow instead), then padding reaching as far
  below (`n_pad_z > 0` fixes it);
- `background_resistivity` (0 = the median apparent resistivity of the data);
- the main domain (the even core columns to the core depth, written as `# main domain
  depth` in the models) is inverted; the padding outside it follows the nearest
  main-domain cell (`invert_padding = true` inverts it too);
- topography from `topo_path` (`topo.dat`, WGS84 lat lon elevation points) or else the
  electrode depths Z: `topography = :staircase` (default, air cells cut into the
  rectangular mesh, `Topography3D`, each electrode on a short tread) or `:flat`;
- geometric factors `:auto` = analytic when flat, numerical over topography (forward cells
  a/`k_refine` at the electrodes); `:analytic`, `:numerical` or `:file` force one;
- `fwd.ctrl` with `refine` (forward cells per spacing at the electrodes; 0 = 4 over flat
  ground, 8 under topography), `growth` and `k_refine`; `inv.ctrl` copied from `inv_ctrl`.

Writes `model.start`, `model.prior`, `fwd.ctrl`, `inv.ctrl`, `data.dat` and `Mesh.png`
into `out_dir` and prints the mesh summary and advice. The input file is never overwritten.
"""
function MakeMesh3D(data_path::AbstractString; out_dir::AbstractString = dirname(abspath(data_path)),
                    topo_path::AbstractString = "",
                    inv_ctrl::AbstractString = joinpath(dirname(@__DIR__), "examples", "ctrl", "3D", "InvCtrl.GN"),
                    kwargs...)
    p = merge(_MAKEMESH3D_DEFAULTS, values(kwargs))
    unknown = setdiff(keys(p), keys(_MAKEMESH3D_DEFAULTS))
    isempty(unknown) || throw(ArgumentError("unknown MakeMesh3D settings $unknown"))
    data_out = joinpath(abspath(out_dir), "data.dat")
    data_out == abspath(data_path) && error("MakeMesh3D writes data.dat into out_dir and would overwrite $data_path")
    ReadInvCtrl3D(inv_ctrl)
    b = _makemesh3d_build(load_data3d(data_path), isempty(topo_path) ? nothing : ReadTopo2D(topo_path), p)
    mkpath(out_dir)
    path(f) = joinpath(out_dir, f)
    paths = (
        start_model_path = WriteModel3D(path("model.start"), b.model),
        prior_model_path = WriteModel3D(path("model.prior"), b.model),
        fwd_path = WriteFwdCtrl3D(path("fwd.ctrl"), b.fwd),
        inv_path = (cp(inv_ctrl, path("inv.ctrl"); force = true); path("inv.ctrl")),
        data_path = write_data3d(data_out, b.data),
        mesh_plot = plot_ert3d_mesh(b.mesh; output_path = path("Mesh.png")),
    )
    println(b.summary)
    foreach(n -> println("  • ", n), b.notes)
    (; paths, summary = b.summary, notes = b.notes, mesh = b.mesh, data = b.data)
end
