# 2D mesh tool
# Author: @pankajkmishra
# Inversion inputs from a data file, as MakeMesh2D.jl of MTGeophysics.jl: a padded profile mesh sized by the
# electrode spacing and the widest quadrupole, topography and water cut in from topo.dat, the geometric factors,
# and model.start, model.prior, cov.ctrl, fwd.ctrl, inv.ctrl and data.dat written next to each other

using Printf
using Statistics

const _MAKEMESH2D_DEFAULTS = (
    cells_per_spacing = 2, core_margin_cells = 4, n_pad = 0, pad_factor = 1.5, pad_reach = 1.0,
    first_layer_div = 0.0, vertical_factor = 1.0, depth_frac = 0.4, n_pad_z = 0, background_resistivity = 0.0,
    cov_smoothing = 0.3, n_smooth = 1, fixed_below_m = Inf, water_resistivity = 100.0,
    geometric_factor = :auto, wavenumbers = 16, refine = 0, growth = 0.25, k_refine = 32, topography = :staircase,
    invert_padding = false,
)

# median apparent resistivity of the valid data, the default background
function _makemesh2d_background(data::DataFile2D)
    K = all(isnan, data.geometric_factor) ? ert2d_geometric_factor(data) : data.geometric_factor
    v = filter(x -> isfinite(x) && x > 0, K .* data.resistance)
    isempty(v) ? 100.0 : 10^median(log10.(v))
end

# the mesh, model, mask, data and checks of one parameter set
function _makemesh2d_build(data::DataFile2D, topo, water, p)
    p.cells_per_spacing >= 1 || throw(ArgumentError("cells_per_spacing must be at least 1"))
    p.geometric_factor in (:auto, :file, :analytic, :numerical) ||
        throw(ArgumentError("geometric_factor must be :auto, :file, :analytic or :numerical"))
    p.topography in (:staircase, :flat) || throw(ArgumentError("topography must be :staircase or :flat"))
    p.pad_factor > 1 && p.pad_reach > 0 || throw(ArgumentError("need pad_factor > 1 and pad_reach > 0"))
    isempty(water) || p.topography == :staircase || throw(ArgumentError("water bodies need topography = :staircase"))
    data = recentre_data2d(data)
    a = ert2d_spacing(data)
    dy = a / p.cells_per_spacing
    span = maximum(ert2d_pseudo(data).span)
    line = maximum(data.electrodes) - minimum(data.electrodes)
    core_depth = max(p.depth_frac * span, 3a)
    # relief of the electrodes: from topo.dat, or from the data's electrode depths Z (e.g. an imported line)
    relief = if topo !== nothing
        ty, te = ert2d_profile_topography(topo, data)
        ee = [_ert2d_interp(ty, te, y) for y in data.electrodes]
        maximum(ee) - minimum(ee)
    else
        maximum(data.z_positions) - minimum(data.z_positions)
    end
    if p.topography == :flat
        relief = 0.0
        data = _ert2d_with(data; z_positions = zeros(nelectrodes(data)))
    end
    # padding: geometric cells reaching pad_reach line lengths beside and below the core (with the mixed boundary
    # condition ρa no longer changes beyond one line length)
    reach = p.pad_reach * max(line, span)
    # the main domain: an even grid of square cells a/cells_per_spacing (first_layer_div = 0, vertical_factor = 1);
    # the cells grow only in the padding outside it
    fld = p.first_layer_div > 0 ? p.first_layer_div : Float64(p.cells_per_spacing)
    layers = ert2d_geometric_layers(a, core_depth; first_layer_div = fld, vertical_factor = p.vertical_factor)
    n_pad = p.n_pad > 0 ? p.n_pad : ert2d_padding_cells(dy, reach, p.pad_factor)
    n_pad_z = p.n_pad_z > 0 ? p.n_pad_z : ert2d_padding_cells(layers[end], reach + relief, p.pad_factor)
    dz = vcat(layers, ert2d_padding(layers[end], n_pad_z, p.pad_factor))
    grid = BuildMesh2D(electrodes = data.electrodes, y_core_cell = dy, core_margin_cells = p.core_margin_cells,
                       n_pad = n_pad, pad_factor = p.pad_factor, layers = dz)
    ρbg = p.background_resistivity > 0 ? Float64(p.background_resistivity) : _makemesh2d_background(data)
    model = ModelFile2D(y_cell_sizes = grid.y_cell_sizes, z_cell_sizes = dz,
                        resistivity = fill(ρbg, length(dz), length(grid.y_cell_sizes)))
    wet = nothing
    if p.topography == :staircase && (topo !== nothing || relief > 0)
        t = topo === nothing ? Topography2D(model, data; water, water_resistivity = p.water_resistivity) :
            Topography2D(model, data, topo; water, water_resistivity = p.water_resistivity)
        model, data, wet = t.model, t.data, t.mask .== ERT2D_MASK_WATER
    end
    # the main domain: the even core, the electrodes ± core_margin_cells, down to the core depth; the padding outside it
    # follows the nearest main-domain cell (not inverted) unless invert_padding
    core_cols = _core_range(grid.y_cell_sizes)
    padding = trues(length(dz), length(grid.y_cell_sizes))
    padding[1:length(layers), core_cols] .= false
    mask = Mask2D(model; water = wet, fixed_below_m = p.fixed_below_m, padding = p.invert_padding ? nothing : padding)
    mesh, _ = Mesh2DFromInputs(model, data; warn = false)

    # geometric factors of the real ground: analytic over flat ground, numerical over topography (the topography changes
    # K by several per cent; the forward normalises against R(1 Ω·m) of the same ground, so the topographic effect
    # stays in the data and is modelled)
    flat = ert2d_is_flat(mesh)
    mode = p.geometric_factor == :auto ? (flat ? :analytic : :numerical) : p.geometric_factor
    K = if mode == :file
        any(isnan, data.geometric_factor) && error("geometric_factor = :file, but the data file has no K for every datum")
        data.geometric_factor
    elseif mode == :analytic
        ert2d_geometric_factor(data)
    else
        ert2d_numerical_geometric_factor(mesh, data; refine = p.k_refine, wavenumbers = max(p.wavenumbers, 20))
    end
    data = _ert2d_with(data; geometric_factor = K)

    # checks
    notes = String[]
    on_node = count(y -> minimum(abs.(mesh.y_nodes .- y)) <= 0.01 * dy, data.electrodes)      # projection round-off is fine
    on_node < nelectrodes(data) &&
        push!(notes, "$(nelectrodes(data) - on_node) electrode(s) between nodes (irregular spacing): they are interpolated")
    length(unique(ert2d_electrode_columns(mesh))) < length(unique(round.(data.electrodes; digits = 6))) &&
        push!(notes, "more than one electrode in a cell: raise cells_per_spacing")
    padded = sum(ert2d_padding(dy, n_pad, p.pad_factor))
    padded < 0.5line && push!(notes, @sprintf("lateral padding reaches %.0f m, less than half the %.0f m line: add padding cells", padded, line))
    dz[1] > a / 2 && push!(notes, @sprintf("first layer %.2f m is thicker than half the spacing %.2f m", dz[1], a / 2))
    if !flat
        off = filter(o -> abs(o.offset) > o.tolerance, ert2d_electrode_offsets(mesh, data))
        isempty(off) || push!(notes, "$(length(off)) electrode(s) lie more than half a cell from their tread")
        mode == :analytic && push!(notes, "geometric_factor = :analytic over topography: K ignores the topographic effect (several per cent); :auto uses the numerical K")
    end
    v = ert2d_valid_data(data)
    count(v.keep) < ndata(data) && push!(notes, "$(ndata(data) - count(v.keep)) of $(ndata(data)) data are unusable " *
                                         "(missing R, K or error, or ρa ≤ 0) and will be skipped")
    flat && topo !== nothing && p.topography == :staircase && push!(notes, "the topography is flat")
    p.topography == :flat && (topo !== nothing || any(!iszero, data.z_positions)) &&
        push!(notes, "topography = :flat: the relief is ignored (electrodes at their horizontal positions, Z = 0)")
    # forward cells a/8 at the electrodes over flat ground, a/16 under topography (the staircase under the sources)
    refine = p.refine > 0 ? p.refine : flat ? 8 : 16
    fwd = FwdCtrl2D(; wavenumbers = p.wavenumbers, refine, growth = p.growth, k_refine = p.k_refine)
    ncore = length(_core_range(grid.y_cell_sizes))
    (; model, mask, data, fwd, mesh, ρbg, dy, notes, geometric_factor = mode, main_depth = sum(layers),
       summary = @sprintf("%d × %d cells; main domain %d × %d even cells of %.2f × %.2f m to %.1f m depth; padding %.0f m beside and %.0f m below it%s; ρ %.0f Ω·m, K %s%s",
                          length(dz), length(grid.y_cell_sizes), length(layers), ncore, dy, layers[1], sum(layers), padded,
                          sum(dz) - sum(layers), p.invert_padding ? " (inverted)" : " (following the main domain)", ρbg, mode,
                          flat ? ", flat" :
                          @sprintf(", staircase over %.1f m of relief, %d air cells%s", relief, count(ert2d_air_mask(mesh)),
                                   any(==(ERT2D_MASK_WATER), mask) ? @sprintf(", %d water cells", count(==(ERT2D_MASK_WATER), mask)) : "")))
end

function _makemesh2d_write(b, out_dir, p, ctrls)
    mkpath(out_dir)
    path(f) = joinpath(out_dir, f)
    m = b.model
    paths = (
        start_model_path = WriteModel2D(path("model.start"), m),
        prior_model_path = WriteModel2D(path("model.prior"), m),
        cov_path = WriteCov2D(path("cov.ctrl"), Cov2D(sy = fill(p.cov_smoothing, length(m.z_cell_sizes)), sz = p.cov_smoothing,
                                                      n_smooth = p.n_smooth, mask = b.mask)),
        fwd_path = WriteFwdCtrl2D(path("fwd.ctrl"), b.fwd),
        inv_path = (cp(ctrls.inv, path("inv.ctrl"); force = true); path("inv.ctrl")),
        data_path = write_data2d(ctrls.data_out, b.data),
    )
    water = b.mask .== ERT2D_MASK_WATER
    plot_ert2d_mesh(b.mesh; output_path = path("Mesh.png"), water, core_depth = b.main_depth)
    plot_ert2d_mesh(b.mesh; output_path = path("MeshCore.png"), region = :core, water, core_depth = b.main_depth)
    plot_ert2d_pseudosection(b.data; output_path = path("Pseudosection.png"))
    paths
end

"""
    MakeMesh2D(data_path; out_dir=dirname(data_path), topo_path="", water=[],
               inv_ctrl=examples/ctrl/2D/InvCtrl.GN, kwargs...) -> (; paths, summary, notes, mesh, data)

Inversion inputs for a 2D ERT data file:
- the data are re-centred on the middle of the electrode line (`recentre_data2d`);
- lateral: a uniform core of a/`cells_per_spacing` (a = median electrode spacing, so
  equally spaced electrodes sit on nodes) over the electrodes plus `core_margin_cells`,
  centred on y = 0, then padding cells growing by `pad_factor` (1.5) that reach
  `pad_reach` (1) line lengths beyond the core (`n_pad > 0` fixes their number);
- vertical: an even grid down to `depth_frac` × the widest quadrupole span: layers of
  a/`cells_per_spacing`, so the main domain has square cells (`first_layer_div > 0` sets
  the first layer to a/`first_layer_div` and `vertical_factor > 1` lets the layers grow
  instead), then padding
  reaching `pad_reach` line lengths (plus the relief) below (`n_pad_z > 0` fixes it);
- `background_resistivity` (0 = the median apparent resistivity of the data) fills the
  start and prior models;
- topography from `topo_path` (`topo.dat`, WGS84 lat lon elevation), or else from the
  electrode depths Z of the data: `topography = :staircase` (default) cuts the air cells
  into the rectangular mesh as in MTGeophysics.jl (`Topography2D`, Z = depth below the
  model top, the highest electrode; each electrode on a flat tread) and keeps the ground
  line for the forward mesh, and also takes `water` bodies `(y_range, level)`; `:flat`
  ignores the relief (Z = 0, analytic K);
- geometric factors `geometric_factor = :auto` = `:analytic` over flat ground (exact) and
  `:numerical` over topography (1/R(1 Ω·m) on a forward mesh with cells a/`k_refine` at
  the electrodes and its own fine staircase); `:file` keeps the data file's K;
- `cov.ctrl` from `Mask2D` (air 0, water 9, padding 2, `fixed_below_m`), smoothing
  `cov_smoothing`: the padding follows the nearest main-domain cell and is not inverted
  (`invert_padding = true` inverts it too);
- `fwd.ctrl` with `wavenumbers`, `refine` (forward cells per spacing at the electrodes;
  0 = 8 over flat ground, 16 under topography), `growth` and `k_refine`; `inv.ctrl` copied
  from `inv_ctrl`.

Writes `model.start`, `model.prior`, `cov.ctrl`, `fwd.ctrl`, `inv.ctrl`, `data.dat` (re-centred, with K, ρa and the electrode depths), `Mesh.png`,
`MeshCore.png` and `Pseudosection.png` into `out_dir`, and prints the mesh summary and
advice. The input data file is never overwritten.
"""
function MakeMesh2D(data_path::AbstractString; out_dir::AbstractString = dirname(abspath(data_path)),
                    topo_path::AbstractString = "", water = NamedTuple[],
                    inv_ctrl::AbstractString = joinpath(dirname(@__DIR__), "examples", "ctrl", "2D", "InvCtrl.GN"),
                    kwargs...)
    p = merge(_MAKEMESH2D_DEFAULTS, values(kwargs))
    unknown = setdiff(keys(p), keys(_MAKEMESH2D_DEFAULTS))
    isempty(unknown) || throw(ArgumentError("unknown MakeMesh2D settings $unknown"))
    data_out = joinpath(abspath(out_dir), "data.dat")
    data_out == abspath(data_path) && error("MakeMesh2D writes data.dat into out_dir and would overwrite $data_path; " *
                                            "rename the data file or choose another out_dir")
    data = load_data2d(data_path)
    topo = isempty(topo_path) ? nothing : ReadTopo2D(topo_path)
    ReadInvCtrl2D(inv_ctrl)
    b = _makemesh2d_build(data, topo, water, p)
    paths = _makemesh2d_write(b, out_dir, p, (inv = inv_ctrl, data_out))
    println(b.summary)
    foreach(n -> println("  • ", n), b.notes)
    (; paths, summary = b.summary, notes = b.notes, mesh = b.mesh, data = b.data)
end
