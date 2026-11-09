# 3D ERT synthetic case generator
# Author: @pankajkmishra
# Writes demo/3D: six parallel lines 10 m apart of 24 electrodes 5 m apart near Jyväskylä, central Finland, flat ground,
# a conductive and a resistive block in a 100 Ω·m host; dipole-dipole along every line and across the lines
# - data.obs: ρa from a finer data mesh with 3% noise, R = ρa / K with the analytic K
# - model.true: the true model on the inversion mesh (plots only)
# - model.start, model.prior, fwd.ctrl, inv.ctrl, data.dat: from MakeMesh3D
# Usage: julia --project=. --threads=8 helpers/make_synthetic3D.jl

using ERTGeophysics
using Printf
using Random

const CASE_DIR = joinpath(dirname(@__DIR__), "demo", "3D")
const CTRL_DIR = joinpath(dirname(@__DIR__), "examples", "ctrl", "3D")

const SURVEY = (
    origin         = [62.25, 25.75],      # WGS84 lat, lon of the grid centre
    n_lines        = 6,                   # east-west lines
    line_spacing   = 10.0,                # m, north-south
    n_per_line     = 24,
    spacing        = 5.0,                 # m, along the lines
    error_fraction = 0.03,
    rng_seed       = 20261019,
)

# rectangular bodies: x (north), y (east), depth below the surface (m), resistivity
const HOST = 100.0
const BODIES = [
    (x_range = (-15.0, 5.0), y_range = (-40.0, -15.0), z_range = (5.0, 15.0), resistivity = 10.0),     # conductor
    (x_range = (0.0, 20.0), y_range = (15.0, 40.0), z_range = (2.5, 10.0), resistivity = 1000.0),      # resistive block
]

# the data mesh is finer than the inversion mesh MakeMesh3D builds
const DATA_MESH = (cells_per_spacing = 4, pad_reach = 1.5, refine = 6)

true_model(mesh) = ert3d_block_model(mesh; background_resistivity = HOST, blocks = BODIES)

mkpath(CASE_DIR)
g = ert3d_grid(SURVEY.n_lines, SURVEY.n_per_line, SURVEY.line_spacing, SURVEY.spacing)
along = ert3d_line_quadrupoles(g.lines_y, n -> ert2d_dipole_dipole(n; a_max = 3, n_max = 6))
across = ert3d_line_quadrupoles(g.lines_x, n -> ert2d_dipole_dipole(n; a_max = 1, n_max = 3))
quadrupoles = map(vcat, along, across)
survey = BuildSurvey3D(origin = SURVEY.origin, x = g.x, y = g.y, quadrupoles = quadrupoles,
                       title = "Synthetic 3D dipole-dipole grid, ERTGeophysics.jl")
println(ndata(survey), " data on ", nelectrodes(survey), " electrodes")

p = merge(ERTGeophysics._MAKEMESH3D_DEFAULTS, DATA_MESH)
fine = ERTGeophysics._makemesh3d_build(survey, nothing, p)
println("Data mesh: ", fine.summary)
t = @elapsed r = ForwardSolve3D(fine.mesh, true_model(fine.mesh), fine.data; refine = DATA_MESH.refine, growth = 0.25)
@printf("Forward on the data mesh: %.1f s, ρa %.1f-%.1f Ω·m\n", t, extrema(r.rhoa)...)
rng = MersenneTwister(SURVEY.rng_seed)
R = r.rhoa ./ fine.data.geometric_factor
R_obs = R .* (1 .+ SURVEY.error_fraction .* randn(rng, length(R)))
observed = ERTGeophysics._ert3d_with(fine.data; resistance = R_obs, resistance_error = SURVEY.error_fraction .* abs.(R_obs))
obs_path = write_data3d(joinpath(CASE_DIR, "data.obs"), observed)

mm = MakeMesh3D(obs_path; out_dir = CASE_DIR, inv_ctrl = joinpath(CTRL_DIR, "InvCtrl.GN"))
ρ = true_model(mm.mesh)
WriteModel3D(joinpath(CASE_DIR, "model.true"), mm.mesh, ρ)
outlines = [(x_range = b.x_range, y_range = b.y_range, z_range = b.z_range) for b in BODIES]
plot_ert3d_slices(mm.mesh, ρ; output_path = joinpath(CASE_DIR, "ModelTrue.png"), resistivity_log10_range = (1.0, 3.0),
                  depths = [2.0, 5.0, 8.0, 11.0, 14.0, 18.0], outlines)
t = @elapsed fit = chi2_rms3d(mm.data, ERTGeophysics._ert3d_predicted(mm.data, ForwardSolve3D(mm.mesh, ρ, mm.data).rhoa))
@printf("The true model on the inversion mesh fits the data at RMS %.2f (forward %.1f s)\n", fit.rms, t)
println("Case written to ", CASE_DIR)
