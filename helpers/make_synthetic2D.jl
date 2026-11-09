# 2D ERT synthetic case generator
# Author: @pankajkmishra
# Writes demo/flat and demo/topo: a dipole-dipole survey of 64 electrodes 5 m apart on a profile near Jyväskylä,
# central Finland (the MTGeophysics.jl 2D-IV location), a conductor and a resistive block in a 100 Ω·m host
# - flat: flat ground
# - topo: gentle relief (12 m over the line), a staircase of air cells on the rectangular mesh
# Each case holds
# - data.obs (and topo.dat): the field-style inputs; R from a finer data mesh (4 cells per spacing, forward cells a/32
#   at the electrodes) with 3% noise: R = ρa / K with the K of the real ground (analytic when flat, numerical over the
#   relief), so the data carry the full topographic effect
# - model.true: the true model on the inversion mesh (plots only)
# - model.start, model.prior, cov.ctrl, fwd.ctrl, inv.ctrl, data.dat: from MakeMesh2D
# Usage: julia --project=. --threads=8 helpers/make_synthetic2D.jl

using ERTGeophysics
using Printf
using Random

const DEMO_DIR = joinpath(dirname(@__DIR__), "demo")
const CTRL_DIR = joinpath(dirname(@__DIR__), "examples", "ctrl", "2D")

#---------- settings ----------

const SURVEY = (
    origin         = [62.25, 25.75],      # WGS84 lat, lon of the line centre
    azimuth        = 70.0,                # line direction, degrees clockwise from north
    n_electrodes   = 64,
    spacing        = 5.0,                 # m
    a_max          = 3,                   # dipole lengths 1..3 spacings
    n_max          = 6,                   # separations 1..6 dipole lengths
    error_fraction = 0.03,                # noise and error, fraction of |R|
    rng_seed       = 20261019,
)

# gentle till relief, m a.s.l.: a 4 m swell over a 7 m rise along the line
elevation(y) = 140.0 + 4.0 * sin(2π * (y + 40.0) / 260.0) + 0.024 * y
const TOPO_Y = collect(-600.0:2.0:600.0)

# rectangular bodies: y range (m along the line), depth range below the model top (m), resistivity, their edges on the
# 2.5 m grid of the main domain; under the relief the model top is the highest electrode, about 9 m above the ground
# over the conductor and 0-2 m over the resistor
const HOST = 100.0
const CASES = (
    flat = (topography = false, bodies = [
        (y_range = (-95.0, -55.0), z_range = (5.0, 22.5), resistivity = 10.0),      # conductor
        (y_range = (25.0, 85.0), z_range = (5.0, 15.0), resistivity = 1000.0),      # resistive block
    ]),
    topo = (topography = true, bodies = [
        (y_range = (-95.0, -55.0), z_range = (15.0, 30.0), resistivity = 10.0),
        (y_range = (25.0, 85.0), z_range = (5.0, 17.5), resistivity = 1000.0),
    ]),
)

# the data mesh is finer than the inversion mesh MakeMesh2D builds
const DATA_MESH = (cells_per_spacing = 4, pad_reach = 2.0, refine = 32, k_refine = 48)

true_model(mesh, bodies) = ert2d_block_model(mesh; background_resistivity = HOST, blocks = bodies)

#---------- one case ----------

function make_case(name, case)
    dir = joinpath(DEMO_DIR, String(name))
    mkpath(dir)
    ne = SURVEY.n_electrodes
    positions = collect(range(-(ne - 1) * SURVEY.spacing / 2, (ne - 1) * SURVEY.spacing / 2, length = ne))
    quadrupoles = ert2d_dipole_dipole(ne; a_max = SURVEY.a_max, n_max = SURVEY.n_max)
    survey = BuildSurvey2D(origin = SURVEY.origin, azimuth = SURVEY.azimuth, positions = positions, quadrupoles = quadrupoles,
                           title = "Synthetic dipole-dipole profile ($name), ERTGeophysics.jl")
    topo, topo_path = nothing, ""
    if case.topography
        lat, lon = ert2d_latlon(SURVEY.origin, SURVEY.azimuth, zeros(length(TOPO_Y)), TOPO_Y)
        topo = Topo2D(latitudes = lat, longitudes = lon, elevations = elevation.(TOPO_Y))
        topo_path = WriteTopo2D(joinpath(dir, "topo.dat"), topo)
    end

    # ρa on the fine data mesh (normalised by its own 1 Ω·m response), R = ρa / K with the K of the real ground
    p = merge(ERTGeophysics._MAKEMESH2D_DEFAULTS, DATA_MESH)
    fine = ERTGeophysics._makemesh2d_build(survey, topo, NamedTuple[], p)
    println("[$name] data mesh: ", fine.summary)
    t = @elapsed r = ForwardSolve2D(fine.mesh, true_model(fine.mesh, case.bodies), fine.data; refine = DATA_MESH.refine)
    @printf("[%s] forward on the data mesh: %d data in %.1f s, ρa %.1f-%.1f Ω·m\n", name, ndata(survey), t, extrema(r.rhoa)...)
    rng = MersenneTwister(SURVEY.rng_seed)
    R = r.rhoa ./ fine.data.geometric_factor
    R_obs = R .* (1 .+ SURVEY.error_fraction .* randn(rng, length(R)))
    observed = ERTGeophysics._ert2d_with(fine.data; resistance = R_obs, resistance_error = SURVEY.error_fraction .* abs.(R_obs),
                                         rhoa = R_obs .* fine.data.geometric_factor)
    obs_path = write_data2d(joinpath(dir, "data.obs"), observed)

    # inversion inputs on the default mesh, and the truth on it
    mm = MakeMesh2D(obs_path; out_dir = dir, topo_path, inv_ctrl = joinpath(CTRL_DIR, "InvCtrl.GN"))
    ρ = true_model(mm.mesh, case.bodies)
    WriteModel2D(joinpath(dir, "model.true"), mm.mesh, ρ)
    plot_ert2d_model(mm.mesh, ρ; output_path = joinpath(dir, "ModelTrue.png"), resistivity_log10_range = (1.0, 3.0),
                     outlines = [(y_range = b.y_range, z_range = b.z_range) for b in case.bodies])
    fit = chi2_rms2d(mm.data, _true_rhoa(mm, ρ))
    @printf("[%s] the true model on the inversion mesh fits the data at RMS %.2f\n", name, fit.rms)
    println("[$name] case written to ", dir)
end

_true_rhoa(mm, ρ) = ERTGeophysics._ert2d_predicted(mm.data, ForwardSolve2D(mm.mesh, ρ, mm.data).rhoa)

foreach(name -> make_case(name, CASES[name]), keys(CASES))
