# Accuracy and cost of the 2.5D and 3D forward solvers
# Author: @pankajkmishra
# - 2D: two-layer earths against the image-series solution on the even-grid default mesh, three arrays incl. long
#   Wenner spreads, forward cells a/8 at the electrodes (default) against a/4 and a uniform 2 × 2 split
# - 2D: padding reach (in line lengths) over a resistive basement
# - 2D topography on the staircase: uniform slopes (a tilted half-space, K exact over the straight distances), and the
#   12 m demo relief: the numerical K as the staircase under the electrodes gets finer, against the analytic K
# - 3D: two-layer earths against the image series; a uniform slope on the staircase
# - sensitivities against finite differences (2D on a staircase, 3D)
# - forward and sensitivity timings
# Usage: julia --project=. --threads=8 helpers/accuracy_report.jl

using ERTGeophysics
using Printf
using Random
using Statistics

const E = ERTGeophysics
const ORIGIN = [62.25, 25.75]
line(n, a) = collect(range(-(n - 1) * a / 2, (n - 1) * a / 2, length = n))
survey(y, q; z = zeros(length(y))) = BuildSurvey2D(origin = ORIGIN, positions = y, quadrupoles = q, z_positions = z)
function mesh_for(y, a, span; reach = 1.0, surface = nothing, bottom = reach)
    L = max(y[end] - y[1], span)
    layers = ert2d_geometric_layers(a, max(0.4span, 3a); first_layer_div = 2, vertical_factor = 1.0)
    BuildMesh2D(electrodes = y, y_core_cell = a / 2, n_pad = ert2d_padding_cells(a / 2, reach * L, 1.5), pad_factor = 1.5,
                layers = vcat(layers, ert2d_padding(layers[end], ert2d_padding_cells(layers[end], bottom * L, 1.5), 1.5)), surface = surface)
end

# image-series ρa of a two-layer earth, any electrode geometry given the distance function
function two_layer_rhoa(dist, d, K, ρ1, ρ2, h)
    κ = (ρ2 - ρ1) / (ρ2 + ρ1)
    V(r) = ρ1 / (2π) * (1 / r + 2 * sum(κ^n / sqrt(r^2 + (2n * h)^2) for n in 1:4000))
    pot(e, s) = (e == 0 || s == 0) ? 0.0 : V(dist(e, s))
    [pot(d.m[i], d.a[i]) - pot(d.m[i], d.b[i]) - pot(d.n[i], d.a[i]) + pot(d.n[i], d.b[i]) for i in eachindex(d.a)] .* K
end

ne, a = 64, 5.0
y = line(ne, a)
arrays = [("dipole-dipole", ert2d_dipole_dipole(ne; a_max = 3, n_max = 6)), ("Wenner", ert2d_wenner(ne; a_max = 21)),
          ("pole-dipole", ert2d_pole_dipole(ne; n_max = 8))]
cases = ((100.0, 10.0, 10.0), (20.0, 500.0, 5.0), (100.0, 10000.0, 10.0), (100.0, 1000.0, 2.5))

println("2D two-layer earths, |ρa / exact − 1| max / median (%), 64 electrodes at 5 m, even 2.5 m model cells")
@printf("  %-14s %-15s %13s %13s %13s\n", "array", "ρ1/ρ2, h", "refine 8", "refine 4", "uniform 2×2")
for (name, q) in arrays
    d = survey(y, q)
    mesh = mesh_for(y, a, maximum(ert2d_pseudo(d).span))
    for (ρ1, ρ2, h) in cases
        hh = mesh.z_nodes[argmin(abs.(mesh.z_nodes .- h))]
        ρ = [z < hh ? ρ1 : ρ2 for z in ert2d_z_centers(mesh), _ in mesh.y_cell_sizes]
        exact = two_layer_rhoa((e, s) -> abs(d.electrodes[e] - d.electrodes[s]), d, ert2d_geometric_factor(d), ρ1, ρ2, hh)
        e8 = abs.(ForwardSolve2D(mesh, ρ, d).rhoa ./ exact .- 1) .* 100
        e4 = abs.(ForwardSolve2D(mesh, ρ, d; refine = 4).rhoa ./ exact .- 1) .* 100
        fm, fρ = RefineMesh2D(mesh, ρ, 2)
        fh, _ = RefineMesh2D(mesh, ones(size(ρ)), 2)
        eu = abs.((ert2d_resistance(ert2d_forward(fm, fρ; keep_fields = false), d) ./
                   ert2d_resistance(ert2d_forward(fh, ones(size(fρ)); keep_fields = false), d)) ./ exact .- 1) .* 100
        @printf("  %-14s %-15s %6.2f / %4.2f %6.2f / %4.2f %6.2f / %4.2f\n", name, @sprintf("%g/%g, %.2f", ρ1, ρ2, hh),
                maximum(e8), median(e8), maximum(e4), median(e4), maximum(eu), median(eu))
    end
end

println("\n2D padding over a resistive basement (100/10000 Ω·m at 10 m), Wenner to the full line: max error (%)")
let d = survey(y, ert2d_wenner(ne; a_max = 21)), span = maximum(ert2d_pseudo(d).span)
    for reach in (0.2, 0.5, 1.0, 2.0)
        mesh = mesh_for(y, a, span; reach)
        hh = mesh.z_nodes[argmin(abs.(mesh.z_nodes .- 10.0))]
        ρ = [z < hh ? 100.0 : 10000.0 for z in ert2d_z_centers(mesh), _ in mesh.y_cell_sizes]
        exact = two_layer_rhoa((e, s) -> abs(d.electrodes[e] - d.electrodes[s]), d, ert2d_geometric_factor(d), 100.0, 10000.0, hh)
        @printf("  padding reaching %.1f line lengths (%3d columns): %.2f%%\n", reach, length(mesh.y_cell_sizes),
                maximum(abs.(ForwardSolve2D(mesh, ρ, d).rhoa ./ exact .- 1)) * 100)
    end
end

println("\n2D uniform slopes on the staircase (a tilted half-space: K exact over the straight distances), K_num / K_exact − 1")
let q = ert2d_dipole_dipole(48; a_max = 3, n_max = 6), y48 = line(48, a)
    right = mesh_for(y48, a, 120.0).y_nodes[end]
    for θ in (5.0, 10.0, 15.0)
        m = mesh_for(y48, a, 120.0; surface = yy -> tand(θ) * (right - yy), bottom = 3.0)
        dz = survey(y48, q; z = tand(θ) .* (right .- y48))
        for r in (16, 32)
            k = ert2d_numerical_geometric_factor(m, dz; refine = r) ./ ert2d_geometric_factor(dz)
            @printf("  slope %4.1f°, cells a/%d at the electrodes: median %+.2f%%, max %.2f%%\n", θ, r, 100median(k .- 1),
                    100maximum(abs.(k .- 1)))
        end
    end
end

println("\n2D topography: the 12 m demo relief, 954 dipole-dipole data")
let y64 = y, elev = 140.0 .+ 4.0 .* sin.(2π .* (y .+ 40.0) ./ 260.0) .+ 0.024 .* y
    datum = maximum(elev)
    d = survey(y64, ert2d_dipole_dipole(ne; a_max = 3, n_max = 6); z = datum .- elev)
    mesh = mesh_for(y64, a, 120.0; surface = yy -> E._ert2d_interp(y64, datum .- elev, yy))
    Kan = ert2d_geometric_factor(d)
    Kref = ert2d_numerical_geometric_factor(mesh, d; refine = 64)
    @printf("  topographic effect K_analytic / K_num(a/64): %.3f … %.3f (%d of %d data beyond 3%%)\n", extrema(Kan ./ Kref)...,
            count(x -> abs(x - 1) > 0.03, Kan ./ Kref), ndata(d))
    for r in (8, 16, 32)
        e = abs.(ert2d_numerical_geometric_factor(mesh, d; refine = r) ./ Kref .- 1) .* 100
        @printf("  numerical K, cells a/%-2d at the electrodes, against a/64: median %.2f%%, max %.2f%%\n", r, median(e), maximum(e))
    end
end

println("\n3D two-layer earths, |ρa / exact − 1| (%), 4 lines × 16 electrodes (10 m × 5 m), DD along the lines")
g = ert3d_grid(4, 16, 10.0, 5.0)
d3 = BuildSurvey3D(origin = ORIGIN, x = g.x, y = g.y, quadrupoles = ert3d_line_quadrupoles(g.lines_y, n -> ert2d_dipole_dipole(n; a_max = 2, n_max = 5)))
l3 = ert2d_geometric_layers(5.0, 30.0; first_layer_div = 2, vertical_factor = 1.0)
m3 = BuildMesh3D(electrode_x = d3.x_positions, electrode_y = d3.y_positions, x_core_cell = 5.0, y_core_cell = 2.5, core_margin_cells = 3,
                 n_pad = ert2d_padding_cells(2.5, 75.0, 1.5), layers = vcat(l3, ert2d_padding(l3[end], ert2d_padding_cells(l3[end], 75.0, 1.5), 1.5)))
dist3(e, s) = hypot(d3.x_positions[e] - d3.x_positions[s], d3.y_positions[e] - d3.y_positions[s])
for (ρ1, ρ2, h) in ((100.0, 10.0, 6.0), (20.0, 500.0, 4.0), (100.0, 10000.0, 8.0))
    hh = m3.z_nodes[argmin(abs.(m3.z_nodes .- h))]
    ρ = [z < hh ? ρ1 : ρ2 for _ in m3.x_cell_sizes, _ in m3.y_cell_sizes, z in ert3d_z_centers(m3)]
    exact = two_layer_rhoa(dist3, d3, ert3d_geometric_factor(d3), ρ1, ρ2, hh)
    for r in (4, 2)
        t = @elapsed f = ForwardSolve3D(m3, ρ, d3; refine = r)
        e = abs.(f.rhoa ./ exact .- 1) .* 100
        @printf("  %g/%g, h %.1f, refine %d: max %.2f%%, median %.2f%% (%d model cells, %.1f s)\n", ρ1, ρ2, hh, r, maximum(e),
                median(e), length(ρ), t)
    end
end
let θ = 5.0
    top = m3.x_nodes[end]
    lz = vcat(l3, ert2d_padding(l3[end], ert2d_padding_cells(l3[end], 250.0, 1.5), 1.5))
    ms = BuildMesh3D(electrode_x = d3.x_positions, electrode_y = d3.y_positions, x_core_cell = 5.0, y_core_cell = 2.5, core_margin_cells = 3,
                     n_pad = ert2d_padding_cells(2.5, 75.0, 1.5), layers = lz, surface = (x, _) -> tand(θ) * (top - x))
    ds = E._ert3d_with(d3; z_positions = tand(θ) .* (top .- d3.x_positions))
    k = ert3d_numerical_geometric_factor(ms, ds; refine = 12) ./ ert3d_geometric_factor(ds)
    @printf("  slope %.0f° across the lines, staircase, cells a/12 at the electrodes: K_num / K_exact − 1 median %+.2f%%, max %.2f%%\n",
            θ, 100median(k .- 1), 100maximum(abs.(k .- 1)))
end

println("\nSensitivities against central finite differences")
small = survey(line(16, a), ert2d_dipole_dipole(16; a_max = 2, n_max = 4))
sm = mesh_for(small.electrodes, a, 40.0; surface = yy -> 1.5 + 1.5sin(yy / 8))
smd = E._ert2d_with(small; z_positions = ert2d_electrode_depths(sm))
ρs = 10 .^ (2 .+ 0.4 .* randn(MersenneTwister(1), length(sm.z_cell_sizes), length(sm.y_cell_sizes)))
ρs[ert2d_air_mask(sm)] .= ERT2D_AIR_TAG
J, cells = FrechetDerivative2D(sm, ρs, smd)
worst = 0.0
for j in rand(MersenneTwister(2), findall(c -> c[1] <= 10 && abs(E.ert2d_y_centers(sm)[c[2]]) < 40, cells), 8)
    ρp, ρm = copy(ρs), copy(ρs)
    ρp[cells[j]] *= 10^1e-4
    ρm[cells[j]] /= 10^1e-4
    fd = (log.(ForwardSolve2D(sm, ρp, smd).rhoa) .- log.(ForwardSolve2D(sm, ρm, smd).rhoa)) ./ 2e-4
    global worst = max(worst, maximum(abs.(fd .- J[:, j])) / maximum(abs.(fd)))
end
@printf("  2D, staircase, forward mesh refined at the sources, 8 cells: largest relative error %.1e\n", worst)
g2 = ert3d_grid(2, 6, 5.0, 5.0)
s3 = BuildSurvey3D(origin = ORIGIN, x = g2.x, y = g2.y, quadrupoles = ert3d_line_quadrupoles(g2.lines_y, n -> ert2d_dipole_dipole(n; a_max = 1, n_max = 3)))
sm3 = BuildMesh3D(electrode_x = s3.x_positions, electrode_y = s3.y_positions, x_core_cell = 2.5, core_margin_cells = 2, n_pad = 3,
                  layers = vcat(ert2d_geometric_layers(5.0, 10.0), ert2d_padding(5.0, 3, 1.4)))
ρ3 = 10 .^ (2 .+ 0.3 .* randn(MersenneTwister(3), length(sm3.x_cell_sizes), length(sm3.y_cell_sizes), length(sm3.z_cell_sizes)))
J3, c3 = FrechetDerivative3D(sm3, ρ3, s3)
Rh = ert3d_homogeneous(sm3, s3)
worst3 = 0.0
for j in rand(MersenneTwister(4), findall(c -> c[3] <= 4, c3), 6)
    ρp, ρm = copy(ρ3), copy(ρ3)
    ρp[c3[j]] *= 10^1e-4
    ρm[c3[j]] /= 10^1e-4
    fd = (log.(ForwardSolve3D(sm3, ρp, s3; homogeneous = Rh).rhoa) .- log.(ForwardSolve3D(sm3, ρm, s3; homogeneous = Rh).rhoa)) ./ 2e-4
    global worst3 = max(worst3, maximum(abs.(fd .- J3[:, j])) / maximum(abs.(fd)))
end
@printf("  3D, forward mesh refined at the sources, 6 cells: largest relative error %.1e\n", worst3)

println("\nCost, $(Threads.nthreads()) threads")
d64 = survey(y, ert2d_dipole_dipole(ne; a_max = 3, n_max = 6))
m64 = mesh_for(y, a, 120.0)
ρ64 = fill(100.0, length(m64.z_cell_sizes), length(m64.y_cell_sizes))
ForwardSolve2D(m64, ρ64, d64)
t2 = minimum(@elapsed(ForwardSolve2D(m64, ρ64, d64; homogeneous = ones(ndata(d64)))) for _ in 1:3)
t2j = @elapsed FrechetDerivative2D(m64, ρ64, d64)
@printf("  2D, 64 electrodes, %d data, %d × %d model cells: forward %.2f s, forward + sensitivity %.1f s\n", ndata(d64),
        size(ρ64)..., t2, t2j)
ρ3d = fill(100.0, length(m3.x_cell_sizes), length(m3.y_cell_sizes), length(m3.z_cell_sizes))
t3 = @elapsed ForwardSolve3D(m3, ρ3d, d3; homogeneous = ones(ndata(d3)))
t3j = @elapsed FrechetDerivative3D(m3, ρ3d, d3)
@printf("  3D, 64 electrodes, %d data, %d model cells: forward %.1f s, forward + sensitivity %.1f s\n", ndata(d3),
        length(ρ3d), t3, t3j)
