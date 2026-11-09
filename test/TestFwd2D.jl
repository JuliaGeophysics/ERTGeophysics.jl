@testset "Fwd2D" begin
    # Bessel functions against reference values
    E = ERTGeophysics
    for (x, k0, k1) in ((0.1, 2.4270690247, 9.8538447809), (1.0, 0.4210244382, 0.6019072302),
                        (2.5, 0.0623475532, 0.0738908163), (10.0, 1.778006232e-5, 1.864877345e-5))
        e0, e1 = E._besselk01e(x)
        @test e0 * exp(-x) ≈ k0 rtol = 2e-6
        @test e1 * exp(-x) ≈ k1 rtol = 2e-6
    end

    # wavenumbers: positive weights, Σ w K₀(kr) = π/(2r) to 1e-4
    k, w, err = ert2d_wavenumbers(5.0, 315.0)
    @test all(>(0), w) && all(>(0), k)
    @test err < 1e-4

    d = survey(24, 5.0; quadrupoles = ert2d_dipole_dipole(24; a_max = 2, n_max = 5))
    mesh = test_mesh(d; depth = 50.0)
    nz, ny = length(mesh.z_cell_sizes), length(mesh.y_cell_sizes)

    # half-space: raw within a few % of the analytic value, normalised exact, reciprocity exact
    f = ert2d_forward(mesh, fill(50.0, nz, ny))
    @test maximum(abs.(f.Φ - f.Φ')) <= 1e-12 * maximum(abs.(f.Φ))
    raw = ert2d_resistance(f, d) .* ert2d_geometric_factor(d)
    @test median(abs.(raw ./ 50 .- 1)) < 0.03
    r = ForwardSolve2D(mesh, fill(50.0, nz, ny), d)
    @test r.rhoa ≈ fill(50.0, ndata(d)) rtol = 1e-10

    # two-layer earth against the image series, with the interface on a node
    zn = mesh.z_nodes
    for (ρ1, ρ2, h) in ((100.0, 10.0, 8.0), (20.0, 500.0, 5.0))
        hh = zn[argmin(abs.(zn .- h))]
        ρ = [z < hh ? ρ1 : ρ2 for z in ert2d_z_centers(mesh), _ in 1:ny]
        for q in (ert2d_dipole_dipole(24; a_max = 2, n_max = 5), ert2d_wenner(24; a_max = 7), ert2d_pole_dipole(24))
            dq = survey(24, 5.0; quadrupoles = q)
            exact = two_layer_rhoa(dq, ρ1, ρ2, hh)
            @test maximum(abs.(ForwardSolve2D(mesh, ρ, dq).rhoa ./ exact .- 1)) < 0.015
        end
    end

    # topography: a uniform slope is a tilted half-space, whose K is exact over the straight electrode distances; the
    # staircase, cut finely around the electrodes, reproduces it
    right = test_mesh(d; depth = 50.0).y_nodes[end]
    slope = test_mesh(d; depth = 50.0, surface = y -> tand(5) * (right - y))
    ds = BuildSurvey2D(origin = ORIGIN, positions = d.electrodes, quadrupoles = (a = d.a, b = d.b, m = d.m, n = d.n),
                       z_positions = tand(5) .* (right .- d.electrodes))
    slopeK = ert2d_numerical_geometric_factor(slope, ds) ./ ert2d_geometric_factor(ds)
    @test abs(median(slopeK) - 1) < 0.005 && maximum(abs.(slopeK .- 1)) < 0.04
    # rectangle element: rows of the stiffness sum to zero, the mass sums to the area
    S, M = ERTGeophysics._ert2d_element(2.0, 0.5)
    @test maximum(abs.(sum(S; dims = 2))) < 1e-12 && sum(M) ≈ 1.0 && issymmetric(S)

    # the forward cache gives the same answer, and rebuilds when the mesh or the air changes
    cache = ERT2DCache()
    ρr = 10 .^ (1.5 .+ rand(MersenneTwister(9), nz, ny))
    r1 = ForwardSolve2D(mesh, ρr, d; cache)
    r2 = ForwardSolve2D(mesh, ρr .* 2, d; cache)
    @test r1.resistance ≈ ForwardSolve2D(mesh, ρr, d).resistance rtol = 1e-10
    @test r2.resistance ≈ ForwardSolve2D(mesh, ρr .* 2, d).resistance rtol = 1e-10
    ρs = fill(50.0, length(slope.z_cell_sizes), length(slope.y_cell_sizes))
    ρs[ert2d_air_mask(slope)] .= ERT2D_AIR_TAG
    @test ForwardSolve2D(slope, ρs, ds; cache).resistance ≈ ForwardSolve2D(slope, ρs, ds).resistance rtol = 1e-10

    # numerical K over flat ground approaches the analytic one
    Kn = ert2d_numerical_geometric_factor(mesh, d)
    @test median(abs.(Kn ./ ert2d_geometric_factor(d) .- 1)) < 0.003

    # file workflow
    mktempdir() do dir
        md = WriteModel2D(joinpath(dir, "model.rho"), mesh, fill(50.0, nz, ny))
        dp = write_data2d(joinpath(dir, "data.dat"), synthetic(mesh, fill(50.0, nz, ny), d; noise = 0.0))
        fp = WriteFwdCtrl2D(joinpath(dir, "fwd.ctrl"), FwdCtrl2D(write_frechet = true))
        pred = ForwardSolve2D(md, dp, fp)
        @test chi2_rms2d(dp, pred).rms < 1e-3      # cell sizes are written to 1 mm
        @test isfile(joinpath(dir, "frechet.dat"))
    end
    @test_throws DimensionMismatch ert2d_forward(mesh, fill(1.0, 3, 3))
    @test_throws ArgumentError ert2d_forward(mesh, fill(-1.0, nz, ny))
end
