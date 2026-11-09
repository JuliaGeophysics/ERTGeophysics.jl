# small 3D survey: nx lines of ny electrodes, dipole-dipole along and across the lines
function survey3d(nx = 4, ny = 8, dx = 4.0, dy = 2.0; z = nothing)
    g = ert3d_grid(nx, ny, dx, dy)
    q = map(vcat, ert3d_line_quadrupoles(g.lines_y, n -> ert2d_dipole_dipole(n; a_max = 1, n_max = 3)),
            ert3d_line_quadrupoles(g.lines_x, n -> ert2d_dipole_dipole(n; a_max = 1, n_max = 1)))
    BuildSurvey3D(origin = ORIGIN, x = g.x, y = g.y, quadrupoles = q, z_positions = z === nothing ? zeros(length(g.x)) : z)
end

mesh3d(d; cps = 2, depth = 8.0, n_pad = 4) =
    BuildMesh3D(electrode_x = d.x_positions, electrode_y = d.y_positions, x_core_cell = 4.0 / cps, y_core_cell = 2.0 / cps,
                core_margin_cells = 2, n_pad = n_pad, layers = vcat(ert2d_geometric_layers(2.0, depth; first_layer_div = 4,
                                                                                           vertical_factor = 1.2),
                                                                     ert2d_padding(2.0, 5, 1.5)))

@testset "3D" begin
    d = survey3d()
    @testset "Data3D" begin
        @test nelectrodes(d) == 32
        @test ndata(d) == 4 * (5 + 4 + 3) + 8 * 1
        @test ert3d_geometric_factor(d)[1] ≈ π * 1 * 2 * 3 * 2.0              # dipole-dipole a = 2 m, n = 1
        mktempdir() do dir
            obs = ERTGeophysics._ert3d_with(d; resistance = fill(0.5, ndata(d)), resistance_error = fill(0.01, ndata(d)),
                                            geometric_factor = ert3d_geometric_factor(d))
            r = load_data3d(write_data3d(joinpath(dir, "d.dat"), obs))
            @test r.x_positions ≈ obs.x_positions atol = 1e-3
            @test r.a == obs.a
            @test r.resistance ≈ obs.resistance rtol = 1e-6
            @test r.latitudes ≈ obs.latitudes atol = 1e-7
            x, y = ert3d_local_frame(r.origin, r.latitudes, r.longitudes)
            @test x ≈ d.x_positions atol = 5e-3                         # lat/lon written to 1e-8 degrees, about 1 mm
            @test y ≈ d.y_positions atol = 5e-3
            s = recentre_data3d(ERTGeophysics._ert3d_with(obs; x_positions = obs.x_positions .+ 10))
            @test (minimum(s.x_positions) + maximum(s.x_positions)) / 2 ≈ 0 atol = 1e-9
            @test_throws ErrorException load_data2d(joinpath(dir, "d.dat"))
        end
    end

    mesh = mesh3d(d)
    @testset "Mesh3D" begin
        @test all(e -> minimum(abs.(mesh.x_nodes .- e)) < 1e-9, d.x_positions)       # electrodes on nodes
        @test all(e -> minimum(abs.(mesh.y_nodes .- e)) < 1e-9, d.y_positions)
        ρ = 10 .^ (1 .+ rand(MersenneTwister(3), ERTGeophysics._ert3d_size(mesh)...))
        mktempdir() do dir
            m = ReadModel3D(WriteModel3D(joinpath(dir, "m.rho"), mesh, ρ))
            @test m.resistivity ≈ ρ rtol = 1e-4
            @test m.origin ≈ [mesh.x_nodes[1], mesh.y_nodes[1], 0.0] atol = 1e-3
            mesh2, ρ2 = Mesh3DFromInputs(m, d)
            @test mesh2.x_nodes ≈ mesh.x_nodes atol = 1e-2
            @test size(ρ2) == size(ρ)
        end
        fm, fρ = RefineMesh3D(mesh, ρ, 2)
        @test size(fρ) == (2, 2, 1) .* size(ρ) && fm.z_nodes == mesh.z_nodes
        b = ert3d_block_model(mesh; blocks = [(x_range = (-2.0, 2.0), y_range = (-3.0, 3.0), z_range = (1.0, 3.0), resistivity = 5.0)])
        @test count(==(5.0), b) > 0

        # staircase from the electrode depths: every electrode on a tread, air tagged, the ground kept in the model file
        z = 0.15 .* (d.y_positions .- minimum(d.y_positions))
        dz = ERTGeophysics._ert3d_with(d; z_positions = z)
        model = ModelFile3D(x_cell_sizes = mesh.x_cell_sizes, y_cell_sizes = mesh.y_cell_sizes, z_cell_sizes = mesh.z_cell_sizes,
                            resistivity = fill(50.0, ERTGeophysics._ert3d_size(mesh)), origin = [mesh.x_nodes[1], mesh.y_nodes[1], 0.0])
        t = Topography3D(model, dz)
        sm, _ = Mesh3DFromInputs(t.model, t.data; warn = false)
        @test !ert3d_is_flat(sm) && any(ert3d_air_mask(sm)) && all(t.mask[ert3d_air_mask(sm)] .== 0)
        @test all(o -> abs(o.offset) <= o.tolerance + 1e-9, ert3d_electrode_offsets(sm, t.data))
        mktempdir() do dir
            m = ReadModel3D(WriteModel3D(joinpath(dir, "t.rho"), sm, fill(50.0, ERTGeophysics._ert3d_size(sm)); main_depth = 6.0))
            @test m.surface ≈ sm.surface atol = 1e-5
            @test m.main_depth ≈ 6.0
            @test count(m.resistivity .> ERT2D_AIR_THRESHOLD) == count(ert3d_air_mask(sm))
            pad = ert3d_padding_mask(sm, m.main_depth)
            @test any(pad) && !all(pad) && all(pad[:, :, end])
        end
        ρs = fill(50.0, ERTGeophysics._ert3d_size(sm))
        gm, gρ, owner = ert3d_forward_mesh(sm, ρs; refine = 4)
        @test minimum(gm.y_cell_sizes) ≈ 0.5
        @test all((owner .== 0) .== ert3d_air_mask(gm))
    end

    @testset "Fwd3D" begin
        nx, ny, nz = ERTGeophysics._ert3d_size(mesh)
        # a homogeneous earth returns its resistivity exactly, the raw response is close to the half-space
        r = ForwardSolve3D(mesh, fill(40.0, nx, ny, nz), d)
        @test r.rhoa ≈ fill(40.0, ndata(d)) rtol = 1e-9
        @test median(abs.(r.homogeneous .* ert3d_geometric_factor(d) .- 1)) < 0.05
        # element: rows sum to zero, symmetric
        E = ERTGeophysics._ert3d_element(1.0, 2.0, 0.5)
        @test maximum(abs.(sum(E; dims = 2))) < 1e-12 && issymmetric(E)
        # two-layer earth against the image series
        ρ1, ρ2 = 100.0, 10.0
        h = mesh.z_nodes[argmin(abs.(mesh.z_nodes .- 3.0))]
        ρ = [zc < h ? ρ1 : ρ2 for _ in 1:nx, _ in 1:ny, zc in ert3d_z_centers(mesh)]
        κ = (ρ2 - ρ1) / (ρ2 + ρ1)
        V(rr) = ρ1 / (2π) * (1 / rr + 2 * sum(κ^n / sqrt(rr^2 + (2n * h)^2) for n in 1:300))
        pot(e, s) = (e == 0 || s == 0) ? 0.0 : V(hypot(d.x_positions[e] - d.x_positions[s], d.y_positions[e] - d.y_positions[s]))
        exact = [pot(d.m[i], d.a[i]) - pot(d.m[i], d.b[i]) - pot(d.n[i], d.a[i]) + pot(d.n[i], d.b[i]) for i in 1:ndata(d)] .*
                ert3d_geometric_factor(d)
        @test maximum(abs.(ForwardSolve3D(mesh, ρ, d).rhoa ./ exact .- 1)) < 0.03
        # reciprocity; a uniform slope (a tilted half-space, K exact over the straight distances) on the staircase; the
        # cache gives the same answer
        f = ert3d_forward(mesh, ρ)
        @test f.Φ ≈ f.Φ'
        top = mesh.y_nodes[end]
        sl = BuildMesh3D(electrode_x = d.x_positions, electrode_y = d.y_positions, x_core_cell = 2.0, y_core_cell = 1.0,
                         core_margin_cells = 2, n_pad = 4, layers = mesh.z_cell_sizes, surface = (x, y) -> tand(5) * (top - y))
        ds = ERTGeophysics._ert3d_with(d; z_positions = tand(5) .* (top .- d.y_positions))
        ratio = ert3d_numerical_geometric_factor(sl, ds; refine = 12) ./ ert3d_geometric_factor(ds)
        @test abs(median(ratio) - 1) < 0.01 && maximum(abs.(ratio .- 1)) < 0.05
        cache = ERT3DCache()
        ForwardSolve3D(mesh, ρ, d; cache)
        @test ForwardSolve3D(mesh, 2 .* ρ, d; cache).resistance ≈ ForwardSolve3D(mesh, 2 .* ρ, d).resistance rtol = 1e-10
        # sensitivities against finite differences, with forward refinement
        ρr = 10 .^ (2 .+ 0.3 .* randn(MersenneTwister(4), nx, ny, nz))
        J, cells = FrechetDerivative3D(mesh, ρr, d)
        Rh = ert3d_homogeneous(mesh, d)
        worst = 0.0
        for j in rand(MersenneTwister(5), findall(c -> c[3] <= 4, cells), 4)
            ρp, ρm = copy(ρr), copy(ρr)
            ρp[cells[j]] *= 10^1e-4
            ρm[cells[j]] /= 10^1e-4
            fd = (log.(ForwardSolve3D(mesh, ρp, d; homogeneous = Rh).rhoa) .- log.(ForwardSolve3D(mesh, ρm, d; homogeneous = Rh).rhoa)) ./ 2e-4
            worst = max(worst, maximum(abs.(fd .- J[:, j])) / maximum(abs.(fd)))
        end
        @test worst < 1e-4
    end

    @testset "Inv3D and file workflow" begin
        mktempdir() do dir
            truth = ert3d_block_model(mesh; blocks = [(x_range = (-4.0, 4.0), y_range = (-4.0, 2.0), z_range = (0.5, 3.0), resistivity = 15.0)])
            r = ForwardSolve3D(mesh, truth, d)
            K = ert3d_geometric_factor(d)
            R = r.rhoa ./ K .* (1 .+ 0.02 .* randn(MersenneTwister(6), ndata(d)))
            obs = ERTGeophysics._ert3d_with(d; resistance = R, resistance_error = 0.02 .* abs.(R), geometric_factor = K)
            dp = write_data3d(joinpath(dir, "data.obs"), obs)
            out = joinpath(dir, "case")
            mm = MakeMesh3D(dp; out_dir = out, background_resistivity = 100.0, n_pad = 4, n_pad_z = 4)
            for f in ("model.start", "model.prior", "fwd.ctrl", "inv.ctrl", "data.dat", "Mesh.png")
                @test isfile(joinpath(out, f))
            end
            inv = WriteInvCtrl3D(joinpath(dir, "inv.ctrl"), InvCtrl3D(target_rms = 1.0, max_iter = 4, error_floor = 0.02))
            @test ReadInvCtrl3D(inv) == ReadInvCtrl3D(WriteInvCtrl3D(joinpath(dir, "inv2.ctrl"), ReadInvCtrl3D(inv)))
            run = Invert3D(joinpath(out, "model.start"), joinpath(out, "data.dat"), joinpath(out, "fwd.ctrl"), inv,
                           joinpath(out, "model.prior"))
            @test run.rms < 0.3 * run.history[1].rms
            for f in ("model.rho", "data.pred", "coverage.rho", "History.csv", "Summary.txt", "inputs/data.dat")
                @test isfile(joinpath(run.run_dir, f))
            end
            @test chi2_rms3d(joinpath(out, "data.dat"), joinpath(run.run_dir, "data.pred"); error_floor = 0.02).rms ≈ run.rms rtol = 1e-6
            @test all(isfile, PlotInversion3D(run; depths = [0.5, 2.0]))
            @test isfinite(ReadModel3D(joinpath(out, "model.start")).main_depth) && any(run.padding)
            @test ReadModel3D(joinpath(run.run_dir, "model.rho")).main_depth ≈ run.main_depth
            # a staircase case through MakeMesh3D from the electrode depths: numerical K (≈ analytic on this uniform slope)
            zp = write_data3d(joinpath(dir, "z.obs"), ERTGeophysics._ert3d_with(obs; z_positions = 0.2 .* (obs.y_positions .+ 7)))
            mz = MakeMesh3D(zp; out_dir = joinpath(dir, "z"), n_pad = 4, n_pad_z = 4, k_refine = 8)
            @test any(ert3d_air_mask(mz.mesh)) && occursin("K numerical", mz.summary)
            @test mz.data.geometric_factor ≈ ert3d_geometric_factor(mz.data) rtol = 0.06
            fl = MakeMesh3D(zp; out_dir = joinpath(dir, "zf"), topography = :flat, n_pad = 4, n_pad_z = 4)
            @test ert3d_is_flat(fl.mesh) && all(iszero, fl.data.z_positions)
        end
    end
end
