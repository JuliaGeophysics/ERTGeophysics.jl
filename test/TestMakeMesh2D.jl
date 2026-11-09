@testset "MakeMesh2D and file workflows" begin
    mktempdir() do dir
        d = survey(20, 2.0; quadrupoles = ert2d_dipole_dipole(20; a_max = 2, n_max = 4), azimuth = 30.0)
        mesh = test_mesh(d; depth = 12.0, n_pad = 8)
        truth = block_model(mesh; blocks = [(y_range = (-8.0, -2.0), z_range = (1.0, 4.0), resistivity = 15.0)])
        obs = synthetic(mesh, truth, d; noise = 0.02)
        # shift the data frame so MakeMesh2D has to re-centre it
        lat, lon = ert2d_latlon(obs.origin, obs.azimuth, obs.x_positions, obs.electrodes .+ 30)
        shifted = ERTGeophysics._ert2d_with(obs; electrodes = obs.electrodes .+ 30, latitudes = lat, longitudes = lon)
        dp = write_data2d(joinpath(dir, "data.obs"), shifted)
        ys = collect(-120.0:1.0:120.0)
        tlat, tlon = ert2d_latlon(ORIGIN, 30.0, zeros(length(ys)), ys .+ 30)
        tp = WriteTopo2D(joinpath(dir, "topo.dat"), Topo2D(latitudes = tlat, longitudes = tlon,
                                                           elevations = 50 .+ 0.05 .* ys))
        out = joinpath(dir, "inputs")
        mm = MakeMesh2D(dp; out_dir = out, topo_path = tp)
        for f in ("model.start", "model.prior", "cov.ctrl", "fwd.ctrl", "inv.ctrl", "data.dat", "Mesh.png", "MeshCore.png",
                  "Pseudosection.png")
            @test isfile(joinpath(out, f))
        end
        dat = load_data2d(joinpath(out, "data.dat"))
        @test (minimum(dat.electrodes) + maximum(dat.electrodes)) / 2 ≈ 0 atol = 1e-6
        @test !any(isnan, dat.geometric_factor)
        @test any(>(0), dat.z_positions)
        @test any(ert2d_air_mask(mm.mesh))                                                 # staircase
        @test all(o -> abs(o.offset) <= o.tolerance + 1e-9, ert2d_electrode_offsets(mm.mesh, dat))
        start = ReadModel2D(joinpath(out, "model.start"))
        @test start.surface ≈ mm.mesh.surface atol = 1e-5
        @test count(start.resistivity .> ERT2D_AIR_THRESHOLD) == count(ert2d_air_mask(mm.mesh))
        # the main domain: even square cells; the padding (mask 2) outside it
        cov = ReadCov2D(joinpath(out, "cov.ctrl"))
        main = cov.mask .== 1
        @test any(main) && any(cov.mask .== ERT2D_MASK_PADDING) && all(cov.mask[ert2d_air_mask(mm.mesh)] .== 0)
        rows = findall(vec(any(main; dims = 2)))
        @test all(mm.mesh.z_cell_sizes[rows] .≈ mm.mesh.y_cell_sizes[end÷2])
        # numerical K over the topography; on this uniform slope it is close to the analytic K over the true distances
        @test occursin("K numerical", mm.summary)
        @test dat.geometric_factor ≈ ert2d_geometric_factor(dat) rtol = 0.03
        @test ReadFwdCtrl2D(joinpath(out, "fwd.ctrl")).refine == 16
        @test_throws ErrorException MakeMesh2D(joinpath(out, "data.dat"); out_dir = out)
        # flat: the relief is ignored, analytic K
        fl = MakeMesh2D(dp; out_dir = joinpath(dir, "flat"), topo_path = tp, topography = :flat)
        df = load_data2d(joinpath(dir, "flat", "data.dat"))
        @test ert2d_is_flat(fl.mesh) && all(iszero, df.z_positions)
        @test ReadFwdCtrl2D(joinpath(dir, "flat", "fwd.ctrl")).refine == 8
        @test df.geometric_factor ≈ ert2d_geometric_factor(df)
        @test_throws ArgumentError MakeMesh2D(dp; out_dir = joinpath(dir, "w"), topo_path = tp, topography = :flat,
                                              water = [(y_range = (-100.0, -90.0), level = 0.0)])

        # the terrain from the electrode depths alone (an imported line, no topo.dat)
        zline = ERTGeophysics._ert2d_with(dat; geometric_factor = fill(NaN, ndata(dat)))
        zp = write_data2d(joinpath(dir, "zline.obs"), zline)
        mz = MakeMesh2D(zp; out_dir = joinpath(dir, "zonly"))
        dz = load_data2d(joinpath(dir, "zonly", "data.dat"))
        line = unique(ert2d_electrode_columns(mm.mesh))
        @test ert2d_topo_air(mz.mesh)[line] == ert2d_topo_air(mm.mesh)[line]
        @test dz.geometric_factor ≈ dat.geometric_factor rtol = 0.02

        # GN file workflow writes its run folder
        inv = WriteInvCtrl2D(joinpath(dir, "inv.ctrl"), InvCtrl2D(algorithm = :gn, target_rms = 1.0, max_iter = 4,
                                                                  error_floor = 0.02))
        run = Invert2D(joinpath(out, "model.start"), joinpath(out, "data.dat"), joinpath(out, "fwd.ctrl"), inv,
                       joinpath(out, "cov.ctrl"), joinpath(out, "model.prior"))
        @test run.rms < 3
        for f in ("model.rho", "data.pred", "coverage.rho", "History.csv", "Summary.txt", "inputs/data.dat")
            @test isfile(joinpath(run.run_dir, f))
        end
        @test chi2_rms2d(joinpath(out, "data.dat"), joinpath(run.run_dir, "data.pred"); error_floor = 0.02).rms ≈ run.rms rtol = 1e-6
        final = ReadModel2D(joinpath(run.run_dir, "model.rho"))
        @test final.surface ≈ mm.mesh.surface atol = 1e-5
        @test all(final.resistivity[ert2d_air_mask(run.mesh)] .> ERT2D_AIR_THRESHOLD)
        @test any(run.padding) && occursin("Padding cells", read(joinpath(run.run_dir, "Summary.txt"), String))

    end
end
