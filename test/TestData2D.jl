@testset "Data2D" begin
    # arrays
    dd = ert2d_dipole_dipole(16; a_max = 2, n_max = 4)
    @test length(dd.a) == sum(max(0, 16 - (n + 2) * s) for s in 1:2 for n in 1:4)
    @test all(dd.b .< dd.a .< dd.m .< dd.n)
    w = ert2d_wenner(16; a_max = 3)
    @test length(w.a) == 13 + 10 + 7
    @test all(ert2d_pole_dipole(16).b .== 0)
    s = ert2d_schlumberger(16; n_max = 3)
    @test all(s.a .< s.m .< s.n .< s.b)

    # analytic K: Wenner 2πa, dipole-dipole πn(n+1)(n+2)a, positive for B A M N
    d = survey(16, 5.0; quadrupoles = w)
    K = ert2d_geometric_factor(d)
    @test K[1] ≈ 2π * 5
    dd1 = survey(16, 5.0; quadrupoles = (a = [2], b = [1], m = [4], n = [5]))   # n = 2, s = 1
    @test ert2d_geometric_factor(dd1)[1] ≈ π * 2 * 3 * 4 * 5
    pd = survey(16, 5.0; quadrupoles = (a = [1], b = [0], m = [2], n = [3]))
    @test ert2d_geometric_factor(pd)[1] ≈ 2π / (1 / 5 - 1 / 10)

    # local frame round trip, any azimuth
    for az in (0.0, 37.0, 90.0, 245.0)
        lat, lon = ert2d_latlon(ORIGIN, az, [3.0, -20.0], [150.0, -310.0])
        x, y = ert2d_local_frame(ORIGIN, az, lat, lon)
        @test x ≈ [3.0, -20.0] atol = 1e-6
        @test y ≈ [150.0, -310.0] atol = 1e-6
    end
    # azimuth 90: y east, x north
    lat, lon = ert2d_latlon(ORIGIN, 90.0, [0.0], [100.0])
    @test lat[1] ≈ ORIGIN[1] atol = 1e-6
    @test lon[1] > ORIGIN[2]

    # profile fit recovers the frame
    sv = survey(16, 5.0; azimuth = 70.0)
    f = ert2d_fit_profile(sv.latitudes, sv.longitudes)
    @test f.azimuth ≈ 70.0 atol = 1e-6
    @test f.y ≈ sv.electrodes atol = 1e-4
    @test maximum(abs.(f.x)) < 1e-4

    # file round trip
    mktempdir() do dir
        d0 = survey()
        R = collect(1.0:ndata(d0))
        dfile = DataFile2D(electrode_names = d0.electrode_names, latitudes = d0.latitudes, longitudes = d0.longitudes,
                           x_positions = d0.x_positions, electrodes = d0.electrodes, z_positions = collect(0.0:15.0) ./ 10,
                           a = d0.a, b = d0.b, m = d0.m, n = d0.n, resistance = R, resistance_error = 0.03R,
                           geometric_factor = ert2d_geometric_factor(d0), origin = d0.origin, azimuth = 70.0)
        p = write_data2d(joinpath(dir, "d.dat"), dfile)
        r = load_data2d(p)
        @test r.electrode_names == dfile.electrode_names
        @test r.latitudes ≈ dfile.latitudes
        @test r.electrodes ≈ dfile.electrodes atol = 1e-3
        @test r.z_positions ≈ dfile.z_positions atol = 1e-3
        @test (r.a, r.b, r.m, r.n) == (dfile.a, dfile.b, dfile.m, dfile.n)
        @test r.resistance ≈ R rtol = 1e-6
        @test r.rhoa ≈ R .* dfile.geometric_factor rtol = 1e-6
        @test r.azimuth ≈ 70.0
        # selection, misfit against itself, invalid data
        sel = select_data2d(r, 1:5)
        @test ndata(sel) == 5 && nelectrodes(sel) == 16
        @test chi2_rms2d(r, r).rms < 1e-5           # ρa written to 8 significant figures
        bad = select_data2d(r, 1:4)
        bad.resistance[1] = NaN; bad.resistance_error[2] = 0.0; bad.resistance[3] = -1.0
        v = ert2d_valid_data(bad)
        @test v.keep == [false, false, false, true]
        @test ert2d_valid_data(bad; error_floor = 0.05).keep == [false, true, false, true]
        # bad electrode index
        lines = readlines(p)
        k = findfirst(l -> occursin(r"^\s+\d+\s+\d+\s+\d+\s+\d+\s", l), lines)
        lines[k] = "    99" * lines[k][7:end]
        write(joinpath(dir, "bad.dat"), join(lines, "\n"))
        @test_throws ArgumentError load_data2d(joinpath(dir, "bad.dat"))
        # a missing K is filled with the flat-earth value
        noK = DataFile2D(electrode_names = d0.electrode_names, latitudes = d0.latitudes, longitudes = d0.longitudes,
                         electrodes = d0.electrodes, a = d0.a, b = d0.b, m = d0.m, n = d0.n, resistance = R,
                         resistance_error = 0.03R, origin = d0.origin)
        @test load_data2d(write_data2d(joinpath(dir, "nok.dat"), noK)).geometric_factor ≈ ert2d_geometric_factor(d0)

        # re-centring keeps the geometry on the ground
        off = BuildSurvey2D(origin = ORIGIN, positions = d0.electrodes .+ 40, quadrupoles = (a = d0.a, b = d0.b, m = d0.m, n = d0.n))
        shifted = recentre_data2d(off)
        @test shifted.electrodes ≈ d0.electrodes
        x, y = ert2d_local_frame(shifted.origin, 90.0, shifted.latitudes, shifted.longitudes)
        @test y ≈ shifted.electrodes atol = 1e-3

        # import from tables
        open(joinpath(dir, "el.csv"), "w") do io
            println(io, "name,lat,lon,elevation")
            foreach(i -> println(io, "E$i,$(d0.latitudes[i]),$(d0.longitudes[i]),$(100 + i)"), 1:16)
        end
        open(joinpath(dir, "m.csv"), "w") do io
            println(io, "a b m n value")
            foreach(i -> println(io, "$(d0.a[i]) $(d0.b[i]) $(d0.m[i]) $(d0.n[i]) 100"), 1:ndata(d0))
        end
        imp = ImportData2D(joinpath(dir, "el.csv"), joinpath(dir, "m.csv"); out_path = joinpath(dir, "imp.dat"),
                           value = :rhoa, topo_path = joinpath(dir, "topo.dat"))
        @test imp.data.rhoa ≈ fill(100.0, ndata(d0)) rtol = 1e-6
        @test imp.data.resistance_error ≈ 0.03 .* abs.(imp.data.resistance)
        @test isfile(imp.topo) && length(ReadTopo2D(imp.topo).elevations) == 16
        @test abs(imp.data.azimuth - 90.0) < 1e-6
    end
    @test_throws ArgumentError BuildSurvey2D(origin = ORIGIN, positions = line(4, 5.0),
                                             quadrupoles = (a = [1], b = [1], m = [2], n = [3]))
end
