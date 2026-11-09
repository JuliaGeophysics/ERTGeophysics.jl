@testset "Topo2D" begin
    d = survey(16, 5.0)
    mesh = test_mesh(d)
    model = ModelFile2D(y_cell_sizes = mesh.y_cell_sizes, z_cell_sizes = mesh.z_cell_sizes,
                        resistivity = fill(100.0, length(mesh.z_cell_sizes), length(mesh.y_cell_sizes)))
    # a ramp: 20 m drop along the line, and a 15 m deep hollow off its left end
    ys = collect(-200.0:1.0:200.0)
    lat, lon = ert2d_latlon(ORIGIN, 90.0, zeros(length(ys)), ys)
    elev = [y < -60 ? 85.0 : 100 - 0.25 * clamp(y, -40, 40) for y in ys]
    topo = Topo2D(latitudes = lat, longitudes = lon, elevations = elev)
    mktempdir() do dir
        t2 = ReadTopo2D(WriteTopo2D(joinpath(dir, "topo.dat"), topo))
        @test t2.elevations ≈ topo.elevations atol = 1e-3
    end
    py, pe = ert2d_profile_topography(topo, d)
    @test issorted(py)
    @test py[1] ≈ -200 atol = 1e-3
    t = Topography2D(model, d, topo; water = [(y_range = (-200.0, -61.0), level = 95.0)], water_resistivity = 30.0)
    @test t.datum ≈ 100 + 0.25 * 37.5 atol = 1e-6                          # highest electrode
    @test t.data.z_positions ≈ t.datum .- (100 .- 0.25 .* d.electrodes) atol = 1e-6
    m2, _ = Mesh2DFromInputs(t.model, t.data; warn = false)
    @test !ert2d_is_flat(m2) && any(ert2d_air_mask(m2))
    @test all(o -> abs(o.offset) <= o.tolerance + 1e-9, ert2d_electrode_offsets(m2, t.data))
    @test all(t.mask[t.model.resistivity .> ERT2D_AIR_THRESHOLD] .== 0)
    # the lake: its surface is the top of the earth, cells above the lake bed are water
    @test ert2d_ground_depth(m2, -150.0) ≈ t.datum - 95 atol = 1e-6
    wet = t.mask .== ERT2D_MASK_WATER
    @test any(wet) && all(t.model.resistivity[wet] .== 30.0)
    yc, zc = ert2d_y_centers(m2), ert2d_z_centers(m2)
    @test all(t.datum - zc[c[1]] > 85 && yc[c[2]] < -60 for c in findall(wet))
    mask = Mask2D(t.model; water = wet, fixed_below_m = 25.0)
    @test all(mask[vcat(0.0, cumsum(mesh.z_cell_sizes)[1:end-1]) .>= 25.0, :] .== 0)
    pad = falses(size(mask)); pad[end, :] .= true
    @test all(Mask2D(t.model; padding = pad)[end, :] .== ERT2D_MASK_PADDING)
    y, g = ert2d_ground(t.model)
    @test length(y) == length(mesh.y_cell_sizes) + 1 && issorted(g[searchsortedfirst(y, -40):searchsortedlast(y, 40)])
    # the same staircase from the electrode depths alone along the line
    tz = Topography2D(model, ERTGeophysics._ert2d_with(d; z_positions = t.data.z_positions))
    @test tz.data.z_positions ≈ t.data.z_positions
    line = unique(ert2d_electrode_columns(m2))
    mz, _ = Mesh2DFromInputs(tz.model, tz.data; warn = false)
    @test ert2d_topo_air(mz)[line] == ert2d_topo_air(m2)[line]
    # a lake under an electrode is an error
    @test_throws ArgumentError Topography2D(model, d, topo; water = [(y_range = (-10.0, 10.0), level = 200.0)])
end
