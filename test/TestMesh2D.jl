@testset "Mesh2D" begin
    d = survey()
    mesh = test_mesh(d)
    @test mesh.y_nodes[1] ≈ -mesh.y_nodes[end]
    @test all(y -> any(abs.(mesh.y_nodes .- y) .< 1e-9), d.electrodes)       # electrodes on nodes
    @test ert2d_electrode_depths(mesh) == zeros(16)
    @test ert2d_is_flat(mesh) && !any(ert2d_air_mask(mesh))

    # staircase of a ground line: air above it, every electrode on a flat tread
    ground(y) = 2.0 + 1.5sin(y / 10)
    tmesh = test_mesh(d; surface = ground)
    air = ert2d_air_mask(tmesh)
    topo = ert2d_topo_air(tmesh)
    @test any(air) && all(air[1:topo[iy], iy] == trues(topo[iy]) for iy in eachindex(topo))
    cols = ert2d_electrode_columns(tmesh)
    for (e, y) in enumerate(d.electrodes)
        left = searchsortedlast(tmesh.y_nodes, y - 1e-9)        # the cells on both sides of an electrode on a node
        @test topo[cols[e]] == topo[left]
        @test abs(ert2d_electrode_depths(tmesh)[e] - ground(y)) <= maximum(tmesh.z_cell_sizes[1:4]) / 2 + 1e-9
    end
    yc, zc = ert2d_y_centers(tmesh), ert2d_z_centers(tmesh)
    far = findall(iy -> all(abs.(d.electrodes .- yc[iy]) .> 2.5), eachindex(yc))
    @test all(count(<(ert2d_ground_depth(tmesh, yc[iy])), zc) == topo[iy] for iy in far)

    # model file round trip, LOGE and LINEAR, air tags and the ground line kept
    ρ = block_model(tmesh)
    @test count(ρ .> ERT2D_AIR_THRESHOLD) == count(air)
    mktempdir() do dir
        p = WriteModel2D(joinpath(dir, "m.rho"), tmesh, ρ)
        @test occursin("# surface:", read(p, String))
        m = ReadModel2D(p)
        @test size(m.resistivity) == size(ρ)
        @test m.resistivity ≈ ρ rtol = 1e-4          # six significant figures of ln ρ, as in MTGeophysics
        @test m.y_cell_sizes ≈ mesh.y_cell_sizes rtol = 1e-4
        @test m.origin ≈ mesh.y_nodes[1] rtol = 1e-4
        @test m.surface ≈ tmesh.surface atol = 1e-5
        mesh2, ρ2 = Mesh2DFromInputs(m, ERTGeophysics._ert2d_with(d; z_positions = ert2d_electrode_depths(tmesh)))
        @test ert2d_air_mask(mesh2) == air
        @test ρ2 ≈ ρ rtol = 1e-4
        p2 = WriteModel2D(joinpath(dir, "l.rho"), mesh.y_cell_sizes, mesh.z_cell_sizes, fill(3.0, size(ρ)); loge = false)
        @test occursin("LINEAR", read(p2, String))
        @test all(ReadModel2D(p2).resistivity .≈ 3.0) && isempty(ReadModel2D(p2).surface)
        # without a ground line: from the air cells of a staircase file, else from the electrode depths
        st = ReadModel2D(WriteModel2D(joinpath(dir, "s.rho"), mesh.y_cell_sizes, mesh.z_cell_sizes, ρ))
        ms, _ = Mesh2DFromInputs(st, d; warn = false)
        @test !ert2d_is_flat(ms) && count(ert2d_air_mask(ms)) > 0
        dz = ERTGeophysics._ert2d_with(d; z_positions = collect(range(0.0, 3.0, length = 16)))
        m3, _ = Mesh2DFromInputs(ReadModel2D(p2), dz)
        @test ert2d_ground_depth(m3, d.electrodes[end]) ≈ 3.0 && ert2d_ground_depth(m3, d.electrodes[1]) ≈ 0.0
    end

    # refinement cuts a finer staircase of the same ground line
    fm, fρ = RefineMesh2D(tmesh, ρ, 2)
    @test length(fm.y_cell_sizes) == 2length(mesh.y_cell_sizes) && fm.y_nodes[end] ≈ mesh.y_nodes[end]
    @test fm.surface ≈ [ERTGeophysics._ert2d_interp(tmesh.y_nodes, tmesh.surface, y) for y in fm.y_nodes]

    # forward mesh: a/8 at the electrodes, fine rows through the ground band, coarse far away; each earth cell owned
    gm, gρ, owner = ert2d_forward_mesh(tmesh, ρ; refine = 8, growth = 0.25)
    @test minimum(gm.y_cell_sizes) ≈ 5.0 / 8
    @test maximum(gm.z_cell_sizes[gm.z_nodes[1:end-1] .< 3.5]) <= 5.0 / 8 + 1e-9
    @test gm.y_cell_sizes[1] > 10
    gair = ert2d_air_mask(gm)
    @test all((owner .== 0) .== gair)
    @test all(gρ[c] == ρ[owner[c]] for c in CartesianIndices(owner) if owner[c] > 0)
    @test all(ρ[owner[.!gair]] .< ERT2D_AIR_THRESHOLD)              # earth cells never take an air value
    @test maximum(abs.(ert2d_electrode_depths(gm) .- ground.(d.electrodes))) <= 5.0 / 16 + 1e-9

    # layers and padding
    l = ert2d_geometric_layers(5.0, 40.0; first_layer_div = 4, vertical_factor = 1.2)
    @test l[1] ≈ 1.25 && sum(l) >= 40 && l[2] / l[1] ≈ 1.2
    @test all(ert2d_geometric_layers(5.0, 20.0; first_layer_div = 2, vertical_factor = 1.0) .≈ 2.5)
    @test_throws ArgumentError ert2d_geometric_layers(5.0, 40.0; vertical_factor = 0.9)
    n = ert2d_padding_cells(2.5, 300.0, 1.5)
    @test sum(ert2d_padding(2.5, n, 1.5)) >= 300 && sum(ert2d_padding(2.5, n - 1, 1.5)) < 300
end
