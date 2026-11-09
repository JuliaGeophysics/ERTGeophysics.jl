@testset "Frechet2D" begin
    d = survey(12, 5.0; quadrupoles = ert2d_dipole_dipole(12; a_max = 2, n_max = 3))
    mesh = test_mesh(d; depth = 20.0, n_pad = 6)
    ρ = 10 .^ (2 .+ 0.4 .* randn(MersenneTwister(3), length(mesh.z_cell_sizes), length(mesh.y_cell_sizes)))
    # graded solve meshes (several solve cells per model cell) and the plain model mesh
    for refine in (8, 2)
        J, cells = FrechetDerivative2D(mesh, ρ, d; refine)
        @test size(J) == (ndata(d), length(cells))
        for j in rand(MersenneTwister(4), 1:length(cells), 4)
            h = 1e-4
            ρp, ρm = copy(ρ), copy(ρ)
            ρp[cells[j]] *= 10^h
            ρm[cells[j]] /= 10^h
            fd = (log.(ForwardSolve2D(mesh, ρp, d; refine).rhoa) .- log.(ForwardSolve2D(mesh, ρm, d; refine).rhoa)) ./ 2h
            @test maximum(abs.(fd .- J[:, j])) <= 1e-4 * max(maximum(abs.(fd)), 1e-3 * maximum(abs.(J)))
        end
        # adjoint product equals Jᵀw
        r = ForwardSolve2D(mesh, ρ, d; refine, keep_fields = true)
        w = randn(MersenneTwister(5), ndata(d))
        Jσ = ert2d_sensitivity(r.fwd, d, cells)
        @test ert2d_sensitivity_transpose(r.fwd, d, w, cells) ≈ Jσ' * w rtol = 1e-10
    end
    # on a staircase too, where model cells hold air and earth of the finer forward staircase
    hm = test_mesh(d; depth = 20.0, n_pad = 6, surface = y -> 1.5 + 1.5sin(y / 8))
    dh = ERTGeophysics._ert2d_with(d; z_positions = ert2d_electrode_depths(hm))
    ρ = copy(ρ); ρ[ert2d_air_mask(hm)] .= ERT2D_AIR_TAG
    J, cells = FrechetDerivative2D(hm, ρ, dh)
    top = ert2d_topo_air(hm)
    mid = length(hm.y_cell_sizes) ÷ 2
    for j in (findfirst(==(CartesianIndex(top[mid] + 1, mid)), cells),
              findfirst(==(CartesianIndex(top[mid+3] + 3, mid + 3)), cells))
        ρp, ρm = copy(ρ), copy(ρ)
        ρp[cells[j]] *= 10^1e-4
        ρm[cells[j]] /= 10^1e-4
        fd = (log.(ForwardSolve2D(hm, ρp, dh).rhoa) .- log.(ForwardSolve2D(hm, ρm, dh).rhoa)) ./ 2e-4
        @test maximum(abs.(fd .- J[:, j])) <= 1e-4 * maximum(abs.(fd))
    end

    # remote electrodes take no terms
    pd = survey(12, 5.0; quadrupoles = ert2d_pole_dipole(12; n_max = 3))
    J, _ = FrechetDerivative2D(mesh, 10 .^ (2 .+ 0.4 .* randn(MersenneTwister(3), length(mesh.z_cell_sizes), length(mesh.y_cell_sizes))), pd)
    @test all(isfinite, J)
    @test_throws ArgumentError ert2d_sensitivity(ert2d_forward(mesh, ρ; keep_fields = false), d, [CartesianIndex(1, 1)])
end
