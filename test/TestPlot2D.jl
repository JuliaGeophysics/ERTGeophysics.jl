@testset "Plots" begin
    mktempdir() do dir
        d = survey(16, 2.0)
        mesh = test_mesh(d; depth = 10.0, n_pad = 6)
        ρ = block_model(mesh)
        obs = synthetic(mesh, ρ, d; noise = 0.02)
        pred = ERTGeophysics._ert2d_predicted(obs, ForwardSolve2D(mesh, ρ, obs).rhoa)
        @test isfile(plot_ert2d_model(mesh, ρ; output_path = joinpath(dir, "m.png")))
        @test isfile(plot_ert2d_model(mesh, ρ; output_path = joinpath(dir, "mc.png"), coverage = fill(1e-5, size(ρ)),
                                      show_padding = true, show_grid = true, annotation = "test"))
        @test isfile(plot_ert2d_mesh(mesh; output_path = joinpath(dir, "mesh.png")))
        @test isfile(plot_ert2d_mesh(mesh; output_path = joinpath(dir, "core.png"), region = :core))
        @test isfile(plot_ert2d_pseudosection(obs; output_path = joinpath(dir, "p.png")))
        @test isfile(plot_ert2d_data_fit(obs, pred; output_path = joinpath(dir, "fit.png")))
        mp = WriteModel2D(joinpath(dir, "m.rho"), mesh, ρ)
        dp = write_data2d(joinpath(dir, "d.dat"), obs)
        @test isfile(PlotModel2D(mp; output_path = joinpath(dir, "pm.png"), data_path = dp))
        @test isfile(PlotModel2D(mp; output_path = joinpath(dir, "pm2.png")))
        @test isfile(PlotData2D(dp; output_path = joinpath(dir, "pd.png")))
        h = [(iteration = i, rms = 3.0 / (i + 1), objective = 10.0 / (i + 1), chi2 = 8.0 / (i + 1), regularization = i * 0.1,
              beta = 1.0 / (i + 1)) for i in 0:4]
        @test isfile(plot_inv2d_convergence(h; output_path = joinpath(dir, "c.png"), target_rms = 1.0))
    end
end
