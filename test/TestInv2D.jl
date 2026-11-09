@testset "Inv2D" begin
    d = survey(24, 2.0; quadrupoles = ert2d_dipole_dipole(24; a_max = 2, n_max = 5))
    mesh = test_mesh(d; depth = 14.0, n_pad = 8)
    truth = block_model(mesh; blocks = [(y_range = (-12.0, -4.0), z_range = (1.0, 5.0), resistivity = 10.0),
                                        (y_range = (4.0, 14.0), z_range = (1.0, 4.0), resistivity = 1000.0)])
    obs = synthetic(mesh, truth, d; noise = 0.02)
    start = fill(100.0, size(truth))
    yc, zc = ert2d_y_centers(mesh), ert2d_z_centers(mesh)
    cond = [CartesianIndex(iz, iy) for iy in eachindex(yc), iz in eachindex(zc) if -10 <= yc[iy] <= -6 && 2 <= zc[iz] <= 4]
    res = [CartesianIndex(iz, iy) for iy in eachindex(yc), iz in eachindex(zc) if 6 <= yc[iy] <= 12 && 1.5 <= zc[iz] <= 3]

    gn = Invert2D(mesh, start, obs; options = Inv2DOptions(max_iter = 15, error_floor = 0.02, verbose = false))
    @test gn.converged && gn.reason == :target_rms
    @test gn.fit.rms <= 1.0
    @test median(log10.(gn.resistivity[cond])) < 1.75             # the conductor shows
    @test median(log10.(gn.resistivity[res])) > 2.25              # and the resistor
    @test all(isfinite, gn.coverage[gn.active_cells])
    @test maximum(gn.coverage[gn.active_cells]) ≈ 1.0
    # β cools to the target, then relaxes (rises) while the fit stays within it: the smoothest fitting model
    betas = [h.beta for h in gn.history]
    k = argmin(betas)
    @test issorted(betas[1:k]; rev = true) && issorted(betas[k:end])
    @test gn.fit.rms > 0.85 || betas[end] == betas[k]
    fixed = Invert2D(mesh, start, obs; options = Inv2DOptions(max_iter = 15, error_floor = 0.02, relax_beta = false, verbose = false))
    @test issorted([h.beta for h in fixed.history]; rev = true) && fixed.fit.rms <= 1.0
    @test gn.history[end].regularization <= fixed.history[end].regularization * (1 + 1e-9)
    @test chi2_rms2d(obs, ERTGeophysics._ert2d_predicted(obs, gn.rhoa); error_floor = 0.02).rms ≈ gn.fit.rms rtol = 1e-6

    # NLCG: slower than GN, but well on its way within a short budget
    nl = Invert2D(mesh, start, obs; algorithm = NLCG2DConfig(),
                  options = Inv2DOptions(max_iter = 60, error_floor = 0.02, beta_cooling = 0.7, verbose = false))
    @test nl.fit.rms < 0.25 * nl.history[1].rms
    @test median(log10.(nl.resistivity[cond])) < 1.9

    # large active models take the conjugate-gradient GN step
    cg = Invert2D(mesh, start, obs; algorithm = GaussNewton2DConfig(dense_limit = 0),
                  options = Inv2DOptions(max_iter = 15, error_floor = 0.02, verbose = false))
    @test cg.fit.rms <= 1.05

    # bounds hold
    bd = Invert2D(mesh, start, obs; options = Inv2DOptions(max_iter = 6, error_floor = 0.02, log_bounds = (1.5, 2.5), verbose = false))
    @test all(1.5 - 1e-9 .<= log10.(bd.resistivity[bd.active_cells]) .<= 2.5 + 1e-9)

    # robust data norm: 5% of the data are outliers ×3; the L1 fit stays closer to the clean data
    bad = select_data2d(obs, 1:ndata(obs))
    idx = 1:20:ndata(bad)
    bad.resistance[idx] .*= 3
    clean = setdiff(1:ndata(obs), idx)
    o = (max_iter = 12, error_floor = 0.02, verbose = false, target_rms = 0.0)
    l2 = Invert2D(mesh, start, bad; options = Inv2DOptions(; o...))
    l1 = Invert2D(mesh, start, bad; options = Inv2DOptions(; o..., data_norm = :L1))
    err(r) = median(abs.(log.(r.rhoa[clean] ./ (obs.geometric_factor[clean] .* obs.resistance[clean]))))
    @test err(l1) < err(l2)
    @test median(log10.(l1.resistivity[cond])) < 1.8

    # blocky model norm runs and fits
    bl = Invert2D(mesh, start, obs; options = Inv2DOptions(max_iter = 15, error_floor = 0.02, model_norm = :L1, verbose = false))
    @test bl.fit.rms < 1.3

    # padding outside the main domain follows the nearest free cell: fewer parameters, the same fit
    pad = trues(size(start))
    pad[1:findlast(<(8.0), zc), ERTGeophysics._core_range(mesh.y_cell_sizes)] .= false
    pd = Invert2D(mesh, start, obs; padding_cells = pad, options = Inv2DOptions(max_iter = 15, error_floor = 0.02, verbose = false))
    @test pd.fit.rms <= 1.0 && length(pd.active_cells) == count(.!pad)
    @test all(isnan, pd.coverage[pad])
    free = pd.active_cells
    for c in findall(pad)[1:37:end]                     # each padding cell carries its nearest free cell's value
        k = argmin([hypot(yc[f[2]] - yc[c[2]], zc[f[1]] - zc[c[1]]) for f in free])
        @test pd.resistivity[c] == pd.resistivity[free[k]]
    end
    @test median(log10.(pd.resistivity[cond])) < 1.75

    # invalid data are dropped; fixed cells stay put
    some = select_data2d(obs, 1:ndata(obs))
    some.resistance[1:3] .= NaN
    active = trues(size(start)); active[1, :] .= false
    r = Invert2D(mesh, start, some; active_cells = active, options = Inv2DOptions(max_iter = 2, verbose = false))
    @test length(r.used) == ndata(obs) - 3
    @test all(r.resistivity[1, :] .== 100.0)
    @test_throws ArgumentError Invert2D(mesh, start, obs; options = Inv2DOptions(data_norm = :L3))
end
