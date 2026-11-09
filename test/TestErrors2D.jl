@testset "Errors2D" begin
    # reciprocal errors: σ = 0.002 + 0.03 |R| imposed on normal and reciprocal measurements
    n = 24
    dd = ert2d_dipole_dipole(n; a_max = 2, n_max = 5)
    q = (a = vcat(dd.a, dd.m), b = vcat(dd.b, dd.n), m = vcat(dd.m, dd.a), n = vcat(dd.n, dd.b))
    d = survey(n, 2.0; quadrupoles = q)
    mesh = test_mesh(d; depth = 14.0, n_pad = 8)
    R = ForwardSolve2D(mesh, block_model(mesh; blocks = NamedTuple[]), d).rhoa ./ ert2d_geometric_factor(d)
    rng = MersenneTwister(7)
    R .+= (0.002 .+ 0.03 .* abs.(R)) .* randn(rng, length(R))
    nd = length(dd.a)
    R[3] *= 2                                                          # a bad normal: its pair is dropped
    raw = ERTGeophysics._ert2d_with(d; resistance = R, resistance_error = fill(NaN, length(R)),
                                    geometric_factor = ert2d_geometric_factor(d))
    rec = Reciprocals2D(raw; max_relative = 0.2)
    @test rec.pairs + rec.dropped == nd
    @test rec.dropped >= 1
    @test ndata(rec.data) == rec.pairs                                  # no unpaired data here
    # |ΔR| of two noisy measurements is about 1.13 σ: the fit lands near the imposed model
    @test 0.02 < rec.e1 < 0.06
    @test all(rec.data.resistance_error .>= 0.01 .* abs.(rec.data.resistance) .- 1e-12)
    @test all(isfinite, rec.data.resistance)
    @test_throws ErrorException Reciprocals2D(survey(n, 2.0; quadrupoles = dd))
end
