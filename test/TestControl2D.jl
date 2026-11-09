@testset "Control2D" begin
    mktempdir() do dir
        ctrl = joinpath(dirname(@__DIR__), "examples", "ctrl", "2D")
        f = ReadFwdCtrl2D(joinpath(ctrl, "FwdCtrl"))
        @test f.wavenumbers == 16 && f.refine == 8 && f.growth ≈ 0.25 && f.k_refine == 32
        @test ReadFwdCtrl2D(WriteFwdCtrl2D(joinpath(dir, "fwd.ctrl"), f)) == f
        for alg in ("GN", "NLCG")
            c = ReadInvCtrl2D(joinpath(ctrl, "InvCtrl.$alg"))
            @test c.algorithm == Symbol(lowercase(alg))
            @test isnan(c.lambda) && c.error_floor ≈ 0.02
            c2 = ReadInvCtrl2D(WriteInvCtrl2D(joinpath(dir, "inv.$alg"), c))
            @test c2.algorithm == c.algorithm && c2.max_iter == c.max_iter && c2.log_bounds == c.log_bounds
            @test isnan(c2.lambda) && c2.error_floor ≈ c.error_floor
        end

        # errors: unknown, duplicate and missing keys, bad values
        write(joinpath(dir, "a"), "Algorithm : GN\nFoo : 1\n")
        @test_throws ErrorException ReadInvCtrl2D(joinpath(dir, "a"))
        write(joinpath(dir, "b"), "Algorithm : GN\nAlgorithm : GN\n")
        @test_throws ErrorException ReadInvCtrl2D(joinpath(dir, "b"))
        write(joinpath(dir, "c"), "Algorithm : GN\n")
        @test_throws ErrorException ReadInvCtrl2D(joinpath(dir, "c"))
        write(joinpath(dir, "d"), "Algorithm : VFSA\nExit search when rms is less than : 1\nMaximum number of iterations : 3\n")
        @test_throws ErrorException ReadInvCtrl2D(joinpath(dir, "d"))
        write(joinpath(dir, "e"), "Algorithm : GN\nExit search when rms is less than : 1\nMaximum number of iterations : 3\nData norm : L3\n")
        @test_throws ErrorException ReadInvCtrl2D(joinpath(dir, "e"))
        write(joinpath(dir, "f"), "Wavenumbers : 2\n")
        @test_throws ArgumentError ReadFwdCtrl2D(joinpath(dir, "f"))

        # covariance round trip
        mask = ones(Int, 4, 6); mask[1, 1:2] .= 0; mask[2, 3] = 9
        cov = Cov2D(sy = fill(0.3, 4), sz = 0.3, n_smooth = 1, exceptions = [(1, 9, 0.0)], mask = mask)
        c2 = ReadCov2D(WriteCov2D(joinpath(dir, "cov.ctrl"), cov))
        @test c2.mask == mask && c2.exceptions == cov.exceptions && c2.sy == cov.sy
        @test size(Cov2D(3, 5).mask) == (3, 5)
    end
end
