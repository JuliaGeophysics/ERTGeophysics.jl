using ERTGeophysics
using Test
using Printf
using LinearAlgebra
using Random
using Statistics

include("Helpers2D.jl")

@testset "ERTGeophysics" begin
    include("TestMesh2D.jl")
    include("TestData2D.jl")
    include("TestControl2D.jl")
    include("TestErrors2D.jl")
    include("TestTopo2D.jl")
    include("TestFwd2D.jl")
    include("TestFrechet2D.jl")
    include("TestInv2D.jl")
    include("TestMakeMesh2D.jl")
    include("TestPlot2D.jl")
    include("Test3D.jl")
end
