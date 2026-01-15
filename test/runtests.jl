using OptimalBath
using Test
# using JET # Latest version is only compatible with Julia v1.12

@testset "OptimalBath.jl" begin
    # @testset "Code linting (JET.jl)" begin
    #     JET.test_package(OptimalBath; target_defined_modules = true)
    # end

    @testset "Adjoint approach gradient" begin
        include("gradient_adjoint_test.jl")
    end


end

nothing
