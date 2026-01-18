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

    @testset "Objective" begin
        include("objective_test.jl")
    end

    @testset "Adjoint solver" begin
        include("adjoint_solver_test.jl")
    end

    @testset "Implementation of interfaces" begin
        include("test_utils.jl")
    end
end

nothing
