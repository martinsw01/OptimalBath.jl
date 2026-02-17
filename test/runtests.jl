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

    @testset "Forward vs reverse mode AD gradients" begin
        include("ad_test.jl")
    end

    @testset "Convert to depth" begin
        include("convert_to_depth_test.jl")
    end

    @testset "Reconstruction tests" begin
        include("reconstruction_test.jl")
    end
end

nothing
