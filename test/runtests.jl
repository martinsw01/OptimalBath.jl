using OptimalBath
using Test
using JET

@testset "OptimalBath.jl" begin
    @testset "Code linting (JET.jl)" begin
        JET.test_package(OptimalBath; target_defined_modules = true)
    end
    # Write your tests here.
end
