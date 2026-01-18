using OptimalBath: GradientType, PrimalSWEProblem
import OptimalBath: solve_primal
using StaticArrays, Test

struct PrimalSWETestProblem <: PrimalSWEProblem end

function lake_at_rest(bathymetry, N, M)
    U = SVector{2, eltype(bathymetry)}(1., 0.0)
    return fill(U, N, M)
end

function solve_primal(::PrimalSWETestProblem, bathymetry)
    N = length(bathymetry) - 1
    M = 10
    U = lake_at_rest(bathymetry, N, M)
    t = range(0.0, stop=1.0, length=M)
    x = range(0.0, stop=1.0, length=N+1)
    return U, t, x
end


@testset "Test ForwardADGradient interface" begin
    using OptimalBath: ForwardADGradient, compute_objective_and_gradient, Objectives, NoObjective

    bathymetry = zeros(11)
    β = ones(4)
    problem = PrimalSWETestProblem()
    objectives = Objectives(design_indices=[1, 2, 5, 8])
    gradient_type = ForwardADGradient(bathymetry, β)

    objective, gradient = compute_objective_and_gradient(β, problem, objectives, gradient_type)
    
    @test objective == 0
    @test gradient == zero(β)
end

@testset "Test AdjointApproachGradient interface" begin
    using OptimalBath: AdjointApproachGradient, compute_objective_and_gradient, Objectives, NoObjective

    bathymetry = zeros(11)
    β = ones(4)
    problem = PrimalSWETestProblem()
    objectives = Objectives(design_indices=[3, 4, 5, 8])
    gradient_type = AdjointApproachGradient(bathymetry)

    objective, gradient = compute_objective_and_gradient(β, problem, objectives, gradient_type)
    
    @test objective == 0
    @test gradient == zero(β)
end