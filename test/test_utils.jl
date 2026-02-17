using OptimalBath: GradientType, PrimalSWEProblem
import OptimalBath: solve_primal, compute_Δx, create_callback, initial_state
using StaticArrays, Test

struct PrimalSWETestProblem <: PrimalSWEProblem
    initial_bathymetry::Vector{Float64}
    function PrimalSWETestProblem(N)
        initial_bathymetry = zeros(Float64, N + 1)
        return new(initial_bathymetry)
    end
end

function compute_Δx(problem::PrimalSWETestProblem)
    N = length(problem.initial_bathymetry) - 1
    return 1 / N
end

function create_callback(f, ::PrimalSWETestProblem)
    return f
end

function initial_state(problem::PrimalSWETestProblem)
    N = length(problem.initial_bathymetry) - 1
    h = first(problem.initial_bathymetry) + 1
    U = State(h, 0.0)
    return States{Average, Elevation}(fill(U, N))
end


function lake_at_rest(bathymetry)
    M = 20
    N = length(bathymetry) - 1
    t = range(0.0, stop=1.0, length=M)
    x = range(0.0, stop=1.0, length=N+1)

    h = first(bathymetry) + 1
    U = State(h, 0.0)
    return fill(U, N, M), t, x
end

function solve_primal(::PrimalSWETestProblem, bathymetry)
    U, t, x = lake_at_rest(bathymetry)
    Ul = States{Left, Depth}(U)
    Ur = States{Right, Depth}(U)
    return (Ul, Ur), t, x
end

function solve_primal(problem::PrimalSWETestProblem, bathymetry, callback)
    U, t, x = lake_at_rest(bathymetry)
    Δt = step(t)
    for n in 2:lastindex(t)
        callback(States{Average, Elevation}(U[:, n]), t[n], Δt)
    end
end


@testset "Test ForwardADGradient interface" begin
    using OptimalBath: ForwardADGradient, compute_objective_and_gradient, Objectives, Mass

    N = 10#rand(10:20)
    bathymetry = zeros(N + 1)
    β = zeros(4)
    problem = PrimalSWETestProblem(N)
    Δx = compute_Δx(problem)
    objectives = Objectives(design_indices=[1, 2, 5, 8], interior_objective=Mass())
    gradient_type = ForwardADGradient(bathymetry, β)

    objective, gradient = compute_objective_and_gradient(β, problem, objectives, gradient_type)

    @test no_error = true
    
    # @test objective ≈ 1
    # @test gradient ≈ [1 + 0.5 * Δx, Δx, Δx, Δx]
end

@testset "Test AdjointApproachGradient interface" begin
    using OptimalBath: AdjointApproachGradient, compute_objective_and_gradient, Objectives, Mass

    N = 10
    bathymetry = zeros(N + 1)
    β = zeros(4)
    problem = PrimalSWETestProblem(N)
    objectives = Objectives(design_indices=[3, 4, 5, 8], interior_objective=Mass())
    gradient_type = AdjointApproachGradient(bathymetry)

    objective, gradient = compute_objective_and_gradient(β, problem, objectives, gradient_type)
    
    @test objective ≈ 1
    @test gradient ≈ [0, 0, 0, 0]
end