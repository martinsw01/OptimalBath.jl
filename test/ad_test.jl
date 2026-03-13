@testset "Test forward equals reverse" begin
    N = 10
    test_problem = PrimalSWETestProblem(N)
    bathymetry = zeros(N + 1)
    β = rand(4) .- 0.5
    objectives = OptimalBath.Objectives(design_indices=[1, 2, 5, 8], interior_objective=OptimalBath.Mass())

    forward_gradient_type = OptimalBath.ForwardADGradient(bathymetry, β)
    reverse_gradient_type = OptimalBath.ReverseADGradient(β)

    forward_objective, forward_gradient = OptimalBath.compute_objective_and_gradient(β, test_problem, objectives, forward_gradient_type)
    reverse_objective, reverse_gradient = OptimalBath.compute_objective_and_gradient(β, test_problem, objectives, reverse_gradient_type)

    @test forward_objective ≈ reverse_objective
    @test forward_gradient ≈ reverse_gradient
end

using OptimalBath: ADGradient, State, States, Depth, Average
import OptimalBath: gradient!, get_objective, get_gradient, update_and_get_bathymetry!

struct TestADGradient <: ADGradient
    objective
    TestADGradient() = new([0.0])
end

function gradient!(ad::TestADGradient, f, β)
    ad.objective[1] = f(β)
end

function get_objective(ad::TestADGradient)
    return ad.objective[1]
end

function get_gradient(::TestADGradient)
    return 1.0
end

function update_and_get_bathymetry!(ad::TestADGradient, swe_problem::PrimalSWEProblem, indices, β)
    bathymetry = copy(swe_problem.initial_bathymetry)
    bathymetry[indices] .+= β
    return bathymetry
end

struct SWETestObjectiveProblem{TimeStepperType} <: PrimalSWEProblem{NoReconstruction, TimeStepperType}
    initial_bathymetry
    U
    t
    x
    function SWETestObjectiveProblem(N, M, timestepper::TimeStepper=ForwardEuler())
        initial_bathymetry = rand(N + 1) .- 0.5
        U = rand(State{Float64}, N, M)
        t = [0.0; cumsum(rand(M - 1))] ./ M
        x = range(0.0, stop=1.0, length=N+1)
        return new{typeof(timestepper)}(initial_bathymetry, U, t, x)
    end
end

compute_Δx(problem::SWETestObjectiveProblem) = step(problem.x)

create_callback(f, ::SWETestObjectiveProblem) = f

initial_state(problem::SWETestObjectiveProblem) = States{Average, Elevation}(problem.U[:, 1])


function solve_primal(problem::SWETestObjectiveProblem, δb, callback)
    x = problem.x
    U = States{Average, Elevation}(copy(problem.U))
    OptimalBath.adjust_to_bathymetry_changes!(U, δb)
    t = problem.t

    N, M = size(U.U)

    for n in 2:M
        Δt = t[n] - t[n-1]
        callback(States{Average, Elevation}(U.U[:, n]), t[n], Δt)
    end
end

function solve_primal(problem::SWETestObjectiveProblem, δb)
    U = States{Average, Elevation}(copy(problem.U))
    OptimalBath.adjust_to_bathymetry_changes!(U, δb)
    return U, problem.t, problem.x
end

function compare_objectives(timestepper::TimeStepper)
    N, M = 10, 20
    test_problem = SWETestObjectiveProblem(N, M, timestepper)
    
    β = rand(4) .- 0.5
    design_indices=[1, 2, 5, 8]
    bathymetry = copy(test_problem.initial_bathymetry)
    δb = zero(bathymetry)
    δb[design_indices] .+= β

    U, t, x = solve_primal(test_problem, δb)
    U = to_depth(U, bathymetry + δb)

    objectives = OptimalBath.Objectives(design_indices=design_indices,
                                        interior_objective=OptimalBath.Mass(),
                                        terminal_objective=OptimalBath.Energy(),
                                        regularization=(β) -> sum(abs2, β)
                                        )

    gradient_type = TestADGradient()
    objective, _ = OptimalBath.compute_objective_and_gradient(β, test_problem, objectives, gradient_type)

    expected_objective = compute_objective(U, t, x, β, objectives, timestepper)

    @test objective ≈ expected_objective
end

@testset "Test ADGradient objective computation" begin
    compare_objectives(ForwardEuler())
    compare_objectives(RK2())
end
