@testset "Test forward equals reverse" begin
    N = 10
    bathymetry = zeros(N + 1)
    problem = PrimalSWEProblem(N, initial_state(bathymetry), 1.0)
    test_spec = SolverSpec(problem, MockBackend())
    β = rand(4) .- 0.5
    objectives = OptimalBath.Objectives(design_indices=[1, 2, 5, 8], interior_objective=OptimalBath.Mass())

    forward_gradient_type = OptimalBath.ForwardADGradient(β, test_spec, objectives)
    reverse_gradient_type = OptimalBath.ReverseADGradient()

    forward_objective, forward_gradient = OptimalBath.compute_objective_and_gradient(β, test_spec, objectives, forward_gradient_type)
    reverse_objective, reverse_gradient = OptimalBath.compute_objective_and_gradient(β, test_spec, objectives, reverse_gradient_type)

    @test forward_objective ≈ reverse_objective
    @test forward_gradient ≈ reverse_gradient
end

struct TestADBackend end
TestADGradient() = ADGradient(TestADBackend())

function OptimalBath.gradient!(f, g, β, spec, objectives, ::ADGradient{Nothing, TestADBackend})
    g .= 0.0
    return f(β, spec, objectives)
end

struct ObjectiveMockBackend <: SolverBackend
    initial_bathymetry
    U
    t
    x
    function ObjectiveMockBackend(N, M)
        initial_bathymetry = -rand(N + 1)
        U = rand(State{2, Float64}, N, M)
        t = [0.0; cumsum(rand(M - 1))] ./ M
        x = range(0.0, stop=1.0, length=N+1)
        return new(initial_bathymetry, U, t, x)
    end
end

function OptimalBath.build_solver(spec::SolverSpec{P, ObjectiveMockBackend, SO}, float_type) where {P, SO}
    return ObjectiveMockSolver(spec.backend, spec.solver_options.timestepper)
end

struct ObjectiveMockSolver{TimeStepperType} <: PrimalSWESolver{NoReconstruction, TimeStepperType, DefaultBathymetrySource}
    initial_bathymetry
    bathymetry
    U
    t
    x
    function ObjectiveMockSolver(backend, timestepper::TimeStepper=ForwardEuler())
        bathymetry = backend.initial_bathymetry
        U = backend.U
        t = backend.t
        x = backend.x
        return new{typeof(timestepper)}(bathymetry, copy(bathymetry), U, t, x)
    end
end

OptimalBath.compute_Δx(problem::ObjectiveMockSolver) = step(problem.x)

OptimalBath.create_callback(f, ::ObjectiveMockSolver) = f

OptimalBath.initial_state(problem::ObjectiveMockSolver) = States{Average, Elevation}(problem.U[:, 1])

OptimalBath.get_bathymetry(solver::ObjectiveMockSolver) = solver.bathymetry


function OptimalBath.solve_primal(solver::ObjectiveMockSolver, δb, callback)
    U, t, x = solve_primal(solver, δb)

    N, M = size(U.U)

    for n in 2:M
        Δt = t[n] - t[n-1]
        callback(States{Average, Elevation}(U.U[:, n]), t[n], Δt)
    end
end

function OptimalBath.solve_primal(solver::ObjectiveMockSolver, δb)
    solver.bathymetry .= solver.initial_bathymetry .+ δb
    U = States{Average, Elevation}(copy(solver.U))
    OptimalBath.adjust_to_bathymetry_changes!(U, δb)
    return U, solver.t, solver.x
end

function create_mock_spec(mock_backend, timestepper)
    N = size(mock_backend.U, 1)
    U0 = States{Average, Elevation}(mock_backend.U[:, 1])
    problem = PrimalSWEProblem(N, U0, last(mock_backend.t); initial_bathymetry=mock_backend.initial_bathymetry)
    return SolverSpec(problem, mock_backend, SolverOptions(NoReconstruction(), timestepper))
end

function compare_objectives(timestepper::TimeStepper)
    N, M = 10, 20
    mock_backend = ObjectiveMockBackend(N, M)
    spec = create_mock_spec(mock_backend, timestepper)
    
    β = rand(4) .- 0.5
    design_indices=[1, 2, 5, 8]
    bathymetry = copy(mock_backend.initial_bathymetry)
    δb = zero(bathymetry)
    δb[design_indices] .+= β

    U, t, x = solve_primal(build_solver(spec), δb)
    U = to_depth(U, bathymetry + δb)

    objectives = OptimalBath.Objectives(design_indices=design_indices,
                                        interior_objective=OptimalBath.Mass(),
                                        terminal_objective=OptimalBath.Energy(),
                                        regularization=L2()
                                        )

    gradient_type = TestADGradient()
    objective, _ = OptimalBath.compute_objective_and_gradient(β, spec, objectives, gradient_type)

    expected_objective = compute_objective(U, t, 0.1, β, objectives, timestepper)

    @test objective ≈ expected_objective
end

@testset "Test ADGradient objective computation" begin
    compare_objectives(ForwardEuler())
    compare_objectives(RK2())
end
