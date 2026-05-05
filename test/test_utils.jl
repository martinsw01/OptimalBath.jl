using OptimalBath: GradientType, PrimalSWEProblem
import OptimalBath: solve_primal, compute_Δx, create_callback, initial_state, build_solver, get_bathymetry
using StaticArrays, Test

struct MockBackend <: SolverBackend end

function build_solver(spec::SolverSpec{P, MockBackend, SO}, float_type) where {P, SO}
    return MockSolver(spec.problem.grid.N[1], spec.solver_options.reconstruction, float_type)
end

struct MockSolver{R<:Reconstruction} <: PrimalSWESolver{R, ForwardEuler, DefaultBathymetrySource}
    initial_bathymetry
    function MockSolver(N, r=NoReconstruction(), float_type=Float64)
        initial_bathymetry = zeros(float_type, N + 1)
        return new{typeof(r)}(initial_bathymetry)
    end
end

function compute_Δx(solver::MockSolver)
    N = length(solver.initial_bathymetry) - 1
    return 1 / N
end

function create_callback(f, ::MockSolver)
    return f
end

function get_bathymetry(solver::MockSolver)
    return solver.initial_bathymetry
end

function initial_state(bathymetry)
    N = length(bathymetry) - 1
    h = first(bathymetry) + 1
    U = State(h, 0.0)
    return States{Average, Elevation}(fill(U, N))
end

function initial_state(problem::MockSolver)
    return initial_state(problem.initial_bathymetry)
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

function solve_primal(problem::MockSolver{R}, δb) where {R<:LinearReconstruction}
    U, t, x = lake_at_rest(problem.initial_bathymetry .+ δb)
    Ul = States{Left, Depth}(U)
    Ur = States{Right, Depth}(U)
    return (Ul, Ur), t, x
end

function solve_primal(problem::MockSolver{NoReconstruction}, δb)
    U, t, x = lake_at_rest(δb .+ problem.initial_bathymetry)
    return States{Average, Elevation}(U), t, x
end

function solve_primal(problem::MockSolver, δb, callback)
    U, t, x = lake_at_rest(problem.initial_bathymetry .+ δb)
    Δt = step(t)
    for n in 2:lastindex(t)
        callback(States{Average, Elevation}(U[:, n]), t[n], Δt)
    end
end


@testset "Test ADGradient interface" begin
    using OptimalBath: ForwardADGradient, compute_objective_and_gradient, Objectives, Mass

    N = 10#rand(10:20)
    bathymetry = zeros(N + 1)
    β = zeros(4)
    problem = PrimalSWEProblem(N, initial_state(bathymetry), 1.0)
    spec = SolverSpec(problem, MockBackend())
    Δx = 1/N
    objectives = Objectives(design_indices=[1, 2, 5, 8], interior_objective=Mass())
    gradient_type = ForwardADGradient(β, spec, objectives)

    ad_backends = [ForwardDiffBackend(), ReverseDiffBackend()] #, MooncakeBackend()] # Skip mooncake due to extremely slow compilation time
    objective_vec = similar(ad_backends, eltype(β))
    gradients = similar(ad_backends, eltype(β), (length(β), length(ad_backends)))

    for (i, ad_backend) in enumerate(ad_backends)
        gradient_type = ADGradient(ad_backend, β, spec, objectives)
        objective, gradient = compute_objective_and_gradient(β, spec, objectives, gradient_type)
        objective_vec[i] = objective
        gradients[:, i] .= gradient
    end

    @test all(objective_vec .≈ objective_vec[1])
    @test all(eachcol(gradients) .≈ Ref(gradients[:, 1]))
end

@testset "Test ContinuousAdjointGradient interface" begin
    using OptimalBath: ContinuousAdjointGradient, compute_objective_and_gradient

    N = 10
    bathymetry = zeros(N + 1)
    β = zeros(4)
    solver = MockSolver(N, MinModSlope())
    objectives = Objectives(design_indices=[3, 4, 5, 8], interior_objective=Mass())
    gradient_type = ContinuousAdjointGradient(bathymetry)

    objective, gradient = compute_objective_and_gradient(β, solver, objectives, gradient_type)
    
    @test objective ≈ 1
    @test gradient ≈ [0, 0, 0, 0]
end