using Revise
using OptimalBath
using GLMakie

function bump(x, a, c)
    return exp(-a * (x - c)^2)
end

function bathymetry(x, y)
    return 1-0.01x +0* 0.3 * bump(x, 0.01, 30) * (bump(y, 0.01, 50))# + bump(y, 0.01, 60))
end

function initial_height(x, y)
    return 2 - 0.05x
end

initial_condition(x, y) = State(initial_height(x, y), 0.0, 0.0)

function create_problem(N)
    domain = [0.0 100.0; 0.0 100.0]
    grid = Grid2D(N..., domain=domain)

    x_faces, y_faces = cell_faces(grid)
    x_centers, y_centers = cell_centers(grid)

    b = bathymetry.(x_faces, y_faces')
    U0 = States{Average, Elevation}(initial_condition.(x_centers, y_centers'))
    T = 20.0

    return PrimalSWEProblem(U0, T, grid, b)
end

function create_spec(problem)
    backend = VolumeFluxesBackend()
    options = SolverOptions()
    return SolverSpec(problem, backend, options)
end

function determine_objective_indices(grid)
    x_centers, y_centers = cell_centers(grid)
    le(a) = x -> x < a
    gt(a) = x -> x > a
    return CartesianIndices((findfirst(gt(50), x_centers):findlast(le(75), x_centers),
                             findfirst(gt(40), y_centers):findlast(le(60), y_centers)))
end


function get_solution_along_x(U, y_index)
    return [State(OptimalBath.height(U), momentum(U, XDIR)) for U in U.U[:, y_index, :]]
end

function solve_and_animate_along_x(N, β=zeros(N .+ 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    solver = build_solver(spec)
    U, t, centers = solve_primal(solver, β)
    y_index = N[2] ÷ 2
    U_x = get_solution_along_x(U, y_index)
    OptimalBath.animate_solution(U_x, U_x, t, cell_faces(problem.grid, XDIR), problem.initial_bathymetry[:, y_index] .+ β[:, y_index], 4.0)
end

function solve_and_animate(N, β=zeros(N .+ 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    solver = build_solver(spec)
    U, t, x = solve_primal(solver, β)
    OptimalBath.animate_solution(U.U, t, problem.initial_bathymetry + β, problem.grid)
end

function compute_and_plot_gradient(N, β=zeros(N .+ 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    objectives = Objectives(objective_indices=determine_objective_indices(problem.grid),
                            interior_objective=Mass(),
                            design_indices=CartesianIndices(β),
                            regularization=SoftL1(20.0))
    solver = build_solver(spec)
    discrete_adjoint = DiscreteAdjointGradient(solver)
    objective, gradient = compute_objective_and_gradient(β, solver, objectives, discrete_adjoint)

    plot_gradient(gradient, problem.initial_bathymetry + β, objectives, problem.grid)
end

function optimize_problem(N, β0=zeros(N.+1))
    problem = create_problem(N)
    spec = create_spec(problem)
    objectives = Objectives(objective_indices=determine_objective_indices(problem.grid),
                            interior_objective=Mass(),
                            design_indices=CartesianIndices(β0),
                            regularization=0.01SoftL1(20.0))
    solver = build_solver(spec)
    discrete_adjoint = DiscreteAdjointGradient(solver)
    inverse_problem = InverseSWEProblem(problem, solver, objectives, discrete_adjoint)
    plot_callback, finalize_plot = OptimalBath.plot_objective_and_gradient_norm(inverse_problem, MakieBackend())
    res = optimize(inverse_problem, BFGSOptimizer(), β0, plot_callback)
    display(finalize_plot())
    return res
end