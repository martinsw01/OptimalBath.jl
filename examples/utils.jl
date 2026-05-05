using Revise
using OptimalBath
using GLMakie

function initial_bathymetry end
function initial_condition end
function create_objectives end

function create_problem(N::Integer)
    domain = [0.0 100.0]
    cell_faces = range(domain..., length=N + 1)
    cell_centers = cell_faces[1:end-1] .+ step(cell_faces) * 0.5
    bathymetry = initial_bathymetry(cell_faces, [75.0])#, 7.0])
    U0 = States{Average, Elevation}(initial_condition.(cell_centers))
    T = 50.
    return PrimalSWEProblem(N, U0, T; initial_bathymetry=bathymetry, domain=domain)
end

function create_problem(N::NTuple{2})
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

function compute_gradient(N, ad_type::Type{<:ADGradient}, β=zeros(N .+ 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    objectives = create_objectives(problem.grid)

    ad_gradient = ad_type(β, spec, objectives)
    return problem, objectives, compute_objective_and_gradient(β, spec, objectives, ad_gradient)
end

function compute_gradient(N, da_type::Type{<:DiscreteAdjointGradient}, β=zeros(N .+ 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    solver = build_solver(spec)
    objectives = create_objectives(problem.grid)
    da_gradient = da_type(solver)

    return problem, objectives, compute_objective_and_gradient(β, solver, objectives, da_gradient)
end

compute_gradient(N, β=zeros(N .+ 1)) = compute_gradient(N, DiscreteAdjointGradient, β)


function compute_and_plot_gradient(N::Integer, gradient_type::Type=DiscreteAdjointGradient, β=zeros(N + 1), plot_backend=MakieBackend())
    problem, objectives, (objective, gradient) = compute_gradient(N, gradient_type, β)
    plot_gradient(gradient, problem.initial_bathymetry, problem.U0.U, objectives, problem.grid, plot_backend)
end

function compute_and_plot_gradient(N::NTuple{2}, gradient_type::Type=DiscreteAdjointGradient, β=zeros(N .+ 1), plot_backend=MakieBackend())
    problem, objectives, (objective, gradient) = compute_gradient(N, gradient_type, β)
    plot_gradient(gradient, problem.initial_bathymetry, objectives, problem.grid, plot_backend)
end

function solve_and_animate(N, β=zeros(N .+ 1), plot_backend=MakieBackend())
    problem = create_problem(N)
    spec = create_spec(problem)
    solver = build_solver(spec)
    U, t, x = solve_primal(solver, β)
    animate_solution(U.U, t, problem.initial_bathymetry + β, problem.grid, plot_backend, duration=4.0)
end

objective_convergence_callback(plot_backend=MakieBackend()) = (inverse_problem, β0) -> begin
    plot_objective_and_gradient_norm(inverse_problem, plot_backend)
end

animation_callback(plot_backend=MakieBackend()) = (inverse_problem, β0) -> begin
    animate_optimization(inverse_problem, β0, plot_backend)
end

function optimize_problem(N, β0=0.01 .* rand(N .+ 1); create_callback=objective_convergence_callback())
    problem = create_problem(N)
    spec = create_spec(problem)
    solver = build_solver(spec)
    objectives = create_objectives(problem.grid)
    gradient_type = DiscreteAdjointGradient(solver)
    inverse_problem = InverseSWEProblem(problem, solver, objectives, gradient_type)
    anim_callback, finalize_anim = create_callback(inverse_problem, β0)
    res = optimize(inverse_problem, BFGSOptimizer(), β0, anim_callback)
    display(finalize_anim())
    return res
end