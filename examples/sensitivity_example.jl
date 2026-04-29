using Revise
using OptimalBath
using Plots
using GLMakie

function bump(x, c)
    return exp(-0.01 * (x - c)^2)
end

function bumps(x, bump_centers)
    return sum(bump_centers) do c
        bump.(x, c)
    end
end

function slope(x)
    return 1 .- 0.01x
end

function initial_bathymetry(x, bump_centers)
    return slope(x) + 0.5 * bumps(x, bump_centers)
end

function initial_condition(x)
    if x < 30.0
        return State(2-0.04x, 0.0)
    else
        return State(1-0.04*30 - 0.2 * (x - 30.0), 0.0)
    end
end

function create_problem(N)
    domain = [0.0 100.0]
    cell_faces = range(domain..., length=N + 1)
    cell_centers = cell_faces[1:end-1] .+ step(cell_faces) * 0.5
    bathymetry = initial_bathymetry(cell_faces, [75.0])#, 7.0])
    U0 = States{Average, Elevation}(initial_condition.(cell_centers))
    T = 50.
    return PrimalSWEProblem(N, U0, T; initial_bathymetry=bathymetry, domain=domain)
end

function create_spec(problem)
    backend = VolumeFluxesBackend()
    options = SolverOptions()
    return SolverSpec(problem, backend, options)
end

function solve_and_animate(N, β=zeros(N + 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    solver = build_solver(spec)
    Ul, t, x = solve_primal(solver, β)

    animate_solution(Ul.U, Ul.U, t, problem.initial_bathymetry .+ β, problem.grid, 4.0, MakieBackend())
end


function compute_and_plot_gradient(N)
    problem = create_problem(N)
    spec = create_spec(problem)
    solver = build_solver(spec)
    β = zero(problem.initial_bathymetry)
    objectives = Objectives(objective_indices=3N÷4:N, interior_objective=Mass())
    discrete_adjoint = DiscreteAdjointGradient(solver)
    objective, gradient = compute_objective_and_gradient(β, solver, objectives, discrete_adjoint)
    plot_gradient(gradient, problem.initial_bathymetry, problem.U0.U, objectives, problem.grid, MakieBackend())
end


function soft_abs(x)
    α = 10.0
    return (log(1 + exp(-2*α * x)) + log(1 + exp(2*α * x)) - log(2)) / α
end

function optimize_problem(N, β0=zeros(N + 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    solver = build_solver(spec)
    objectives = Objectives(objective_indices=3N÷4:N, interior_objective=Mass(), regularization=(b) -> sum(soft_abs, b)/N)
    gradient_type = DiscreteAdjointGradient(solver)
    inverse_problem = InverseSWEProblem(problem, solver, objectives, gradient_type)
    # anim_callback, finalize_anim = plot_objective_and_gradient_norm(inverse_problem, MakieBackend())
    anim_callback, finalize_anim = OptimalBath.animate_optimization(inverse_problem, β0, MakieBackend())
    res = optimize(inverse_problem, BFGSOptimizer(), β0, anim_callback)
    display(finalize_anim())
    return res
end

# display(solve_and_animate(200))
# display(compute_and_plot_gradient(50))


# OptimalBath.plot_reconstruction_comparison(sin.(range(0, stop=2π, length=20)); ε=1)




# 