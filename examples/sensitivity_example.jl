using Revise
using OptimalBath

function bump(x, c)
    return exp(-30 * (x - c)^2)
end

function bumps(x, bump_centers)
    return sum(bump_centers) do c
        bump.(x, c)
    end
end

function slope(x)
    return 1 .- 0.1x
end

function initial_bathymetry(x, bump_centers)
    return slope(x) + 0*0.15 * bumps(x, bump_centers)
end

function initial_condition(x)
    if x < 3.0
        return State(1-0.08x, 0.0)
    else
        return State(1-0.08*3 - 0.2 * (x - 3.0), 0.0)
    end
end

function create_problem(N)
    domain = [0.0 10.0]
    cell_faces = range(domain..., length=N + 1)
    cell_centers = cell_faces[1:end-1] .+ step(cell_faces) * 0.5
    bathymetry = initial_bathymetry(cell_faces, [4.5])#, 7.0])
    U0 = States{Average, Elevation}(initial_condition.(cell_centers))
    T = 4.0
    return PrimalSWEProblem(N, U0, T; initial_bathymetry=bathymetry, domain=domain)
end

function create_spec(problem)
    backend = VolumeFluxesBackend()
    options = SolverOptions()
    # options = SolverOptions(SoftMinModSlope())
    # options = SolverOptions(MinModSlope())
    return SolverSpec(problem, backend, options)
end

function solve_and_animate(N, β=zeros(N + 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    solver = build_solver(spec)
    Ul, t, x = solve_primal(solver, β)
    for n in axes(Ul.U, 2)
        @show sum(Ul.U[:, n])
    end
    # (Ul, Ur), t, x = solve_primal(solver, β)
    # Ul = unsafe_to_elevation!(Ul, problem.initial_bathymetry)
    # Ur = unsafe_to_elevation!(Ur, problem.initial_bathymetry)
    # for n in axes(Ul.U, 2)
    #     if n % 10 == 0
    #         @show sum(Ul.U[:, n]) #+ sum(Ur.U[:, n])
    #     end
    # end

    # OptimalBath.animate_solution(Ul.U, Ul.U, t, x, problem.initial_bathymetry .+ β, 4.0)
    # OptimalBath.animate_solution(Ul.U, Ur.U, t, x, problem.initial_bathymetry, 4.0)
end


function compute_and_plot_gradient(N)
    problem = create_problem(N)
    spec = create_spec(problem)
    β = zero(problem.initial_bathymetry)
    objectives = Objectives(objective_indices=N÷2:N, interior_objective=Mass())
    # forward_ad = ForwardADGradient(β)
    discrete_adjoint = DiscreteAdjointGradient()
    objective, gradient = compute_objective_and_gradient(β, spec, objectives, discrete_adjoint)
    x = range(problem.grid.domain..., length=N + 1)
    # objective, gradient = compute_objective_and_gradient(β, spec, objectives, forward_ad)
    display(plot_gradient(gradient, x, problem.U0, problem.initial_bathymetry))
    # display(gradient)
    # display(objective)
end


function soft_abs(x)
    α = 10.0
    return (log(1 + exp(-2*α * x)) + log(1 + exp(2*α * x)) - log(2)) / α
end

function optimize_problem(N, β0=zeros(N + 1))
    problem = create_problem(N)
    spec = create_spec(problem)
    objectives = Objectives(objective_indices=N÷2:N, interior_objective=Mass(), regularization=(b) -> 10sum(soft_abs, b)/N)
    gradient_type = DiscreteAdjointGradient()
    # gradient_type = ForwardADGradient(β0)
    inverse_problem = InverseSWEProblem(problem, spec, objectives, gradient_type)
    anim_callback, finalize_anim = OptimalBath.animate_optimization(inverse_problem, β0)
    res = optimize(inverse_problem, BFGSOptimizer(), β0, anim_callback)
    # res = optimize(inverse_problem, GradientDescent(), β0, anim_callback)
    display(finalize_anim())#;fps=10))#;loop=0, fps=1))
    return res
end

# display(solve_and_animate(200))
# display(compute_and_plot_gradient(50))


# OptimalBath.plot_reconstruction_comparison(sin.(range(0, stop=2π, length=20)); ε=1)