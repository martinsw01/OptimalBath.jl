function solve_and_compute_objective(β, spec::SolverSpec, objectives::Objectives)
    db = extrapolate_β_to_full_domain(β, objectives.design_indices, size(spec.problem.initial_bathymetry))
    
    objective = Ref(objectives.regularization(β))

    solver = build_solver(spec, eltype(β))
    Δx = compute_Δx(solver)

    U_depth = similar(spec.problem.U0, Depth, Average, eltype(objective))
    to_depth!(U_depth, spec.problem.U0, spec.problem.initial_bathymetry)

    f_prev = Ref(zero(objective[]))
    f_prev[] = sum(objective_density(objectives.interior_objective, U_depth, objectives.objective_indices))
    function integrate_objective_one_step(U_n, t_n, Δt)
        adjusted_bathymetry = get_bathymetry(solver)
        to_depth!(U_depth, U_n, adjusted_bathymetry)
        f_next = sum(objective_density(objectives.interior_objective, U_depth, objectives.objective_indices))
        objective[] += interior_objective_increment(f_prev[], f_next, Δt, Δx, spec.solver_options.timestepper)
        f_prev[] = f_next
    end

    integration_callback = create_callback(integrate_objective_one_step, solver)

    solve_primal(solver, db, integration_callback)

    densities = objective_density(objectives.terminal_objective,
                                        U_depth,
                                        objectives.objective_indices)

    objective[] += sum(densities) * prod(Δx)

    return objective[]
end

function interior_objective_increment(f_prev, f_next, Δt, Δx, ::ForwardEuler)
    return f_prev * Δt * prod(Δx)
end

function interior_objective_increment(f_prev, f_next, Δt, Δx, ::RK2)
    return 0.5 * (f_next + f_prev) * Δt * prod(Δx)
end

@views function extrapolate_β_to_full_domain(β, design_indices, bathymetry_size)
    full_β = zeros(eltype(β), bathymetry_size)
    full_β[design_indices] .= β
    return full_β
end
