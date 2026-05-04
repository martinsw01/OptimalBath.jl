function objective_increment(objectives, U, n, Δt, Δx, ::Type{ForwardEuler})
    interior_objective = objectives.interior_objective
    objective_indices = objectives.objective_indices
    return sum(@view U.U[objective_indices, n]) do Uj
        objective_density(interior_objective, Uj)
    end * Δt * prod(Δx)
end

function objective_increment(objectives, U, n, Δt, Δx, ::Type{RK2})
    interior_objective = objectives.interior_objective
    objective_indices = objectives.objective_indices
    return 0.5 * sum(@view U.U[objective_indices, n:n+1]) do Uj
        objective_density(interior_objective, Uj)
    end * Δt * prod(Δx)
end

function _compute_objective(U::States, t, Δx, β, objectives, timestepper)
    M = lastindex(t)

    interior_integral = objectives.regularization(β)
    for n in 1:lastindex(t)-1
        Δt = t[n+1] - t[n]

        interior_integral += objective_increment(objectives, U, n, Δt, Δx, timestepper)
    end

    terminal_integral = sum(U.U[objectives.objective_indices, M]) do Uj
        objective_density(objectives.terminal_objective, Uj)
    end * prod(Δx)

    return interior_integral + terminal_integral
end

function compute_objective(U::States{S, Depth}, t, Δx, β, objectives::Objectives, timestepper::Type{<:TimeStepper}) where S
    return _compute_objective(U, t, Δx, β, objectives, timestepper)
end

function compute_objective(U::States{S, Depth}, t, Δx, β, objectives::Objectives, timestepper::TimeStepper) where S
    return compute_objective(U, t, Δx, β, objectives, typeof(timestepper))
end

function compute_objective(Ul::States{Left, Depth}, Ur::States{Right, Depth}, t, Δx, β, objectives::Objectives, timestepper::Type{<:TimeStepper})
    return 0.5 * (_compute_objective(Ul, t, Δx, β, objectives, timestepper) +
                  _compute_objective(Ur, t, Δx, β, objectives, timestepper))
end

function compute_objective(Ul::States{Left, Depth}, Ur::States{Right, Depth}, t, Δx, β, objectives::Objectives, timestepper::TimeStepper)
    return compute_objective(Ul, Ur, t, Δx, β, objectives, typeof(timestepper))
end