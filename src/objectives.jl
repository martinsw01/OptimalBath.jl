export Energy, KineticEnergy, SquaredMomentum, Mass, NoObjective

struct Energy <: Objective end
struct KineticEnergy <: Objective end
struct SquaredMomentum <: Objective end
struct Mass <: Objective end
struct NoObjective <: Objective end

function Base.Broadcast.broadcastable(obj::Objective) 
    return Ref(obj)
end

function objective_density(obj::Objective, U::States{S, Depth, T}, I...) where {S, T}
    return objective_density.(obj, U.U[I...])
end

function objective_density(::Energy, U)
    h, p = U
    u = desingularize(h, p)
    return 0.5 * (p*u + 9.81 * h^2)
end

function objective_density(::KineticEnergy, (h, hu...))
    u = desingularize.(h, hu)
    return 0.5 * h * sum(u.^2)
end

function objective_density_gradient(::KineticEnergy, (h, hu...))
    u = desingularize.(h, hu)
    SVector(-0.5 * sum(u.^2), u...)
end

function objective_density(::SquaredMomentum, U)
    return sum(U[2:end].^2)
end

function objective_density(::Mass, U)
    return U[1]
    # return U[1]^2
end

function objective_density(::NoObjective, U)
    return zero(eltype(U))
end


function objective_density_gradient(obj::Objective, U::States{S, Depth}, I...) where S
    return objective_density_gradient.(obj, @view U.U[I...])
end

function objective_density_gradient(obj::Objective, U)
    ForwardDiff.gradient(U) do u
        objective_density(obj, u)
    end
end

function objective_increment(objectives, U, n, Δt, Δx, ::Type{ForwardEuler})
    interior_objective = objectives.interior_objective
    objective_indices = objectives.objective_indices
    return sum(objective_density(interior_objective, U, objective_indices, n)) * Δt * prod(Δx)
end

function objective_increment(objectives, U, n, Δt, Δx, ::Type{RK2})
    interior_objective = objectives.interior_objective
    objective_indices = objectives.objective_indices
    return 0.5 * sum(objective_density(interior_objective, U, objective_indices, n:n+1)) * Δt * prod(Δx)
end

function _compute_objective(U::States, t, Δx, β, objectives, timestepper)
    M = lastindex(t)

    interior_integral = objectives.regularization(β)
    for n in 1:lastindex(t)-1
        Δt = t[n+1] - t[n]

        interior_integral += objective_increment(objectives, U, n, Δt, Δx, timestepper)
    end

    terminal_integral = sum(objective_density(objectives.terminal_objective,
                                              U,
                                              objectives.objective_indices,
                                              M)) * prod(Δx)

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