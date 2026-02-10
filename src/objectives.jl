struct Energy <: Objective end
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
    return 0.5 * (U[2]^2 / U[1] + 9.81 * U[1]^2)
end

function objective_density(::SquaredMomentum, U)
    return U[2]^2
end

function objective_density(::Mass, U)
    return U[1]
end

function objective_density(::NoObjective, U)
    return zero(eltype(U))
end


function objective_density_gradient(obj::Objective, U::States{S, Depth, T, N, A}, I...) where {S, T, N, A}
    return objective_density_gradient.(obj, U.U[I...])
end

function objective_density_gradient(obj::Objective, U)
    ForwardDiff.gradient(U) do u
        objective_density(obj, u)
    end
end


function _compute_objective(U::States, t, x, β, objectives)
    N, M = size(U.U)
    @assert (N, M) == (length(x)-1, length(t))
    Δx = x[2] - x[1]

    interior_integral = objectives.regularization(β)
    for n in 1:lastindex(t)-1
        Δt = t[n+1] - t[n]

        interior_integral += 0.5 * sum(objective_density(objectives.interior_objective,
                                                         U,
                                                         objectives.objective_indices,
                                                         n:n+1)) * Δt * Δx
    end

    terminal_integral = sum(objective_density(objectives.terminal_objective,
                                              U,
                                              objectives.objective_indices,
                                              M)) * Δx

    return interior_integral + terminal_integral
end


function compute_objective(U::States{S, Depth, T, D, A}, t, x, β, objectives::Objectives) where {S, T, D, A}
    return _compute_objective(U, t, x, β, objectives)
end

function compute_objective(Ul::States{Left, Depth, T, D, A}, Ur::States{Right, Depth, T, D, A}, t, x, β, objectives::Objectives) where {T, D, A}
    return 0.5 * (_compute_objective(Ul, t, x, β, objectives) +
                  _compute_objective(Ur, t, x, β, objectives))
end