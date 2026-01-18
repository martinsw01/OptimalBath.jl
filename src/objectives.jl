struct Energy <: Objective end
struct SquaredMomentum <: Objective end
struct Mass <: Objective end
struct NoObjective <: Objective end

function Base.Broadcast.broadcastable(obj::Objective) 
    return Ref(obj)
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


function objective_density_gradient(obj::Objective, U)
    ForwardDiff.gradient(U) do u
        objective_density(obj, u)
    end
end



@views function compute_objective(U, t, x, β, gradient_data::Objectives)
    @assert size(U) == (length(x)-1, length(t))
    Δx = x[2] - x[1]

    function f(U)
        return objective_density(gradient_data.interior_objective, U)
    end
    function g(U)
        return objective_density(gradient_data.terminal_objective, U)
    end

    interior_integral = gradient_data.regularization(β)
    for n in eachindex(t)[1:end-1]
        Δt = t[n+1] - t[n]
        U0 = U[gradient_data.objective_indices, n]
        U1 = U[gradient_data.objective_indices, n+1]
        interior_integral += 0.5 * sum(f, U0) * Δt
        interior_integral += 0.5 * sum(f, U1) * Δt
    end

    terminal_integral = sum(g, U[gradient_data.objective_indices, end]) * Δx
    return interior_integral * Δx + terminal_integral
end