abstract type IntegrationRegion end
struct Volume <: IntegrationRegion end
struct Terminal <: IntegrationRegion end

function Base.Broadcast.broadcastable(region::IntegrationRegion) 
    return Ref(region)
end

abstract type Objective end
struct Energy <: Objective end
struct SquaredMomentum <: Objective end
struct Mass <: Objective end
struct NoObjective <: Objective end

function Base.Broadcast.broadcastable(obj::Objective) 
    return Ref(obj)
end

function objective_density(::IntegrationRegion, ::Objective, U) end

function objective_density(::Volume, ::Energy, U)
    return 0.5 * (U[2]^2 / U[1] + 9.81 * U[1]^2)
end

function objective_density(::Volume, ::SquaredMomentum, U)
    return U[2]^2
end

function objective_density(::Volume, ::Mass, U)
    return U[1]
end

function objective_density(::Terminal, ::NoObjective, U)
    return zero(eltype(U))
end

function objective_density(::Volume, ::NoObjective, U)
    return zero(eltype(U))
end

function objective_density_gradient(region::IntegrationRegion, obj::Objective, U)
    ForwardDiff.gradient(U) do u
        objective_density(region, obj, u)
    end
end



@views function compute_objective(U, t, x, β, interior_objective::Objective, terminal_objective::Objective, parameter_objective)
    @assert size(U) == (length(x)-1, length(t))
    Δx = x[2] - x[1]

    function f(U)
        return objective_density(Volume(), interior_objective, U)
    end
    function g(U)
        return objective_density(Terminal(), terminal_objective, U)
    end

    interior_integral = parameter_objective(β)
    for n in eachindex(t)[1:end-1]
        Δt = t[n+1] - t[n]
        U0 = U[:,n]
        U1 = U[:,n+1]
        interior_integral += 0.5 * sum(f, U0) * Δt
        interior_integral += 0.5 * sum(f, U1) * Δt
    end

    terminal_integral = sum(g, U[:,end]) * Δx
    return interior_integral * Δx + terminal_integral
end