export Energy, KineticEnergy, SquaredMomentum, Mass, NoObjective

struct Energy <: Objective end
struct KineticEnergy <: Objective end
struct SquaredMomentum <: Objective end
struct Mass <: Objective end
struct NoObjective <: Objective end

struct ScaledObjective{ObjType, FloatType} <: Objective
    scale::FloatType
    objective::ObjType
end

function Base.:*(scale::Number, objective::Objective)
    return ScaledObjective(scale, objective)
end

function objective_density(obj::ScaledObjective, U)
    return obj.scale * objective_density(obj.objective, U)
end

function objective_density_gradient(obj::ScaledObjective, U)
    return obj.scale .* objective_density_gradient(obj.objective, U)
end

function Base.show(io::IO, objective::ScaledObjective)
    return print(io, "$(objective.scale) * $(repr(objective.objective))")
end

function Base.Broadcast.broadcastable(obj::Objective) 
    return Ref(obj)
end

function objective_density(obj::Objective, U::States{S, Depth, T}, I...) where {S, T}
    return objective_density.(obj, @view U.U[I...])
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
    return  U[1]
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

include("streaming_objective_evaluations.jl")
include("batch_objective_evaluations.jl")