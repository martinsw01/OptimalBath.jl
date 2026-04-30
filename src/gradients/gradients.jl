export Objectives

@kwdef struct Objectives{InteriorObjective<:Objective, TerminalObjective<:Objective, RegType, DesignIndices, ObjectiveIndices}
    interior_objective::InteriorObjective = NoObjective()
    terminal_objective::TerminalObjective = NoObjective()
    regularization::RegType = NoRegularization()
    design_indices::DesignIndices = Colon()
    objective_indices::ObjectiveIndices = Colon()
end

function compute_objective_and_gradient!(G, β, spec::SolverSpec, objectives::Objectives, gradient_type::GradientType)
    return compute_objective_and_gradient!(G, β, build_solver(spec, eltype(β)), objectives, gradient_type)
end

function compute_objective_and_gradient(β, spec::SolverSpec, objectives::Objectives, gradient_type::GradientType)
    G = allocate_gradient(β)
    objective = compute_objective_and_gradient!(G, β, spec, objectives, gradient_type)
    return objective, G
end

function compute_objective_and_gradient(β, solver::PrimalSWESolver, objectives::Objectives, gradient_type::GradientType)
    G = allocate_gradient(β)
    objective = compute_objective_and_gradient!(G, β, solver, objectives, gradient_type)
    return objective, G
end

function allocate_gradient(β)
    return similar(β)
end

include("automatic_differentiation/ad_gradients.jl")
include("adjoints/adjoint_gradients.jl")

using .DiscreteAdjoints: DiscreteAdjointGradient, DiscreteAdjointSWE

export DiscreteAdjointGradient, DiscreteAdjointSWE