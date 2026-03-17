export Objectives

@kwdef struct Objectives
    interior_objective::Objective = NoObjective()
    terminal_objective::Objective = NoObjective()
    regularization = no_regularization
    design_indices = Colon()
    objective_indices = Colon()
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