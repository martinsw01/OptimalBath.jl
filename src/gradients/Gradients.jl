export Objectives

@kwdef struct Objectives
    interior_objective::Objective = NoObjective()
    terminal_objective::Objective = NoObjective()
    regularization = no_regularization
    design_indices = Colon()
    objective_indices = Colon()
end


function compute_objective_and_gradient(β, primal_swe_problem::PrimalSWEProblem, objectives::Objectives, gradient_type::GradientType)
    G = preallocate_gradient(β)
    objective = compute_objective_and_gradient!(G, β, primal_swe_problem, objectives, gradient_type)
    return objective, G
end

function preallocate_gradient(β)
    return similar(β)
end

include("AdjointApproachGradients.jl")
include("ad_gradients.jl")