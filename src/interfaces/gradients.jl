export GradientType, compute_objective_and_gradient!, compute_objective_and_gradient, Objective, objective_density, objective_density_gradient, compute_objective

abstract type GradientType end

"""
    compute_objective_and_gradient!(G, β, primal_swe_problem::PrimalSWEProblem, data::GradientData, gradient_type::GradientType)

Assigns the gradient of the objective function with respect to the parameters `β` into `G`, and returns the objective function value.
"""
function compute_objective_and_gradient! end

"""
    compute_objective_and_gradient(β, primal_swe_problem::PrimalSWEProblem, data::GradientData, ::GradientType)
"""
function compute_objective_and_gradient end


abstract type Objective end

"""
    objective_density(::Objective, U)
"""
function objective_density end

"""
    objective_density_gradient(obj::Objective, U)
"""
function objective_density_gradient end

"""
    compute_objective(U, t, x, β, gradient_data::GradientData)
"""
function compute_objective end