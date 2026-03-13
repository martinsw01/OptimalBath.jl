export AdjointGradient

include("adjoint_swe.jl")

abstract type AdjointGradient <: GradientType end

"""
    adjoint_solver(primal_swe_problem::PrimalSWEProblem, ag::AdjointGradient) -> AdjointSolver
"""
function adjoint_solver end

"""
    compute_gradient!(G, Λ, U, t, Δx, objectives::Objectives, ag::AdjointGradient)
    compute_gradient!(G, Λ, Ul, Ur, t, Δx, objectives::Objectives, ag::AdjointGradient)
"""
function compute_gradient! end


function compute_objective_and_gradient!(G, β, primal_swe_problem::PrimalSWEProblem{NoReconstruction, TS}, objectives::Objectives, ag::AdjointGradient) where {TS}
    δb = extrapolate_β_to_full_domain(β, objectives.design_indices, length(initial_state(primal_swe_problem).U))

    adjusted_bathymetry = δb .+ primal_swe_problem.initial_bathymetry
    U, t, x = solve_primal(primal_swe_problem, δb)
    U = unsafe_to_depth!(U, adjusted_bathymetry)

    Δx = x[2] - x[1]
    Λ_end = zero(U.U[:, end])
    Λ_end[objectives.objective_indices] .+= objective_density_gradient(objectives.terminal_objective, U, objectives.objective_indices, lastindex(t)) * Δx

    adjoint = adjoint_solver(primal_swe_problem, ag)
    Λ = solve_adjoint(Λ_end, U, objectives, adjusted_bathymetry, t, Δx, adjoint)

    compute_gradient!(G, Λ, U, t, Δx, objectives, adjoint)
    objective = compute_objective(U, t, x, β, objectives, TS)
    return objective
end

function compute_objective_and_gradient!(G, β, primal_swe_problem::PrimalSWEProblem{LinearReconstruction, TS}, objectives::Objectives, ag::AdjointGradient) where {TS}
    δb = extrapolate_β_to_full_domain(β, objectives.design_indices, length(initial_state(primal_swe_problem).U))

    adjusted_bathymetry = δb .+ primal_swe_problem.initial_bathymetry
    (Ul, Ur), t, x = solve_primal(primal_swe_problem, δb)

    Δx = x[2] - x[1]
    Λ_end = zero(Ul.U[:, end])
    Λ_end[objectives.objective_indices] .+= 0.5 * objective_density_gradient(objectives.terminal_objective, Ul, objectives.objective_indices, lastindex(t))
    Λ_end[objectives.objective_indices] .+= 0.5 * objective_density_gradient(objectives.terminal_objective, Ur, objectives.objective_indices, lastindex(t))

    adjoint = adjoint_solver(primal_swe_problem, ag)
    Λ = solve_adjoint(Λ_end, Ul, Ur, objectives, adjusted_bathymetry, t, Δx, adjoint)

    compute_gradient!(G, Λ, Ul, Ur, t, Δx, objectives, ag)
    objective = compute_objective(Ul, Ur, t, x, β, objectives, TS)
    return objective
end


include("discrete_adjoint_gradients.jl")
include("continuous_adjoint_gradients.jl")

