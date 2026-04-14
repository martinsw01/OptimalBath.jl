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


function time_frame(U, n)
    return selectdim(U, ndims(U), n)
end

function compute_objective_and_gradient!(G, β, solver::PrimalSWESolver{NoReconstruction, TS, BS}, objectives::Objectives, ag::AdjointGradient) where {TS, BS}
    δb = extrapolate_β_to_full_domain(β, objectives.design_indices, size(get_bathymetry(solver)))
    
    U, t, x = solve_primal(solver, δb)
    adjusted_bathymetry = get_bathymetry(solver)
    U = unsafe_to_depth!(U, adjusted_bathymetry)

    Δx = compute_Δx(solver)
    Λ_end = zero(time_frame(U.U, 1))
    Λ_end[objectives.objective_indices] .+= objective_density_gradient(objectives.terminal_objective, U, objectives.objective_indices, lastindex(t)) .* prod(Δx)

    adjoint = adjoint_solver(solver, ag)
    Λ = solve_adjoint(Λ_end, U, objectives, adjusted_bathymetry, t, Δx, adjoint)

    compute_gradient!(G, Λ, U, β, t, Δx, objectives, adjoint)
    objective = compute_objective(U, t, Δx, β, objectives, TS)
    return objective
end

function compute_objective_and_gradient!(G, β, solver::PrimalSWESolver{R, TS, BS}, objectives::Objectives, ag::AdjointGradient) where {R<:LinearReconstruction, TS, BS}
    δb = extrapolate_β_to_full_domain(β, objectives.design_indices, size(get_bathymetry(solver)))

    (Ul, Ur), t, x = solve_primal(solver, δb)
    adjusted_bathymetry = get_bathymetry(solver)

    Δx = compute_Δx(solver)
    Λ_end = zero(time_frame(Ul.U, 1))
    Λ_end[objectives.objective_indices] .+= 0.5 * objective_density_gradient(objectives.terminal_objective, Ul, objectives.objective_indices, lastindex(t)) .* prod(Δx)
    Λ_end[objectives.objective_indices] .+= 0.5 * objective_density_gradient(objectives.terminal_objective, Ur, objectives.objective_indices, lastindex(t)) .* prod(Δx)

    adjoint = adjoint_solver(solver, ag)
    Λ = solve_adjoint(Λ_end, Ul, Ur, objectives, adjusted_bathymetry, t, Δx, adjoint)

    compute_gradient!(G, Λ, Ul, Ur, t, Δx, objectives, ag)
    objective = compute_objective(Ul, Ur, t, Δx, β, objectives, TS)
    return objective
end


include("discrete_adjoint_gradients.jl")
include("continuous_adjoint_gradients.jl")

