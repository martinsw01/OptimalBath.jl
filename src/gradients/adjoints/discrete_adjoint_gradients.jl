module DiscreteAdjoints

using ..OptimalBath
using ..OptimalBath: extrapolate_β_to_full_domain, initial_state
import ..OptimalBath: compute_objective_and_gradient!

using ForwardDiff: derivative
using SinFVM: XDIR

include("discrete_adjoint_swe.jl")

export DiscreteAdjointGradient, DiscreteAdjoint

struct DiscreteAdjointGradient <: GradientType end

function compute_objective_and_gradient!(G, β, primal_swe_problem::PrimalSWEProblem{NoReconstruction, ForwardEuler}, objectives::Objectives, ::DiscreteAdjointGradient)
    δb = extrapolate_β_to_full_domain(β, objectives.design_indices, length(initial_state(primal_swe_problem).U))
    adjusted_bathymetry = δb .+ primal_swe_problem.initial_bathymetry
    U, t, x = solve_primal(primal_swe_problem, δb)
    U = unsafe_to_depth!(U, adjusted_bathymetry)

    Δx = x[2] - x[1]
    Λ0 = zero(U.U[:, end])
    Λ0[objectives.objective_indices] .+= objective_density_gradient(objectives.terminal_objective, U, objectives.objective_indices, lastindex(t)) * Δx

    da = DiscreteAdjoint(primal_swe_problem)
    Λ = solve_adjoint(Λ0, U, objectives, adjusted_bathymetry, t, Δx, da)

    compute_gradient!(G, Λ, U.U, t, Δx, objectives, da)
    objective = compute_objective(U, t, x, β, objectives, ForwardEuler)
    return objective
end

function compute_gradient!(G, Λ, U, t, Δx, objectives::Objectives, da::DiscreteAdjoint)
    G .= zero(eltype(G))
    add_bottom_source_gradient_contribution!(G, Λ, U, t, Δx, objectives.design_indices, da)
end

function integrate_gradient(U, Λ, t, Δx, ::DiscreteAdjoint)
    return sum(momentum.(Λ[2:end]) .* height.(U[1:end-1]) .* diff(t)) * 9.81 / Δx
end

function add_bottom_source_gradient_contribution!(G, Λ, U, t, Δx, design_indices::Colon, da::DiscreteAdjoint)
    N, M = size(U)
    G[1] += integrate_gradient(U[1,:], Λ[1,:], t, Δx, da)
    for j in 2:N
        G[j] -= integrate_gradient(U[j-1,:], Λ[j-1,:], t, Δx, da)
        G[j] += integrate_gradient(U[j,:], Λ[j,:], t, Δx, da)
    end
    G[end] -= integrate_gradient(U[end,:], Λ[end,:], t, Δx, da)
end

end