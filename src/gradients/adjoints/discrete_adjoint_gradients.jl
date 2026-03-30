module DiscreteAdjoints

using ..OptimalBath
using ..OptimalBath: extrapolate_β_to_full_domain, initial_state
import ..OptimalBath: compute_objective_and_gradient!, compute_gradient!, adjoint_solver

using ForwardDiff
using VolumeFluxes: XDIR

include("discrete_adjoint_swe.jl")

export DiscreteAdjointGradient

struct DiscreteAdjointGradient <: AdjointGradient end

function adjoint_solver(primal_swe_problem::PrimalSWESolver{NoReconstruction, ForwardEuler, BS}, ::DiscreteAdjointGradient) where BS
    return DiscreteAdjointSWE(primal_swe_problem)
end

function compute_gradient!(G, Λ, U::AverageDepthStates, β, t, Δx, objectives::Objectives, da::DiscreteAdjointSWE)
    ForwardDiff.gradient!(G, objectives.regularization, β)
    add_bottom_source_gradient_contribution!(G, Λ, U.U, t, Δx, objectives.design_indices, da)
end

@views function integrate_gradient(U, Λ, t, Δx, ::DiscreteAdjointSWE)
    return sum(momentum.(Λ[2:end]) .* height.(U[1:end-1]) .* diff(t)) * 9.81 / Δx
end

@views function add_bottom_source_gradient_contribution!(G, Λ, U, t, Δx, design_indices::Colon, da::DiscreteAdjointSWE)
    N, M = size(U)
    G[1] += integrate_gradient(U[1,:], Λ[1,:], t, Δx, da)
    for j in 2:N
        G[j] -= integrate_gradient(U[j-1,:], Λ[j-1,:], t, Δx, da)
        G[j] += integrate_gradient(U[j,:], Λ[j,:], t, Δx, da)
    end
    G[end] -= integrate_gradient(U[end,:], Λ[end,:], t, Δx, da)
end

end