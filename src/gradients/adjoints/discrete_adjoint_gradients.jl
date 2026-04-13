module DiscreteAdjoints

using ..OptimalBath
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
    add_bottom_source_gradient_contribution!(G, Λ, U.U, t, objectives.design_indices, da)
end

@views function integrate_gradient(U, Λ, t, dir, grid::Grid)
    Δx = grid.Δx[dir]
    return sum(momentum.(Λ[2:end], dir) .* height.(U[1:end-1]) .* diff(t)) * 9.81 / Δx
end

function add_bottom_source_gradient_contribution!(G, Λ, U, t, design_indices, da::DiscreteAdjointSWE)
    add_bottom_source_gradient_contribution!(G, Λ, U, t, design_indices, da.grid)
end

@views function add_bottom_source_gradient_contribution!(G, Λ, U, t, design_indices::Colon, grid::Grid{1})
    N, M = size(U)
    G[1] += integrate_gradient(U[1,:], Λ[1,:], t, 1, grid)
    for j in 2:N
        G[j] -= integrate_gradient(U[j-1,:], Λ[j-1,:], t, 1, grid)
        G[j] += integrate_gradient(U[j,:], Λ[j,:], t, 1, grid)
    end
    G[end] -= integrate_gradient(U[end,:], Λ[end,:], t, 1, grid)
end

@views function add_bottom_source_gradient_contribution!(G, Λ, U, t, design_indices::CartesianIndices, grid::Grid{2})
    @assert all(iszero, G)
    for i in 2:grid.N[1]
        for j in 2:grid.N[2]
            G[i, j] -= 0.5*integrate_gradient(U[i, j, :], Λ[i, j, :], t, 1, grid)
            G[i, j] -= 0.5*integrate_gradient(U[i-1, j, :], Λ[i-1, j, :], t, 1, grid)
            G[i, j] += 0.5*integrate_gradient(U[i, j-1, :], Λ[i, j-1, :], t, 1, grid)
            G[i, j] += 0.5*integrate_gradient(U[i-1, j-1, :], Λ[i-1, j-1, :], t, 1, grid)
        end
    end
    for i in 2:grid.N[1]
        for j in 2:grid.N[2]
            G[i, j] -= 0.5*integrate_gradient(U[i, j, :], Λ[i, j, :], t, 2, grid)
            G[i, j] -= 0.5*integrate_gradient(U[i, j-1, :], Λ[i, j-1, :], t, 2, grid)
            G[i, j] += 0.5*integrate_gradient(U[i-1, j, :], Λ[i-1, j, :], t, 2, grid)
            G[i, j] += 0.5*integrate_gradient(U[i-1, j-1, :], Λ[i-1, j-1, :], t, 2, grid)
        end
    end
    for i in 2:grid.N[1]
        j=1
        G[i, j] -= 0.5*integrate_gradient(U[i, j, :], Λ[i, j, :], t, 2, grid)
        G[i, j] -= 0.5*integrate_gradient(U[i-1, j, :], Λ[i-1, j, :], t, 2, grid)
        j = grid.N[2] + 1
        G[i, j] += 0.5*integrate_gradient(U[i-1, j-1, :], Λ[i-1, j-1, :], t, 2, grid)
        G[i, j] += 0.5*integrate_gradient(U[i, j-1, :], Λ[i, j-1, :], t, 2, grid)
    end
    for j in 2:grid.N[2]
        i = 1
        G[i, j] -= 0.5*integrate_gradient(U[i, j, :], Λ[i, j, :], t, 1, grid)
        G[i, j] -= 0.5*integrate_gradient(U[i, j-1, :], Λ[i, j-1, :], t, 1, grid)
        i = grid.N[1] + 1
        G[i, j] += 0.5*integrate_gradient(U[i-1, j, :], Λ[i-1, j, :], t, 1, grid)
        G[i, j] += 0.5*integrate_gradient(U[i-1, j-1, :], Λ[i-1, j-1, :], t, 1, grid)
    end
    G[1, 1] -= 0.5*integrate_gradient(U[1, 1, :], Λ[1, 1, :], t, 1, grid)
    G[1, 1] -= 0.5*integrate_gradient(U[1, 1, :], Λ[1, 1, :], t, 2, grid)

    G[1, grid.N[2]+1] -= 0.5*integrate_gradient(U[1, grid.N[2], :], Λ[1, grid.N[2], :], t, 1, grid)
    G[1, grid.N[2]+1] += 0.5*integrate_gradient(U[1, grid.N[2], :], Λ[1, grid.N[2], :], t, 2, grid)

    G[grid.N[1]+1, 1] += 0.5*integrate_gradient(U[grid.N[1], 1, :], Λ[grid.N[1], 1, :], t, 1, grid)
    G[grid.N[1]+1, 1] -= 0.5*integrate_gradient(U[grid.N[1], 1, :], Λ[grid.N[1], 1, :], t, 2, grid)

    G[grid.N[1]+1, grid.N[2]+1] += 0.5*integrate_gradient(U[grid.N[1], grid.N[2], :], Λ[grid.N[1], grid.N[2], :], t, 1, grid)
    G[grid.N[1]+1, grid.N[2]+1] += 0.5*integrate_gradient(U[grid.N[1], grid.N[2], :], Λ[grid.N[1], grid.N[2], :], t, 2, grid)
end
end