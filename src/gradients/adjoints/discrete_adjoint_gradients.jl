module DiscreteAdjoints

using ..OptimalBath
import ..OptimalBath: compute_objective_and_gradient!, compute_gradient!, adjoint_solver

using ForwardDiff
using VolumeFluxes: XDIR

include("discrete_adjoint_swe.jl")

export DiscreteAdjointGradient

struct DiscreteAdjointGradient{PrimalSolver, AdjointSolver} <: AdjointGradient
    primal_solver::PrimalSolver
    adjoint_solver::AdjointSolver
    function DiscreteAdjointGradient(primal_solver::VolumeFluxesSolver)
        adjoint_solver = DiscreteAdjointSWE(primal_solver)
        return new{typeof(primal_solver), typeof(adjoint_solver)}(primal_solver, adjoint_solver)
    end
end

function adjoint_solver(::PrimalSWESolver{NoReconstruction, ForwardEuler}, da::DiscreteAdjointGradient)
    return da.adjoint_solver
end

function compute_gradient!(G, Λ, U::AverageDepthStates, β, t, Δx, objectives::Objectives, da::DiscreteAdjointSWE)
    ForwardDiff.gradient!(G, objectives.regularization, β)
    add_bottom_source_gradient_contribution!(G, Λ, U.U, t, objectives.design_indices, da)
end

@views function integrate_gradient(U, Λ, t, dir, grid::Grid)
    Δx = get_Δx(grid, dir)
    time_indices = 1:(length(t)-1)
    return sum(time_indices) do n
        Δt = t[n+1] - t[n]
        momentum(Λ[n+1], dir) * height(U[n]) * Δt
    end * 9.81 / Δx
end

function add_bottom_source_gradient_contribution!(G, Λ, U, t, design_indices, da::DiscreteAdjointSWE)
    add_bottom_source_gradient_contribution!(G, Λ, U, t, design_indices, da.grid)
end

signs(::XDIRT, ::Grid{2}) = (1.0, -1.0,  1.0, -1.0)
signs(::YDIRT, ::Grid{2}) = (1.0,  1.0, -1.0, -1.0)
offsets(::Grid{2}) = map(CartesianIndex{2}, ((0, 0), (-1, 0), (0, -1), (-1, -1)))

signs(::XDIRT, ::Grid{1}) = (1.0, -1.0)
offsets(::Grid{1}) = map(CartesianIndex{1}, (0, -1))

in_bounds(i, N) = i in CartesianIndices(N)

contribution_weight(::Grid{D}) where D = 2.0^(1-D)



@views function add_bottom_source_gradient_contribution!(G, Λ, U, t, design_indices::CartesianIndices, grid::Grid)
    N_corners = size(G)
    N_cells = N_corners .- 1

    w = contribution_weight(grid)

    for dir in directions(grid)
        for (s, offset) in zip(signs(dir, grid), offsets(grid))
            for i in CartesianIndices(G)
                ui = i + offset
                if in_bounds(ui, N_cells)
                    I = design_indices[ui]
                    G[i] += w * s * integrate_gradient(U[I, :], Λ[I, :], t, dir, grid)
                end
            end
        end
    end
end

function add_bottom_source_gradient_contribution!(G, Λ, U, t, design_indices::Colon, grid::Grid{1})
    add_bottom_source_gradient_contribution!(G, Λ, U, t, CartesianIndices(G), grid)
end
end