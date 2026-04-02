export AdjointSWE, solve_adjoint

abstract type AdjointSWE end

struct Grid{Dims, StepSizes, TotalCells}
    Δx::StepSizes
    N::TotalCells
    function Grid(Δx, N)
        @assert length(Δx) == length(N)
        @assert eltype(N) <: Integer
        new{length(Δx), typeof(Δx), typeof(N)}(Δx, N)
    end
end

function interior_cell_indices(grid::Grid{1})
    return 2:grid.N[1]-1
end

function for_each_interior_directional_stencil(f, dir, grid::Grid{1})
    for i in interior_cell_indices(grid)
        f(i-1, i, i+1)
    end
end

function for_each_cell(f, grid::Grid{1})
    indices = 1:grid.N[1]
    for i in indices
        f(i)
    end
end

function for_each_left_boundary_directional_stencil(f, dir, ::Grid{1})
    f(1, 2)
end

function for_each_right_boundary_directional_stencil(f, dir, grid::Grid{1})
    N = grid.N[1]
    f(N-1, N)
end

function compute_ghost_cell(U::State{1}, dir)
    return typeof(U)(U[1], -U[2])
end

function compute_ghost_cell(U)
    typeof(U)(U[1], -U[2])
end


"""
    solve_adjoint(Λ_end, U::AverageDepthStates, objectives::Objectives, b, t, Δx, da::AdjointSWE)
    solve_adjoint(Λ0, Ul::LeftDepthStates, Ur::RightDepthStates, objectives::Objectives, b, t, Δx, da::AdjointSWE)
"""
function solve_adjoint end


function add_objective_source!(Λ, U, Δt, Δx, objectives::Objectives, weight=1.0)
    indices = objectives.objective_indices
    objective = objectives.interior_objective
    Λ[indices] .+= weight * objective_density_gradient.(objective, @view U[indices]) * Δt * Δx
end

function add_objective_source!(Λ, Ul, Ur, Δt, Δx, objectives::Objectives)
    add_objective_source!(Λ, Ul, Δt, Δx, objectives, 0.5)
    add_objective_source!(Λ, Ur, Δt, Δx, objectives, 0.5)
end