export ContinuousAdjointSWE

struct ContinuousAdjointSWE{PrimalSolver<:PrimalSWESolver, GridT} <: AdjointSWE
    primal::PrimalSolver
    grid::GridT
    function ContinuousAdjointSWE(primal::PrimalSWESolver)
        grid = get_grid(primal)
        return new{typeof(primal), typeof(grid)}(primal, grid)
    end
end

include("numerical_adjoint_fluxes.jl")
include("flux_jacobian_sources.jl")
include("bottom_source_terms.jl")

function solve_adjoint(Λ_end, U::AverageDepthStates, objectives::Objectives, b, t, da::ContinuousAdjointSWE)
    grid = da.grid
    Δx = grid.Δx

    M = length(t)
    Λ = similar(U.U)
    time_frame(Λ, M) .= Λ_end

    for n in M-1:-1:1
        Δt = t[n+1] - t[n]
        time_frame(Λ, n) .= time_frame(Λ, n+1)
        for dir in directions(grid)
            add_adjoint_flux!(Λ, U.U, n, Δt, dir, grid, da)
            add_flux_spatial_derivative!(Λ, U.U, n, Δt, dir, grid, da)
            add_bottom_source!(Λ, n, Δt, b, dir, grid, da)
        end
        add_objective_source!(Λ, U.U, n, Δt, Δx, objectives, da)
    end
    return Λ
end

function add_objective_source!(Λ, U, n, Δt, Δx, objectives, ::ContinuousAdjointSWE)
    OptimalBath.add_objective_source!(time_frame(Λ, n), time_frame(U, n), Δt, Δx, objectives)
end


function primal_solver(da::ContinuousAdjointSWE)
    return da.primal
end