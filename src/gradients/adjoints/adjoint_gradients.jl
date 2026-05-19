export AdjointGradient

include("adjoint_swe.jl")

struct AdjointApproachGradient{AdjointSolver<:AdjointSWE} <: GradientType
    adjoint_solver::AdjointSolver
end

function time_frame(U, n)
    return selectdim(U, ndims(U), n)
end

function compute_objective_and_gradient!(G, β, solver::PrimalSWESolver{NoReconstruction, TS, BS}, objectives::Objectives, ag::AdjointApproachGradient) where {TS, BS}
    δb = extrapolate_β_to_full_domain(β, objectives.design_indices, size(get_bathymetry(solver)))
    
    U, t, x = solve_primal(solver, δb)
    adjusted_bathymetry = get_bathymetry(solver)
    U = unsafe_to_depth!(U, adjusted_bathymetry)

    Δx = compute_Δx(solver)
    Λ_end = zero(time_frame(U.U, 1))
    Λ_end[objectives.objective_indices] .+= objective_density_gradient(objectives.terminal_objective, U, objectives.objective_indices, lastindex(t)) .* prod(Δx)

    adjoint = ag.adjoint_solver
    Λ = solve_adjoint(Λ_end, U, objectives, adjusted_bathymetry, t, adjoint)

    regularization_gradient!(G, β, objectives.regularization)
    adjoint_based_gradient!(G, Λ, U, t, objectives.design_indices, get_grid(solver), ag)
    objective = compute_objective(U, t, Δx, β, objectives, TS)
    return objective
end

function compute_objective_and_gradient!(G, β, solver::PrimalSWESolver{R, TS, BS}, objectives::Objectives, ag::AdjointApproachGradient) where {R<:LinearReconstruction, TS, BS}
    δb = extrapolate_β_to_full_domain(β, objectives.design_indices, size(get_bathymetry(solver)))

    (Ul, Ur), t, x = solve_primal(solver, δb)
    adjusted_bathymetry = get_bathymetry(solver)

    Δx = compute_Δx(solver)
    Λ_end = zero(time_frame(Ul.U, 1))
    Λ_end[objectives.objective_indices] .+= 0.5 * objective_density_gradient(objectives.terminal_objective, Ul, objectives.objective_indices, lastindex(t)) .* prod(Δx)
    Λ_end[objectives.objective_indices] .+= 0.5 * objective_density_gradient(objectives.terminal_objective, Ur, objectives.objective_indices, lastindex(t)) .* prod(Δx)

    adjoint = ag.adjoint_solver
    Λ = solve_adjoint(Λ_end, Ul, Ur, objectives, adjusted_bathymetry, t, adjoint)

    regularization_gradient!(G, β, objectives.regularization)
    adjoint_based_gradient!(G, Λ, Ul, Ur, t, objectives.design_indices, get_grid(solver), ag)
    objective = compute_objective(Ul, Ur, t, Δx, β, objectives, TS)
    return objective
end


function adjoint_based_gradient!(G, Λ, U::States{Average, Depth}, t, design_indices, grid::Grid, ag::AdjointApproachGradient)
    _adjoint_based_gradient!(G, Λ, U.U, t, design_indices, grid, ag)
end

function adjoint_based_gradient!(G, Λ, Ul::States{Left, Depth}, Ur::States{Right, Depth}, t, design_indices, grid::Grid, ag::AdjointApproachGradient)
    _adjoint_based_gradient!(G, Λ, 0.5 .* (Ul.U .+ Ur.U), t, design_indices, grid, ag)
end

include("discrete_adjoint_gradients.jl")
include("continuous_adjoint_gradients.jl")

function _adjoint_based_gradient!(G, Λ, U, t, design_indices::Colon, grid::Grid{1}, ca::AdjointApproachGradient)
    _adjoint_based_gradient!(G, Λ, U, t, eachindex(G), grid, ca)
end

function _adjoint_based_gradient!(G, Λ, U, t, design_indices::AbstractVector{Int}, grid::Grid{1}, ca::AdjointApproachGradient)
    N, M = size(U)
    for (i, j) in enumerate(design_indices)
        if j == 1
            G[i] += integrate_gradient(U[j,:], Λ[j,:], t, XDIR, grid)
        elseif j == N + 1
            G[i] -= integrate_gradient(U[j-1,:], Λ[j-1,:], t, XDIR, grid)
        else
            G[i] -= integrate_gradient(U[j-1,:], Λ[j-1,:], t, XDIR, grid)
            G[i] += integrate_gradient(U[j,:], Λ[j,:], t, XDIR, grid)
        end
    end
end

function _adjoint_based_gradient!(G, Λ, U, t, design_indices::AbstractUnitRange{<:Integer}, grid::Grid{1}, ca::AdjointApproachGradient)
    start = first(design_indices)
    stop = last(design_indices) - 1
    for j in start:stop
        temp_integral = integrate_gradient(U[j,:], Λ[j,:], t, XDIR, grid)
        G[j] += temp_integral
        G[j+1] -= temp_integral
    end
end

function integrate_gradient(U, Λ, t, dir, grid::Grid)
    time_indices = firstindex(t):lastindex(t)-1

    Δx = get_Δx(grid, dir)
    gradient = zero(eltype(eltype(U)))
    @inbounds @simd for n in time_indices
        Δt = t[n+1] - t[n]
        gradient += momentum(Λ[n+1], dir) * height(U[n]) * Δt
    end
    return gradient * 9.81 / Δx
end

signs(::XDIRT, ::Grid{2}) = (1.0, -1.0,  1.0, -1.0)
signs(::YDIRT, ::Grid{2}) = (1.0,  1.0, -1.0, -1.0)
offsets(::Grid{2}) = map(CartesianIndex{2}, ((0, 0), (-1, 0), (0, -1), (-1, -1)))

signs(::XDIRT, ::Grid{1}) = (1.0, -1.0)
offsets(::Grid{1}) = map(CartesianIndex{1}, (0, -1))

in_bounds(i, N) = i in CartesianIndices(N)

contribution_weight(::Grid{D}) where D = 2.0^(1-D)

@views function _adjoint_based_gradient!(G, Λ, U, t, design_indices::CartesianIndices, grid::Grid, ::AdjointApproachGradient)
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