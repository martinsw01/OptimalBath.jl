export AdjointSWE, solve_adjoint

abstract type AdjointSWE end

function compute_ghost_cell(U::State, ::Val{dir}=XDIR) where dir
    setindex(U, -momentum(U, dir), 1+dir)
end


"""
    solve_adjoint(Λ_end, U::AverageDepthStates, objectives::Objectives, b, t, Δx, da::AdjointSWE)
    solve_adjoint(Λ0, Ul::LeftDepthStates, Ur::RightDepthStates, objectives::Objectives, b, t, Δx, da::AdjointSWE)
"""
function solve_adjoint end


function add_objective_source!(Λ, U, Δt, Δx, objectives::Objectives, weight=1.0)
    indices = objectives.objective_indices
    objective = objectives.interior_objective
    scale = weight * Δt * prod(Δx)
    @views @. Λ[indices] += scale * objective_density_gradient(objective, U[indices])
end

function add_objective_source!(Λ, Ul, Ur, Δt, Δx, objectives::Objectives)
    add_objective_source!(Λ, Ul, Δt, Δx, objectives, 0.5)
    add_objective_source!(Λ, Ur, Δt, Δx, objectives, 0.5)
end


function time_frame(U, n)
    return selectdim(U, ndims(U), n)
end


"""
    primal_solver(da::AdjointSWE)
"""
function primal_solver end

function desingularize(h, da::AdjointSWE)
    return OptimalBath.desingularize(h, primal_solver(da))
end

function desingularize(h, p, da::AdjointSWE)
    return OptimalBath.desingularize(h, p, primal_solver(da))
end

function depth_cutoff(da::AdjointSWE)
    return da |> primal_solver |> OptimalBath.depth_cutoff
end