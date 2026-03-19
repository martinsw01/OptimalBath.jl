export AdjointSWE, solve_adjoint

abstract type AdjointSWE end

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