export AdjointApproachGradient, ContinuousAdjointGradient, DiscreteAdjointGradient

include("adjoint_swe.jl")
include("continuous_adjoint/continuous_adjoint_swe.jl")
include("discrete_adjoint_swe.jl")

struct AdjointApproachGradient{AdjointSolver<:AdjointSWE} <: GradientType
    adjoint_solver::AdjointSolver
end

function ContinuousAdjointGradient(primal_solver::PrimalSWESolver; adjoint_options...)
    return AdjointApproachGradient(ContinuousAdjointSWE(primal_solver; adjoint_options...))
end

function DiscreteAdjointGradient(primal_solver::VolumeFluxesSolver)
    return AdjointApproachGradient(DiscreteAdjointSWE(primal_solver))
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

include("adjoint_approach_gradient_computation.jl")