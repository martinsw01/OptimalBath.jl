abstract type ADGradient <: GradientType end

"""
    gradient!(ad::ADGradient, f, β)
"""
function gradient! end

"""
    get_objective(ad::ADGradient)
"""
function get_objective end

"""
    get_gradient(ad::ADGradient)
"""
function get_gradient end

"""
    update_and_get_bathymetry!(ad::ADGradient, swe_problem::PrimalSWEProblem, indices, β)
"""
function update_and_get_bathymetry! end

function extrapolate_β_to_full_domain(β, design_indices, bathymetry_length)
    full_β = zeros(eltype(β), bathymetry_length)
    full_β[design_indices] .= β
    return full_β
end

function to_real(x::Float64)
    return x
end

function to_real(x)
    return x.value
end

const NoReconstructionSolverSpec{P<:PrimalSWEProblem, B<:SolverBackend, TS<:TimeStepper, BS<:BathymetrySourceTerm} = SolverSpec{P, B, SolverOptions{NoReconstruction, TS, BS}}

# Automatic differentiation requires the solver to be constructed inside the AD framework, so we can only accept SolverSpecs.
function compute_objective_and_gradient!(G, β, p::PrimalSWESolver, o::Objectives, ad::ADGradient)
    @error "compute_objective_and_gradient! with ADGradient requires a SolverSpec, but got a PrimalSWESolver."
    throw(MethodError(compute_objective_and_gradient!, (G, β, p, o, ad)))
end

function compute_objective_and_gradient!(G, β, spec::NoReconstructionSolverSpec, objectives::Objectives, ad::ADGradient)
    function solve_and_compute_objective(β)
        db = extrapolate_β_to_full_domain(β, objectives.design_indices, length(spec.problem.initial_bathymetry))
        objective = objectives.regularization(β)

        solver = build_solver(spec, eltype(β))
        Δx = compute_Δx(solver)

        U_depth = similar(spec.problem.U0, Depth, Average, typeof(objective))
        to_depth!(U_depth, spec.problem.U0, spec.problem.initial_bathymetry)

        f_prev = typeof(objective)(sum(objective_density(objectives.interior_objective, U_depth, objectives.objective_indices)))
        function integrate_objective_one_step(U_n, t_n, Δt)
            adjusted_bathymetry = get_bathymetry(solver)
            to_depth!(U_depth, U_n, adjusted_bathymetry)
            f_next = sum(objective_density(objectives.interior_objective, U_depth, objectives.objective_indices))
            objective += interior_objective_increment(f_prev, f_next, Δt, Δx, spec.solver_options.timestepper)
            f_prev = f_next
        end

        integration_callback = create_callback(integrate_objective_one_step, solver)

        solve_primal(solver, db, integration_callback)

        densities = objective_density(objectives.terminal_objective,
                                           U_depth,
                                           objectives.objective_indices)

        objective += sum(densities) * Δx

        return objective
    end

    gradient!(ad, solve_and_compute_objective, β)
    objective = get_objective(ad)
    G .= get_gradient(ad)

    return objective
end

function interior_objective_increment(f_prev, f_next, Δt, Δx, ::ForwardEuler)
    return f_prev * Δt * Δx
end

function interior_objective_increment(f_prev, f_next, Δt, Δx, ::RK2)
    return 0.5 * (f_next + f_prev) * Δt * Δx
end

include("forward_ad_gradients.jl")
include("reverse_ad_gradients.jl")