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

function extrapolate_β_to_full_domain(β, design_indices, N)
    full_β = zeros(eltype(β), N+1)
    full_β[design_indices] .= β
    return full_β
end

function to_real(x::Float64)
    return x
end

function to_real(x)
    return x.value
end

function compute_objective_and_gradient!(G, β, primal_swe_problem::PrimalSWEProblem{NoReconstruction, TimeStepperType}, objectives::Objectives, ad::ADGradient) where TimeStepperType
    function solve_and_compute_objective(β)
        db = extrapolate_β_to_full_domain(β, objectives.design_indices, length(initial_state(primal_swe_problem).U))
        adjusted_bathymetry = db .+ primal_swe_problem.initial_bathymetry

        objective = objectives.regularization(β)

        Δx = compute_Δx(primal_swe_problem)
        U_depth = to_depth(initial_state(primal_swe_problem), adjusted_bathymetry)
        f_prev = typeof(objective)(sum(to_real, objective_density(objectives.interior_objective, U_depth, objectives.objective_indices)))

        function integrate_objective_one_step(U_n, t_n, Δt)
            to_depth!(U_depth, U_n, adjusted_bathymetry)
            f_next = sum(objective_density(objectives.interior_objective, U_depth, objectives.objective_indices))
            # objective += 0.5 * (f_next + f_prev) * Δt * Δx
            objective += interior_objective_increment(f_prev, f_next, Δt, Δx, TimeStepperType)
            f_prev = f_next
        end

        integration_callback = create_callback(integrate_objective_one_step, primal_swe_problem)

        solve_primal(primal_swe_problem, db, integration_callback)

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

function interior_objective_increment(f_prev, f_next, Δt, Δx, ::Type{ForwardEuler})
    return f_prev * Δt * Δx
end

function interior_objective_increment(f_prev, f_next, Δt, Δx, ::Type{RK2})
    return 0.5 * (f_next + f_prev) * Δt * Δx
end

include("ForwardADGradients.jl")
include("ReverseADGradients.jl")