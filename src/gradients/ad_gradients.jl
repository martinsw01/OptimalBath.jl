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


function compute_objective_and_gradient!(G, β, primal_swe_problem::PrimalSWEProblem, objectives::Objectives, ad::ADGradient)
    function solve_and_compute_objective(β)
        b = update_and_get_bathymetry!(ad, primal_swe_problem, objectives.design_indices, β)

        objective = objectives.regularization(β)

        Δx = compute_Δx(primal_swe_problem)
        U_depth = to_depth(initial_state(primal_swe_problem), b)
        f_prev = sum(objective_density(objectives.interior_objective, U_depth, objectives.objective_indices))

        function integrate_objective_one_step(U_n, t_n, Δt)
            to_depth!(U_depth, U_n, b)
            f_next = sum(objective_density(objectives.interior_objective, U_depth, objectives.objective_indices))
            objective += 0.5 * (f_next + f_prev) * Δt * Δx
            f_prev = f_next
        end

        integration_callback = create_callback(integrate_objective_one_step, primal_swe_problem)

        solve_primal(primal_swe_problem, b, integration_callback)

        return objective
    end

    gradient!(ad, solve_and_compute_objective, β)
    objective = get_objective(ad)
    G .= get_gradient(ad)

    return objective
end

include("ForwardADGradients.jl")
# include("ReverseADGradients.jl")