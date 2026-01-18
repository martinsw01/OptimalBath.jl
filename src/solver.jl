export solve

function no_regularization(β)
    return zero(eltype(β))
end

struct SweOptimizationProblem
    gradient_data::Objectives
    initial_bathymetry
    primal_swe_problem::PrimalSWEProblem
    function SweOptimizationProblem(primal_swe_problem::PrimalSWEProblem;
                                    interior_objective = NoObjective(),
                                    terminal_objective = NoObjective(),
                                    regularization = no_regularization,
                                    design_indices = Colon(),
                                    objective_indices = Colon(),
                                    initial_bathymetry=zero(primal_swe_problem.initial_bathymetry))
        # @assert length(initial_bathymetry) == N + 1 "Bathymetry must have length N+1=$(N + 1) ≠ $(length(initial_bathymetry))"
        return new(
            Objectives(interior_objective, terminal_objective, regularization, design_indices, objective_indices),
            initial_bathymetry,
            primal_swe_problem
        )
    end       
end

function solve(problem::SweOptimizationProblem, gradient::GradientType, initial)

    fg! = Optim.only_fg!() do F, G, β
        objective = compute_objective_and_gradient!(G, β, problem.primal_swe_problem, problem.gradient_data, gradient)
        return objective
    end
    
    res = Optim.optimize(fg!, initial, Optim.BFGS(), Optim.Options(iterations = 6, show_trace = true))
    
    return res
end