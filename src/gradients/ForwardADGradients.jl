using ForwardDiff, DiffResults

struct ForwardADGradient{Buffer} <: GradientType
    buffer::Buffer
    function ForwardADGradient(β::AbstractArray)
        value = first(β)
        gradient = similar(β)
        buffer = DiffResults.DiffResult(value, gradient)
        return new{typeof(buffer)}(buffer)
    end
    ForwardADGradient(parameters::Integer) = ForwardADGradient(Vector{Float64}(undef, parameters))
end


function compute_objective_and_gradient!(G, β, problem::SweOptimizationProblem, ad::ForwardADGradient)
    function solve_and_compute_objective(β)
        U, t, x = solve(problem, β)
        objective = compute_objective(U, t, x, β, problem.interior_objective, problem.terminal_objective, problem.parameter_objective)
        return objective
    end
    ForwardDiff.gradient!(ad.buffer, solve_and_compute_objective, β)
    objective = ad.buffer.value
    G .= ad.buffer.derivs[1]
    return objective
end
