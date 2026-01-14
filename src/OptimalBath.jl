module OptimalBath

using Optim

include("solver.jl")
include("AdjointSWE.jl")
include("objectives.jl")
include("gradients/Gradients.jl")

function solve(problem::SweOptimizationProblem, gradient::GradientType, initial)

    fg! = Optim.only_fg!() do F, G, β
        objective = compute_objective_and_gradient!(G, β, problem, gradient)
        return objective
    end
    
    res = Optim.optimize(fg!, initial, Optim.BFGS(), Optim.Options(iterations = 6, show_trace = true))
    
    return res
end


# function compute_objective_and_gradient(β, problem, gradient)
#     G = similar(β)
#     obj = compute_objective_and_gradient!(G, β, problem, gradient)
#     return obj, G
# end

# N = 30


# ω = π * sqrt(9.81)
# ε = 0.05
# function height(x, t)
#         return 1.0 + ε * cos(π * x) * cos(ω * t)
#     end

# function momentum(x, t)
#     return sqrt(9.81) * ε * sin(π * x) * sin(ω * t)
# end

# problem = SweOptimizationProblem(
#     N,
#     # (x) -> [1., 0.],
#     # (x) -> [1-0.1x, 0.1x],
#     (x) -> [height(x, 0.0), momentum(x, 0.0)],
#     1.;
#     # interior_objective_density = (U, β1, β2) -> U[2] ^ 2,
#     # terminal_objective_density = (U, β1, β2) -> sum(U .^ 2),
#     interior_objective = Energy(),
#     # terminal_objective_density = (U, β1, β2) -> 0.5 * sum(U .^ 2)
#     )
#     # parameter_objective = (β) -> sum(β .^ 2))

# initial = zeros(N+1)

# solve(problem, ForwardADGradient(initial), initial)
# solve(problem, AdjointApproachGradient(), initial);


end
