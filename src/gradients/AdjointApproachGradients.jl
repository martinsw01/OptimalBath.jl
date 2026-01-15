using StaticArrays

struct AdjointApproachGradient <: GradientType end

function compute_objective_and_gradient!(G, β, problem::SweOptimizationProblem, ::AdjointApproachGradient)
    U, t, x = solve_primal(problem, β)

    dJdU = objective_density_gradient.(Volume(), problem.interior_objective, U)
    Δx = x[2] - x[1]
    Λ0 = objective_density_gradient.(Terminal(), problem.terminal_objective, U[:, end])
    Λ = solve_adjoint(Λ0, U, dJdU, β, t, Δx)
    
    _compute_gradient!(G, Λ, U, t, Δx)
    objective = compute_objective(U, t, x, β, problem.interior_objective, problem.terminal_objective, problem.parameter_objective)
    # return objective
    # # return plot_solution(U)
    # return animate_solution(U, t, x, 4)
    # return animate_solution(Λ, t, x, 4)
    # return plot_solution(Λ)
end

# U, t, cell_faces, animation_duration=t[end]

function _height(U)
    return U[1]
end

function _adjoint_momentum(Λ)
    return Λ[2]
end

function _compute_gradient!(G, Λ, U, t, Δx)
    G .= zero(eltype(G))
    N, M = size(U)
    g = 9.81
    for j in 1:N
        temp_integral = sum(_height.(U[j,2:end]) .* _adjoint_momentum.(Λ[j, 2:end]) .* diff(t)) * g
        G[j] -= temp_integral
        G[j+1] += temp_integral
    end
end





# function integrate_time(U, Λ, t)
#     integral = zero(eltype(eltype(U)))

#     for j in 1:length(U) - 1
#         Δt = t[j+1] - t[j]
#         (h1, _), (h2, _) = U[j:j+1]
#         (_, λ21), (_, λ22) = Λ[j:j+1]
#         integral += 0.5 * (h1*λ21 + h2*λ22) * Δt
#     end
#     return integral
# end

# function compute_gradient(U, Λ, t, β, parameter_objective)

#     gradient = zero(β)
#     g = 9.81


    
#     for j in axes(U, 1)
#         gradient[j:j+1] .+= g * integrate_time(U[j, :], Λ[j, :], t) .* [-1, 1]
#     end

#     gradient .+= ForwardDiff.gradient(parameter_objective, β)

#     return gradient
# end

# # import Base: one
# # function one(::Type{SVector{N, T}}) where {N, T}
# #     return SVector{N, T}(fill(one(T), N))
# # end