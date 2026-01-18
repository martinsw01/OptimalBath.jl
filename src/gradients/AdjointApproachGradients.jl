export AdjointApproachGradient, compute_gradient!

struct AdjointApproachGradient{BathymetryBuffer} <: GradientType 
    bathymetry_buffer::BathymetryBuffer
    function AdjointApproachGradient(bathymetry)
        buffer = copy(bathymetry)
        return new{typeof(buffer)}(buffer)
    end
end

function compute_objective_and_gradient!(G, β, primal_swe_problem::PrimalSWEProblem, objectives::Objectives, aag::AdjointApproachGradient)
    update_bathymetry!(aag, objectives.design_indices, β)
    U, t, x = solve_primal(primal_swe_problem, aag.bathymetry_buffer)

    dJdU = zero(U)
    dJdU[objectives.objective_indices, :] .= objective_density_gradient.(objectives.interior_objective, U[objectives.objective_indices, :])
    
    Δx = x[2] - x[1]
    Λ0 = objective_density_gradient.(objectives.terminal_objective, U[:, end])
    Λ = solve_adjoint(Λ0, U, dJdU, aag.bathymetry_buffer, t, Δx)
    
    compute_gradient!(G, Λ, U, t, Δx, objectives.design_indices)
    objective = compute_objective(U, t, x, β, objectives)
end

function update_bathymetry!(aag::AdjointApproachGradient, indices, β)
    aag.bathymetry_buffer[indices] .= β
end

# U, t, cell_faces, animation_duration=t[end]

function _height(U)
    return U[1]
end

function _adjoint_momentum(Λ)
    return Λ[2]
end

function compute_gradient!(G, Λ, U, t, Δx, design_indices::Colon)
    G .= zero(eltype(G))
    N, M = size(U)
    for j in 1:N
        temp_integral = integrate(U, Λ, t, j)
        # temp_integral = sum(_height.(U[j,2:end]) .* _adjoint_momentum.(Λ[j, 2:end]) .* diff(t)) * g
        G[j] -= temp_integral
        G[j+1] += temp_integral
    end
end


function integrate(U, Λ, t, j)
    g = 9.81
    return sum(_height.(U[j,2:end]) .* _adjoint_momentum.(Λ[j, 2:end]) .* diff(t)) * g
end

function compute_gradient!(G, Λ, U, t, Δx, design_indices)
    N, M = size(U)
    g = 9.81
    for (i, j) in enumerate(design_indices)
        if j == 1
            G[i] = -integrate(U, Λ, t, j)
        elseif j == N + 1
            G[i] = integrate(U, Λ, t, j - 1)
        else
            G[i] = integrate(U, Λ, t, j - 1) - integrate(U, Λ, t, j)
        end
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