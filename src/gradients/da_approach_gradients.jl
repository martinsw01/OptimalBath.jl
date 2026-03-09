module DiscreteAdjoints

using ..OptimalBath
using ..OptimalBath: extrapolate_β_to_full_domain, initial_state
import ..OptimalBath: compute_objective_and_gradient!

using ForwardDiff: derivative
using SinFVM: XDIR

include("../discrete_adjoints.jl")

export DiscreteAdjointGradient, DiscreteAdjoint

struct DiscreteAdjointGradient <: GradientType end

function compute_objective_and_gradient!(G, β, primal_swe_problem::PrimalSWEProblem, objectives::Objectives, ::DiscreteAdjointGradient)
    δb = extrapolate_β_to_full_domain(β, objectives.design_indices, length(initial_state(primal_swe_problem).U))
    U, t, x = solve_primal(primal_swe_problem, β)
    U = unsafe_to_depth!(U, primal_swe_problem.initial_bathymetry .+ δb)

    @show size(t)

    Δx = x[2] - x[1]
    Λ0 = zero(U.U[:, end])
    Λ0[objectives.objective_indices] .+= objective_density_gradient(objectives.terminal_objective, U, objectives.objective_indices, lastindex(t)) * Δx

    Λ = solve_adjoint(Λ0, U, objectives, β, t, Δx, DiscreteAdjoint(primal_swe_problem))

    # display(height.(Λ))
    # display(momentum.(Λ))

    compute_gradient!(G, Λ, U.U, t, Δx, objectives, DiscreteAdjoint(primal_swe_problem))
    objective = compute_objective(U, t, x, β, objectives)
    return objective
end

function compute_gradient!(G, Λ, U, t, Δx, objectives::Objectives, da::DiscreteAdjoint)
    G .= zero(eltype(G))
    add_bottom_source_gradient_contribution!(G, Λ, U, t, Δx, objectives.design_indices, da)
    # add_adjoint_height_contribution!(G, Λ, t, objectives.design_indices, da)
end

# function compute_gradient!(G, Λ, U, t, Δx, objectives::Objectives, da::DiscreteAdjoint)
#     G .= zero(eltype(G))
#     G_copy = zero(G)
#     # add_bottom_source_gradient_contribution!(G, Λ, U, t, Δx, objectives.design_indices, da)
#     # @assert (≈)(G, G_copy, atol=1e-17)
#     # add_gradient_flux_contribution!(G, Λ, U, t, Δx, objectives.design_indices, da)
#     # @assert (≈)(G, G_copy, atol=1e-17)
#     # add_timestep_gradient_contribution!(G, Λ, U, t, Δx, objectives, da)
#     # @assert (≈)(G, G_copy, atol=1e-17)
#     # add_objective_gradient_contribution!(G, U, t, Δx, objectives, da)
#     # # @assert (≈)(G, G_copy, atol=1e-17)
#     # add_initial_condition_gradient_contribution!(G, Λ, objectives.design_indices, da)
# end

# function add_initial_condition_gradient_contribution!(G, Λ, design_indices::Colon, da::DiscreteAdjoint)
#     G[1:end-1] .-= 0.5 * height.(Λ[:, 1])
#     G[2:end] .-= 0.5 * height.(Λ[:, 1])
# end

# @warn "`add_adjoint_height_contribution!` Should not depend on Δt"
# function add_adjoint_height_contribution!(G, Λ, t, design_indices::Colon, da::DiscreteAdjoint)
#     # G[1:end-1] .-= 0.5 * sum(height.(Λ[:,1:end-1]) * diff(t), dims=2)
#     # G[2:end] .-= 0.5 * sum(height.(Λ[:,1:end-1]) * diff(t), dims=2)
#     G[1:end-1] .-= 0.5 * sum(height.(Λ[:,2:end]) * diff(t), dims=2)
#     G[2:end] .-= 0.5 * sum(height.(Λ[:,2:end]) * diff(t), dims=2)
#     # G[1:end-1] .-= 0.5 * sum(height, Λ[:,2:end], dims=2)
#     # G[2:end] .-= 0.5 * sum(height, Λ[:,2:end], dims=2)
# end

function integrate_gradient(U, Λ, t, Δx, ::DiscreteAdjoint)
    return sum(momentum.(Λ[2:end]) .* height.(U[1:end-1]) .* diff(t)) * 9.81 / Δx
end

function add_bottom_source_gradient_contribution!(G, Λ, U, t, Δx, design_indices::Colon, da::DiscreteAdjoint)
    N, M = size(U)
    G[1] += integrate_gradient(U[1,:], Λ[1,:], t, Δx, da) # sum(momentum.(Λ[1,2:end]) .* height.(U[1, 1:end-1]) .* diff(t)) * 9.81 / Δx
    for j in 2:N
        G[j] -= integrate_gradient(U[j-1,:], Λ[j-1,:], t, Δx, da)
        G[j] += integrate_gradient(U[j,:], Λ[j,:], t, Δx, da)
    end
    G[end] -= integrate_gradient(U[end,:], Λ[end,:], t, Δx, da)
end


# function integrate_objective_contribution(U, t, Δx, objectives::Objectives, da::DiscreteAdjoint)
#     int_obj = objectives.interior_objective
#     term_obj = objectives.terminal_objective
#     int_obj_contribution = -0.5 * sum(height.(objective_density_gradient.(int_obj, @view U[1:end-1])) .* diff(t)) * Δx
#     term_obj_contribution = 0.5 * height(objective_density_gradient(term_obj, U[end])) * Δx
#     return int_obj_contribution + term_obj_contribution
# end

# function add_objective_gradient_contribution!(G, U, t, Δx, objectives::Objectives, da::DiscreteAdjoint)
#     @assert objectives.objective_indices isa Colon
#     N, M = size(U)
#     G[1] += integrate_objective_contribution(U[1,:], t, Δx, objectives, da)
#     for j in 2:N
#         G[j] += integrate_objective_contribution(U[j-1,:], t, Δx, objectives, da)
#         G[j] += integrate_objective_contribution(U[j,:], t, Δx, objectives, da)
#     end
#     G[end] += integrate_objective_contribution(U[end,:], t, Δx, objectives, da)
# end


# function flux_left_deriv_center(Ul, Uc)
#     hc = height(Uc)
#     return derivative(hc) do h
#         eq = _numerical_flux.eq
#         U = State(h, momentum(Uc))
#         F = _numerical_flux(eq, Ul, U, XDIR)[1]
#         return F
#     end
# end

# function flux_right_deriv_center(Uc, Ur)
#     hc = height(Uc)
#     return derivative(hc) do h
#         eq = _numerical_flux.eq
#         U = State(h, momentum(Uc))
#         F = _numerical_flux(eq, U, Ur, XDIR)[1]
#         return F
#     end
# end

# @views function add_timestep_gradient_contribution!(G, Λ, U, t, Δx, objectives::Objectives, da::DiscreteAdjoint)
#     N, M = size(U)
#     Δt = t[end] - t[end-1]
#     J_final = compute_objective_step(U[:, end-1], Δx, Δt, objectives)
#     μ_final = compute_timestep_correction(Λ[:, end], U[:, end-1], U[:, end], Δt)
#     CFL = 0.25
#     for n in 1:M-1
#         Un = U[:, n]
#         Δt = t[n+1] - t[n]
#         i, source = compute_timestep_source(Λ[:,n+1], U[:,n+1], Un, μ_final, J_final, Δt, Δx, CFL, objectives, compute_timestep_derivative_h)
#         G[i] += source
#         G[i+1] += source
#     end
# end

# function add_gradient_flux_contribution!(G, Λ, U, t, Δx, design_indices::Colon, da::DiscreteAdjoint)
#     G_copy = copy(G)
#     compute_left_gradient_flux_contribution!(G, Λ, U, t, Δx, design_indices, da)
#     compute_inner_left_gradient_flux_contribution!(G, Λ, U, t, Δx, design_indices, da)
#     compute_interior_gradient_flux_contribution!(G, Λ, U, t, Δx, design_indices, da)
#     compute_inner_right_gradient_flux_contribution!(G, Λ, U, t, Δx, design_indices, da)
#     compute_right_gradient_flux_contribution!(G, Λ, U, t, Δx, design_indices, da)
#     # if !(≈)(G, G_copy, atol=1e-17)
#     #     @show G - G_copy
#     #     error("Gradient flux contribution modified G")
#     # end
# end

# function compute_left_gradient_flux_contribution!(G, Λ, U, t, Δx, design_indices::Colon, ::DiscreteAdjoint)
#     N, M = size(U)
#     A = @SMatrix [1 0; 0 -1]
#     for n in 1:M-1
#         Δt = t[n+1] - t[n]
#         U_ghost = compute_ghost_cell(U[1, n])
#         dF12_dh0 = A * flux_right_deriv_center(U_ghost, U[1, n])
#         dF12_dh1 = flux_left_deriv_center(U_ghost, U[1, n])
#         dF32_dh1 = flux_right_deriv_center(U[1, n], U[2, n])
#         @show dF12_dh0 + dF12_dh1 - dF32_dh1
#         @show dF32_dh1
#         G[1] -= 0.5*Δt/Δx * ((dF12_dh0 + dF12_dh1 - dF32_dh1)' * Λ[1, n+1] - dF32_dh1' * Λ[2, n+1])
#     end
# end

# function compute_right_gradient_flux_contribution!(G, Λ, U, t, Δx, design_indices::Colon, da::DiscreteAdjoint)
#     N, M = size(U)
#     A = @SMatrix [1 0; 0 -1]
#     for n in 1:M-1
#         Δt = t[n+1] - t[n]
#         U_ghost = compute_ghost_cell(U[end, n])
#         dFNm12_dhN = flux_left_deriv_center(U[end-1, n], U[end, n])
#         dFnp12_dhN = flux_right_deriv_center(U[end, n], U_ghost)
#         dFNp12_dhNp1 = A * flux_left_deriv_center(U[end, n], U_ghost)
#         G[end] -= 0.5*Δt/Δx * (dFNm12_dhN' * Λ[end-1, n+1] + (dFnp12_dhN + dFNp12_dhNp1 - dFNm12_dhN)' * Λ[end, n+1])
#     end
# end

# function compute_inner_left_gradient_flux_contribution!(G, Λ, U, t, Δx, design_indices::Colon, da::DiscreteAdjoint)
#     N, M = size(U)
#     A = @SMatrix [1 0; 0 -1]
#     for n in 1:M-1
#         Δt = t[n+1] - t[n]
#         U1, U2, U3 = U[1:3, n]
#         Λ1, Λ2, Λ3 = Λ[1:3, n+1]
#         U_ghost = compute_ghost_cell(U[1, n])
#         dF12_dh0 = A * flux_right_deriv_center(U_ghost, U1)
#         dF12_dh1 = flux_left_deriv_center(U_ghost, U1)
#         dF32_dh1 = flux_right_deriv_center(U1, U2)
#         dF32_dh2 = flux_left_deriv_center(U1, U2)
#         dF52_dh2 = flux_right_deriv_center(U2, U3)
#         G[2] -= 0.5*Δt/Δx * ((dF12_dh0 + dF12_dh1)' * Λ1 + (dF32_dh1 + dF32_dh2)' * (Λ2 - Λ1) + dF52_dh2' * (Λ3 - Λ2))
#     end
# end

# function compute_inner_right_gradient_flux_contribution!(G, Λ, U, t, Δx, design_indices::Colon, da::DiscreteAdjoint)
#     N, M = size(U)
#     A = @SMatrix [1 0; 0 -1]
#     for n in 1:M-1
#         Δt = t[n+1] - t[n]
#         U_Nm2, U_Nm1, U_N = U[end-2:end, n]
#         Λ_Nm2, Λ_Nm1, Λ_N = Λ[end-2:end, n+1]
#         U_ghost = compute_ghost_cell(U_N)
#         dFNm32_dhNm1 = flux_left_deriv_center(U_Nm2, U_Nm1)
#         dFNm12_dhNm1 = flux_right_deriv_center(U_Nm1, U_N)
#         dFNm12_dhN = flux_left_deriv_center(U_Nm1, U_N)
#         dFNp12_dhN = flux_right_deriv_center(U_N, U_ghost)
#         dFNp12_dhNp1 = A * flux_left_deriv_center(U_N, U_ghost)
#         G[end-1] -= 0.5*Δt/Δx * (dFNm32_dhNm1' * (Λ_Nm2 - Λ_Nm1)
#                                 + (dFNm12_dhNm1 + dFNm12_dhN)' * (Λ_Nm1 - Λ_N)
#                                 + (dFNp12_dhN + dFNp12_dhNp1)' * Λ_N)
#     end
# end

# function compute_interior_gradient_flux_contribution!(G, Λ, U, t, Δx, design_indices::Colon, ::DiscreteAdjoint)
#     N, M = size(U)
#     for n in 1:M-1
#         Δt = t[n+1] - t[n]
#         for j in 3:N-2
#             for i in j:j+1
#                 dFldh = flux_left_deriv_center(U[i-1, n], U[i, n])
#                 dFrdh = flux_right_deriv_center(U[i, n], U[i+1, n])
#                 G[i] += 0.5*Δt/Δx * (dFrdh - dFldh)' * (Λ[i, n+1] - Λ[i-1, n+1])
#             end
#         end
#     end
# end

end